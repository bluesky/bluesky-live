import collections
import collections.abc
from datetime import datetime
import functools
import threading

import event_model

from .document import Start, Stop, Descriptor
from .event import EmitterGroup, Event
from .conversion import documents_to_xarray, documents_to_xarray_config
from ._utils import (
    coerce_dask,
    discover_handlers,
    parse_transforms,
    parse_handler_registry,
)


def _write_locked(method):
    "This is used by DocumentCache to holds it write_lock during method calls."

    @functools.wraps(method)
    def inner(self, *args, **kwargs):
        with self.write_lock:
            return method(self, *args, **kwargs)

    return inner


class DocumentCache(event_model.SingleRunDocumentRouter):
    """
    An in-memory cache of documents from one Run.

    Examples
    --------

    >>> dc = DocumentCache()
    >>> dc('start', {'time': ..., 'uid': ...})
    >>> list(dc)
    [('start', {'time': ..., 'uid': ...})]
    """

    def __init__(self):
        self.write_lock = threading.RLock()
        self.descriptors = {}
        self.resources = {}
        self.event_pages = collections.defaultdict(list)
        self.datum_pages_by_resource = collections.defaultdict(list)
        self.resource_uid_by_datum_id = {}
        self.start_doc = None
        self.stop_doc = None
        self.events = EmitterGroup(
            source=self,
            started=Event,
            new_stream=Event,
            new_data=Event,
            completed=Event,
            new_doc=Event,
        )
        # maps stream name to list of descriptors
        self._streams = {}
        # list of (name, doc) pairs in the order they were consumed
        self._ordered = []
        super().__init__()

    def __iter__(self):
        """Yield (name, doc) pairs in the order they were consumed."""
        yield from self._ordered

    @property
    def streams(self):
        return self._streams

    @_write_locked
    def start(self, doc):
        self.start_doc = doc
        self._ordered.append(("start", doc))
        self.events.new_doc(name="start", doc=doc)
        self.events.started()
        super().start(doc)

    @_write_locked
    def stop(self, doc):
        self._ordered.append(("stop", doc))
        self.stop_doc = doc
        self.events.new_doc(name="stop", doc=doc)
        self.events.completed()
        super().stop(doc)

    @_write_locked
    def event_page(self, doc):
        self._ordered.append(("event_page", doc))
        self.event_pages[doc["descriptor"]].append(doc)
        self.events.new_doc(name="event_page", doc=doc)
        self.events.new_data(
            updated={self.descriptors[doc["descriptor"]]["name"]: len(doc["seq_num"])}
        )
        super().event_page(doc)

    @_write_locked
    def datum_page(self, doc):
        self._ordered.append(("datum_page", doc))
        self.datum_pages_by_resource[doc["resource"]].append(doc)
        for datum_id in doc["datum_id"]:
            self.resource_uid_by_datum_id[datum_id] = doc["resource"]
        self.events.new_doc(name="datum_page", doc=doc)
        super().datum_page(doc)

    @_write_locked
    def descriptor(self, doc):
        self._ordered.append(("descriptor", doc))
        name = doc.get("name")  # Might be missing in old documents
        self.descriptors[doc["uid"]] = doc
        if name is not None and name not in self._streams:
            self._streams[name] = [doc]
            self.events.new_stream(name=name)
        else:
            self._streams[name].append(doc)
        self.events.new_doc(name="descriptor", doc=doc)
        super().descriptor(doc)

    @_write_locked
    def resource(self, doc):
        self._ordered.append(("resource", doc))
        self.resources[doc["uid"]] = doc
        self.events.new_doc(name="resource", doc=doc)
        super().resource(doc)


class BlueskyRun(collections.abc.Mapping):
    """
    Push-based BlueskyRun
    """

    def __init__(
        self,
        document_cache,
        *,
        handler_registry=None,
        root_map=None,
        filler_class=event_model.Filler,
        transforms=None,
    ):

        if document_cache.start_doc is None:
            raise ValueError(
                "The document_cache must at least have a 'start' doc before a BlueskyRun can be created from it."
            )
        self._document_cache = document_cache
        self.write_lock = self._document_cache.write_lock
        self._streams = {}

        self._root_map = root_map or {}
        self._filler_class = filler_class
        self._transforms = parse_transforms(transforms)
        if handler_registry is None:
            handler_registry = discover_handlers()
        self._handler_registry = parse_handler_registry(handler_registry)
        self.handler_registry = event_model.HandlerRegistryView(self._handler_registry)

        self._get_filler = functools.partial(
            self._filler_class,
            handler_registry=self.handler_registry,
            root_map=self._root_map,
            inplace=False,
        )
        self.metadata = {
            "start": Start(self._transforms["start"](document_cache.start_doc))
        }

        # Re-emit Events emitted by self._document_cache.events. The only
        # difference is that these Events include a reference to self, and
        # thus subscribers will get a reference to this BlueskyRun.
        self.events = EmitterGroup(
            source=self,
            new_stream=Event,
            new_data=Event,
            completed=Event,
            new_doc=Event,
        )
        # We intentionally do not re-emit self._document_cache.started, because
        # by definition that will have fired already (and will never fire
        # again) since we already have a 'start' document.

        self._document_cache.events.new_data.connect(
            lambda event: self.events.new_data(run=self, updated=event.updated)
        )
        self._document_cache.events.new_doc.connect(
            lambda event: self.events.new_doc(run=self, name=event.name, doc=event.doc)
        )
        # The `completed` and `new_stream` Events are emitted below *after* we
        # update our internal state.

        # Wire up notification for when 'stop' doc is emitted or add it now if
        # it is already present.
        if self._document_cache.stop_doc is None:
            self.metadata["stop"] = None

            def on_completed(event):
                self.metadata["stop"] = Stop(
                    self._transforms["stop"](self._document_cache.stop_doc)
                )
                self.events.completed(run=self)

            self._document_cache.events.completed.connect(on_completed)
        else:
            self.metadata["stop"] = Stop(
                self._transforms["stop"](self._document_cache.stop_doc)
            )

        # Create any streams already present.
        for name in document_cache.streams:
            stream = BlueskyEventStream(
                name, document_cache, self._get_filler, self._transforms["descriptor"]
            )
            self._streams[name] = stream

        # ...and wire up notification for future ones.

        def on_new_stream(event):
            stream = BlueskyEventStream(
                event.name,
                document_cache,
                self._get_filler,
                self._transforms["descriptor"],
            )
            self._streams[event.name] = stream
            self.events.new_stream(name=event.name, run=self)

        self._document_cache.events.new_stream.connect(on_new_stream)

    # The following three methods are required to implement the Mapping interface.

    def __getitem__(self, key):
        return self._streams[key]

    def __len__(self):
        return len(self._streams)

    def __iter__(self):
        yield from self._streams

    def __getattr__(self, key):
        # Allow dot access for any stream names, as long as they are (of
        # course) valid Python identifiers and do not collide with existing
        # method names.
        try:
            return self._streams[key]
        except KeyError:
            raise AttributeError(key)

    def __repr__(self):
        # This is intentially a *single-line* string, suitable for placing in
        # logs. See _repr_pretty_ for a string better suited to interactive
        # computing.
        try:
            start = self.metadata["start"]
            return f"<{self.__class__.__name__} uid={start['uid']!r}>"
        except Exception as exc:
            return f"<{self.__class__.__name__} *REPR RENDERING FAILURE* {exc!r}>"

    def _repr_pretty_(self, p, cycle):
        # This hook is used by IPython. It provides a multi-line string more
        # detailed than __repr__, suitable for interactive computing.
        try:
            start = self.metadata["start"]
            stop = self.metadata["stop"] or {}
            out = (
                "BlueskyRun\n"
                f"  uid={start['uid']!r}\n"
                f"  exit_status={stop.get('exit_status')!r}\n"
                f"  {_ft(start['time'])} -- {_ft(stop.get('time', '?'))}\n"
            )
            if len(self):
                out += "  Streams:\n"
                for stream_name in self:
                    out += f"    * {stream_name}\n"
            else:
                out += "  (No Streams)"
        except Exception as exc:
            out = f"<{self.__class__.__name__} *REPR_RENDERING_FAILURE* {exc!r}>"
        p.text(out)

    def documents(self, *, fill):
        """
        Give Bluesky's streaming representation.

        Parameters
        ----------
        fill: {'yes', 'no', 'delayed'}
            Whether and how to resolve references to external data, if any.

        Yields
        ------
        (name, doc)
        """
        FILL_OPTIONS = {"yes", "no", "delayed"}
        if fill not in FILL_OPTIONS:
            raise ValueError(
                f"Invalid fill option: {fill}, fill must be: {FILL_OPTIONS}"
            )

        if fill == "yes":
            filler = self._get_filler(coerce="force_numpy")
        elif fill == "no":
            filler = event_model.NoFiller(self._handler_registry, inplace=True)
        else:  # fill == 'delayed'
            filler = self._get_filler(coerce="delayed")
        for name, doc in self._document_cache:
            yield filler(name, doc)


class ConfigAccessor(collections.abc.Mapping):
    def __init__(self, stream):
        self._stream = stream
        # Cache of object names mapped to BlueskyConfig objects
        self._config = {}

    # The following three methods are required to implement the Mapping interface.

    def __getitem__(self, key):
        # Return BlueskyConfig object for this key. Lazily create them as
        # needed.
        try:
            config = self._config[key]
        except KeyError:
            if key not in self._objects:
                raise
            config = BlueskyConfig(self._stream, key)
            self._config[key] = config
        return config

    def __len__(self):
        return len(self._objects)

    def __iter__(self):
        yield from self._objects

    def __getattr__(self, key):
        # Allow dot access for any object names, as long as they are (of
        # course) valid Python identifiers and do not collide with existing
        # method names.
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __repr__(self):
        return f"<Config Accessor: {sorted(self._objects)}>"

    @property
    def _objects(self):
        "Names of valid objects (i.e. devices)"
        objects = set()
        for d in self._stream._descriptors:
            objects.update(set(d["object_keys"]))
        return objects


class BlueskyConfig:
    def __init__(self, stream, object_name):
        self._stream = stream
        self._object_name = object_name

    def __repr__(self):
        return f"<BlueskyConfig for {self._object_name} in stream {self._stream.name}>"

    def to_dask(self):

        document_cache = self._stream._document_cache

        def get_event_pages(descriptor_uid, skip=0, limit=None):
            if skip != 0 and limit is not None:
                raise NotImplementedError
            return document_cache.event_pages[descriptor_uid]

        # def get_event_count(descriptor_uid):
        #     return sum(len(page['seq_num'])
        #                for page in (document_cache.event_pages[descriptor_uid]))

        def get_resource(uid):
            return document_cache.resources[uid]

        # def get_resources():
        #     return list(document_cache.resources.values())

        def lookup_resource_for_datum(datum_id):
            return document_cache.resource_uid_by_datum_id[datum_id]

        def get_datum_pages(resource_uid, skip=0, limit=None):
            if skip != 0 and limit is not None:
                raise NotImplementedError
            return document_cache.datum_pages_by_resource[resource_uid]

        # This creates a potential conflict with databroker, which currently
        # tries to register the same thing. For now our best effort to avoid
        # this is to register this at the last possible moment to give
        # databroker every chance of running first.
        try:
            event_model.register_coersion("delayed", coerce_dask)
        except event_model.EventModelValueError:
            # Already registered by databroker (or ourselves, earlier)
            pass

        filler = self._stream._get_filler(coerce="delayed")

        ds = documents_to_xarray_config(
            object_name=self._object_name,
            start_doc=document_cache.start_doc,
            stop_doc=document_cache.stop_doc,
            descriptor_docs=self._stream._descriptors,
            get_event_pages=get_event_pages,
            filler=filler,
            get_resource=get_resource,
            lookup_resource_for_datum=lookup_resource_for_datum,
            get_datum_pages=get_datum_pages,
        )
        return ds

    def read(self):
        return self.to_dask().load()


class BlueskyEventStream:
    def __init__(self, stream_name, document_cache, get_filler, transform):
        self._stream_name = stream_name
        self._document_cache = document_cache
        self._get_filler = get_filler
        self._transform = transform
        self.config = ConfigAccessor(self)

    @property
    def name(self):
        return self._stream_name

    def __repr__(self):
        return f"<BlueskyEventStream {self.name}>"

    @property
    def _descriptors(self):
        return [
            Descriptor(self._transform(descriptor))
            for descriptor in self._document_cache.streams[self._stream_name]
        ]

    def to_dask(self):

        document_cache = self._document_cache

        def get_event_pages(descriptor_uid, skip=0, limit=None):
            if skip != 0 and limit is not None:
                raise NotImplementedError
            return document_cache.event_pages[descriptor_uid]

        # def get_event_count(descriptor_uid):
        #     return sum(len(page['seq_num'])
        #                for page in (document_cache.event_pages[descriptor_uid]))

        def get_resource(uid):
            return document_cache.resources[uid]

        # def get_resources():
        #     return list(document_cache.resources.values())

        def lookup_resource_for_datum(datum_id):
            return document_cache.resource_uid_by_datum_id[datum_id]

        def get_datum_pages(resource_uid, skip=0, limit=None):
            if skip != 0 and limit is not None:
                raise NotImplementedError
            return document_cache.datum_pages_by_resource[resource_uid]

        # This creates a potential conflict with databroker, which currently
        # tries to register the same thing. For now our best effort to avoid
        # this is to register this at the last possible moment to give
        # databroker every chance of running first.
        try:
            event_model.register_coersion("delayed", coerce_dask)
        except event_model.EventModelValueError:
            # Already registered by databroker (or ourselves, earlier)
            pass

        filler = self._get_filler(coerce="delayed")

        ds = documents_to_xarray(
            start_doc=document_cache.start_doc,
            stop_doc=document_cache.stop_doc,
            descriptor_docs=self._descriptors,
            get_event_pages=get_event_pages,
            filler=filler,
            get_resource=get_resource,
            lookup_resource_for_datum=lookup_resource_for_datum,
            get_datum_pages=get_datum_pages,
        )
        return ds

    def read(self):
        return self.to_dask().load()


def _ft(timestamp):
    "format timestamp"
    if isinstance(timestamp, str):
        return timestamp
    # Truncate microseconds to miliseconds. Do not bother to round.
    return (datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f"))[:-3]
