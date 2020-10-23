import collections
import collections.abc
from datetime import datetime
import functools

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


class DocumentCache(event_model.SingleRunDocumentRouter):
    def __init__(self):
        self.descriptors = {}
        self.resources = {}
        self.event_pages = collections.defaultdict(list)
        self.datum_pages_by_resource = collections.defaultdict(list)
        self.resource_uid_by_datum_id = {}
        self.start_doc = None
        self.stop_doc = None
        self.events = EmitterGroup(
            started=Event, new_stream=Event, new_data=Event, completed=Event
        )
        # maps stream name to list of descriptors
        self._streams = {}
        self._ordered = []
        super().__init__()

    @property
    def streams(self):
        return self._streams

    def start(self, doc):
        self.start_doc = doc
        self._ordered.append(doc)
        self.events.started()
        super().start(doc)

    def stop(self, doc):
        self.stop_doc = doc
        self._ordered.append(doc)
        self.events.completed()
        super().stop(doc)

    def event_page(self, doc):
        self.event_pages[doc["descriptor"]].append(doc)
        self._ordered.append(doc)
        self.events.new_data()
        super().event_page(doc)

    def datum_page(self, doc):
        self.datum_pages_by_resource[doc["resource"]].append(doc)
        self._ordered.append(doc)
        for datum_id in doc["datum_id"]:
            self.resource_uid_by_datum_id[datum_id] = doc["resource"]
        super().datum_page(doc)

    def descriptor(self, doc):
        name = doc.get("name")  # Might be missing in old documents
        self.descriptors[doc["uid"]] = doc
        self._ordered.append(doc)
        if name is not None and name not in self._streams:
            self._streams[name] = [doc]
            self.events.new_stream(name=name)
        else:
            self._streams[name].append(doc)
        super().descriptor(doc)

    def resource(self, doc):
        self.resources[doc["uid"]] = doc
        self._ordered.append(doc)
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

        self._document_cache = document_cache
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

        # Wire up notification for when 'stop' doc is emitted or add it now if
        # it is already present.
        if self._document_cache.stop_doc is None:
            self.metadata["stop"] = None

            def on_completed(event):
                self.metadata["stop"] = Stop(
                    self._transforms["stop"](self._document_cache.stop_doc)
                )

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

    @property
    def events(self):
        return self._document_cache.events


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
