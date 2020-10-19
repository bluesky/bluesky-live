import collections
import collections.abc
import itertools
from datetime import datetime
import functools

import event_model
import numpy

from .document import Start, Stop, Descriptor
from .event import EmitterGroup, Event
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

        ds = _documents_to_xarray_config(
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

        ds = _documents_to_xarray(
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


def _documents_to_xarray(
    *,
    start_doc,
    stop_doc,
    descriptor_docs,
    get_event_pages,
    filler,
    get_resource,
    lookup_resource_for_datum,
    get_datum_pages,
    include=None,
    exclude=None,
):
    """
    Represent the data in one Event stream as an xarray.

    Parameters
    ----------
    start_doc: dict
        RunStart Document
    stop_doc : dict
        RunStop Document
    descriptor_docs : list
        EventDescriptor Documents
    filler : event_model.Filler
    get_resource : callable
        Expected signature ``get_resource(resource_uid) -> Resource``
    lookup_resource_for_datum : callable
        Expected signature ``lookup_resource_for_datum(datum_id) -> resource_uid``
    get_datum_pages : callable
        Expected signature ``get_datum_pages(resource_uid) -> generator``
        where ``generator`` yields datum_page documents
    get_event_pages : callable
        Expected signature ``get_event_pages(descriptor_uid) -> generator``
        where ``generator`` yields event_page documents
    include : list, optional
        Fields ('data keys') to include. By default all are included. This
        parameter is mutually exclusive with ``exclude``.
    exclude : list, optional
        Fields ('data keys') to exclude. By default none are excluded. This
        parameter is mutually exclusive with ``include``.

    Returns
    -------
    dataset : xarray.Dataset
    """
    import xarray

    if include is None:
        include = []
    if exclude is None:
        exclude = []
    if include and exclude:
        raise ValueError(
            "The parameters `include` and `exclude` are mutually exclusive."
        )
    # Data keys must not change within one stream, so we can safely sample
    # just the first Event Descriptor.
    if descriptor_docs:
        data_keys = descriptor_docs[0]["data_keys"]
        if include:
            keys = list(set(data_keys) & set(include))
        elif exclude:
            keys = list(set(data_keys) - set(exclude))
        else:
            keys = list(data_keys)

    # Collect a Dataset for each descriptor. Merge at the end.
    datasets = []
    dim_counter = itertools.count()
    event_dim_labels = {}
    for descriptor in descriptor_docs:
        events = list(_flatten_event_page_gen(get_event_pages(descriptor["uid"])))
        if not events:
            continue
        if any(data_keys[key].get("external") for key in keys):
            filler("descriptor", descriptor)
            filled_events = []
            for event in events:
                filled_event = _fill(
                    filler,
                    event,
                    lookup_resource_for_datum,
                    get_resource,
                    get_datum_pages,
                )
                filled_events.append(filled_event)
        else:
            filled_events = events
        times = [ev["time"] for ev in events]
        seq_nums = [ev["seq_num"] for ev in events]
        uids = [ev["uid"] for ev in events]
        data_table = _transpose(filled_events, keys, "data")
        # external_keys = [k for k in data_keys if 'external' in data_keys[k]]

        # Collect a DataArray for each field in Event, 'uid', and 'seq_num'.
        # The Event 'time' will be the default coordinate.
        data_arrays = {}

        # Make DataArrays for Event data.
        for key in keys:
            field_metadata = data_keys[key]
            # if the EventDescriptor doesn't provide names for the
            # dimensions (it's optional) use the same default dimension
            # names that xarray would.
            try:
                dims = tuple(field_metadata["dims"])
            except KeyError:
                ndim = len(field_metadata["shape"])
                # Reuse dim labels.
                try:
                    dims = event_dim_labels[key]
                except KeyError:
                    dims = tuple(f"dim_{next(dim_counter)}" for _ in range(ndim))
                    event_dim_labels[key] = dims
            attrs = {}
            # Record which object (i.e. device) this column is associated with,
            # which enables one to find the relevant configuration, if any.
            for object_name, keys_ in descriptor.get("object_keys", {}).items():
                for item in keys_:
                    if item == key:
                        attrs["object"] = object_name
                        break
            data_arrays[key] = xarray.DataArray(
                data=data_table[key],
                dims=("time",) + dims,
                coords={"time": times},
                name=key,
                attrs=attrs,
            )

        # Finally, make DataArrays for 'seq_num' and 'uid'.
        data_arrays["seq_num"] = xarray.DataArray(
            data=seq_nums, dims=("time",), coords={"time": times}, name="seq_num"
        )
        data_arrays["uid"] = xarray.DataArray(
            data=uids, dims=("time",), coords={"time": times}, name="uid"
        )

        datasets.append(xarray.Dataset(data_vars=data_arrays))
    # Merge Datasets from all Event Descriptors into one representing the
    # whole stream. (In the future we may simplify to one Event Descriptor
    # per stream, but as of this writing we must account for the
    # possibility of multiple.)
    return xarray.merge(datasets)


def _documents_to_xarray_config(
    *,
    object_name,
    start_doc,
    stop_doc,
    descriptor_docs,
    get_event_pages,
    filler,
    get_resource,
    lookup_resource_for_datum,
    get_datum_pages,
    include=None,
    exclude=None,
):
    """
    Represent the data in one Event stream as an xarray.

    Parameters
    ----------
    object_name : str
        Object (i.e. device) name of interest
    start_doc: dict
        RunStart Document
    stop_doc : dict
        RunStop Document
    descriptor_docs : list
        EventDescriptor Documents
    filler : event_model.Filler
    get_resource : callable
        Expected signature ``get_resource(resource_uid) -> Resource``
    lookup_resource_for_datum : callable
        Expected signature ``lookup_resource_for_datum(datum_id) -> resource_uid``
    get_datum_pages : callable
        Expected signature ``get_datum_pages(resource_uid) -> generator``
        where ``generator`` yields datum_page documents
    get_event_pages : callable
        Expected signature ``get_event_pages(descriptor_uid) -> generator``
        where ``generator`` yields event_page documents
    include : list, optional
        Fields ('data keys') to include. By default all are included. This
        parameter is mutually exclusive with ``exclude``.
    exclude : list, optional
        Fields ('data keys') to exclude. By default none are excluded. This
        parameter is mutually exclusive with ``include``.

    Returns
    -------
    dataset : xarray.Dataset
    """
    import xarray

    if include is None:
        include = []
    if exclude is None:
        exclude = []
    if include and exclude:
        raise ValueError(
            "The parameters `include` and `exclude` are mutually exclusive."
        )

    # Collect a Dataset for each descriptor. Merge at the end.
    datasets = []
    dim_counter = itertools.count()
    config_dim_labels = {}
    for descriptor in descriptor_docs:
        events = list(_flatten_event_page_gen(get_event_pages(descriptor["uid"])))
        if not events:
            continue
        times = [ev["time"] for ev in events]
        # external_keys = [k for k in data_keys if 'external' in data_keys[k]]

        # Collect a DataArray for each field configuration. The Event 'time'
        # will be the default coordinate.
        data_arrays = {}

        # Make DataArrays for configuration data.
        config = descriptor["configuration"][object_name]
        data_keys = config["data_keys"]
        for key in data_keys:
            field_metadata = data_keys[key]
            ndim = len(field_metadata["shape"])
            # if the EventDescriptor doesn't provide names for the
            # dimensions (it's optional) use the same default dimension
            # names that xarray would.
            try:
                dims = tuple(field_metadata["dims"])
            except KeyError:
                try:
                    dims = config_dim_labels[key]
                except KeyError:
                    dims = tuple(f"dim_{next(dim_counter)}" for _ in range(ndim))
                    config_dim_labels[key] = dims
            data_arrays[key] = xarray.DataArray(
                # TODO Once we know we have one Event Descriptor
                # per stream we can be more efficient about this.
                data=numpy.tile(config["data"][key], (len(times),) + ndim * (1,) or 1),
                dims=("time",) + dims,
                coords={"time": times},
                name=key,
            )

        datasets.append(xarray.Dataset(data_vars=data_arrays))
    # Merge Datasets from all Event Descriptors into one representing the
    # whole stream. (In the future we may simplify to one Event Descriptor
    # per stream, but as of this writing we must account for the
    # possibility of multiple.)
    return xarray.merge(datasets)


def _ft(timestamp):
    "format timestamp"
    if isinstance(timestamp, str):
        return timestamp
    # Truncate microseconds to miliseconds. Do not bother to round.
    return (datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f"))[:-3]


def _flatten_event_page_gen(gen):
    """
    Converts an event_page generator to an event generator.

    Parameters
    ----------
    gen : generator

    Returns
    -------
    event_generator : generator
    """
    for page in gen:
        yield from event_model.unpack_event_page(page)


def _fill(
    filler,
    event,
    lookup_resource_for_datum,
    get_resource,
    get_datum_pages,
    last_datum_id=None,
):
    try:
        _, filled_event = filler("event", event)
        return filled_event
    except event_model.UnresolvableForeignKeyError as err:
        datum_id = err.key
        if datum_id == last_datum_id:
            # We tried to fetch this Datum on the last trip
            # trip through this method, and apparently it did not
            # work. We are in an infinite loop. Bail!
            raise

        # try to fast-path looking up the resource uid if this works
        # it saves us a a database hit (to get the datum document)
        if "/" in datum_id:
            resource_uid, _ = datum_id.split("/", 1)
        # otherwise do it the standard way
        else:
            resource_uid = lookup_resource_for_datum(datum_id)

        # but, it might be the case that the key just happens to have
        # a '/' in it and it does not have any semantic meaning so we
        # optimistically try
        try:
            resource = get_resource(uid=resource_uid)
        # and then fall back to the standard way to be safe
        except ValueError:
            resource = get_resource(lookup_resource_for_datum(datum_id))

        filler("resource", resource)
        # Pre-fetch all datum for this resource.
        for datum_page in get_datum_pages(resource_uid=resource_uid):
            filler("datum_page", datum_page)
        # TODO -- When to clear the datum cache in filler?

        # Re-enter and try again now that the Filler has consumed the
        # missing Datum. There might be another missing Datum in this same
        # Event document (hence this re-entrant structure) or might be good
        # to go.
        return _fill(
            filler,
            event,
            lookup_resource_for_datum,
            get_resource,
            get_datum_pages,
            last_datum_id=datum_id,
        )


def _transpose(in_data, keys, field):
    """Turn a list of dicts into dict of lists

    Parameters
    ----------
    in_data : list
        A list of dicts which contain at least one dict.
        All of the inner dicts must have at least the keys
        in `keys`

    keys : list
        The list of keys to extract

    field : str
        The field in the outer dict to use

    Returns
    -------
    transpose : dict
        The transpose of the data
    """
    import dask.array
    import numpy

    out = {k: [None] * len(in_data) for k in keys}
    for j, ev in enumerate(in_data):
        dd = ev[field]
        for k in keys:
            out[k][j] = dd[k]
    for k in keys:
        try:
            # compatibility with dask < 2
            if hasattr(out[k][0], "shape"):
                out[k] = dask.array.stack(out[k])
            else:
                out[k] = dask.array.array(out[k])
        except NotImplementedError:
            # There are data structured that dask auto-chunking cannot handle,
            # such as an list of list of variable length. For now, let these go
            # out as plain numpy arrays. In the future we might make them dask
            # arrays with manual chunks.
            out[k] = numpy.asarray(out[k])

    return out
