import itertools

import event_model
import numpy


def documents_to_xarray(
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
    sub_dict="data",
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
    sub_dict : {"data", "timestamps"}, optional
        Which sub-dict in the EventPage to use

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
        # seq_nums = [ev["seq_num"] for ev in events]
        # uids = [ev["uid"] for ev in events]
        data_table = _transpose(filled_events, keys, data_keys, sub_dict)
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
        # data_arrays["seq_num"] = xarray.DataArray(
        #     data=seq_nums, dims=("time",), coords={"time": times}, name="seq_num"
        # )
        # data_arrays["uid"] = xarray.DataArray(
        #     data=uids, dims=("time",), coords={"time": times}, name="uid"
        # )

        datasets.append(xarray.Dataset(data_vars=data_arrays))
    # Merge Datasets from all Event Descriptors into one representing the
    # whole stream. (In the future we may simplify to one Event Descriptor
    # per stream, but as of this writing we must account for the
    # possibility of multiple.)
    return xarray.merge(datasets)


def documents_to_xarray_config(
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
    sub_dict="data",
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
    sub_dict : {"data", "timestamps"}, optional
        Which sub-dict in the EventPage to use

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
                data=numpy.tile(
                    config[sub_dict][key], (len(times),) + ndim * (1,) or 1
                ),
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


def _transpose(in_data, keys, data_keys, field):
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
            if len(out[k]):
                out[k] = dask.array.stack(out[k])
            else:
                # Case of no Events yet
                out[k] = dask.array.array([]).reshape(0, *data_keys[k]["shape"])
        except NotImplementedError:
            # There are data structured that dask auto-chunking cannot handle,
            # such as an list of list of variable length. For now, let these go
            # out as plain numpy arrays. In the future we might make them dask
            # arrays with manual chunks.
            out[k] = numpy.asarray(out[k])

    return out
