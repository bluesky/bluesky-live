import event_model


def coerce_dask(handler_class, filler_state):
    # If the handler has its own delayed logic, defer to that.
    if hasattr(handler_class, "return_type"):
        if handler_class.return_type["delayed"]:
            return handler_class

    # Otherwise, provide best-effort dask support by wrapping each datum
    # payload in dask.array.from_delayed. This means that each datum will be
    # one dask task---it cannot be rechunked into multiple tasks---but that
    # may be sufficient for many handlers.
    class Subclass(handler_class):
        def __call__(self, *args, **kwargs):
            descriptor = filler_state.descriptor
            key = filler_state.key
            shape = extract_shape(descriptor, key)
            # there is an un-determined size (-1) in the shape, abandon
            # lazy as it will not work
            if any(s <= 0 for s in shape):
                return dask.array.from_array(super().__call__(*args, **kwargs))
            else:
                dtype = extract_dtype(descriptor, key)
                load_chunk = dask.delayed(super().__call__)(*args, **kwargs)
                return dask.array.from_delayed(load_chunk, shape=shape, dtype=dtype)

    return Subclass


def extract_shape(descriptor, key):
    """
    Work around bug in https://github.com/bluesky/ophyd/pull/746
    """
    # Ideally this code would just be
    # descriptor['data_keys'][key]['shape']
    # but we have to do some heuristics to make up for errors in the reporting.

    # Broken ophyd reports (x, y, 0). We want (num_images, y, x).
    data_key = descriptor["data_keys"][key]
    if len(data_key["shape"]) == 3 and data_key["shape"][-1] == 0:
        object_keys = descriptor.get("object_keys", {})
        for object_name, data_keys in object_keys.items():
            if key in data_keys:
                break
        else:
            raise RuntimeError(f"Could not figure out shape of {key}")
        for k, v in descriptor["configuration"][object_name]["data"].items():
            if k.endswith("num_images"):
                num_images = v
                break
        else:
            num_images = -1
        x, y, _ = data_key["shape"]
        shape = (num_images, y, x)
    else:
        shape = descriptor["data_keys"][key]["shape"]
    return shape


def extract_dtype(descriptor, key):
    """
    Work around the fact that we currently report jsonschema data types.
    """
    reported = descriptor["data_keys"][key]["dtype"]
    if reported == "array":
        return float  # guess!
    else:
        return reported


def register():
    "This is run in bluesky_live.__init__"
    # This adds a 'delayed' option to event_model.Filler's `coerce` parameter.
    # By adding it via plugin, we avoid adding a dask.array dependency to
    # event-model and we keep the fiddly hacks into extract_shape here in
    # databroker, a faster-moving and less fundamental library than event-model.
    try:
        event_model.register_coercion('delayed', coerce_dask)
    except event_model.EventModelValueError:
        # If databroker is imported and an older version, it has already registered
        # its copy of this. Let is pass.
        databroker = sys.modules.get("databroker")
        if databroker is None:
            # Databroker is not imported, so there must be a different issue in
            # play here.
            raise
        from distutils.version import LooseVersion

        if LooseVersion(databroker.__version__) > LooseVersion("1.2.0"):
            # Databroker is imported but a new enough version that it should *not*
            # be stepping on us.
            raise
        # Do nothing. Databroker has stepped on us.
