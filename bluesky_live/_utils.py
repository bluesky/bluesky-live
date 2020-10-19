import collections
import importlib
import itertools
import warnings

import dask
import dask.array
import entrypoints


def discover_handlers(entrypoint_group_name="databroker.handlers", skip_failures=True):
    """
    Discover handlers via entrypoints.

    Parameters
    ----------
    entrypoint_group_name: str
        Default is 'databroker.handlers', the "official" databroker entrypoint
        for handlers.
    skip_failures: boolean
        True by default. Errors loading a handler class are converted to
        warnings if this is True.

    Returns
    -------
    handler_registry: dict
        A suitable default handler registry
    """
    group = entrypoints.get_group_named(entrypoint_group_name)
    group_all = entrypoints.get_group_all(entrypoint_group_name)
    if len(group_all) != len(group):
        # There are some name collisions. Let's go digging for them.
        for name, matches in itertools.groupby(group_all, lambda ep: ep.name):
            matches = list(matches)
            if len(matches) != 1:
                winner = group[name]
                warnings.warn(
                    f"There are {len(matches)} entrypoints for the "
                    f"databroker handler spec {name!r}. "
                    f"They are {matches}. The match {winner} has won the race."
                )
    handler_registry = {}
    for name, entrypoint in group.items():
        try:
            handler_class = entrypoint.load()
        except Exception as exc:
            if skip_failures:
                warnings.warn(
                    f"Skipping {entrypoint!r} which failed to load. "
                    f"Exception: {exc!r}"
                )
                continue
            else:
                raise
        handler_registry[name] = handler_class

    return handler_registry


def parse_handler_registry(handler_registry):
    """
    Parse mapping of spec name to 'import path' into mapping to class itself.

    Parameters
    ----------
    handler_registry : dict
        Values may be string 'import paths' to classes or actual classes.

    Examples
    --------
    Pass in name; get back actual class.

    >>> parse_handler_registry({'my_spec': 'package.module.ClassName'})
    {'my_spec': <package.module.ClassName>}

    """
    result = {}
    for spec, handler_str in handler_registry.items():
        if isinstance(handler_str, str):
            module_name, _, class_name = handler_str.rpartition(".")
            class_ = getattr(importlib.import_module(module_name), class_name)
        else:
            class_ = handler_str
        result[spec] = class_
    return result


def parse_transforms(transforms):
    """
    Parse mapping of spec name to 'import path' into mapping to class itself.

    Parameters
    ----------
    transforms : collections.abc.Mapping or None
        A collections.abc.Mapping or subclass, that maps any subset of the
        keys {start, stop, resource, descriptor} to a function (or a string
        import path) that accepts a document of the corresponding type and
        returns it, potentially modified. This feature is for patching up
        erroneous metadata. It is intended for quick, temporary fixes that
        may later be applied permanently to the data at rest (e.g via a
        database migration).

    Examples
    --------
    Pass in name; get back actual class.

    >>> parse_transforms({'descriptor': 'package.module.function_name'})
    {'descriptor': <package.module.function_name>}

    """
    transformable = {"start", "stop", "resource", "descriptor"}

    if transforms is None:
        result = {key: _no_op for key in transformable}
        return result
    elif isinstance(transforms, collections.abc.Mapping):
        if len(transforms.keys() - transformable) > 0:
            raise NotImplementedError(
                f"Transforms for {transforms.keys() - transformable} "
                f"are not supported."
            )
        result = {}

        for name in transformable:
            transform = transforms.get(name)
            if isinstance(transform, str):
                module_name, _, class_name = transform.rpartition(".")
                function = getattr(importlib.import_module(module_name), class_name)
            elif transform is None:
                function = _no_op
            else:
                function = transform
            result[name] = function
        return result
    else:
        raise ValueError(
            f"Invalid transforms argument {transforms}. "
            f"transforms must be None or a dictionary."
        )


def _no_op(doc):
    # Implemented as a function (not a lambda)
    # so that objects that have it are pickle-able.
    return doc


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
