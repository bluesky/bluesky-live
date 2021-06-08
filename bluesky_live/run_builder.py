import collections
import collections.abc

import event_model
import numpy

from .bluesky_run import DocumentCache, BlueskyRun


class StreamExists(event_model.EventModelRuntimeError):
    ...


class StreamDoesNotExist(event_model.EventModelRuntimeError):
    ...


class RunBuilder:
    """
    Construct a BlueskyRun from xarray.Dataset, pandas.DataFrame, or dict-of-lists.

    Parameters
    ----------
    metadata : dict, optional
        This should include metadata about what will be done and any adminstrative info.
    uid : string, optional
        By default, a UUID4 is generated.
    time : float, optional
        POSIX epoch time. By default, the current time is used.

    Examples
    --------

    Minimal example

    >>> builder = RunBuilder()
    >>> run = builder.get_run()  # -> BluekyRun. This may be called at any point.
    >>> builder.add_stream("primary", data={"x": [1, 2, 3]})
    >>> builder.close()

    Context manager closes automatically on exit from context, and marks

    >>> with RunBuilder() as builder:
    >>> ... builder.add_stream("primary", data={"x": [1, 2, 3]})
    >>> ...

    Add data incrementally.

    >>> with RunBuilder() as builder:
    >>> ... builder.add_stream("primary", data={"x": [1, 2, 3]})
    >>> ... builder.add_data("primary", data={"x": [4, 5, 6]})
    >>> ... builder.add_data("primary", data={"x": [7, 8, 9]})
    >>> ...

    Define the stream without adding data at first.

    >>> with RunBuilder() as builder:
    >>> ... builder.add_stream("primary",
    >>> ...     data_keys={"x": {"dtype": "number", "shape": [], "source": "whatever"}}
    >>> ... builder.add_data("primary", data={"x": [1, 2, 3]})


    Additional metadata (placed in Run Start document)

    >>> with RunBuilder({"sample": "Cu"}) as builder:
    >>> ... builder.add_stream("primary", data={"x": [1, 2, 3]})
    >>> ...

    """

    def __init__(self, metadata=None, uid=None, time=None):
        self._cache = DocumentCache()
        self._run_bundle = event_model.compose_run(
            uid=uid, time=time, metadata=metadata
        )
        self._cache.start(self._run_bundle.start_doc)
        # maps stream name to bundle returned by compose_descriptor
        self._streams = {}

    def add_stream(
        self,
        name,
        *,
        data=None,
        data_keys=None,
        uid=None,
        time=None,
        object_keys=None,
        configuration=None,
        hints=None,
    ):
        """
        Define a new stream (table of data) and, optionally add data to it.

        Parameters
        ----------
        name : string
            Any string. The name "primary" is conventional (but not required)
            for the primary table of interest, if any.
        data : xarray.Dataset, pandas.DataFrame, or dict-of-arrays-or-lists, optional
        data_keys : dict, optional
            Metadata about the columns. If not specified, it will be inferred.
            Required items are:

                * source --- a freeform string
                * dtype --- one of "array", "bool", "string", "integer", "number"
                * shape --- list of dimensions (empty for scalars)

        uid : string, optional
            By default, a UUID4 is generated.
        time : float, optional
            POSIX epoch time. By default, the current time is used.
        object_keys : dict, optional
            Map a device map to a list of the associated keys in data_keys
        configuration : dict, optional
            Map each device in object_keys to a dict of data_keys, data, and
            timestamps.
        hints : dict, optional
            Dict mapping "fields" to list of most commonly interesting
            keys in data_keys to aid in producing best-effort
            visualizations and other downstream tools.

        Raises
        ------
        StreamExists
            If the name has already been used. Use :meth:`add_data` to add data
            to an existing stream.
        """
        if data is None and data_keys is None:
            raise event_model.EventModelValueError(
                "Neither 'data' nor 'data_keys' was given. At least one is " "required."
            )
        if name in self._streams:
            raise StreamExists(
                f"The stream {name!r} exists. Use the method `add_data` to add "
                "more data to the stream."
            )
        if data_keys is None:
            # Infer the data_keys from the data.
            data_keys = {
                k: {
                    "source": "RunBuilder",
                    "dtype": _infer_dtype(next(iter(v))),
                    "shape": _infer_shape(v),
                }
                for k, v in data.items()
            }
        bundle = self._run_bundle.compose_descriptor(
            name=name,
            data_keys=data_keys,
            time=time,
            object_keys=object_keys,
            configuration=configuration,
            hints=hints,
        )
        self._streams[name] = bundle
        self._cache.descriptor(bundle.descriptor_doc)
        if data is not None:
            self.add_data(name=name, data=data, time=time)

    def add_data(self, name, data, time=None, timestamps=None, seq_num=None, uid=None):
        """
        Add data to an existing stream.

        Parameters
        ----------
        name : string
            Name of a stream previously created using :meth:`add_stream`
        data : xarray.Dataset, pandas.DataFrame, or dict-of-arrays-or-lists
        time : array, optional
            POSIX epoch times. By default, an array of the current time is used.
        timestamps : xarray.Dataset, pandas.DataFrame, or dict-of-arrays-or-lists
            Individual timestamps for every element in data. This level of
            detail is sometimes available from data acquisition systems, but it
            is rarely available or useful in other contexts.
        seq_num : array, optional
            Sequence numbers.
        uid : string, optional
            By default, a UUID4 is generated.

        Raises
        ------
        StreamDoesNotExist
            If stream with the given name has not yet been created using
            :meth:`add_stream`.
        EventModelRuntimeErrror
            If the run has been closed.
        """
        if name not in self._streams:
            raise StreamDoesNotExist(
                "First use the method `add_stream` to create a stream named "
                f"{name!r}. You can add data there and/or add it later separately "
                "using this method, `add_data`."
            )
        if self.closed:
            raise event_model.EventModelRuntimeError("Run is closed.")
        import time as time_module

        len_ = len(data[next(iter(data))])
        now = time_module.time()
        if time is None:
            time = [now] * len_
        if timestamps is None:
            timestamps = {k: [now] * len_ for k in data}
        if seq_num is None:
            seq_num = (1 + numpy.arange(len_, dtype=int)).tolist()
        bundle = self._streams[name]
        doc = bundle.compose_event_page(
            time=time,
            data=_normalize_dataframe_like(data),
            timestamps=_normalize_dataframe_like(timestamps),
            seq_num=seq_num,
            uid=uid,
        )
        self._cache.event_page(doc)

    def update_configuration(self, name, configuration):
        """
        Update the configuration in a stream.

        This issues a new Event Descriptor for a stream.

        Paramters
        ---------
        name : string
        configuration : dict
            See add_stream for expected structure.

        Raises
        ------
        StreamDoesNotExist
            If stream with the given name has not yet been created using
            :meth:`add_stream`.
        EventModelRuntimeErrror
            If the run has been closed.
        """
        if self.closed:
            raise event_model.EventModelRuntimeError("Run is closed.")
        if name not in self._streams:
            raise StreamDoesNotExist(
                "First use the method `add_stream` to create a stream named "
                f"{name!r}."
            )
        # Fill in everything except configuration using the previous Event
        # Descriptor in this stream.
        old_bundle = self._streams[name]
        doc = old_bundle.descriptor_doc
        bundle = self._compose_descriptor(
            name=doc["name"],
            data_keys=doc["data_keys"],
            time=doc["time"],
            object_keys=doc["object_keys"],
            configuration=configuration,
            hints=doc["hints"],
        )
        self._streams[name] = bundle
        self._cache.descriptor(bundle.descriptor_doc)

    def close(self, exit_status="success", reason="", uid=None, time=None):
        """
        Mark the Run as complete.

        It will not be possible to add new streams or new data once this is
        called.

        Parameters
        ----------
        exit_status : {"success", "abort", "fail"}, optional
        reason : string, optional
        uid : string, optional
            By default, a UUID4 is generated.
        time : float
            POSIX epoch time. By default, the current time is used.
        """
        if self.closed:
            raise event_model.EventModelRuntimeError("Run is already closed.")
        doc = self._run_bundle.compose_stop(
            exit_status=exit_status, reason=reason, uid=uid, time=time
        )
        self._cache.stop(doc)

    @property
    def closed(self):
        """
        True if the Run has been closed and cannot accept new data.
        """
        return self._cache.stop_doc is not None

    def get_run(
        self,
        *,
        handler_registry=None,
        root_map=None,
        filler_class=event_model.Filler,
        transforms=None,
    ):
        """
        Provide a BlueskyRun built by this RunBuilder.

        Returns
        -------
        BlueskyRun
        """
        return BlueskyRun(
            document_cache=self._cache,
            # TODO These arguments are *proabably* not needed.
            handler_registry=handler_registry,
            root_map=root_map,
            filler_class=filler_class,
            transforms=transforms,
        )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if type is not None:
            reason = repr(value)
            if not self.closed:
                self.close(exit_status="fail", reason=reason)
        else:
            if not self.closed:
                self.close()  # success


def build_simple_run(
    data, metadata=None, uid=None, time=None, exit_status="success", reason=""
):
    """
    Build a simple run from a dataset.

    By "simple", we mean it has a single "stream" of data and does not expose
    some of the options available from the more advanced RunBuilder, such as
    adding data incrementally or including configuration and more detailed
    timestamps.

    This function is suitable for consuming data that could be represented as a
    single "spreadsheet" plus a dictionary of metadata.

    Parameters
    ----------
    data : xarray.Dataset, pandas.DataFrame, or dict-of-arrays-or-lists, optional
    metadata : dict, optional
        This should include metadata about what will be done and any adminstrative info.
    uid : string, optional
        By default, a UUID4 is generated.
    time : float, optional
        POSIX epoch time. By default, the current time is used.
    exit_status : {"success", "abort", "fail"}, optional
    reason : string, optional

    Returns
    -------
    BlueskyRun

    Examples
    --------

    >>> build_simple_run({'x': [1, 2, 3], 'y': [4, 5, 6], metadata={'sample': 'Cu'})
    """
    with RunBuilder(metadata=metadata, uid=uid, time=time) as builder:
        builder.add_stream("primary", data=data)
        builder.close(exit_status=exit_status, reason=reason)
    return builder.get_run()


def build_one_run_from_documents(document_generator):
    """
    Build one BlueskyRun from a stream of documents --- (name, doc) pairs.

    This assumes document_generator contains documents for one run.
    It will fail if:

    * There is no 'start' document (e.g. empty generator); or
    * There is more than one 'start' document (e.g. a stream of multiple
      sequenced or interleaved runs).

    Parameters
    ----------
    document_generator: Iterable[Tuple[String, Dict]]
        Iterable of ``(name, doc)`` pairs.

    Returns
    -------
    BlueskyRun
    """
    cache = DocumentCache()
    for item in document_generator:
        cache(*item)
    return BlueskyRun(cache)


def build_runs_from_documents(document_generator):
    """
    Build BlueskyRuns from a stream of documents --- (name, doc) pairs.

    This will work for any number of runs, including 0 (empty generator).

    Parameters
    ----------
    document_generator: Iterable[Tuple[String, Dict]]
        Iterable of ``(name, doc)`` pairs.

    Returns
    -------
    List[BlueskyRun]
    """
    runs = []

    def factory(name, doc):
        dc = DocumentCache()

        def add_run_to_list(event):
            run = BlueskyRun(dc)
            runs.append(run)

        dc.events.started.connect(add_run_to_list)
        return [dc], []

    rr = event_model.RunRouter([factory])
    for item in document_generator:
        rr(*item)
    return runs


def _infer_dtype(obj):
    "Infer the dtype for Event Descriptor data_keys based on the data."
    if isinstance(obj, (numpy.generic, numpy.ndarray)):
        if numpy.isscalar(obj):
            obj = obj.item()
        else:
            return "array"
    if isinstance(obj, str):
        return "string"
    elif isinstance(obj, collections.abc.Iterable):
        return "array"
    elif isinstance(obj, bool):
        return "boolean"
    elif isinstance(obj, int):
        return "integer"
    else:
        return "number"


def _infer_shape(obj):
    "Infer the shape for Event Descriptor data_keys based on the data."
    if hasattr(obj, "shape"):
        # The first axis is the "Event" axis. We want the shape of the data
        # within a given Event, so we trim off the first axis.
        return obj.shape[1:]
    else:
        return []


def _normalize_dataframe_like(df):
    "Normalize xarray.Dataset and pandas.DataFrame to be dict-of-arrays."
    import xarray
    import pandas

    if isinstance(df, xarray.Dataset):
        return {k: v["data"] for k, v in df.to_dict()["data_vars"].items()}
    elif isinstance(df, pandas.DataFrame):
        # Is there a better way?
        return {k: v.values for k, v in df.items()}
    else:
        return df
