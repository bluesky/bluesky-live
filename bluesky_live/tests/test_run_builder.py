import numpy
import pandas
import pytest
import xarray

from ..run_builder import RunBuilder


def test_minimal_run():
    builder = RunBuilder()
    run = builder.get_run()
    builder.close()
    assert run.metadata["start"] is not None
    assert run.metadata["stop"] is not None


def test_minimal_run_using_context_manager():
    with RunBuilder() as builder:
        run = builder.get_run()
    assert run.metadata["start"] is not None
    assert run.metadata["stop"] is not None


def test_error_within_context_manager():
    class CustomException(Exception):
        ...

    err = CustomException("some message")
    with pytest.raises(CustomException):
        with RunBuilder() as builder:
            run = builder.get_run()
            raise err
    assert run.metadata["stop"]["exit_status"] == "fail"
    assert run.metadata["stop"]["reason"] == repr(err)


def test_custom_metadata_in_start_doc():
    with RunBuilder(metadata={"sample": "pigeon"}) as builder:
        ...
    run = builder.get_run()
    assert run.metadata["start"]["sample"] == "pigeon"


def test_close_with_failure():
    builder = RunBuilder()
    run = builder.get_run()
    builder.close(exit_status="abort", reason="sample on fire")
    assert run.metadata["stop"]["exit_status"] == "abort"
    assert run.metadata["stop"]["reason"] == "sample on fire"


def test_close_with_failure_inside_context():
    with RunBuilder() as builder:
        builder.close(exit_status="abort", reason="sample on fire")
    run = builder.get_run()
    assert run.metadata["stop"]["exit_status"] == "abort"
    assert run.metadata["stop"]["reason"] == "sample on fire"


def test_add_stream_from_data_keys():
    with RunBuilder() as builder:
        builder.add_stream(
            "primary",
            data_keys={
                "x": {"source": "made up", "dtype": "number", "shape": []},
                "y": {"source": "made up", "dtype": "number", "shape": []},
            },
        )
    run = builder.get_run()
    run.primary


data_param = pytest.mark.parametrize(
    "data",
    [
        {"x": [1, 2, 3], "y": [10, 20, 30]},
        {"x": numpy.array([1, 2, 3]), "y": numpy.array([10, 20, 30])},
        pandas.DataFrame({"x": [4, 5, 6], "y": [40, 50, 60]}),
        xarray.Dataset(
            {"x": xarray.DataArray([7, 8, 9]), "y": xarray.DataArray([70, 80, 90])}
        ),
    ],
    ids=["dict-of-lists", "dict-of-arrays", "pandas.DataFrame", "xarray.Dataset"],
)


@data_param
def test_add_stream_and_add_data(data):
    "Declare a new stream with data_keys and then add data separately."
    with RunBuilder() as builder:
        builder.add_stream(
            "primary",
            data_keys={
                "x": {"source": "made up", "dtype": "number", "shape": []},
                "y": {"source": "made up", "dtype": "number", "shape": []},
            },
        )
        builder.add_data("primary", data)


@data_param
def test_add_stream_with_data_keys_and_data(data):
    "Declare a new stream with data_keys and some data."
    with RunBuilder() as builder:
        builder.add_stream(
            "primary",
            data_keys={
                "x": {"source": "made up", "dtype": "number", "shape": []},
                "y": {"source": "made up", "dtype": "number", "shape": []},
            },
            data=data,
        )


@data_param
def test_add_stream_with_data_only(data):
    "Declare a new stream with data only, inferring the data_keys."
    with RunBuilder() as builder:
        builder.add_stream(
            "primary",
            data=data,
        )


def test_add_stream_from_data_keys_with_extras():
    with RunBuilder() as builder:
        builder.add_stream(
            "primary",
            data_keys={
                "x": {"source": "made up", "dtype": "number", "shape": []},
                "y": {"source": "made up", "dtype": "number", "shape": []},
            },
            object_keys={
                "stage": ["x", "y"]
            },  # The 'x' and 'y' fields above are associated with one device, a 'stage'.
            configuration={
                "stage": {  # The 'stage' has some additional readings which help interpret 'x' and 'y'.
                    "data_keys": {
                        "tilt": {"source": "made up", "dtype": "number", "shape": []},
                    },
                    "data": {"tilt": 5.0},
                    "timestamps": {"tilt": 0.0},
                }
            },
        )
        builder.add_data(
            "primary",
            data={"x": [1, 2, 3], "y": [10, 20, 30]},
            timestamps={"x": [1, 2, 3], "y": [10, 20, 30]},
        )
    builder.get_run()


def test_boolean_data():
    with RunBuilder() as builder:
        builder.add_stream("primary", data=[True, False, True])
    builder.get_run()
