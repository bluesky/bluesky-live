import pytest

from ..bluesky_run import BlueskyRun, DocumentCache
from ..run_builder import RunBuilder


# BlueskyRun is mostly tested via test_run_builder.py. This exercises corner
# cases.


def test_empty():
    "Bluesky needs a DocumentCache with at *least* a 'start' doc."
    dc = DocumentCache()
    with pytest.raises(ValueError):
        BlueskyRun(dc)


def test_read_empty_stream():
    "An empty stream should return a xarray with no data but the right columns."

    with RunBuilder() as builder:
        builder.add_stream(
            "primary",
            data_keys={"a": {"shape": [10, 10], "dtype": "number", "source": "stuff"}},
        )
    run = builder.get_run()
    ds = run.primary.read()
    assert "a" in ds
    assert ds["a"].shape == (0, 10, 10)


def test_events():
    "Test the Event EmitterGroup on BlueskyRun."
    # We will subscribe callbacks that appent Events to these lists.
    new_stream_events = list()
    new_data_events = list()
    completed_events = list()

    with RunBuilder() as builder:
        run = builder.get_run()
        run.events.new_stream.connect(lambda event: new_stream_events.append(event))
        run.events.new_data.connect(lambda event: new_data_events.append(event))
        run.events.completed.connect(lambda event: completed_events.append(event))
        assert not new_stream_events
        assert not new_data_events
        assert not completed_events

        builder.add_stream("primary", data={"a": [1, 2, 3]})
        assert len(new_stream_events) == 1
        assert new_stream_events[0].run is run
        assert new_stream_events[0].name == "primary"
        assert len(new_data_events) == 1
        assert new_data_events[0].run is run
        assert len(completed_events) == 0

        builder.add_data("primary", data={"a": [1, 2, 3]})
        assert len(new_stream_events) == 1
        assert len(new_data_events) == 2
        assert new_data_events[1].run is run
        assert len(completed_events) == 0

    # Exiting the context issues a 'stop' document....

    assert len(new_stream_events) == 1
    assert len(new_data_events) == 2
    assert len(completed_events) == 1
    assert completed_events[0].run is run


def test_access_stream_in_callback():
    "Test that the stream is accessible when new_stream fires."

    with RunBuilder() as builder:
        run = builder.get_run()

        def access_stream(event):
            run[event.name]

        run.events.new_stream.connect(access_stream)
        builder.add_stream(
            "primary",
            data_keys={"b": {"shape": [10, 10], "dtype": "number", "source": "stuff"}},
        )
