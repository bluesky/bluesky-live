import time
import threading

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
    new_doc_events = list()

    with RunBuilder() as builder:
        run = builder.get_run()
        run.events.new_stream.connect(lambda event: new_stream_events.append(event))
        run.events.new_data.connect(lambda event: new_data_events.append(event))
        run.events.completed.connect(lambda event: completed_events.append(event))
        run.events.new_doc.connect(lambda event: new_doc_events.append(event))
        assert not new_stream_events
        assert not new_data_events
        assert not completed_events

        builder.add_stream("primary", data={"a": [1, 2, 3]})
        assert len(new_stream_events) == 1
        assert new_stream_events[0].run is run
        assert new_stream_events[0].name == "primary"
        assert len(new_data_events) == 1
        assert new_data_events[0].run is run
        assert new_data_events[0].updated == {"primary": 3}
        assert len(completed_events) == 0

        builder.add_data("primary", data={"a": [1, 2, 3]})
        assert len(new_stream_events) == 1
        assert len(new_data_events) == 2
        assert new_data_events[1].run is run
        assert new_data_events[0].updated == {"primary": 3}
        assert len(completed_events) == 0

    # Exiting the context issues a 'stop' document....

    assert len(new_stream_events) == 1
    assert len(new_data_events) == 2
    assert len(completed_events) == 1
    assert completed_events[0].run is run

    actual_docs = [(ev.name, ev.doc) for ev in new_doc_events]
    # Omit first ('start') doc because subscriber is too late to see it.
    expected_docs = list(run.documents(fill="no"))[1:]
    assert actual_docs == expected_docs


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


def test_write_lock():
    "It should not be possible to add data without holding the write_lock."

    def worker(run, locked, failed):
        "Check that no data is added while the write_lock is held."
        with run.write_lock:
            # Signal that we have the lock.
            locked.set()
            if len(run.primary.read()["a"]):
                failed.set()
            time.sleep(0.1)
            if len(run.primary.read()["a"]):
                failed.set()
            time.sleep(0.1)
            if len(run.primary.read()["a"]):
                failed.set()

    with RunBuilder() as builder:
        run = builder.get_run()
        locked = threading.Event()
        failed = threading.Event()
        builder.add_stream(
            "primary",
            data_keys={"a": {"shape": [], "dtype": "number", "source": "stuff"}},
        )
        thread = threading.Thread(target=worker, args=(run, locked, failed))
        thread.start()
        locked.wait()
        # This should be blocked until the lock is released.
        # We'll check below that it was.
        builder.add_data("primary", {"a": [1, 2, 3]})
        time.sleep(0.1)
        # But it should have data now...
        assert bool(len(run.primary.read()["a"]))
        thread.join()
        assert not failed.is_set()
