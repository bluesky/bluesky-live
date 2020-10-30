import pytest

from ..bluesky_run import BlueskyRun, DocumentCache


# BlueskyRun is mostly tested via test_run_builder.py. This exercises corner
# cases.


def test_empty():
    "Bluesky needs a DocumentCache with at *least* a 'start' doc."
    dc = DocumentCache()
    with pytest.raises(ValueError):
        BlueskyRun(dc)
