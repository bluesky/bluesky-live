from ..run_builder import (
    RunBuilder,
    build_runs_from_documents,
    build_one_run_from_documents,
)


def test_build_one_run_from_docs():
    "Test build_one_run_from_documents."
    with RunBuilder(metadata={"a": "b"}) as builder:
        builder.add_stream("primary", data={"x": [1, 2, 3]})
        builder.add_stream("baseline", data={"y": [10, 20]})
    run_from_builder = builder.get_run()
    expected = list(run_from_builder.documents(fill="no"))

    run_from_docs = build_one_run_from_documents(expected)
    actual = list(run_from_docs.documents(fill="no"))
    assert actual == expected


def test_build_runs_from_docs():
    "Test build_one_run_from_documents."
    with RunBuilder(metadata={"a": "b"}) as builder1:
        builder1.add_stream("primary", data={"x": [1, 2, 3]})
        builder1.add_stream("baseline", data={"y": [10, 20]})
    with RunBuilder(metadata={"a": "b"}) as builder2:
        builder2.add_stream("primary", data={"x": [1, 2, 3]})
        builder2.add_stream("baseline", data={"y": [10, 20]})
    run_from_builder1 = builder1.get_run()
    run_from_builder2 = builder2.get_run()
    expected1 = list(run_from_builder1.documents(fill="no"))
    expected2 = list(run_from_builder2.documents(fill="no"))

    def combined():
        "Yield the documents from the two runs, mixed up."
        gen1 = iter(expected1)
        gen2 = iter(expected2)
        # Do enough mixing to exercise the RunRouter in a nontrivial fashion.
        yield next(gen1)
        yield next(gen2)
        yield next(gen1)
        yield next(gen2)
        yield from gen1
        yield from gen2

    runs = build_runs_from_documents(combined())
    actual1 = list(runs[0].documents(fill="no"))
    actual2 = list(runs[1].documents(fill="no"))
    assert actual1 == expected1
    assert actual2 == expected2
