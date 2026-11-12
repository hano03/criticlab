"""last_returns.py: the same 'end-of-run training return' from CleanRL logs and our TensorBoard runs."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "last_returns", Path(__file__).resolve().parents[1] / "scripts" / "last_returns.py")
last_returns = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(last_returns)


def test_parses_both_cleanrl_print_formats(tmp_path):
    log = tmp_path / "ref.log"
    log.write_text("SPS: 120\n"
                   "global_step=95000, episodic_return=[310.5]\n"
                   "global_step=99000, episodic_return=290.0\n"
                   "global_step=80000, episodic_return=[100.0]\n", encoding="utf-8")
    pairs = last_returns.from_log(log)
    assert pairs == [(95000, 310.5), (99000, 290.0), (80000, 100.0)]
    last_step, n, mean, best = last_returns.summarise(pairs, window=10_000)
    assert (last_step, n, mean, best) == (99000, 2, 300.25, 310.5)  # 80k is outside the window


def test_reads_our_tensorboard_runs(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary

    run = tmp_path / "runs" / "Hopper-v5__td3__1__0"
    writer = EventFileWriter(str(run))
    for step, value in [(50_000, 100.0), (95_000, 300.0), (100_000, 340.0)]:
        summary = Summary(value=[Summary.Value(tag="charts/episodic_return", simple_value=value)])
        writer.add_event(Event(step=step, summary=summary))
    writer.close()
    (name, pairs), = last_returns.sources([str(tmp_path / "runs")])
    assert name == "Hopper-v5__td3__1__0"
    assert last_returns.summarise(pairs, 10_000)[1:3] == (2, 320.0)


def test_empty_log_reports_nothing(tmp_path):
    log = tmp_path / "empty.log"
    log.write_text("nothing here\n", encoding="utf-8")
    assert last_returns.summarise(last_returns.from_log(log), 10_000) is None
