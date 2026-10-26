import pytest
from conftest import Flaky

from vlmbench.models import load_model
from vlmbench.runner import RunConfig, Runner, read_records


def _cfg(tmp_path, dataset, **kw):
    return RunConfig(dataset=dataset.name, data_dir=str(dataset.data_dir), output_dir=str(tmp_path / "runs"), **kw)


def test_run_resume_and_identity_check(tmp_path, vistaqa):
    samples = vistaqa.load()
    cfg = _cfg(tmp_path, vistaqa, model="dummy:echo", concurrency=2)
    summary = Runner(load_model(cfg.model), vistaqa, cfg).run(samples)
    assert summary["n_questions"] == 12 and summary["n_errors"] == 0
    assert cfg.run_dir().name == "vistaqa__dummy__echo__default"

    lines = (cfg.run_dir() / "responses.jsonl").read_text().count("\n")
    Runner(load_model(cfg.model), vistaqa, cfg).run(samples)  # resume: nothing re-run
    assert (cfg.run_dir() / "responses.jsonl").read_text().count("\n") == lines

    other = _cfg(tmp_path, vistaqa, model="dummy:random", run_name=cfg.run_dir().name)
    with pytest.raises(RuntimeError):  # different model under the same run name
        Runner(load_model(other.model), vistaqa, other).run(samples)


def test_retries_then_error_then_retry_errors(tmp_path, vistaqa):
    one = vistaqa.load()[:1]
    cfg = _cfg(tmp_path, vistaqa, model="dummy:x", max_retries=2, batch_size=4)
    Runner(Flaky(fail_times=2), vistaqa, cfg).run(one)
    rec = read_records(cfg.run_dir() / "responses.jsonl")["1"]
    assert rec["error"] is None and rec["attempts"] == 3

    cfg2 = _cfg(tmp_path, vistaqa, model="dummy:y", max_retries=1)
    assert Runner(Flaky(fail_times=99), vistaqa, cfg2).run(one)["n_errors"] == 1
    cfg2.retry_errors = True
    assert Runner(Flaky(fail_times=0), vistaqa, cfg2).run(one)["n_errors"] == 0
