import importlib.util
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager

import pytest

spec = importlib.util.spec_from_file_location(
    "mme_realworld", Path(__file__).parents[1] / "scripts/mme_realworld.py")
mme = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mme)


def example(index, category, image):
    return {"Question_id": f"q{index}", "Image": image, "Text": "Which?",
            "Answer choices": [f"({c}) choice {c.lower()}" for c in "ABCDE"],
            "Ground truth": "B", "Task": "Perception", "Subtask": "Monitoring",
            "Category": category}


def test_sampling_balances_tasks_and_keeps_images_disjoint():
    # Deliberate sharing between tasks exercises augmenting-path assignment.
    rows = [example(i, "person/counting", f"image-{i}") for i in range(30)]
    rows += [example(30+i, "vehicle/counting", f"image-{i}") for i in range(15, 45)]
    rows += [example(90+i, "vehicle/location", f"image-{60+i}") for i in range(3)]
    rows += [example(200, "person/counting", "duplicate-1"),
             example(200, "person/counting", "duplicate-2")]
    batches, audit = mme.select_sets(rows, seed=42, size=15)
    assert [len(b) for b in batches] == [15, 15]
    assert len({r["Image"] for b in batches for r in b}) == 30
    assert not any(r["Question_id"] == "q200" for b in batches for r in b)
    assert audit["excluded_duplicate_ids"] == {"q200": 2}
    scarce = next(r for r in audit["tasks"] if r["task"] == "vehicle/location")
    assert scarce["batch_1"] == scarce["batch_2"] == 1
    assert mme.select_sets(rows, seed=42, size=15) == (batches, audit)


def test_official_monitoring_attribute_groups():
    assert mme.task_key(example(1, "Person/Attribute/Color", "a")) == (
        "Monitoring", "Perception", "person/attribute")
    assert mme.task_key(example(2, "vehicle/attribute/orientation", "b"))[-1] == "vehicle/attribute"


@pytest.mark.parametrize("text,expected", [("", ""), ("B", "B"),
    ("(B)", "B"), ("The best answer is B.", "B"),
    ("The correct option is C", "C"), ("choice b", "B")])
def test_extraction(text, expected):
    assert mme.extract_answer(text, example(0, "x", "x")["Answer choices"]) == expected


def test_aggregates_distinguish_pending_errors_and_wrong_answers(tmp_path):
    rows = [{**example(i, "person/counting", str(i)), "sample_id": f"s{i}",
             "task_key": ["Monitoring", "Perception", "person/counting"]} for i in range(4)]
    for sid, status, correct in [("s0", "completed", True), ("s1", "completed", False), ("s2", "error", False)]:
        mme.append(tmp_path / "responses.jsonl", {"sample_id": sid, "status": status,
            "correct": correct, "latency_s": 2, "prediction": "B" if correct else "A"})
    summary = mme.summarize(tmp_path, rows)
    assert summary["overall"]["accuracy_completed"] == .5
    assert summary["overall"]["accuracy_selected"] == .25
    assert summary["overall"]["errors"] == 1
    assert summary["overall"]["pending"] == 1
    assert summary["per_domain"]["Monitoring"] == summary["overall"]
    assert len(json.loads((tmp_path / "per_example.json").read_text())) == 4


@pytest.mark.parametrize("model,budget,expected,provider", [(mme.MODEL, 5, 2, None), (mme.PAID_MODEL, 5, 2, None),
    (mme.PAID_MODEL, .001, 0, None), (mme.PAID_MODEL, 5, 2, "bf16-fallback")])
def test_runner_saves_reasoning_and_resumes_without_repeating_requests(tmp_path, monkeypatch, model, budget, expected, provider):
    import httpx
    monkeypatch.setattr(mme, "ROOT", tmp_path)
    monkeypatch.setattr(mme, "DATA", tmp_path / "data")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-never-log")
    data = tmp_path / "data"
    data.mkdir()
    image_bytes = b"already-verified-test-image"
    (data / "image.jpg").write_bytes(image_bytes)
    rows = [{**example(i, "person/counting", str(i)), "sample_id": f"s{i}",
             "task_key": ["Monitoring", "Perception", "person/counting"]} for i in range(2)]
    mme.dump(data / "batch_1.json", rows)
    mme.dump(data / "images_batch_1.json", {r["sample_id"]: {
        "path": "image.jpg", "mime": "image/jpeg", "sha256": hashlib.sha256(image_bytes).hexdigest()}
        for r in rows})
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def get(self, url):
            return httpx.Response(200, json={"data": {"is_free_tier": False}})

        def post(self, url, json):
            calls.append(json)
            return httpx.Response(200, json={"id": "fake-response", "choices": [{
                "finish_reason": "stop", "message": {"content": "B", "reasoning": "test reasoning"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                          "completion_tokens_details": {"reasoning_tokens": 9}}})

        @contextmanager
        def stream(self, method, url, json, **kwargs):
            yield self.post(url, json)

        def close(self):
            pass

    monkeypatch.setattr(httpx, "Client", Client)
    args = SimpleNamespace(batch=1, output="run", interval=0, concurrency=2, max_requests=2,
                           limit=None, model=model, max_cost_usd=budget, provider=provider)
    mme.run(args)
    mme.run(args)
    assert len(calls) == expected
    assert all(c["reasoning"]["enabled"] for c in calls)
    if provider == "bf16-fallback":
        assert all(c["provider"]["allow_fallbacks"] for c in calls)
        assert all(c["provider"]["quantizations"] == ["bf16"] for c in calls)
    assert all("Ground truth" not in str(c["messages"]) for c in calls)
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["overall"]["completed"] == expected
    assert summary["overall"]["accuracy_completed"] == (1 if expected else None)
    assert summary["usage"]["reasoning_tokens"] == expected * 9
    outputs = list((tmp_path / "run").iterdir())
    assert all("test-secret-never-log" not in p.read_text() for p in outputs)


@pytest.mark.parametrize("text,expected", [
    ("Based on the image, the correct answer is **(E)**.", "E"),
    ("There is a B sign and a C sign.", ""),
    ("The answer is A. The answer is B.", ""),
    ("(C)", "C"), ("", "")])
def test_explicit_answer_supplements_the_official_regex(text, expected):
    assert mme.explicit_answer(text) == expected


def test_pacer_spaces_every_attempt_and_applies_global_pause(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(mme.time, "monotonic", lambda: clock[0])

    class Stop:
        def is_set(self):
            return False

        def wait(self, seconds):
            clock[0] += seconds

    pacer = mme.Pacer(3.5)
    times = []
    for _ in range(3):
        assert pacer.wait(Stop())
        times.append(clock[0])
    assert times == [0, 3.5, 7]
    pacer.pause(10)
    assert pacer.wait(Stop())
    assert clock[0] == 17


@pytest.mark.parametrize("status,raw,expected", [
    (520, {}, True), (522, {}, True), (503, {}, True), (429, {}, True),
    (None, None, True), (400, {}, False), (401, {}, False), (402, {}, False),
    (402, {"error": {"code": 402, "metadata": {"limit_source": "openrouter_in_flight_budget"}}}, True),
    (200, {"error": {"code": 429}}, True), (200, {"error": {"code": 403}}, False),
])
def test_retry_classification_includes_provider_520_and_transient_budgets(status, raw, expected):
    assert mme.retryable_error(status, raw) == expected


def test_live_controls_increase_slots_without_restarting_or_ignoring_rate_limiter(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    pacer = mme.Pacer(3.5)
    changes = []
    slots = mme.LiveSlots(1, pacer, tmp_path / "control.json", lambda n, i: changes.append((n, i)), paid=True)
    stop = threading.Event()
    assert slots.enter(stop)
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(slots.enter, stop)
        with pytest.raises(TimeoutError):
            waiting.result(timeout=.02)
        mme.dump(tmp_path / "control.json", {"concurrency": 2, "interval_s": 2})
        assert waiting.result(timeout=1)
    assert slots.active == 2
    assert pacer.interval == 2
    assert changes == [(2, 2)]
    slots.leave()
    slots.leave()
    assert slots.active == 0


def test_live_slots_obey_available_budget_even_when_configured_concurrency_is_higher(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    capacity = [1]
    slots = mme.LiveSlots(4, mme.Pacer(2), tmp_path / "control.json", lambda *_: None,
                          paid=True, budget_capacity=lambda: capacity[0])
    stop = threading.Event()
    assert slots.enter(stop)
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(slots.enter, stop)
        with pytest.raises(TimeoutError):
            waiting.result(timeout=.02)
        capacity[0] = 2
        assert waiting.result(timeout=1)
    assert slots.active == 2
    slots.leave()
    slots.leave()


def test_combined_results_keep_batch_request_numbers_distinct(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import mme_campaign
    monkeypatch.setattr(mme_campaign, "ROOT", tmp_path)
    monkeypatch.setattr(mme_campaign.mme, "DATA", tmp_path / "data")
    mme.dump(tmp_path / "data/sampling_audit.json", {"task_count": 1})
    for batch in (1, 2):
        row = {**example(batch, "person/counting", f"image-{batch}"),
               "sample_id": f"b{batch}", "task_key": ["Monitoring", "Perception", "person/counting"]}
        mme.dump(tmp_path / f"data/batch_{batch}.json", [row])
        run = tmp_path / f"runs/mme_gemma4_reasoning_paid_32k_batch{batch}"
        run.mkdir(parents=True)
        mme.append(run / "responses.jsonl", {"sample_id": row["sample_id"], "status": "completed",
                   "response": "B", "correct": True, "prediction": "B", "latency_s": 1})
        for event in ("started", "finished"):
            mme.append(run / "attempts.jsonl", {"event": event, "request_number": 1,
                       "sample_id": row["sample_id"]})
    summary = mme_campaign.combine()
    assert summary["overall"]["completed"] == 2
    assert summary["requests_started"] == 2
    assert summary["requests_without_durable_outcome"] == 0
    assert summary["overall"]["accuracy_completed"] == 1


def test_campaign_resumes_only_when_settled_reservations_leave_funds(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    import mme_campaign
    mme.dump(tmp_path / "config.json", {"max_cost_usd": 1, "max_tokens": 32768})
    mme.append(tmp_path / "attempts.jsonl", {"event": "started", "request_number": 1})
    mme.append(tmp_path / "attempts.jsonl", {"event": "finished", "request_number": 1,
                                          "accounted_cost_usd": .90})
    assert mme_campaign.can_resume_reservation_stop(tmp_path)
    # A request with no durable outcome must retain its full reservation.
    mme.append(tmp_path / "attempts.jsonl", {"event": "started", "request_number": 2})
    assert not mme_campaign.can_resume_reservation_stop(tmp_path)


def test_midstream_provider_error_is_retried_and_never_scored_correct():
    r = {"sample_id": "x", "status": "completed", "correct": True, "prediction": "B",
         "finish_reason": "error", "response": "B"}
    normalized = mme.normalize_record(r)
    assert normalized["status"] == "error"
    assert normalized["correct"] is False
    assert normalized["prediction"] == ""
    assert r["status"] == "completed"  # preserve the original history record


def test_historical_http200_interruption_is_reclassified_and_repair_keeps_failure_count(tmp_path):
    sample = {**example(1, "person/counting", "image"), "sample_id": "s1",
              "task_key": ["Monitoring", "Perception", "person/counting"]}
    mme.append(tmp_path / "responses.jsonl", {"sample_id": "s1", "status": "completed",
        "request_number": 1, "finish_reason": "error", "correct": True,
        "prediction": "B", "response": "B", "latency_s": 1})
    for event in ("started", "finished"):
        mme.append(tmp_path / "attempts.jsonl", {"sample_id": "s1", "event": event,
                   "request_number": 1})
    failed = mme.summarize(tmp_path, [sample])
    assert failed["overall"]["completed"] == 0
    assert failed["overall"]["errors"] == 1
    assert failed["requests_failed"] == 1
    mme.append(tmp_path / "responses.jsonl", {"sample_id": "s1", "status": "completed",
        "request_number": 2, "finish_reason": "stop", "correct": True,
        "prediction": "B", "response": "B", "latency_s": 1})
    repaired = mme.summarize(tmp_path, [sample])
    assert repaired["overall"]["completed"] == 1
    assert repaired["overall"]["correct"] == 1
    assert repaired["overall"]["errors"] == 0
    assert repaired["requests_failed"] == 1


def test_retry_changes_backend_after_midstream_failure_without_changing_precision_or_price():
    options = {"only": ["CoreWeave", "Io Net", "NextBit"],
               "order": ["CoreWeave", "Io Net", "NextBit"], "quantizations": ["bf16"],
               "max_price": {"prompt": .1, "completion": .33}}
    raw = {"provider": "Io Net", "choices": [{"finish_reason": "error"}]}
    retry = mme.retry_provider_options(options, raw)
    assert retry["only"] == ["CoreWeave", "NextBit"]
    assert retry["quantizations"] == ["bf16"]
    assert retry["max_price"] == options["max_price"]
    assert "Io Net" in options["only"]
    stalled = mme.retry_provider_options(options, {"error": {"code": 408}, "choices": [{"finish_reason": "error"}]})
    assert stalled["order"] == ["Io Net", "NextBit", "CoreWeave"]
    assert stalled["only"] == options["only"]


def test_stream_reader_preserves_reasoning_final_answer_usage_and_raw_trace(tmp_path):
    import httpx, gzip
    events = [
        {"id": "g1", "provider": "CoreWeave", "choices": [{"delta": {"reasoning": "Think "}}]},
        {"choices": [{"delta": {"reasoning": "carefully."}}]},
        {"choices": [{"delta": {"content": "B"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": .001}},
    ]
    body = "\n\n".join("data: " + json.dumps(e) for e in events) + "\n\ndata: [DONE]\n\n"
    response = httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    trace = tmp_path / "trace.gz"
    raw = mme.read_completion_stream(response, trace, mme.time.monotonic())
    assert raw["choices"][0]["message"]["content"] == "B"
    assert raw["choices"][0]["message"]["reasoning"] == "Think carefully."
    assert raw["choices"][0]["finish_reason"] == "stop"
    assert raw["usage"]["cost"] == .001
    assert len(gzip.open(trace, "rt").readlines()) == 4


def test_stream_reader_allows_long_active_reasoning_but_aborts_a_stalled_provider(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(mme.time, "monotonic", lambda: clock[0])

    class Response:
        headers = {"content-type": "text/event-stream"}

        def __init__(self, stall, steps=11):
            self.stall = stall
            self.steps = steps

        def iter_lines(self):
            for n in range(self.steps):
                clock[0] += 100 if not self.stall else 121
                yield "data: " + json.dumps({"id": "g1", "provider": "CoreWeave",
                    "choices": [{"delta": {"reasoning": "thinking "}}]})
            yield 'data: {"choices":[{"delta":{"content":"B"},"finish_reason":"stop"}]}'
            yield "data: [DONE]"

    active = mme.read_completion_stream(Response(False), tmp_path / "active.gz", 0)
    assert active["choices"][0]["finish_reason"] == "stop"
    assert active["stream_metrics"]["elapsed_s"] == 1100
    clock[0] = 0
    overlong = mme.read_completion_stream(Response(False, 17), tmp_path / "overlong.gz", 0)
    assert overlong["skipped_over_20m"]
    assert mme.normalize_record({"raw_response": overlong})["status"] == "skipped"
    clock[0] = 0
    stalled = mme.read_completion_stream(Response(True), tmp_path / "stalled.gz", 0)
    assert stalled["error"]["code"] == 408
    assert stalled["choices"][0]["finish_reason"] == "error"


def test_finished_answer_is_preserved_if_usage_delivery_stalls(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(mme.time, "monotonic", lambda: clock[0])
    class Response:
        headers = {"content-type": "text/event-stream"}
        def iter_lines(self):
            clock[0] = 1
            yield 'data: {"choices":[{"delta":{"content":"B"},"finish_reason":"stop"}]}'
            clock[0] = 201
            yield ': keepalive'
    raw = mme.read_completion_stream(Response(), tmp_path / 'complete.gz', 0)
    assert raw['choices'][0]['message']['content'] == 'B'
    assert raw['choices'][0]['finish_reason'] == 'stop'
    assert 'error' not in raw
    assert 'stream_warning' in raw
