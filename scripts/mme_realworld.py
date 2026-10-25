"""Reproducible, task-balanced MME-RealWorld sampling and OpenRouter evaluation.

Run with the repository's virtualenv. Credentials are read only from environment/.env.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import gzip
import json
import os
from pathlib import Path
import random
import re
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/mme-realworld"
MODEL = "google/gemma-4-26b-a4b-it:free"
PAID_MODEL = "google/gemma-4-26b-a4b-it"
REPO = "yifanzhang114/MME-RealWorld-Lmms-eval"
REVISION = "f157390129e8d2b60757eb68d340aade9446b52b"
ANNOTATIONS_URL = "https://huggingface.co/datasets/yifanzhang114/MME-RealWorld/resolve/main/MME_RealWorld.json"


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def append(path, record):
    with Path(path).open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def task_key(row):
    category = row["Category"].lower()
    # The official evaluator pools color/orientation for monitoring attributes.
    if category.startswith(("person/attribute/", "vehicle/attribute/")):
        category = "/".join(category.split("/")[:2])
    return (row["Subtask"], row["Task"], category)


def download(url, path, expected_size=None):
    path = Path(path)
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=180) as response, temp.open("wb") as out:
                while chunk := response.read(4 * 1024 * 1024):
                    out.write(chunk)
            if expected_size is not None and temp.stat().st_size != expected_size:
                raise ValueError(f"Incomplete download: {path.name}")
            temp.replace(path)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def select_sets(rows, seed=42, size=1000):
    """Allocate equally per evaluation task, then match slots to unique images."""
    rng = random.Random(seed)
    counts = Counter(r["Question_id"] for r in rows)
    # Exclude both records of conflicting IDs; don't silently choose one of them.
    clean = [(i, r) for i, r in enumerate(rows) if counts[r["Question_id"]] == 1]
    groups = defaultdict(list)
    for i, r in clean:
        groups[task_key(r)].append((i, r))
    keys = sorted(groups)
    allocations = []
    for batch in range(2):
        quota = {k: 0 for k in keys}
        order = keys[:]
        rng.shuffle(order)
        # Water filling also handles small groups, preserving equal quotas between sets.
        while sum(quota.values()) < size:
            changed = False
            for key in order:
                capacity = len({r["Image"] for _, r in groups[key]}) // 2
                if quota[key] < capacity and sum(quota.values()) < size:
                    quota[key] += 1
                    changed = True
            if not changed:
                raise ValueError("Insufficient examples for two balanced sets")
        allocations.append(quota)
    choices = {}
    row_by_image = {}
    for key in keys:
        images = defaultdict(list)
        for i, row in groups[key]:
            images[row["Image"]].append((i, row))
        choices[key] = sorted(images)
        rng.shuffle(choices[key])
        row_by_image[key] = images
    # Bipartite augmenting-path matching enforces image disjointness across ALL slots.
    slots = [(batch, key, slot) for batch in range(2) for key in keys
             for slot in range(allocations[batch][key])]
    slots.sort(key=lambda s: len(choices[s[1]]))
    image_owner = {}
    assigned = {}
    sys.setrecursionlimit(max(10000, len(slots) * 3))

    def match(slot, visited):
        for image in choices[slot[1]]:
            if image in visited:
                continue
            visited.add(image)
            owner = image_owner.get(image)
            if owner is None or match(owner, visited):
                image_owner[image] = slot
                assigned[slot] = image
                return True
        return False

    for slot in slots:
        if not match(slot, set()):
            raise ValueError("Cannot satisfy image-disjoint quotas; revise allocation explicitly")
    result = [[], []]
    for slot in slots:
        batch, key, _ = slot
        i, row = rng.choice(row_by_image[key][assigned[slot]])
        result[batch].append({**row, "source_index": i, "task_key": list(key),
                              "sample_id": f"mme-{i:05d}"})
    for batch in result:
        rng.shuffle(batch)
    assert all(len(batch) == size for batch in result)
    assert len({r["Question_id"] for batch in result for r in batch}) == 2 * size
    assert len({r["Image"] for batch in result for r in batch}) == 2 * size
    audit = {
        "seed": seed, "source_rows": len(rows), "eligible_rows": len(clean),
        "excluded_duplicate_ids": {k: n for k, n in counts.items() if n > 1},
        "task_count": len(keys), "image_overlap": 0, "question_overlap": 0,
        "tasks": [{"domain": k[0], "type": k[1], "task": k[2],
                   "available": len(groups[k]),
                   "available_images": len(choices[k]),
                   "batch_1": allocations[0][k], "batch_2": allocations[1][k]} for k in keys],
        "domains": [dict(Counter(r["Subtask"] for r in batch)) for batch in result],
    }
    return result, audit


def prepare(seed):
    download(ANNOTATIONS_URL, DATA / "annotations.json")
    source = DATA / "annotations.json"
    rows = json.loads(source.read_text())
    batches, audit = select_sets(rows, seed)
    audit.update({"annotations_url": ANNOTATIONS_URL,
                  "annotations_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "image_repo": REPO, "image_revision": REVISION})
    for i, batch in enumerate(batches, 1):
        dump(DATA / f"batch_{i}.json", batch)
    dump(DATA / "sampling_audit.json", audit)
    import csv
    with (DATA / "sampling_task_counts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit["tasks"][0]))
        writer.writeheader()
        writer.writerows(audit["tasks"])
    print(json.dumps(audit, indent=2), flush=True)


def cache_images(batch_number):
    import pyarrow.parquet as pq
    from PIL import Image
    batch = json.loads((DATA / f"batch_{batch_number}.json").read_text())
    wanted = {r["source_index"]: r for r in batch}
    image_dir = DATA / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA / f"images_batch_{batch_number}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    wanted = {i: r for i, r in wanted.items()
              if r["sample_id"] not in manifest or not (DATA / manifest[r["sample_id"]]["path"]).exists()}
    tree = json.load(urllib.request.urlopen(
        f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data"))
    # Fetch one shard at a time and discard after extracting selected images.
    for entry in tree:
        if not wanted:
            break
        name = entry["path"]
        if not name.endswith(".parquet"):
            continue
        print(f"Image cache: {len(batch)-len(wanted)}/{len(batch)}; reading {name} ({entry['size']/1e6:.0f} MB)", flush=True)
        shard = DATA / "shards" / Path(name).name
        download(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}", shard, entry["size"])
        parquet = pq.ParquetFile(shard)
        for chunk in parquet.iter_batches(batch_size=4):
            for row in chunk.to_pylist():
                index = row["index"]
                if index not in wanted:
                    continue
                original = wanted[index]
                assert row["question"] == original["Text"], (index, "question mismatch")
                assert row["multi-choice options"] == original["Answer choices"], (index, "choices mismatch")
                assert row["answer"] == original["Ground truth"], (index, "answer mismatch")
                image_bytes = base64.b64decode(row["bytes"], validate=True)
                with Image.open(io.BytesIO(image_bytes)) as im:
                    width, height = im.size
                    mime = Image.MIME[im.format]
                    suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(mime)
                    if suffix is None:
                        raise ValueError(f"Unsupported image format {mime}")
                    im.verify()
                path = image_dir / (original["sample_id"] + suffix)
                path.write_bytes(image_bytes)
                manifest[original["sample_id"]] = {
                    "path": str(path.relative_to(DATA)), "width": width, "height": height,
                    "mime": mime, "bytes": len(image_bytes),
                    "sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "source_shard": name, "source_index": index,
                }
                del wanted[index]
        parquet.close()
        dump(manifest_path, manifest)
        shard.unlink()
    if wanted:
        raise RuntimeError(f"Missing {len(wanted)} images")
    print(f"Verified and cached all {len(batch)} images; no client resizing or re-encoding", flush=True)


def extract_answer(text, choices):
    """Official evaluator extraction, with its accidentally joined prefixes corrected."""
    text = (text or "").strip()
    for prefix in ["The best answer is", "The correct answer is", "The answer is", "The answer",
                   "The best option is", "The correct option is", "Best answer:", "Best option:"]:
        text = text.replace(prefix, "")
    if len(text.split()) > 10 and not re.search("[ABCDE]", text):
        return ""
    match = re.search("[ABCDE]", text)
    if match:
        return match[0]
    if text:
        for choice in choices:
            if text.lower() in choice.lower():
                return choice[1]
    return ""


def explicit_answer(text):
    """Supplemental, conservative extraction when a model ignores letter-only format.

    Keep the official score separately; never infer an answer from arbitrary prose.
    """
    text = (text or "").replace("**", "").strip()
    match = re.fullmatch(r"\(?([A-E])\)?[.\s]*", text)
    if match:
        return match[1]
    matches = re.findall(r"(?:correct|best|final)?\s*(?:answer|option)\s*(?:is\s*:?\s*|[:=]\s*)\(?([A-E])\)?\b", text, re.I)
    if matches and len({m.upper() for m in matches}) == 1:
        return matches[-1].upper()
    return ""


def prompt(row):
    return (row["Text"] + "\nThe choices are listed below:\n" + "\n".join(row["Answer choices"])
            + "\nSelect the best answer to the above multiple-choice question based on the image. "
            "Respond with only the letter (A, B, C, D, or E) of the correct option.\nThe best answer is:")


def api_error_status(http_status, raw):
    error = (raw or {}).get("error")
    if isinstance(error, dict):
        try:
            return int(error["code"])
        except (KeyError, ValueError, TypeError):
            pass
    return 502 if http_status == 200 else http_status


def retryable_error(http_status, raw):
    code = api_error_status(http_status, raw)
    error = (raw or {}).get("error")
    metadata = error.get("metadata", {}) if isinstance(error, dict) else {}
    return (code is None or code in (408, 429) or 500 <= code < 600
            or (code == 402 and metadata.get("limit_source") == "openrouter_in_flight_budget"))


def normalize_record(record):
    """Provider interruptions are infrastructure failures, including HTTP 200 errors."""
    if record.get("status") == "skipped" or (record.get("raw_response") or {}).get("skipped_over_20m"):
        return {**record, "status": "skipped", "correct": False, "prediction": ""}
    choice = (((record.get("raw_response") or {}).get("choices")) or [{}])[0]
    if record.get("finish_reason") == "error" or choice.get("finish_reason") == "error":
        return {**record, "status": "error", "correct": False, "prediction": "",
                "error": record.get("error") or "Provider interruption: finish_reason=error",
                "classification_note": "Provider interruption requires a retry, not model scoring"}
    return record


def retry_provider_options(options, raw):
    """Try another equivalent backend after an identified mid-generation failure."""
    choices = (raw or {}).get("choices") or [{}]
    failed = (raw or {}).get("provider")
    allowed = options.get("only", [])
    if choices[0].get("finish_reason") == "error" and failed in allowed and len(allowed) > 1:
        remaining = [name for name in allowed if name != failed]
        return {**options, "only": remaining,
                "order": [name for name in options.get("order", remaining) if name != failed]}
    error = (raw or {}).get("error") or {}
    if not failed and error.get("code") == 408 and len(allowed) > 1:
        order = list(options.get("order", allowed))
        return {**options, "order": order[1:] + order[:1]}
    return options


def read_completion_stream(response, trace_path, t0, first_token_s=120, idle_s=120, max_wall_s=1200):
    """Read SSE reasoning/content with progress deadlines and a compressed raw trace."""
    if "text/event-stream" not in response.headers.get("content-type", ""):
        body = b"".join(response.iter_bytes()).decode("utf-8", errors="replace")
        try:
            return json.loads(body)
        except ValueError:
            return {"body": body[:4000]}
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    raw = {"object": "chat.completion", "choices": []}
    content, reasoning, details = [], [], []
    finish = None
    first_output = None
    last_progress = t0
    count = 0
    try:
        with gzip.open(trace_path, "wt", compresslevel=1) as trace:
            for line in response.iter_lines():
                now = time.monotonic()
                if now-t0 > max_wall_s or (first_output is None and now-t0 > first_token_s) or (first_output is not None and now-last_progress > idle_s):
                    if finish is not None and finish != "error":
                        raw["stream_warning"] = "Usage/stream termination stalled after the model finish marker"
                        break
                    raw["error"] = {"code": 408, "message": "Stream stalled or exceeded its progress deadline"}
                    if now-t0 > max_wall_s:
                        raw["skipped_over_20m"] = True
                        raw["error"]["message"] = "Skipped: exceeded user-requested 20-minute request limit"
                    finish = "error"
                    break
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if not data:
                    continue
                event = json.loads(data)
                trace.write(json.dumps({"received_at": datetime.now(timezone.utc).isoformat(), "event": event}, ensure_ascii=False) + "\n")
                count += 1
                if count % 100 == 0:
                    trace.flush()
                for name in ("id", "model", "provider", "created", "system_fingerprint"):
                    if event.get(name) is not None:
                        raw[name] = event[name]
                if event.get("usage") is not None:
                    raw["usage"] = event["usage"]
                if event.get("error"):
                    raw["error"] = event["error"]
                    finish = "error"
                for choice in event.get("choices") or []:
                    if choice.get("index", 0) != 0:
                        continue
                    delta = choice.get("delta") or choice.get("message") or {}
                    text = delta.get("content") or ""
                    thought = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    chunks = delta.get("reasoning_details") or []
                    if not thought:
                        thought = "".join(d.get("text") or d.get("summary") or "" for d in chunks)
                    details.extend(d for d in chunks if d.get("type") == "reasoning.encrypted")
                    if text:
                        content.append(text if isinstance(text, str) else json.dumps(text))
                    if thought:
                        reasoning.append(thought)
                    if text or thought or chunks:
                        last_progress = now
                        if first_output is None:
                            first_output = now-t0
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
                if finish is not None and finish != "error" and raw.get("usage") is not None:
                    break
    except Exception as exc:
        if finish is not None and finish != "error":
            raw["stream_warning"] = f"Connection closed after finish marker: {type(exc).__name__}"
        else:
            raw["error"] = {"code": 408, "message": f"Streaming {type(exc).__name__}: {str(exc)[:500]}"}
            finish = "error"
    if finish is None:
        finish = "error"
        raw["error"] = {"code": 502, "message": "Stream ended without a completion finish marker"}
    raw["choices"] = [{"index": 0, "finish_reason": finish, "message": {
        "role": "assistant", "content": "".join(content), "reasoning": "".join(reasoning),
        "reasoning_details": details}}]
    raw["stream_trace"] = str(trace_path)
    raw["stream_metrics"] = {"events": count, "time_to_first_output_s": first_output,
                             "elapsed_s": round(time.monotonic()-t0, 3)}
    return raw


def summarize(run_dir, samples):
    response_history = [normalize_record(r) for r in read_jsonl(run_dir / "responses.jsonl")]
    records = {r["sample_id"]: r for r in response_history}
    attempts = read_jsonl(run_dir / "attempts.jsonl")
    by_sample = defaultdict(list)
    for attempt in attempts:
        by_sample[attempt["sample_id"]].append(attempt)
    rows = []
    for sample in samples:
        result = records.get(sample["sample_id"])
        if result:
            # Derive supplemental scores from the retained response, including on resume.
            result = {**result, "explicit_prediction": explicit_answer(result.get("response")),
                      "explicit_correct": explicit_answer(result.get("response")) == sample["Ground truth"]
                                          if result["status"] == "completed" else False,
                      "provider": (result.get("raw_response") or {}).get("provider", "unknown")}
        history = by_sample[sample["sample_id"]]
        usages = [a.get("usage") or {} for a in history if a["event"] == "finished"]
        rows.append({**sample, **(result or {}), "status": result["status"] if result else "pending",
                     "attempts": sum(a["event"] == "started" for a in history),
                     "total_request_latency_s": sum(a.get("latency_s", 0) for a in history),
                     "all_attempt_usage": {k: sum(u.get(k) or 0 for u in usages) for k in
                                           ("prompt_tokens", "completion_tokens", "total_tokens", "cost")},
                     "all_attempt_reasoning_tokens": sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0 for u in usages)})

    def metrics(group):
        completed = [r for r in group if r["status"] == "completed"]
        latencies = sorted(r["latency_s"] for r in completed)
        correct = sum(bool(r.get("correct")) for r in completed)
        return {"selected": len(group), "completed": len(completed),
                "pending": sum(r["status"] == "pending" for r in group),
                "errors": sum(r["status"] == "error" for r in group),
                "skipped": sum(r["status"] == "skipped" for r in group),
                "correct": correct,
                "accuracy_completed": correct / len(completed) if completed else None,
                "explicit_answer_accuracy_completed": sum(r["explicit_correct"] for r in completed)/len(completed) if completed else None,
                "official_vs_explicit_disagreements": sum(bool(r.get("explicit_prediction")) and r["explicit_prediction"] != r.get("prediction") for r in completed),
                "accuracy_selected": correct / len(group) if group else None,
                "unparsed": sum(not r.get("prediction") for r in completed),
                "letter_only_format_violations": sum(not r.get("strict_answer_format", False) for r in completed),
                "truncated": sum(r.get("finish_reason") == "length" for r in completed),
                "reasoning_evidence": sum(bool(r.get("reasoning_present")) for r in completed),
                "requests_started": sum(r["attempts"] for r in group),
                "prompt_tokens": sum(r["all_attempt_usage"]["prompt_tokens"] for r in group),
                "completion_tokens": sum(r["all_attempt_usage"]["completion_tokens"] for r in group),
                "reasoning_tokens": sum(r["all_attempt_reasoning_tokens"] for r in group),
                "reported_cost": sum(r["all_attempt_usage"]["cost"] for r in group),
                "latency_mean_s": sum(latencies)/len(latencies) if latencies else None,
                "latency_p50_s": latencies[len(latencies)//2] if latencies else None,
                "latency_p95_s": latencies[min(len(latencies)-1, int(len(latencies)*.95))] if latencies else None}

    def grouped(key):
        groups = defaultdict(list)
        for row in rows:
            groups[key(row)].append(row)
        return {key: metrics(value) for key, value in sorted(groups.items())}

    summary = {"updated_at": datetime.now(timezone.utc).isoformat(),
               "overall": metrics(rows),
               "per_domain": grouped(lambda r: r["Subtask"]),
               "per_task": grouped(lambda r: " / ".join(r["task_key"])),
               "per_raw_task": grouped(lambda r: " / ".join((r["Subtask"], r["Task"], r["Category"].lower()))),
               "per_type": grouped(lambda r: r["Task"]),
               "per_provider": grouped(lambda r: r.get("provider", "pending")),
               "requests_started": sum(r["event"] == "started" for r in attempts),
               "requests_failed": len({r["request_number"] for r in attempts
                    if r["event"] == "finished" and r.get("error") is not None} |
                    {r["request_number"] for r in response_history if r.get("finish_reason") == "error" and r.get("request_number") is not None})}
    finished_numbers = {a["request_number"] for a in attempts if a["event"] == "finished"}
    summary["requests_without_durable_outcome"] = sum(
        a["event"] == "started" and a["request_number"] not in finished_numbers for a in attempts)
    timestamps = sorted(a["timestamp"] for a in attempts if a.get("timestamp"))
    summary["first_request_at"] = timestamps[0] if timestamps else None
    summary["last_request_event_at"] = timestamps[-1] if timestamps else None
    summary["elapsed_wall_s"] = ((datetime.fromisoformat(timestamps[-1]) - datetime.fromisoformat(timestamps[0])).total_seconds()
                                 if timestamps else None)
    scored_tasks = [v["accuracy_completed"] for v in summary["per_task"].values()
                    if v["accuracy_completed"] is not None]
    summary["macro_task_accuracy_completed"] = sum(scored_tasks)/len(scored_tasks) if scored_tasks else None
    usages = [r.get("usage") or {} for r in attempts if r["event"] == "finished"]
    summary["usage"] = {k: sum(u.get(k) or 0 for u in usages)
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")}
    summary["usage"]["reasoning_tokens"] = sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0 for u in usages)
    dump(run_dir / "summary.json", summary)
    write_report(run_dir, summary)
    dump(run_dir / "per_example.json", rows)
    dump(run_dir / "skipped_examples.json", [r for r in rows if r["status"] == "skipped"])
    dump(run_dir / "over_20_minute_attempts.json", [a for a in attempts
         if a["event"] == "finished" and a.get("latency_s", 0) > 1200])
    # CSV files make all aggregates convenient to inspect without custom tooling.
    import csv
    for name in ("per_domain", "per_task", "per_raw_task", "per_type", "per_provider"):
        values = [{"group": k, **v} for k, v in summary[name].items()]
        with (run_dir / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    with (run_dir / "per_example.csv").open("w", newline="") as f:
        fields = ["sample_id", "Question_id", "Subtask", "Task", "Category", "Image", "Text",
                  "Ground truth", "status", "prediction", "correct", "explicit_prediction", "explicit_correct", "latency_s", "finish_reason",
                  "reasoning_present", "attempts", "error", "response"]
        fields.append("provider")
        fields.extend(("source_index", "Answer choices", "input_tokens", "output_tokens", "reasoning_tokens",
                       "api_cost_usd", "total_request_latency_s"))
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**r, "Answer choices": "\n".join(r["Answer choices"]),
            "input_tokens": r["all_attempt_usage"]["prompt_tokens"],
            "output_tokens": r["all_attempt_usage"]["completion_tokens"],
            "reasoning_tokens": r["all_attempt_reasoning_tokens"],
            "api_cost_usd": r["all_attempt_usage"]["cost"]} for r in rows)
    dump(run_dir / "official_results.json", [{**r, "Output": r.get("response") or ""}
                                             for r in rows if r["status"] == "completed"])
    return summary


def write_report(run_dir, summary):
    def percent(value):
        return f"{value:.2%}" if value is not None else "—"

    overall = summary["overall"]
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    lines = ["# MME-RealWorld benchmark results", "",
             f"Model: `{config.get('model', 'unknown')}`; reasoning enabled. "
             f"Completion limit: {config.get('max_tokens', 'unknown')} tokens.", "",
             f"Task-balanced English subset: {overall['selected']:,} questions across {len(summary['per_task'])} evaluation tasks.", "",
             f"Completed: **{overall['completed']}/{overall['selected']}**. "
             f"Errors: {overall['errors']}; skipped over 20 minutes: {overall['skipped']}; pending: {overall['pending']}; "
             f"truncated: {overall['truncated']}.", "",
             f"Official-extraction accuracy on completed examples: **{percent(overall['accuracy_completed'])}**. "
             f"Explicit-final-answer accuracy: **{percent(overall['explicit_answer_accuracy_completed'])}**. "
             f"Macro task accuracy: {percent(summary['macro_task_accuracy_completed'])}.", "",
             f"Reported API cost: **${summary['usage']['cost']:.6f}**. "
             f"Reasoning tokens: {summary['usage']['reasoning_tokens']:,}. "
             f"Failed API attempts: {summary['requests_failed']}.", "",
             "Scores are provisional while pending examples or infrastructure errors remain. "
             "Requests exceeding the user-requested 20-minute limit are tracked as skipped and excluded from completed-only accuracy. "
             "Official extraction can select a letter from prose before the final answer; "
             "the explicit-final-answer score is retained separately. "
             "Unparsed or truncated responses count as incorrect unless a valid answer was returned.", ""]
    for label, field in [("Domains", "per_domain"), ("Tasks", "per_task")]:
        lines.extend([f"## {label}", "", "| Group | Selected | Completed | Correct | Official accuracy | Explicit answer accuracy | Errors |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for group, values in summary[field].items():
            lines.append(f"| {group} | {values['selected']} | {values['completed']} | {values['correct']} | "
                         f"{percent(values['accuracy_completed'])} | {percent(values['explicit_answer_accuracy_completed'])} | {values['errors']} |")
        lines.append("")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


class Pacer:
    def __init__(self, interval):
        self.interval = interval
        self.next_start = time.monotonic()
        self.lock = threading.Lock()

    def wait(self, stop):
        while not stop.is_set():
            with self.lock:
                remaining = self.next_start - time.monotonic()
                if remaining <= 0:
                    self.next_start = time.monotonic() + self.interval
                    return True
            stop.wait(min(remaining, 1))
        return False

    def pause(self, seconds):
        with self.lock:
            self.next_start = max(self.next_start, time.monotonic() + seconds)


class LiveSlots:
    """Bound actual requests while allowing safe live concurrency/pacing changes."""
    def __init__(self, limit, pacer, path, on_change, paid=False, budget_capacity=None):
        self.limit = limit
        self.pacer = pacer
        self.path = path
        self.on_change = on_change
        self.minimum_interval = .5 if paid else 3.5
        self.active = 0
        self.budget_capacity = budget_capacity
        self.lock = threading.Lock()
        self.refresh_lock = threading.Lock()
        self.last_mtime = None
        if not path.exists():
            dump(path, {"concurrency": limit, "interval_s": pacer.interval})

    def refresh(self):
        with self.refresh_lock:
            try:
                mtime = self.path.stat().st_mtime_ns
                if mtime == self.last_mtime:
                    return
                control = json.loads(self.path.read_text())
                limit, interval = int(control["concurrency"]), float(control["interval_s"])
                if not 1 <= limit <= 64 or interval < self.minimum_interval:
                    return
                self.last_mtime = mtime
                if limit != self.limit or interval != self.pacer.interval:
                    with self.lock:
                        self.limit = limit
                    with self.pacer.lock:
                        self.pacer.interval = interval
                    self.on_change(limit, interval)
            except (OSError, ValueError, TypeError, KeyError):
                return

    def enter(self, stop):
        while not stop.is_set():
            self.refresh()
            funded = self.budget_capacity() if self.budget_capacity else 64
            with self.lock:
                if self.active < min(self.limit, funded):
                    self.active += 1
                    return True
            stop.wait(.25)
        return False

    def leave(self):
        with self.lock:
            self.active -= 1


def run(args):
    import httpx
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY in .env before running (key will never be logged).")
    samples_path = DATA / f"batch_{args.batch}.json"
    samples = json.loads(samples_path.read_text())
    manifest = json.loads((DATA / f"images_batch_{args.batch}.json").read_text())
    for row in (samples[:args.limit] if args.limit else samples):
        if row["sample_id"] not in manifest and getattr(args, "wait_for_images", False):
            continue
        meta = manifest[row["sample_id"]]
        if hashlib.sha256((DATA / meta["path"]).read_bytes()).hexdigest() != meta["sha256"]:
            raise SystemExit(f"Image integrity check failed: {row['sample_id']}")
    model = getattr(args, "model", MODEL)
    paid = model == PAID_MODEL
    max_tokens = getattr(args, "max_tokens", 32768)
    max_cost = getattr(args, "max_cost_usd", None)
    default_output = (f"runs/mme_gemma4_reasoning_paid_{max_tokens//1024}k_batch{args.batch}" if paid
                      else f"runs/mme_gemma4_reasoning_batch{args.batch}")
    run_dir = ROOT / (args.output or default_output)
    run_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    run_lock = (run_dir / ".run.lock").open("a")
    try:
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        run_lock.close()
        raise SystemExit("Another process is already writing this run; refusing duplicate inference.")
    prior_config_path = run_dir / "config.json"
    previous = json.loads(prior_config_path.read_text()) if prior_config_path.exists() else None
    provider_mode = getattr(args, "provider", None) or (previous or {}).get("provider_mode", "DekaLLM")
    fallback = provider_mode == "bf16-fallback"
    price = {"free": (0, 0), "DekaLLM": (.06, .33), "CoreWeave": (.1, .30), "Io Net": (.09, .31),
             "NextBit": (.0675, .225), "bf16-fallback": (.1, .33)}[provider_mode]
    input_price, output_price = (p / 1_000_000 for p in price)
    # Reserve the full context/output cost at the maximum allowed route prices.
    request_reservation = 262144 * input_price + max_tokens * output_price if paid else 0
    if fallback:
        providers = ["Io Net", "DekaLLM", "CoreWeave", "NextBit"]
        provider = {"only": providers, "order": providers, "allow_fallbacks": True,
                    "quantizations": ["bf16"], "require_parameters": True,
                    "max_price": {"prompt": price[0], "completion": price[1]}}
    else:
        provider = {"only": [provider_mode], "allow_fallbacks": False, "require_parameters": True}
    if max_cost is None:
        prior_config = run_dir / "config.json"
        max_cost = json.loads(prior_config.read_text()).get("max_cost_usd", 5.0) if prior_config.exists() else 5.0
    config = {"model": model, "reasoning": {"enabled": True, "exclude": False},
              "temperature": 0, "max_tokens": max_tokens, "interval_s": args.interval,
              "concurrency": args.concurrency, "batch": args.batch,
              "sample_sha256": hashlib.sha256(samples_path.read_bytes()).hexdigest(),
              "image_source": REPO, "image_revision": REVISION,
              "prompt_version": "mme-official-v1", "image_detail": "high",
              "image_transform": "none; official lmms-eval release bytes",
              "max_requests": args.max_requests,
              "transport": "SSE with token-progress deadlines and compressed raw event traces",
              "transport_deadlines": {"first_output_s": 120, "idle_output_s": 120, "total_s": 1200},
              "provider": provider if paid else {"require_parameters": True},
              "provider_mode": provider_mode if paid else "free",
              "max_cost_usd": max_cost if paid else 0,
              "price_per_million_tokens": {"input": price[0], "output": price[1]} if paid else {"input": 0, "output": 0},
              "score_protocol": "official extraction with corrected joined prefixes; empty output unparsed"}
    config_path = run_dir / "config.json"
    if previous:
        operational = {"max_requests", "interval_s", "concurrency", "provider_mode", "transport", "transport_deadlines"}
        if getattr(args, "allow_provider_change", False):
            operational.update(("provider", "price_per_million_tokens"))
        identity = [k for k in config if k not in operational]
        if any(previous.get(k) != config[k] for k in identity):
            raise SystemExit("Run configuration differs; choose another output directory.")
        if previous != config:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            dump(run_dir / "config_history" / f"{stamp}.json", previous)
    dump(config_path, config)
    dump(run_dir / "image_manifest.json", manifest)
    client = httpx.Client(timeout=httpx.Timeout(600, connect=30),
                          headers={"Authorization": f"Bearer {key}", "X-Title": "vlm-project MME-RealWorld research"})
    # This read-only check consumes no inference request.
    auth = client.get("https://openrouter.ai/api/v1/key")
    if auth.status_code != 200:
        raise SystemExit(f"Key validation failed: HTTP {auth.status_code}")
    (run_dir / "status.json").unlink(missing_ok=True)
    info = auth.json().get("data", {})
    dump(run_dir / "account_check.json", {k: info.get(k) for k in
         ("limit", "limit_remaining", "is_free_tier", "rate_limit", "usage_daily")})
    records = {r["sample_id"]: normalize_record(r) for r in read_jsonl(run_dir / "responses.jsonl")}
    todo = [r for r in samples if records.get(r["sample_id"], {}).get("status") not in ("completed", "skipped")]
    if args.limit:
        todo = todo[:args.limit]
    if getattr(args, "wait_for_images", False):
        # Evaluate cached images while preparation finishes; sample membership is unchanged.
        todo.sort(key=lambda r: r["sample_id"] not in manifest)
    starts = sum(r["event"] == "started" for r in read_jsonl(run_dir / "attempts.jsonl"))
    finished_attempts = [r for r in read_jsonl(run_dir / "attempts.jsonl") if r["event"] == "finished"]
    spent = sum(r.get("accounted_cost_usd", 0) for r in finished_attempts)
    # A started request without a durable outcome may have been billed before a crash.
    completed_numbers = {r["request_number"] for r in finished_attempts}
    unknown = sum(r["event"] == "started" and r["request_number"] not in completed_numbers
                  for r in read_jsonl(run_dir / "attempts.jsonl"))
    state = {"starts": starts, "stop_reason": None, "completed": sum(r.get("status") == "completed" for r in records.values()),
             "accounted_cost_usd": spent + unknown * request_reservation,
             "reserved_cost_usd": 0.0}
    lock = threading.Lock()
    stop = threading.Event()
    pacer = Pacer(args.interval)
    def control_change(concurrency, interval):
        with lock:
            config.update(concurrency=concurrency, interval_s=interval)
            dump(config_path, config)
            append(run_dir / "live_control_changes.jsonl", {"timestamp": datetime.now(timezone.utc).isoformat(),
                   "concurrency": concurrency, "interval_s": interval})
    def funded_slots():
        return max(1, int((max_cost-state["accounted_cost_usd"]-.05) / request_reservation))
    slots = LiveSlots(args.concurrency, pacer, run_dir / "control.json", control_change, paid,
                      budget_capacity=funded_slots if paid else None)
    summarize(run_dir, samples)

    prior_elapsed_by_sample = defaultdict(float)
    for a in read_jsonl(run_dir / "attempts.jsonl"):
        if a["event"] == "finished":
            prior_elapsed_by_sample[a["sample_id"]] += a.get("latency_s", 0)

    def evaluate(row):
        sid = row["sample_id"]
        elapsed_total = prior_elapsed_by_sample[sid]
        if elapsed_total >= 1200:
            previous = records.get(sid, {})
            append(run_dir / "responses.jsonl", {**previous, "sample_id": sid, "status": "skipped",
                   "skip_reason": "cumulative_example_time_over_20_minutes", "correct": False, "prediction": "",
                   "error": "Skipped: cumulative request time across attempts exceeded 20 minutes",
                   "timestamp": datetime.now(timezone.utc).isoformat()})
            return
        deadline = time.monotonic() + 1800
        while sid not in manifest:
            if stop.wait(1):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Image preparation did not finish for {sid}")
            latest = json.loads((DATA / f"images_batch_{args.batch}.json").read_text())
            with lock:
                manifest.update(latest)
        meta = manifest[sid]
        image_bytes = (DATA / meta["path"]).read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != meta["sha256"]:
            raise RuntimeError(f"Image integrity check failed for {sid}")
        image = base64.b64encode(image_bytes).decode()
        payload = {"model": model, "reasoning": config["reasoning"], "temperature": 0,
                   "max_tokens": config["max_tokens"],
                   "stream": True, "stream_options": {"include_usage": True},
                   "provider": config["provider"],
                   "messages": [{"role": "user", "content": [
                       {"type": "image_url", "image_url": {"url": f"data:{meta['mime']};base64,{image}", "detail": "high"}},
                       {"type": "text", "text": prompt(row)}]}]}
        previous = records.get(sid)
        if previous and previous.get("status") == "error":
            payload["provider"] = retry_provider_options(payload["provider"], previous.get("raw_response"))
        for attempt in range(1, 5):
            if not slots.enter(stop):
                return
            if not pacer.wait(stop):
                slots.leave()
                return
            with lock:
                if state["starts"] >= args.max_requests:
                    state["stop_reason"] = "request_budget_exhausted"
                    stop.set()
                    slots.leave()
                    return
                if paid and state["accounted_cost_usd"] + state["reserved_cost_usd"] + request_reservation > max_cost:
                    state["stop_reason"] = "spending_cap_reached"
                    stop.set()
                    slots.leave()
                    return
                state["starts"] += 1
                state["reserved_cost_usd"] += request_reservation
                request_number = state["starts"]
                append(run_dir / "attempts.jsonl", {"event": "started", "sample_id": sid,
                       "request_number": request_number, "attempt": attempt,
                       "provider_options": payload["provider"],
                       "inflight_requests": round(state["reserved_cost_usd"] / request_reservation) if paid else None,
                       "timestamp": datetime.now(timezone.utc).isoformat()})
            t0 = time.monotonic()
            remaining_wall_s = max(.1, 1200 - elapsed_total)
            status_code = None
            raw = None
            retry_after = None
            error = None
            try:
                with client.stream("POST", "https://openrouter.ai/api/v1/chat/completions", json=payload,
                                   timeout=httpx.Timeout(120, connect=30)) as response:
                    status_code = response.status_code
                    retry_after = response.headers.get("retry-after")
                    raw = read_completion_stream(response, run_dir / "streams" / f"request_{request_number:06d}.jsonl.gz", t0,
                                                 max_wall_s=remaining_wall_s)
                if status_code != 200 or raw.get("error"):
                    error = f"HTTP {status_code}: {json.dumps(raw.get('error', raw))[:1500]}"
                elif not raw.get("choices"):
                    error = "Response contains no choices"
                elif raw["choices"][0].get("finish_reason") == "error":
                    error = "Provider interruption: finish_reason=error"
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:1000]}"
                status_code = None
            elapsed = round(time.monotonic()-t0, 3)
            elapsed_total += elapsed
            if error and elapsed_total >= 1200:
                raw = {**(raw or {}), "skipped_over_20m": True}
                error = "Skipped: exceeded user-requested 20-minute request limit"
            usage = (raw or {}).get("usage")
            accounted_cost = 0.0
            if paid:
                if usage and usage.get("cost") is not None:
                    accounted_cost = float(usage["cost"])
                elif usage and usage.get("prompt_tokens") is not None and usage.get("completion_tokens") is not None:
                    accounted_cost = usage["prompt_tokens"] * input_price + usage["completion_tokens"] * output_price
                elif status_code not in (400, 401, 402, 403, 404, 429):
                    accounted_cost = request_reservation
            event = {"event": "finished", "sample_id": sid, "request_number": request_number,
                     "attempt": attempt, "latency_s": elapsed, "http_status": status_code,
                     "error": error, "retry_after": retry_after, "usage": usage,
                     "timestamp": datetime.now(timezone.utc).isoformat(), "accounted_cost_usd": accounted_cost}
            event["api_error_code"] = api_error_status(status_code, raw) if error else None
            message = (((raw or {}).get("choices") or [{}])[0].get("message") or {})
            text = message.get("content") or ""
            if not isinstance(text, str):
                text = json.dumps(text)
            prediction = extract_answer(text, row["Answer choices"]) if not error else ""
            choice = ((raw or {}).get("choices") or [{}])[0]
            reasoning_tokens = ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
            result = {"sample_id": sid, "question_id": row["Question_id"],
                      "status": "skipped" if (raw or {}).get("skipped_over_20m") else ("error" if error else "completed"), "response": text,
                      "prediction": prediction, "correct": prediction == row["Ground truth"] if not error else False,
                      "strict_answer_format": bool(re.fullmatch(r"\s*\(?[ABCDE]\)?[.\s]*", text)),
                      "finish_reason": choice.get("finish_reason"), "usage": usage,
                      "reasoning_present": bool(message.get("reasoning") or message.get("reasoning_details") or reasoning_tokens),
                      "latency_s": elapsed, "attempts": attempt, "error": error,
                      "timestamp": event["timestamp"], "request_number": request_number,
                      "prompt": prompt(row), "image_metadata": meta,
                      "raw_response": raw}
            with lock:
                state["reserved_cost_usd"] -= request_reservation
                state["accounted_cost_usd"] += accounted_cost
                state["completed"] += not error
                append(run_dir / "attempts.jsonl", event)
                append(run_dir / "responses.jsonl", result)
                summary = summarize(run_dir, samples)
                print(json.dumps({"sample_id": sid, "request": request_number,
                                  **summary["overall"], "last_error": error}), flush=True)
            slots.leave()
            if result["status"] == "skipped":
                return
            if not error:
                # Fail closed if the endpoint silently ignores enabled reasoning.
                if not result["reasoning_present"]:
                    with lock:
                        state["stop_reason"] = "reasoning_not_observed; inspect raw_response before continuing"
                    stop.set()
                return
            error_lower = error.lower()
            code = api_error_status(status_code, raw)
            if code in (401, 403, 404) or (code == 402 and not retryable_error(status_code, raw)) or (code == 429 and any(
                    s in error_lower for s in ("per-day", "per day", "daily", "quota", "free-models-per-day"))):
                with lock:
                    state["stop_reason"] = error
                stop.set()
                return
            if not retryable_error(status_code, raw):
                with lock:
                    state["stop_reason"] = f"Non-retryable API error: {error}"
                stop.set()
                return
            if attempt == 4:
                break
            payload["provider"] = retry_provider_options(payload["provider"], raw)
            delay = min(120, 10 * 2 ** (attempt-1))
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    from email.utils import parsedate_to_datetime
                    try:
                        delay = max(delay, parsedate_to_datetime(retry_after).timestamp()-time.time())
                    except (TypeError, ValueError):
                        pass
            if code in (429, 503, 402):
                pacer.pause(delay)
            if stop.wait(delay):
                return
        # A persistent error on one question must not prevent the other questions
        # from running. Stop the initial probe if the provider is unavailable.
        if state["completed"] == 0:
            with lock:
                state["stop_reason"] = "Repeated transient/provider errors; resume later without repeating completed examples"
            stop.set()

    try:
        # Validate one real sampled question before spending the remaining quota.
        if todo and not any(r.get("status") == "completed" for r in records.values()):
            evaluate(todo.pop(0))
        if not stop.is_set():
            with ThreadPoolExecutor(max_workers=max(64, args.concurrency)) as pool:
                futures = [pool.submit(evaluate, row) for row in todo]
                try:
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    stop.set()
                    for future in futures:
                        future.cancel()
                    raise
    except KeyboardInterrupt:
        stop.set()
        state["stop_reason"] = "interrupted"
    finally:
        client.close()
        dump(run_dir / "image_manifest.json", json.loads((DATA / f"images_batch_{args.batch}.json").read_text()))
        summary = summarize(run_dir, samples)
        dump(run_dir / "status.json", {**state, "overall": summary["overall"],
             "updated_at": datetime.now(timezone.utc).isoformat()})
        run_lock.close()
    print(json.dumps({**state, "overall": summary["overall"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("images")
    p.add_argument("--batch", type=int, default=1, choices=[1, 2])
    p = sub.add_parser("run")
    p.add_argument("--batch", type=int, default=1, choices=[1, 2])
    p.add_argument("--output")
    p.add_argument("--model", choices=[MODEL, PAID_MODEL], default=MODEL)
    p.add_argument("--provider", choices=["DekaLLM", "CoreWeave", "Io Net", "NextBit", "bf16-fallback"])
    p.add_argument("--allow-provider-change", action="store_true")
    p.add_argument("--max-cost-usd", type=float)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--interval", type=float, default=3.5)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--max-requests", type=int, default=1250)
    p.add_argument("--limit", type=int)
    p.add_argument("--wait-for-images", action="store_true")
    args = parser.parse_args()
    if args.command == "run" and (args.interval < (.5 if args.model == PAID_MODEL else 3.5) or
                                  not 1 <= args.concurrency <= 64 or args.max_requests < 1):
        parser.error("Use a valid pacing interval (paid >=0.5s; free >=3.5s), 1–64 workers, and a positive request budget")
    if args.command == "run" and (not 1 <= args.max_tokens <= 32768 or
                                  (args.max_cost_usd is not None and args.max_cost_usd <= 0)):
        parser.error("Use 1–32768 completion tokens and a positive spending cap")
    if args.command == "prepare":
        prepare(args.seed)
    elif args.command == "images":
        cache_images(args.batch)
    else:
        run(args)


if __name__ == "__main__":
    main()
