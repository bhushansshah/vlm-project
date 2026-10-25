"""Run both prepared MME batches sequentially, resuming errors and saving combined results."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import argparse
from datetime import datetime, timezone

import mme_realworld as mme

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "runs/mme_campaign"


def status(**values):
    values.update(updated_at=datetime.now(timezone.utc).isoformat(), supervisor_pid=os.getpid())
    mme.dump(CAMPAIGN / "status.json", values)
    print(json.dumps(values), flush=True)


def can_resume_reservation_stop(run_dir):
    """A concurrent reservation stop can clear after active calls settle."""
    config = json.loads((run_dir / "config.json").read_text())
    prices = config.get("provider", {}).get("max_price", {})
    reservation = (262144 * prices.get("prompt", .1) +
                   config.get("max_tokens", 32768) * prices.get("completion", .33)) / 1_000_000
    attempts = mme.read_jsonl(run_dir / "attempts.jsonl")
    finished = {a["request_number"] for a in attempts if a["event"] == "finished"}
    accounted = sum(a.get("accounted_cost_usd", 0) for a in attempts if a["event"] == "finished")
    accounted += reservation * sum(a["event"] == "started" and a["request_number"] not in finished
                                   for a in attempts)
    return config.get("max_cost_usd", 0) - accounted >= reservation + .05


def combine():
    out = ROOT / "runs/mme_gemma4_reasoning_combined_2000"
    out.mkdir(parents=True, exist_ok=True)
    samples = []
    responses = []
    attempts = []
    for batch in (1, 2):
        run = ROOT / f"runs/mme_gemma4_reasoning_paid_32k_batch{batch}"
        samples.extend(json.loads((mme.DATA / f"batch_{batch}.json").read_text()))
        responses.extend({**response, "batch": batch,
                          "request_number": f"batch{batch}:{response.get('request_number', 'unknown')}"}
                         for response in mme.read_jsonl(run / "responses.jsonl"))
        for attempt in mme.read_jsonl(run / "attempts.jsonl"):
            attempts.append({**attempt, "batch": batch,
                             "request_number": f"batch{batch}:{attempt['request_number']}"})
    for name, records in (("responses.jsonl", responses), ("attempts.jsonl", attempts)):
        temp = out / (name + ".tmp")
        with temp.open("w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp.replace(out / name)
    mme.dump(out / "config.json", {"model": mme.PAID_MODEL, "max_tokens": 32768,
             "reasoning": {"enabled": True}, "batches": [1, 2],
             "provider_note": "BF16 providers; actual provider recorded per example and in source config histories"})
    result = mme.summarize(out, samples)
    mme.dump(out / "sampling_audit.json", json.loads((mme.DATA / "sampling_audit.json").read_text()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attach-pid", type=int)
    parser.add_argument("--attach-batch", type=int, default=1)
    args = parser.parse_args()
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    lock = (CAMPAIGN / ".campaign.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Campaign already running")
    if args.attach_pid:
        status(stage="attached", batch=args.attach_batch, child_pid=args.attach_pid,
               reason="Supervisor updated while preserving existing inference process")
        while True:
            try:
                os.kill(args.attach_pid, 0)
            except ProcessLookupError:
                break
            probe = (ROOT / f"runs/mme_gemma4_reasoning_paid_32k_batch{args.attach_batch}/.run.lock").open("a")
            try:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                probe.close()
                break
            except BlockingIOError:
                probe.close()
            time.sleep(1)
    for batch in (1, 2):
        if batch == 2 and (CAMPAIGN / "DO_NOT_START_BATCH_2").exists():
            status(stage="paused_before_batch2", batch=2,
                   reason="User explicitly instructed not to start batch 2")
            return
        run_dir = ROOT / f"runs/mme_gemma4_reasoning_paid_32k_batch{batch}"
        round_number = 0
        stagnant = 0
        before = -1
        while True:
            summary_path = run_dir / "summary.json"
            if (run_dir / "responses.jsonl").exists():
                samples = json.loads((mme.DATA / f"batch_{batch}.json").read_text())
                mme.summarize(run_dir, samples)
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
            count = summary["overall"]["completed"] if summary else 0
            skipped = summary["overall"].get("skipped", 0) if summary else 0
            if count + skipped == 1000:
                status(stage="batch_complete", batch=batch, completed=count, skipped=skipped)
                break
            if count == before:
                stagnant += 1
            else:
                stagnant = 0
            if stagnant >= 3:
                status(stage="needs_attention", batch=batch, completed=count,
                       reason="Three resumptions made no progress; inspect provider or local failure")
                return
            before = count
            round_number += 1
            # Paid endpoint has no daily request quota; spending remains bounded.
            starts = summary.get("requests_started", 0) if summary else 0
            request_budget = max(2000, starts + (1000-count)*2 + 100)
            control_path = run_dir / "control.json"
            if not control_path.exists() and batch == 2:
                control_path = ROOT / "runs/mme_gemma4_reasoning_paid_32k_batch1/control.json"
            control = json.loads(control_path.read_text()) if control_path.exists() else {"concurrency": 20, "interval_s": 3.5}
            command = [sys.executable, str(ROOT / "scripts/mme_realworld.py"), "run",
                       "--batch", str(batch), "--model", mme.PAID_MODEL,
                       "--provider", "bf16-fallback", "--allow-provider-change",
                       "--concurrency", str(control["concurrency"]), "--interval", str(control["interval_s"]), "--max-tokens", "32768",
                       "--max-requests", str(request_budget), "--wait-for-images"]
            logfile = CAMPAIGN / f"batch{batch}_round{round_number}_{int(time.time())}.log"
            with logfile.open("w") as output:
                child = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
                status(stage="running", batch=batch, completed=count, child_pid=child.pid,
                       log=str(logfile.relative_to(ROOT)), round=round_number)
                code = child.wait()
            end_path = run_dir / "status.json"
            end = json.loads(end_path.read_text()) if end_path.exists() else {}
            reason = end.get("stop_reason") or ""
            if "spending_cap" in reason and can_resume_reservation_stop(run_dir):
                # The latest runner gates concurrent reservations to available
                # funds. Keep the original cap; resume only if a request fits.
                reason = "concurrent_reservations_settled; resuming within original spending cap"
            if any(value in reason for value in ("spending_cap", "reasoning_not_observed", "HTTP 401", "HTTP 402", "HTTP 403")):
                status(stage="needs_attention", batch=batch, reason=reason, return_code=code)
                return
            latest = json.loads(summary_path.read_text())["overall"] if summary_path.exists() else {}
            if latest.get("completed", 0) + latest.get("skipped", 0) == 1000:
                continue
            status(stage="retrying_remaining", batch=batch, return_code=code, reason=reason)
            time.sleep(30)
    combined = combine()
    status(stage="complete", completed=combined["overall"]["completed"],
           skipped=combined["overall"].get("skipped", 0),
           results="runs/mme_gemma4_reasoning_combined_2000")


if __name__ == "__main__":
    main()
