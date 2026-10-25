# Tanish: MME-RealWorld benchmark handoff

<!-- LIVE_MME_STATUS_START -->
## Latest live status (automatic)

Updated **2026-10-17T19:33:17.600608+00:00**. This section supersedes older snapshots below.

| Batch | Completed | Skipped >20 min | Errors awaiting repair | Never attempted |
|---|---:|---:|---:|---:|
| 1 | 996 | 4 | 0 | 0 |
| 2 | 0 | 0 | 0 | 1000 |

Campaign stage: `paused_before_batch2`; supervisor PID `None`; child PID `None`.
Latest child log: `see runs/mme_campaign/`.

**Current policy:** paid Gemma, reasoning ON, 32,768 output tokens; skip individual requests over 1,200 seconds and retain them in `skipped_examples.json`. Do not retry skips. Finish each 1,000-selection batch when completed + skipped = 1,000.

Counts are derived from normalized raw responses; error records may be in flight. Campaign counts can be round-start snapshots. Live controls and locks remain authoritative.

Automatic status history: `runs/mme_campaign/handoff_status_history.jsonl`. Authored policy/fix updates remain below.
<!-- LIVE_MME_STATUS_END -->

Last updated: **2026-10-17 18:57 UTC / 14:57 America/New_York**. This is a snapshot; inspect the files and processes below for live state.

## Objective and authorization

Complete **two sequential batches of 1,000 examples each** from MME-RealWorld English using **paid OpenRouter `google/gemma-4-26b-a4b-it`**, with reasoning enabled. Finish batch 1, then immediately start and finish batch 2. Monitor continuously, fix recoverable failures, and optimize throughput while retaining detailed per-example responses and aggregate results per domain and task.

The user explicitly authorized the paid endpoint and both batches. The original free endpoint was unusable because of repeated upstream 429 responses. Do not revert to free or use multiple accounts. The user's key is already in the git-ignored `.env` as `OPENROUTER_API_KEY`. **Never print it, commit it, copy it into this document, or include it in logs.** Load it through dotenv. No further permission is needed for routine monitoring, retries, or the authorized evaluation.

Workspace: `/Users/tanish/Desktop/Repos/vlm-project`. Origin: `https://github.com/bhushansshah/vlm-project.git`. Branch **`tanish-main`** was created from freshly fetched `origin/main`. No commits or pushes have been made. README changes, `scripts/`, this file, and `tests/test_mme_realworld.py` are local work. `.env`, `.venv`, `data/`, and `runs/` are ignored.

An active unbudgeted goal exists in this agent thread: complete and verify both batches. Mark it complete only after both batches and final artifacts are verified. Do not spawn subagents unless the user or applicable instructions explicitly ask.

## Current execution state

At the snapshot:

| Batch | Valid API completions | Latest infrastructure-error records | Not yet evaluated |
|---|---:|---:|---:|
| 1 | **944 / 1,000** | 56, some actively being repaired | 0 |
| 2 | **0 / 1,000** | 0 | 1,000 |

A valid completion includes a response that reached the fixed 32,768-token limit; that is a model-budget outcome and must be scored, not retried to improve its answer. An HTTP 200 response with `finish_reason=error` is an infrastructure error and must be retried. Always normalize records using `mme_realworld.normalize_record`; old raw records and stale summaries can otherwise overcount successes.

Active supervisor PID **41708**, active batch-1 repair child PID **37788**. Previous supervisor 29187 was replaced without interrupting the child. Verify PIDs before sending any signal. Supervisor originally attached to earlier child 27456 and subsequently launched 37788. Current repair started `2026-10-17T18:43:14.638730+00:00`.

- Campaign state: `runs/mme_campaign/status.json`
- Current child log: `runs/mme_campaign/batch1_round1_1792262594.log`
- Supervisor log: `data/mme-realworld/campaign_supervisor.log`
- Unified terminal session for supervisor: 81359 (may be inaccessible to another session; OS processes/files are authoritative).
- `status.json` campaign `completed=912` is the **round-start snapshot**, not the live count.
- Active child writes final `runs/.../status.json` only when it exits; absence during execution is expected.

### Immediate operational issue to watch

The current child loaded just **before** the new budget-aware concurrency gate was added. It started with concurrency 48 and may have set its internal stop flag when concurrent worst-case cost reservations approached the cap. **No new starts have appeared after request 1115 at 18:44:27 UTC**, although existing streams keep completing. At 18:56 there were 14 active streams and 944 completions. This strongly suggests it is draining after a reservation-cap stop; verify the eventual final `stop_reason` rather than assuming it is hung.

Do not kill healthy ongoing reasoning streams merely because they take many minutes. A healthy stream can run up to 1,800 seconds. There may be up to ~15 minutes more drain time from this snapshot. When this child finishes, the existing supervisor may report `needs_attention` on `spending_cap_reached`. This can represent **temporary concurrent reservations**, not actual exhausted funds. Inspect the ledger, then restart the campaign with the latest code and lower concurrency if adequate cap remains. The latest runner's budget-aware gate should prevent this recurrence.

Current batch-1 `control.json`: concurrency **31**, interval **1.5 seconds**. Batch-2 control is precreated at **64**, interval **1.5 seconds**; the next process will load the latest budget-aware gate. Native images for both batches are fully prepared.

## Dataset, sampling, and image preparation completed

Official annotations: `https://huggingface.co/datasets/yifanzhang114/MME-RealWorld/resolve/main/MME_RealWorld.json`, saved as `data/mme-realworld/annotations.json`.

- Actual annotation rows: **23,609**.
- ID `perception/ocr_cc/license/h1_0085` occurs twice with different questions/images; both excluded, leaving 23,607 eligible rows.
- Annotation SHA-256: `502eae251a7a11257b94d36fde51e9171144184a522017637be99eea03393037`.
- There are 39 normalized raw task groups. Following the official evaluation grouping, monitoring color/orientation labels pool into person/attribute and vehicle/attribute, giving **37 evaluation task groups**.
- Group key includes domain (`Subtask`), perception/reasoning (`Task`), and normalized category. Do not accidentally merge unrelated attributes in autonomous driving.
- Seed **42**. Each batch has 27 examples per task and one task receives one extra: 37×27+1=1,000.
- Batch 1 domain allocation: autonomous driving 406; monitoring 216; OCR 189; diagrams/tables 108; remote sensing 81.
- Batch 2: autonomous driving 405; monitoring 217; OCR 189; diagrams/tables 108; remote sensing 81.
- Both batches have exactly 1,000 unique question IDs and image paths, disjoint across batches.
- A later SHA-256 check found two identical native images under different paths across batches. `scripts/mme_image_overlap.py` repaired this deterministically while preserving task quotas, with one augmenting reassignment. Replacements: `mme-17131`→`mme-17234`, `mme-11577`→`mme-11657`, `mme-13558`→`mme-13606`.
- **All 2,000 selected native image SHA-256 values are now unique across both batches.** Audit records the repair. Batch-2 backup before repair exists; do not use it for inference.

Native bytes come from official author dataset `yifanzhang114/MME-RealWorld-Lmms-eval`, pinned revision `f157390129e8d2b60757eb68d340aade9446b52b`. Its 70 parquet shards (~30.9 GB scanned) were downloaded sequentially, selected images extracted, then shards removed. Row index, question, choices, and ground truth were checked against annotations. Images are not resized or reencoded. Both manifests and selected image files are complete; largest batch-2 image ~8.823 MB. Some unused cached images are harmless.

Authoritative local inputs:

- `data/mme-realworld/batch_1.json`, `batch_2.json`
- `data/mme-realworld/images_batch_1.json`, `images_batch_2.json`
- `data/mme-realworld/images/`
- `data/mme-realworld/sampling_audit.json`, `sampling_task_counts.csv`
- `data/mme-realworld/image_overlap_repair.log`

Do not rerun sampling or image repair while inference is active.

## Code in place

### `scripts/mme_realworld.py`

Standalone preparation, image extraction, inference, persistence, scoring, and reporting:

```sh
.venv/bin/python scripts/mme_realworld.py prepare --seed 42
.venv/bin/python scripts/mme_realworld.py images --batch 1
.venv/bin/python scripts/mme_realworld.py run --help
```

Preparation has already finished; the first two commands are reference only.

Final benchmark settings: paid model above; `temperature=0`; `reasoning={"enabled":true,"exclude":false}`; **`max_tokens=32768`**. The official image-first question/choices/letter-only prompt is used with native base64 images and `detail=high`; no system prompt. Do not change model, prompt, reasoning, temperature, resolution, or token budget mid-benchmark.

Early 8,192-token/free/provider pilots were isolated into different run directories. They are **not** included in final benchmark accuracy. 8k was too small for reasoning, so the final benchmark restarted at uniform 32k.

Operational routing is `--provider bf16-fallback --allow-provider-change`: only **Io Net, DekaLLM, CoreWeave, NextBit**, BF16, reasoning parameter support required, input ceiling **$0.10/M**, output ceiling **$0.33/M**. Current route priority is Io Net, DekaLLM, CoreWeave, NextBit, based on measured throughput. Actual provider is recorded per example; config history preserves changes. Known failing backends are excluded on a question's retry; unknown stalls rotate ordering. Operational routing changes preserve generation settings.

Durable append-only `attempts.jsonl` has started/finished events with fsync; `responses.jsonl` retains all outcome history. Resume skips valid completed examples. `.run.lock` prevents simultaneous writers. There is a 64-thread pool, live concurrency gate, and global request-start pacer. Both starts and retries share pacing. Live controls are in each run's `control.json`; changing it does not require restart:

```json
{"concurrency": 32, "interval_s": 1.5}
```

Paid interval minimum 0.5s, concurrency 1–64. There is no claimed fixed published provider concurrency guarantee. Global pauses honor retry timing for 429/503/transient inflight-funding 402. Other transient errors retry per question. 408, all 5xx including 520, and embedded provider errors are retryable. Four transient attempts per question per pass; campaign resumes errors. Permanent auth/funding failures stop. Successful responses without any reasoning evidence stop for inspection.

**SSE streaming is active**: includes usage; reconstructs final text, reasoning, metadata and usage; gzip raw event traces saved under `streams/request_######.jsonl.gz`. First token deadline 120s, idle-output deadline 120s, healthy total deadline 1,800s. Reasoning tokens count as progress. Completed finish markers survive later usage-stream stalls. Token progress distinguishes slow healthy thinking from provider queue hangs. No credentials/headers are saved in traces.

### Scoring and output interpretation

Primary score uses official-style final-content answer extraction. The official parser can interpret letters in verbose prose (e.g. B from “Based…”); therefore a separate conservative explicit-answer score is also saved. Keep and report both; do not silently replace the primary score. Never derive the answer from hidden reasoning.

- `prediction`, `correct`: primary official-style extraction.
- `explicit_prediction`, `explicit_correct`: conservative explicit final answer.
- `finish_reason=length`: completed model-budget outcome, often unparsed/incorrect, **not retried**.
- `finish_reason=error`, aborted stream, API error: infrastructure failure, retried.
- Per-example records retain original output, reasoning, usage, costs, provider, image hash/dimensions, metadata, latency, and errors.

### `scripts/mme_campaign.py`

Supervisor resumes batch 1, completes it, then does batch 2, and builds the combined result. `.campaign.lock` prevents duplicate supervisors. Three resumptions with no progress or an auth/funding/reasoning-evidence problem causes `needs_attention` for inspection. `--attach-pid PID --attach-batch 1` updates a supervisor without canceling inference. It waits for PID/run-lock release before starting another child.

Combined output prefixes request numbers `batch1:n`/`batch2:n` in **both** responses and attempts, avoiding collisions.

### Other changed files

- `scripts/mme_image_overlap.py`: successful byte-identical image repair described above.
- `tests/test_mme_realworld.py`: targeted suite passed **40 tests** after the reservation-stop recovery test; the full suite passed **71 tests** at approximately 19:01 UTC (one upstream google/genai deprecation warning). Run the full suite again at final verification only if code changes warrant it.
- `README.md`: setup, protocol, campaign, controls, provider routing, streaming, and audit documentation.

Tests cover stratification, image disjointness, persistence/resume, cost caps, answer parsing, retry classification, live controls/budget gate, combined IDs, legacy provider-error normalization, SSE assembly, reasoning evidence, deadlines, and finish preservation. No real paid calls are made by tests.

## Cost accounting and concurrency caveat

The key initially reported a $10 API-key spending cap, not necessarily account credit balance. Batch-1 cap is **$4.914612**, preserving the previously announced $5 budget including pilot allowance. Batch-2 default cap is $5. Do not blindly raise either cap.

Worst-case per-request reservation is **$0.03702784**: full 262,144 context at input ceiling plus 32,768 output at output ceiling. Reported actual API cost is much lower. Accounting uses API `usage.cost` where available, otherwise token ceiling estimates, otherwise worst-case reservation. **26 legacy started requests have unknown billing outcomes after earlier controlled restarts** and retain worst-case reservations; corresponding questions were retried. They do not mean 26 unresolved dataset examples.

At 18:56 the conservative batch-1 settled/unknown ledger was about **$3.4656**, with 14 current requests active, versus reported successful-result API cost around $1.8. These figures change; do not describe conservative reservations as actual charges.

The new runner dynamically gates funded slots:

```python
max(1, int((max_cost - accounted_cost - .05) / request_reservation))
```

The current child lacks this newest gate; future children load it. Manual controls were reduced as the conservative ledger grew. If necessary calculate safe current-child concurrency from `(cap - settled_ledger - .30) / reservation`, with settled ledger including legacy unknowns, and leave existing streams running.

## Files to inspect and deliver

Final batch directories:

- `runs/mme_gemma4_reasoning_paid_32k_batch1/`
- `runs/mme_gemma4_reasoning_paid_32k_batch2/`

Combined directory: `runs/mme_gemma4_reasoning_combined_2000/` (created when both finish).

Each contains generated `per_example.json/.csv`, `summary.json`, `per_domain.csv`, `per_task.csv`, `per_raw_task.csv`, `per_type.csv`, `per_provider.csv`, `official_results.json`, and `report.md`, plus raw responses/attempts, config histories and manifests/audits. Stream traces live in the source batch directories. Per-example CSV includes source index, choices, prediction/GT, correctness, provider, token counts, reported cost, and request latency. Aggregates include task macro accuracy, reasoning evidence, truncation/unparsed counts, errors, latency and usage/cost.

Older `runs/mme_gemma4_reasoning_paid_batch1`, free/pilot directories, and route pilots are troubleshooting history, **not final 32k results**.

## Safe live status check

Run from the repo:

```sh
.venv/bin/python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import mme_realworld as m
for batch in (1, 2):
    p = Path(f'runs/mme_gemma4_reasoning_paid_32k_batch{batch}')
    rows = {r['sample_id']: m.normalize_record(r)
            for r in m.read_jsonl(p / 'responses.jsonl')}
    print(batch, 'completed', sum(r['status'] == 'completed' for r in rows.values()),
          'errors', sum(r['status'] == 'error' for r in rows.values()))
print(Path('runs/mme_campaign/status.json').read_text())
PY
```

Also inspect OS processes and the latest child log. Do not launch a second runner into an active run. If nothing is running and the blocking issue has been fixed:

```sh
.venv/bin/python scripts/mme_campaign.py
```

Keep the terminal session alive and monitor. If replacing only the supervisor, preserve its child, release the old supervisor lock, then launch `scripts/mme_campaign.py --attach-pid <verified-child-pid> --attach-batch <batch>`; do not start duplicate calls.

## Required work remaining

1. Finish active batch-1 repair/drain. Inspect any eventual reservation-cap stop and resume with latest budget-aware runner if sufficient budget remains. Continue repairing infrastructure errors only.
2. Finish all 1,000 batch-1 valid API completions, then automatically run batch 2 with the same benchmark settings. Monitor stalls, provider failures, actual usage and reservations.
3. Finish batch 2. Both manifests/images are ready and fully disjoint; no more dataset preparation is needed.
4. **Fresh finalization is essential**: the previous supervisor imported an older module before a diagnostic response-history shadowing bug and some richer reporting fields were fixed on disk. The supervisor was replaced at approximately 19:05 UTC and now imports the newest module; a fresh finalization remains a useful verification step. After all children/supervisors finish, run:

```sh
.venv/bin/python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import mme_realworld as m
import mme_campaign as c
for b in (1, 2):
    m.summarize(Path(f'runs/mme_gemma4_reasoning_paid_32k_batch{b}'),
                json.loads((m.DATA / f'batch_{b}.json').read_text()))
c.combine()
PY
```

5. Copy the final global sampling audit into source run directories if their snapshots predate byte deduplication. Verify 1,000 valid completions per batch, 2,000 unique IDs and native image hashes, preserved 37-task quotas, matching JSON/CSV counts, no unresolved infrastructure failures, and unchanged benchmark settings. Historical failed/unknown billing attempts remain documented.
6. Run `.venv/bin/python -m pytest` and `git diff --check`. Check for accidental credential exposure without printing secrets. Do not commit ignored datasets, keys, or large result files.
7. Deliver concise results with links to the combined report, per-example CSV, per-domain CSV, and per-task CSV. State official and explicit-answer accuracy, token-limit/unparsed outcomes, and actual API-reported cost separately from estimates/reservations. Optionally package reports/raw responses without `.env` or images.
8. Mark the active goal complete only after verification. No commit, push, PR, or deployment was requested.

## Reference sources used

- Official MME-RealWorld evaluation: `https://github.com/MME-Benchmarks/MME-RealWorld`
- Dataset annotations and pinned image dataset above.
- `https://openrouter.ai/google/gemma-4-26b-a4b-it`
- `https://openrouter.ai/docs/api_reference/limits`
- `https://openrouter.ai/docs/guides/routing/provider-selection`
- `https://openrouter.ai/docs/api_reference/errors-and-debugging`
- `https://openrouter.ai/docs/guides/best-practices/reasoning-tokens`

Paid OpenRouter requests do not have the free-model 20/minute platform cap; upstream capacity and funding reservations can still limit throughput. Earlier runtime estimates were provisional and proved too optimistic for long reasoning/provider stalls. Prefer measured live progress over repeating those estimates.

## Update at approximately 19:05 UTC

Batch 1 reached **952 valid completions**: 2 still generating (~20 minutes each), 46 waiting for infrastructure-error retries. Replaced only the supervisor, preserving child 37788. New supervisor uses `can_resume_reservation_stop`: after active calls settle, it retries automatically only when the unchanged cost cap can fund another full reservation plus $0.05 buffer. Unknown outcomes still retain full reservations. Targeted tests now 40 passed. No model settings changed and no active inference was canceled. Campaign state currently says `attached`; this is expected until child 37788 releases its lock.

## User policy change and restart at approximately 19:07 UTC

The user explicitly instructed: **skip and track examples taking more than 20 minutes**. This supersedes the earlier 1,800-second healthy-stream allowance and the requirement for 1,000 valid responses per batch. Finish processing all 1,000 selected examples in each batch with terminal status `completed` or `skipped`; do not replace skipped samples or retry them. Deadline now **1,200 seconds per request**. Early idle/provider failures still retry normally. Existing completed long requests are preserved; `over_20_minute_attempts.json` tracks all historical attempts over the threshold.

Stopped only the two currently active requests after verifying both exceeded 20 minutes: `mme-03018` / request1082 (NextBit, ~1,367s), and `mme-02812` / request1102 (NextBit, ~1,337s). Their responses now have `status=skipped`, a reason, elapsed time, provider, and partial trace path. Raw gzip traces may lack their final footer because the old child was terminated; existing bytes are retained. Billing is unknown and the full reservation is conservatively retained in durable finished events. No younger request was canceled.

Latest runner recognizes skips on resume, writes `skipped_examples.json`, counts skipped examples in every aggregate, and never retries them. Campaign completion condition is `completed + skipped == 1000`. Primary completed-only accuracy excludes skips; selected accuracy retains all 1,000 in its denominator. Report both counts honestly. Tests cover the new timeout/skip classification.

Previous supervisor41708 and child37788 have ended. Fresh campaign is active; terminal session **2099**. Live campaign status at this update: {"stage": "running", "batch": 1, "completed": 952, "child_pid": 42035, "log": "runs/mme_campaign/batch1_round1_1792263988.log", "round": 1, "updated_at": "2026-10-17T19:06:28.631693+00:00", "supervisor_pid": 42029}. Current batch1 counts at restart: **952 completed, 2 skipped, 46 remaining**. Future children use the budget-aware gate and 20-minute policy.

## Update at approximately 19:09 UTC: automatic handoff maintenance

User requested that all new changes and statuses continue to be recorded here. Added `scripts/mme_handoff.py --watch`, running independently (terminal session50345), which refreshes the marked live status section at the top every 60 seconds from normalized raw response records and campaign state. It saves timestamped history in `runs/mme_campaign/handoff_status_history.jsonl` and exits after campaign completion. `.handoff.lock` prevents duplicate monitors. It never reads credentials or changes inference state. To restart when no monitor is active: `.venv/bin/python scripts/mme_handoff.py --watch`. Keep the manually authored updates below the live section; do not overwrite the whole document.

Full test suite after reservation recovery and 20-minute policy changes: **72 passed**, with one upstream google/genai deprecation warning. Batch1 restarted successfully using the budget-aware gate; started events now include `inflight_requests`.

## Update at approximately 19:14 UTC: batch-2 throughput setting

Batch1 reached 981 completed, 2 skipped, 17 in flight. Set prepared batch2 `control.json` to **64 concurrency / 0.5-second global request-start spacing** to pursue the user's fastest-completion objective. The latest runner's funded-slot gate and all retry/provider backoff remain active. Paid requests are not governed by the former free-model 20/minute cap; provider capacity still applies. This is an operational pacing change only, with no benchmark generation changes or spending-cap increase. Current batch1 controls remain31 /1.5s because its tail is already running.

## USER INSTRUCTION — 2026-10-17T19:31:24.561404+00:00 — DO NOT START BATCH 2

**The user explicitly said “dont start batch 2.” This supersedes the previous automatic two-batch execution objective.** Supervisor42029 was stopped while still on batch1. Batch2 has not been started. A persistent guard `runs/mme_campaign/DO_NOT_START_BATCH_2` was created; `scripts/mme_campaign.py` now refuses to transition into batch2 while this marker exists. Do not remove it or start batch2 without a later explicit user instruction. Finish only batch1 bookkeeping/reporting, then pause.

Also clarified the 20-minute policy as a **cumulative per-example request-time budget across retries**: a retry must not reset the clock. Future runner workers subtract prior finished-attempt latencies from the remaining time. Existing completed results are preserved. The last active batch1 example `mme-20782` already exceeded this cumulative budget before its latest retry; it is being marked skipped rather than allowed to delay completion further. This change applies prospectively to unfinished work; do not retroactively discard completed answers.

## Batch 1 final status — 2026-10-17T19:38 UTC

**Batch 1 is complete.** 996 completed + 4 skipped = 1,000 / 1,000. Zero errors, zero pending.

Campaign stage: `paused_before_batch2`. Batch 2 has not started; `DO_NOT_START_BATCH_2` guard is in place.

### Final accuracy

- **Official-extraction accuracy (completed):** 37.25% (371 / 996)
- **Explicit-answer accuracy (completed):** 42.57% (424 / 996)
- **Macro task accuracy (official, 37 tasks):** 37.23%
- **Macro task accuracy (explicit, 37 tasks):** 42.53%

### Final cost

- Reported API cost: **$2.14**
- Reasoning tokens: 8,040,711 (96.4% of completions)
- Total tokens: 8,786,980
- Wall time: ~2 hr 7 min

### Reporting artifacts generated

All per-example, per-domain, per-task, per-provider, and summary files are present in `runs/mme_gemma4_reasoning_paid_32k_batch1/`. Full results report artifact created in the agent conversation directory.

## Batch 1 final-answer report — 2026-10-17

Generated by `scripts/mme_final_answer_report.py` into `runs/mme_gemma4_reasoning_paid_32k_batch1/final_answer_report/` (`report.md`, `summary.json`, `per_example.csv`, `per_domain.csv`, `per_task.csv`, `truncated_examples.json`).

Finding: the official parser is the unreliable one (it strips "The best answer is" everywhere, then takes the first A-E letter in the prose, e.g. from "Based"). The explicit parser matches a last-stated-answer extractor on every response, so it is used here. Official-vs-explicit disagreements: 311.

Outcomes of 1,000: **821 answered**, 174 truncated (173 `length` plus 1 response cut off with no finish reason), 1 stop without a final answer, 4 skipped. Truncated and skipped examples are excluded from accuracy and listed separately.

**Accuracy on answered only: 51.64% (424/821)**, macro task 50.45%. Counting truncated/skipped as wrong: 42.4%. By domain (answered only): Diagram and Table 79.79%, OCR 79.17%, Remote Sensing 48.21%, Autonomous Driving 38.39%, Monitoring 36.11%. Remote Sensing and Autonomous Driving lose the most to truncation (28% and 20%).
