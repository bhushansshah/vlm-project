# vlmbench: VLM baselines and agents on VisualOverload

Course project: *Studying the Performance of VLM Agents on VQA Tasks for Dense Datasets*.
This repo currently implements **pipeline 1, the base VLM** (`A = VLM(I, Q)`) on
[VisualOverload](https://github.com/paulgavrikov/visualoverload) (CVPR 2026). It works with any
backend: OpenAI, Anthropic, Gemini, OpenRouter, a local OpenAI-compatible server (vLLM / SGLang /
Ollama), or in-process HF transformers.

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[api,dev]"        # API backends
uv pip install -e ".[local]"          # + torch/transformers for in-process local models
cp .env.example .env                  # add the API keys you need
vlmbench prepare                      # download + extract the dataset (~630 MB, once)
```

`prepare` writes the 150 paintings to `data/visualoverload/images/` plus a `questions.jsonl` index
(2,720 questions). Set `VLMBENCH_DATA_DIR` to put the data on scratch storage instead.

## Running the baseline

```bash
# quick, cheap smoke test on a random subset
vlmbench run --model openrouter:qwen/qwen2.5-vl-72b-instruct --limit 20 --shuffle-seed 0

# full benchmark from a config file (CLI flags override config values)
vlmbench run --config configs/openrouter_qwen25vl_72b.yaml
vlmbench run --config configs/gemini_25_flash.yaml --categories counting ocr
```

Model spec = `<provider>:<model>`:

| provider | example | notes |
|---|---|---|
| `openai` | `openai:gpt-5-mini` | Responses API (as in the paper). `-o image_detail=high` |
| `anthropic` | `anthropic:claude-sonnet-5-5` | images > 5 MB are re-encoded to fit (logged) |
| `gemini` | `gemini:gemini-2.5-flash` | safety filters off (as in the paper). `-o media_resolution=MEDIA_RESOLUTION_HIGH` |
| `openrouter` | `openrouter:qwen/qwen2.5-vl-72b-instruct` | any OpenRouter vision model |
| `vllm` / `ollama` / `openai-compat` | `vllm:Qwen/Qwen2.5-VL-7B-Instruct -o base_url=http://node:8000/v1` | locally served models |
| `local` | `local:Qwen/Qwen2.5-VL-7B-Instruct` | in-process HF `AutoModelForImageTextToText`; greedy, 512 tokens, bf16 |
| `dummy` | `dummy:random` | offline, chance-level; for testing the pipeline |

Useful flags: `-o key=value` (backend option), `-g key=value` (provider-specific generation
option, e.g. `-g 'reasoning={effort: low}'`), `--max-tokens`, `--temperature`, `--prompt`,
`--concurrency` (API), `--batch-size` (local), `--categories`, `--difficulties`, `--limit`,
`--question-ids FILE`, `--retry-errors`.

### Outputs and scoring

Each run writes `runs/<run_name>/`:

- `config.json`: the full run configuration and backend description
- `responses.jsonl`: one record per question with the raw response, extracted answer, finish reason, token usage, latency, and error
- `submission.json`: what you upload to the evaluation server
- `summary.json`: local diagnostics (errors, parse rate per category, answer-letter distribution, tokens)

Runs are **resumable**: re-running the same command skips finished questions, so an interrupted or
preempted job just continues. `--retry-errors` re-runs only the questions that failed.

**Ground truth is private.** Accuracy comes only from the official evaluator:
upload `submission.json` at https://huggingface.co/spaces/paulgavrikov/visualoverload-submit
(HF login; rate-limited per day, so don't burn submissions on partial runs: missing questions count as wrong).
Locally we use a port of the server's answer extraction (`src/vlmbench/extract.py`), so the
`parse_rate` in `summary.json` tells you whether the server will be able to read your answers.

## Faithfulness to the paper's protocol

- Prompt `default` = the dataset's `default_prompt` (the authors' prompts), image first, no system prompt.
- Open-weight models: greedy decoding, `max_new_tokens=512`, processor-default resolution (as in the authors' `models.py`).
- APIs: provider defaults, **no reasoning/thinking** unless you ask for it. OpenRouter enables reasoning
  by default for some models (e.g. Claude Haiku 5.5); disable it with `-g 'reasoning={enabled: false}'`.
- Images are sent at full resolution. Downscaling happens only if you set `-o max_side=...`/`-o max_pixels=...`,
  or to meet a hard API limit (Anthropic 5 MB: 48 of 150 images get re-encoded at JPEG q=85, same resolution).
- Submission: raw responses, with markdown emphasis stripped (`**1557**` → `1557`), which the server's
  OCR cleaning doesn't handle. The paper explicitly allows client-side preprocessing.
- Binary yes/no questions are scored **pairwise** with their logical opposite by the server.

Reference numbers (paper Table 1, total accuracy): o3 69.5, Gemini 2.0 Flash 68.1,
InternVL3-38B 67.6, Qwen2.5-VL-72B 65.7, Qwen2.5-VL-7B 61.5, random 16.0.

## Code layout

```
src/vlmbench/
  types.py         Message / ImagePart / TextPart / GenerationConfig / Completion (provider-neutral)
  data.py          dataset preparation + filtered loading -> Sample
  prompts.py       prompt strategies (register new ones with @register_prompt)
  images.py        image encoding with explicit, logged resize / byte-limit policies
  extract.py       answer extraction ported from the official judge
  runner.py        concurrent / batched, resumable execution + submission & summary
  cli.py           `vlmbench` command
  models/          one backend per provider behind the VLM interface; registry in __init__.py
configs/           example run configs
tests/             offline tests (SDK clients stubbed; no network)
```

Backends take a *list of multi-turn messages with any number of images*, not just `(image, question)`,
so the upcoming tool-augmented and Python-interpreter agents (pipelines 2 and 3) can reuse them unchanged.
To add a backend: subclass `VLM`, implement `generate(messages, config) -> Completion`, and register it in
`models/__init__.py`.

## On the cluster (Slurm)

The login node caps each user at 4 GB RAM, so run model inference only inside Slurm jobs
(partition `defq`, GPUs via `--gres=gpu:N`). API-backed runs are lightweight HTTP clients.

```bash
srun -p defq --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=4:00:00 --pty bash
source .venv/bin/activate
vlmbench run --config configs/local_qwen25vl_7b.yaml
```

Tests: `pytest`.

## MME-RealWorld with Gemma 4 reasoning

`scripts/mme_realworld.py` prepares two deterministic, image-disjoint sets of
1,000 English questions. It balances the 37 evaluation tasks (domain +
perception/reasoning + normalized category), pooling monitoring attribute
subcategories as in the official evaluator. Each set has 27 examples per task
and one task with 28. Both occurrences of a conflicting duplicate question ID
are excluded. The full allocation, source hashes, and seed are recorded in
`data/mme-realworld/sampling_audit.json`.

```bash
python scripts/mme_realworld.py prepare --seed 42
python scripts/mme_realworld.py images --batch 1
# Set OPENROUTER_API_KEY in the git-ignored .env first.
python scripts/mme_realworld.py run --batch 1
```

Image preparation uses the authors' pinned `MME-RealWorld-Lmms-eval` release.
It downloads approximately 31 GB of Parquet shards sequentially, retains only
the selected images, and deletes each shard after extraction. It verifies the
question, choices, and answer against the original annotation row. Images are
sent as supplied by that release, with no client resizing or re-encoding.

The runner uses `google/gemma-4-26b-a4b-it:free`, enables reasoning, and allows
32,768 total completion tokens (including reasoning). It checks the first real
sample before launching the rest, with at most 10 in-flight requests. All
attempts, including retries, share a 3.5-second minimum submission interval.
Rate-limit responses pause submissions and honor `Retry-After`. Authentication,
daily-quota, or unsupported-request errors stop the run. It also stops if a
response provides no evidence that reasoning ran, so that setting can be checked.
Streaming requests retain compressed raw event traces. They have two-minute
first-output and output-idle deadlines, plus a 30-minute total deadline, so
provider keepalives cannot hide a stalled request while active reasoning can
continue through the full completion-token allowance.

Outputs are saved incrementally under `runs/mme_gemma4_reasoning_batch1/`:

- `config.json`: model, generation settings, source revision, and sample manifest hash.
- `image_manifest.json`: image dimensions, byte sizes, and SHA-256 hashes.
- `attempts.jsonl`: durable request starts and outcomes, errors, timing, and usage.
- `responses.jsonl`: append-only detailed responses, raw provider JSON, returned
  reasoning, extracted answers, correctness, image metadata, and prompts.
- `per_example.json` / `.csv`: all 1,000 selected questions and latest results,
  including pending/error states and accumulated attempt counts.
- `summary.json`: overall, domain, task, raw category, and perception/reasoning
  accuracy, completion counts, errors, latency, tokens, and reported costs.
- `per_domain.csv`, `per_task.csv`, `per_raw_task.csv`, `per_type.csv`: aggregate tables.
- `official_results.json`: completed results in the official evaluator's format.
- `status.json`: final completion counts and any stop reason.

Accuracy is reported both over completed responses and over the selected set;
the latter counts unfinished examples as incorrect and is provisional until all
1,000 finish. Truncated and unparsed responses are explicitly counted. Answer
extraction follows the official evaluator, correcting its accidentally joined
answer-prefix strings and treating empty responses as unparsed. Because the
official extractor selects the first capital A–E even in prose, supplemental
`explicit_prediction` / `explicit_correct` fields and
`explicit_answer_accuracy_completed` metrics also score an explicitly stated
final answer. The official score remains the primary accuracy; disagreements
and letter-only formatting violations are counted.

Re-run the same command to resume without repeating completed questions. The
default retry budget permits 1,250 total request starts across resumes; the
account's daily free quota can stop the run sooner and leave questions unfinished.
After quota replenishment, the total budget can be raised with `--max-requests`
when needed. Credentials are never written to run outputs. Dataset files and
run outputs are git-ignored.

For the paid endpoint, use:

```bash
python scripts/mme_realworld.py run --batch 1 \
  --model google/gemma-4-26b-a4b-it --max-cost-usd 5
```

Paid runs are stored separately in `runs/mme_gemma4_reasoning_paid_32k_batch1/` and
pin DekaLLM's BF16 endpoint, with reasoning enabled. At setup, its listed rates
were $0.06 per million input tokens and $0.33 per million output tokens. Each
in-flight request reserves its worst-case context/output cost against the cap;
reported actual costs release the unused reservation. Unknown outcomes after
a crash retain their reservation when resuming. The paid pilot uses selected
benchmark questions, and completed pilot examples are retained in the full run.
Use `--concurrency 20` to resume with more requests in flight while retaining the
same submission interval and generation settings. The run logs operational
concurrency changes separately; there is no published guaranteed concurrency
limit for the pinned provider.

`python scripts/mme_campaign.py` resumes batch 1, retries remaining API errors,
then starts batch 2 automatically. It uses the same Gemma model with a pool of
BF16 providers (CoreWeave, io.net, NextBit, and DekaLLM), preserving provider
history and adding `per_provider.csv` diagnostics. The pool enforces price
ceilings of $0.10/M input and $0.33/M output tokens. Model, prompts, reasoning,
image handling, and completion limits are retained across the two batches.

While the current runner is active, edit its `control.json` atomically to change
`concurrency` (1–64) and `interval_s` (paid minimum 0.5 seconds). These controls
apply without restarting or cancelling requests; operational changes are logged.
Batch 2 inherits the latest controls from batch 1. Spending caps and provider
backoff remain in force. Finished campaigns also write a combined 2,000-example
report to `runs/mme_gemma4_reasoning_combined_2000/`.

Before evaluating the second set, cache its images with
`python scripts/mme_realworld.py images --batch 2`, then run
`python scripts/mme_image_overlap.py`. The latter replaces any byte-identical
images shared across the sets with deterministic same-task alternatives and
records the replacements in the sampling audit. Provider responses marked
`finish_reason: error` are infrastructure failures requiring retries, even if
their HTTP status was 200. They are excluded from completed-model accuracy.
