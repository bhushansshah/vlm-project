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
