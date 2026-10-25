# vlmbench: VLM baselines and agents on dense-scene VQA

Course project: *Studying the Performance of VLM Agents on VQA Tasks for Dense Datasets*.
This repo currently implements **pipeline 1, the base VLM** (`A = VLM(I, Q)`) on two benchmarks:

- [VisualOverload](https://github.com/paulgavrikov/visualoverload) (CVPR 2026): 2,720 questions on dense paintings; ground truth is **private** (scored by the authors' server).
- [VistaQA](https://vistaqa.github.io) (NeurIPS 2026 D&B): 1,157 free-form questions, six task types, six domains, 27% "trick" questions about things not in the image; answers are **public**, so we grade locally with an LLM judge.

It works with any backend: OpenAI, Anthropic, Gemini, OpenRouter, a local OpenAI-compatible server (vLLM / SGLang /
Ollama), or in-process HF transformers.

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[api,dev]"        # API backends
uv pip install -e ".[local]"          # + torch/transformers for in-process local models
cp .env.example .env                  # add the API keys you need
vlmbench prepare                      # VisualOverload (~630 MB, once)
vlmbench prepare --dataset vistaqa    # VistaQA (~440 MB, once)
```

Each dataset goes to `data/<dataset>/` with a `questions.jsonl` index. Set `VLMBENCH_DATA_DIR` to put
the data on scratch storage instead.

## Running the baseline

```bash
# quick, cheap smoke test on a random subset
vlmbench run --model openrouter:qwen/qwen2.5-vl-72b-instruct --limit 20 --shuffle-seed 0

# full benchmark from a config file (CLI flags override config values)
vlmbench run --config configs/openrouter_qwen25vl_72b.yaml
vlmbench run --config configs/gemini_25_flash.yaml --categories counting ocr
```

### VistaQA: run, judge, report

```bash
# balanced subset: 100 random questions per task type (seed 0), same 600 for every model
vlmbench run --config configs/vistaqa/qwen3.8-27b.yaml --per-category 100 --sample-seed 0
# full benchmark (1,157); questions already answered in the same run folder are skipped
vlmbench run --config configs/vistaqa/qwen3.8-27b.yaml
vlmbench judge runs/vistaqa__openrouter__qwen_qwen3.8-27b__default     # grade answers (cached, resumable)
vlmbench report runs/vistaqa__* --out runs/vistaqa_report               # tables by task / domain / trick
```

The judge uses the authors' exact prompt and their default judge model (Qwen2.5-72B-Instruct, here via
OpenRouter; `--judge <spec>` to change). We report their text-accuracy metric only: the GROVE metric also
needs segmentation masks, which a plain VLM does not produce. `predictions.json` in each run is in the official
evaluator's format (with empty masks).

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
`--concurrency` (API), `--batch-size` (local), `--retry-errors`.
Question selection (combinable): `--dataset`, `--categories`, `--question-ids ID... | FILE`,
`--filter KEY=V1,V2` (e.g. `--filter difficulty=hard`, `--filter domain=av,robotics`, `--filter hallucination=1`),
`--per-category N`, `--limit N`, `--sample-seed S`.

### Outputs and scoring

Each run writes `runs/<run_name>/`:

- `config.json`: the full run configuration and backend description
- `responses.jsonl`: one record per question with the raw response, finish reason, token usage, latency, and error
- `summary.json`: errors, tokens, latency, plus dataset-specific diagnostics
- VisualOverload: `submission.json` (upload to the evaluation server); VistaQA: `predictions.json`, and `judgments.jsonl` after `vlmbench judge`

**Cost.** Every saved usage record (answers, agent steps, judgments) has `cached_tokens` and `cost` (USD)
next to the token counts. A field is `null` when the provider does not report it: OpenRouter reports both;
the OpenAI, Anthropic and Gemini APIs report cached tokens but no cost; vLLM/Ollama and local models report
neither. `summary.json` and `vlmbench judge` print the total, and `vlmbench report` adds a Cost table
(`–` = not reported, never shown as $0). Runs made before this change have no cost saved.

Runs are **resumable**: re-running the same command skips finished questions, so an interrupted or
preempted job just continues. `--retry-errors` re-runs only the questions that failed.

**VisualOverload ground truth is private.** Accuracy comes only from the official evaluator:
upload `submission.json` at https://huggingface.co/spaces/paulgavrikov/visualoverload-submit
(HF login; rate-limited per day, so don't burn submissions on partial runs: missing questions count as wrong).
Locally we use a port of the server's answer extraction (`src/vlmbench/datasets/visualoverload_extract.py`), so the
`parse_rate` in `summary.json` tells you whether the server will be able to read your answers.

## Faithfulness to the VisualOverload protocol

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

## Tool-augmented agent (pipeline 2)

`--agent tools` wraps any model in a ReAct-style agent: the model thinks, calls a tool by writing
`<action>{"name": ..., "arguments": {...}}</action>`, sees the result (text + image), and finishes with
`<answer>...</answer>`. The tools are also declared through the API's native tool-calling field, and
native calls are converted to the same `<action>` form: some models/providers only call tools natively
(e.g. Muse Glimmer on DeepInfra), others strip `<tool_call>` text (one Qwen provider), so the agent
accepts both.

| tool | what it does | backend | needs |
|---|---|---|---|
| `zoom` | crop a region and show it enlarged (optional coordinate grid) | PIL | nothing |
| `grid` | show the image / a region with labeled 0-1000 grid lines | PIL | nothing |
| `detect` | find, locate and count objects named in words | Grounding DINO tiny | `[tools]`, GPU recommended |
| `ocr` | read text in the image or a region | EasyOCR | `[tools]` |
| `depth` | order objects from nearest to farthest | Depth Anything V2 small | `[tools]` |

All tools use boxes `{"left": .., "top": .., "right": .., "bottom": ..}` on a 0-1000 scale of the
original image. Named fields on purpose: with plain lists, Gemma silently swapped x and y (its own
`[y1, x1, y2, x2]` convention). Lists are still accepted, read as `[left, top, right, bottom]`.

**GPU on our cluster:** the GPU nodes' driver supports CUDA ≤ 12.4, but the default PyTorch from PyPI is
built for CUDA 13 and silently falls back to CPU (tools then take ~15 s per call instead of <1 s).
Install the CUDA 12.4 build: `uv pip install "torch==2.6.*" "torchvision==0.21.*" --index-url
https://download.pytorch.org/whl/cu124`. Run agent experiments inside a GPU job, e.g.
`srun -p defq --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=4:00:00 vlmbench run ...`
(the tool models need about 2 GB of GPU memory; compute nodes can reach OpenRouter).

```bash
uv pip install -e ".[tools]"     # only for detect / ocr / depth
# all tools (run on a GPU node: the detector, OCR and depth models run locally)
vlmbench run --config configs/vistaqa/qwen3.8-27b.yaml --agent tools --per-category 100 --sample-seed 0
# a subset of tools and options (an ablation)
vlmbench run --config configs/vistaqa/qwen3.8-27b.yaml --agent tools -a 'tools=[zoom, grid]' -a max_steps=4
vlmbench judge runs/<agent run>  &&  vlmbench report runs/<baseline run> runs/<agent run>
```

Agent options (`-a key=value`): `tools` (default: all five), `max_steps` (default 6), `tool_options`
(e.g. `-a 'tool_options={detect: {box_threshold: 0.35}}'`), `keep_last_images` (drop older tool images to
save tokens), `step_max_tokens` (default 1024). Each answer's full trajectory (model text, tool calls,
results) is saved under `trace` in `responses.jsonl`; `summary.json` reports tool usage. An agent run
uses several model calls per question (about 4-5x the baseline's input tokens with zoom/grid).

## Code layout

```
src/vlmbench/
  types.py         Message / ImagePart / TextPart / GenerationConfig / Completion (provider-neutral)
  images.py        image encoding with explicit, logged resize / byte-limit policies
  runner.py        concurrent / batched, resumable execution + summary
  retry.py         retries with backoff (shared by runner and judge)
  judge.py         LLM-as-a-judge engine (prompt comes from the dataset)
  report.py        accuracy tables by category and dataset-specific groups
  cli.py           `vlmbench` command
  datasets/        one module per benchmark behind the Dataset interface (base.py):
                   download, load, prompts, outputs, judge prompt; registry in __init__.py
  agents/          agents that wrap a model (tool_agent.py: the ReAct tool agent); registry in __init__.py
  tools/           agent tools behind the Tool interface (base.py); registry in __init__.py
  models/          one backend per provider behind the VLM interface; registry in __init__.py
configs/           example run configs
tests/             offline tests (SDK clients stubbed; no network)
```

Backends take a *list of multi-turn messages with any number of images*, not just `(image, question)`,
so the upcoming tool-augmented and Python-interpreter agents (pipelines 2 and 3) can reuse them unchanged.
To add a backend: subclass `VLM`, implement `generate(messages, config) -> Completion`, and register it in
`models/__init__.py`. To add a benchmark: subclass `Dataset` (`prepare`, `load_all`, `prompts`, and
optionally `judge_prompt` / `write_outputs`) and list it in `datasets/__init__.py`.

## On the cluster (Slurm)

The login node caps each user at 4 GB RAM, so run model inference only inside Slurm jobs
(partition `defq`, GPUs via `--gres=gpu:N`). API-backed runs are lightweight HTTP clients.

```bash
srun -p defq --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=4:00:00 --pty bash
source .venv/bin/activate
vlmbench run --config configs/local_qwen25vl_7b.yaml
```

Tests: `pytest`.
