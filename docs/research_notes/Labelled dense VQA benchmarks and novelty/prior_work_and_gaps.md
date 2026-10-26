# Prior work and gaps: tool-using and agentic VLMs for dense / high-resolution VQA (2023 – Oct 2026)

Scope: what has already been published on fixed-tool (ReAct / JSON function-calling) agents, Python/code agents and plain VLMs for dense or high-resolution VQA. The aim is to find gaps that a 3-person, 7-week course project can fill (report due 10 Dec 2026), with 2× RTX A6000 (48 GB each) plus some API budget.

Status: checked 15 Oct 2026. These notes build on `../Dense scene VQA training datasets/agentic_tool_use.md`, which covers training sets, tool interfaces, licences and headline V*/HR-Bench numbers. That material is not repeated here.

How claims were checked:
- Statements marked "(abstract)" come from the arXiv abstract page.
- Statements marked "(HTML)" come from the arXiv full-text HTML.
- Statements marked "(snippet only)" come from a search-engine snippet. I did not open those papers, so treat them as **UNVERIFIED**.

Abbreviations:
- P1 = base VLM, no tools
- P2 = ReAct / function-calling agent with a fixed tool set
- P3 = Python-interpreter (code) agent that can compose the same tools
- V* = V*Bench
- HR4K / HR8K = HR-Bench 4K / 8K
- MME-RW = MME-RealWorld
- CWL = "Calling Without Looking"

---

## Q1. Which papers compare fixed-tool agents and code agents on the same benchmark, with the same base model and the same tools? What did they find, and what did they leave untested?

### Takeaway
Only three published comparisons come close to "same model, same tools, interface varied", and none of them is on dense natural-scene VQA with specialist tools and open small VLMs:
- **VTC-Bench** (2026): same model, 32 OpenCV operations, interface call vs code. The two interfaces are comparable overall, and Qwen3-VL mostly gets *worse* with tools.
- **VTOS** (EMNLP 2026): ReAct vs CodeAct with Claude Sonnet 4.6 and Grounding-DINO tools on 100 LVIS counting images. CodeAct has lower error than ReAct, but both are worse than the bare VLM.
- **m&m's** (ECCV 2024): text-only LLM planners, JSON vs code plans. Tool prediction is comparable, and JSON plans execute more reliably.

The large "thinking-with-images" families do not isolate the interface, because each also changes data, reward and base model:
- crop/JSON: DeepEyes, Mini-o3, Chain-of-Focus, Pixel Reasoner
- code: Thyme, PyVision, CodeV, CodeVision, CodeDance, DeepEyesV2

CodeV and CodeVision both admit this: they argue for code qualitatively but run no same-backbone interface ablation.

### Cited Findings

**Near-controlled interface comparisons**
- **VTC-Bench** (arXiv 2603.15030, Mar 2026; venue not listed)
  - 680 problems, 32 OpenCV-based operations, a nine-category hierarchy, 19 MLLMs.
  - It compares "interface-based" with "code-based" tool invocation on the same models. The authors conclude that interface-based invocation is "comparable to code-based approaches overall"; three of five proprietary models did better with the interface (HTML).
  - Examples (base → code / interface): GPT-4o 24.26 → 31.62 / 33.82; Gemini-2.5-Pro 38.09 → 36.03 / 39.85; GPT-5.2 36.03 → 44.56 / 40.74; Gemini-3.0-Pro 44.41 → 51.18 / 51.03.
  - Open "tool-use" models gain little or degrade (DeepEyes-v2 −1.47 with the interface). With tools, most Qwen3-VL variants decline or barely change: Qwen3-VL-8B-Thinking −3.97 with code, Qwen3-VL-235B-Thinking +2.94 with the interface — [arXiv HTML](https://arxiv.org/html/2603.15030v1); [abs](https://arxiv.org/abs/2603.15030)
  - Untested: natural dense or high-resolution photos, specialist detectors, segmentation, OCR and pose tools, faithfulness, and cost.
- **VTOS** (arXiv 2606.20728, EMNLP 2026 Main per the comments field), on LVIS-Count: 100 test images, Normal tier 10–50 objects and Extreme tier 51–100 (HTML). MAE (lower is better) and bias:

  | Method | MAE | Bias |
  |---|---|---|
  | Sonnet 4.6 count-only (no tools) | 35.13 | — |
  | ReAct (Sonnet 4.6 + DINO) | 46.31 | −46.03 |
  | CodeAct (Sonnet 4.6 + tools) | 39.38 | −32.40 |
  | Reflexion | 44.28 | — |
  | VTOS's offline program search | 23.36 | — |
  | Qwen3-VL-8B direct | 54.26 | — |
  | Qwen3-VL-32B direct | 48.32 | — |
  | Grounding DINO | 52.76 | +27.44 |
  | CountGD | 43.01 | +32.05 |

  - Cost: ReAct uses up to 6 LLM calls per image and CodeAct 1. Wall-clock latency is not reported — [arXiv HTML](https://arxiv.org/html/2606.20728); [abs](https://arxiv.org/abs/2606.20728)
  - Untested: open VLMs as the agent, VQA beyond counting, per-category analysis, and faithfulness. Only one closed LLM and 100 images.
- **m&m's** (ECCV 2024, arXiv 2403.11085): 4K+ multi-step multimodal tasks, 33 tools, 10 LLMs.
  - Models are "comparable on tool prediction with JSON-format and Python code generation", but most models produce more executable plans in JSON, and the gap is smaller for code LLMs.
  - Multi-step JSON planning with feedback was best overall.
  - The planners are text LLMs that call vision tools; this is not VLM perception on dense images — [arXiv](https://arxiv.org/abs/2403.11085v4); [ECCV](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1486_ECCV_2024_paper.php)
- **CodeAct** (ICML 2024, arXiv 2402.01030), text-only LLM agents: code actions match or beat JSON even for atomic actions, and the gains are larger for open-source models. This is the classic LLM-side result that has not been replicated for VLM perception on dense images — [arXiv](https://arxiv.org/pdf/2402.01030)
- **VisualNeedle** (arXiv 2605.26380, v2 Sep 2026): Appendix J compares a crop-only tool with a multi-tool OpenCV set (geometry, enhancement, analysis, drawing), run once per model on 9 mostly closed models. It has no Python interpreter and no specialist detectors (HTML) — [arXiv HTML](https://arxiv.org/html/2605.26380)

**Older and agentic systems (each uses one interface only)**

| System | Interface | Tools | What it found / what it left untested |
|---|---|---|---|
| VisProg (CVPR 2023) — [arXiv 2211.11559](https://arxiv.org/abs/2211.11559) | LLM-generated modular program | detectors, VQA, crop, etc. | Program synthesis only. Visual Sketchpad later reports that VisProg *underperformed* base GPT-4o on all its tasks (32.4 on V* vs GPT-4o 66.0) — [Sketchpad HTML](https://arxiv.org/html/2406.09403v3) |
| ViperGPT (ICCV 2023) — [arXiv 2303.08128](https://arxiv.org/abs/2303.08128) | Python via Codex + vision API | GLIP, X-VLM, depth, BLIP-2 | Code only. No ReAct comparison on the same tools (not checked in the full text) |
| MM-ReAct (2023) — [arXiv 2303.11381](https://arxiv.org/abs/2303.11381) | ReAct prompting with ChatGPT | vision experts (captioning, OCR, etc.) | Fixed tools only, with no code comparison (I did not open the full text) |
| Visual Sketchpad (NeurIPS 2024) — [arXiv 2406.09403](https://arxiv.org/html/2406.09403v3) | GPT-4o writes Python that calls tools | Grounding-DINO, SAM / Semantic-SAM, DepthAnything, sliding-window search, crop | V* 66.0 → 80.3. BLINK relative depth 71.8 → 83.9. Failures "often come from vision specialists or simple visual QA errors rather than planning". No ReAct or tool-subset ablation. Open models were tested only with *oracle* GPT-4o sketches |
| HAMMR (arXiv 2404.05465) | hierarchical ReAct | crop, OCR, detection | Function-call style only (snippet only) — [arXiv](https://arxiv.org/pdf/2404.05465) |
| TACO / CoTA (arXiv 2412.05479) | JSON chain-of-thought-and-action | 15 specialists | Trains a multi-tool ReAct-style VLM; no code arm (see agentic_tool_use.md) |
| PyVision (NeurIPS 2025, arXiv 2507.07998) | Python, training-free | dynamic, self-written | GPT-4.1 +7.8 on V*. Builds a taxonomy of the generated tools by domain. No same-model fixed-tool arm (abstract and snippets) — [arXiv](https://arxiv.org/pdf/2507.07998) |
| CodeV (arXiv 2511.19661) | Python | crop and others in code | Says its comparison with DeepEyes and Pixel-Reasoner is not controlled; its ablations vary training stage and reward, not the interface (HTML) — [arXiv HTML](https://arxiv.org/html/2511.19661) |
| CodeVision (CVPR 2026) | code only, no registry | rotate, flip, crop, enhance, plus unseen operations | Argues qualitatively that registries "break when tools are renamed". No direct code-vs-registry experiment — [paper note](https://en.papernotes.org/CVPR2026/vlm_reasoning/thinking_with_programming_vision_towards_a_unified_view_for_thinking_with_images/) |
| Self-Evolving Code-with-Image (arXiv 2608.11292, Aug 2026) | Python + learned skills | — | Open 27B: tool-free 9% → bare Python 33% → evolved skills 56% on CwI-Bench. No fixed-tool arm (abstract) — [arXiv](https://arxiv.org/abs/2608.11292) |
| ToolVision (arXiv 2608.08907, Aug 2026) | code | not named | ToolVision-8B beats Thyme-7B, CodeVision-8B and CodeDance-7B on three high-resolution benchmarks and beats Qwen3-VL-32B-Thinking on V* and HRBench-8K. No interface comparison (abstract) — [arXiv](https://arxiv.org/abs/2608.08907) |
| OpenVisTool (arXiv 2608.08557, Aug 2026) | function-call crop and others | — | Filters SFT trajectories by "causal utility" of tool observations; 4B–27B backbones improve. No code arm (abstract) — [arXiv](https://arxiv.org/abs/2608.08557) |
| DeepEyes / Mini-o3 / CoF / Pixel Reasoner / ZoomEye / OpenThinkIMG / VisTA / AdaTooler-V | fixed crop or zoom (OpenThinkIMG adds SAM and point tools; VisTA does tool *selection*) | — | Single interface. Numbers are in agentic_tool_use.md |

**Audits (2026) that span both interfaces only by accident**
- **"The Illusion of Visual Tool-Use"** (arXiv 2608.06270, EMNLP 2026 Findings)
  - Audits DeepEyes, Pixel Reasoner, Mini-o3, Qwen3-VL-8B, Qwen3-VL-4B and Thyme; only Thyme is code-based.
  - Benchmarks: V*, HR4K, HR8K, VisualProbe and MME-RW-Lite.
  - Stated limitation: "only crop-and-zoom was studied" (HTML) — [arXiv HTML](https://arxiv.org/html/2608.06270)
- **"Do Multimodal Agents Really Benefit from Tool Use?"** (arXiv 2606.02357, v2 Aug 2026)
  - Compares Thyme (code) and DeepEyesV2 (code + search) with tool-free and pure-text counterparts trained on the same source pool.
  - 93% of DeepEyesV2's and 96% of Thyme's tool-solved problems are also solved by at least one non-tool setting (abstract) — [arXiv](https://arxiv.org/abs/2606.02357)

### Inferences
- The specific comparison this project wants (same open VLM; same specialist tool backends for crop, detection, segmentation, pose and OCR; ReAct vs Python; dense natural-scene VQA with per-category results) appears **not to have been published as of 15 Oct 2026**. The closest are:
  - VTC-Bench: different tool type (OpenCV operations) and task type.
  - VTOS: counting only, one closed LLM, 100 images.
  - VisualNeedle Appendix J: crop vs OpenCV multi-tool, closed models, no code.
- Prior evidence points in different directions:
  - CodeAct (text) and VTOS (counting) favour code.
  - m&m's and VTC-Bench find parity, or a JSON/interface advantage for some models.
  - For open Qwen3-VL models, VTC-Bench shows code often *hurts*.
  
  So the result is open, and both a positive and a negative finding would be informative.

### Gaps
- I did not open the full texts of ViperGPT, MM-ReAct or HAMMR to confirm they lack any interface ablation.
- The VTC-Bench table of the code-vs-interface comparison for open models was only partly read; the page was truncated at 100K characters.
- I did not check whether VTOS's LVIS-Count split is public.

---

## Q2. When do tools help or hurt, by skill category, image resolution or density, and base-model size? Is there a per-category, controlled analysis on dense scenes?

### Takeaway
Per-category evidence exists but is fragmented, and none of it is controlled across interfaces:
- **VisualNeedle** gives per-category gains for crop tools on dense scenes, but only for 9 mostly closed models. Average gains are OCR +26.6, spatial +21.6, entity +15.4, occluded +13.5 and colour +12.0 points.
- **VTC-Bench** gives tier-level gains. Measurement, colour and counting are mixed or negative, while chart, math and spatial show large gains.
- **The Illusion audit** gives per-benchmark gains for open models. Tool gains are small or negative on MME-RW-Lite for Qwen3-VL and larger on VisualProbe.

On resolution and density:
- Zoom helps on easy and medium items but plateaus on hard ones (XLRS-Bench pilot in "Beyond Zooming").
- Sparse counting scenes saturate even without agents (VTOS).

On model size, smaller models tend to fail at *planning* and larger ones at *perception* (MM-ToolSandBox). There is no single controlled study that varies category × density × interface × model size on dense natural scenes.

### Cited Findings
- **VisualNeedle** (arXiv 2605.26380)
  - 300 questions; mean image size 7.46 MP; 99% of target boxes cover ≤1% of the image.
  - Five categories: OCR 71, colour 73, entity 64, spatial 58, occluded 34.
  - Crop-only tool, up to 100 calls.
  - Average across 9 models, without → with tools:

    | Category | Without | With | Gain |
    |---|---|---|---|
    | OCR | 11.11 | 37.66 | +26.55 |
    | Spatial | 12.13 | 33.72 | +21.58 |
    | Entity | 14.29 | 29.69 | +15.39 |
    | Occluded | 9.91 | 23.42 | +13.51 |
    | Colour | 17.86 | 29.83 | +11.97 |

  - Colour gains least, which the authors attribute to "subtle attribute discrimination".
  - The best model (Gemini 3.1 Pro) reaches 56.00% with tools vs 16.33% without; Qwen3-VL-235B-A22B-Thinking only goes 8.33 → 11.44 (HTML) — [arXiv HTML](https://arxiv.org/html/2605.26380)
- **The Illusion audit** (arXiv 2608.06270): tool minus direct accuracy, in points:

  | Model | V* | HR4K | HR8K | VisualProbe | MME-RW-Lite |
  |---|---|---|---|---|---|
  | Qwen3-VL-8B | +6.9 | +5.0 | +3.8 | +11.7 | −3.7 |
  | Qwen3-VL-4B | +4.4 | 0.0 | −1.6 | +6.8 | −1.3 |
  | Mini-o3 | +5.5 | +4.8 | +5.8 | +21.3 | +11.9 |
  | DeepEyes | 0.0 | −1.4 | −0.6 | 0.0 | −0.2 |

  The paper has no OCR/attribute/counting breakdown (HTML) — [arXiv HTML](https://arxiv.org/html/2608.06270)
- **MED** ("What Does Vision Tool-Use RL Really Learn?", arXiv 2602.01334, ICML 2026)
  - Models: Qwen2.5-VL-7B and Qwen3-VL-8B. Benchmarks: V*, HR4K, HR8K and VisualProbe E/M/H.
  - Tool-induced share of RL improvement: 0.30 for Qwen2.5-VL and 0.22 for Qwen3-VL, i.e. more than 70% of the gain is intrinsic.
  - Averages over six benchmarks (tool-free / schema-only / tool-available):
    - Qwen2.5-VL: 48.4 / 42.6 / 42.2
    - Qwen3-VL: 53.0 / 40.0 / 61.2
  - No analysis by object size or resolution; per-benchmark results are in an appendix (HTML) — [arXiv HTML](https://arxiv.org/html/2602.01334v2)
- **VTC-Bench tiers** (arXiv 2603.15030)
  - Tier 1 (OCR, restoration, attention focusing): strong, because only coarse operations are needed.
  - Tier 2 (measurement, colour, counting): "mixed, with noticeable drops in high-precision tasks", blamed on tool selection and parameter setting.
  - Tier 3 (chart, math, spatial): large gains; for example Gemini-3.0-Pro spatial +17.78 with the interface (HTML) — [arXiv HTML](https://arxiv.org/html/2603.15030v1)
- **"Beyond Zooming"** (arXiv 2607.25993, Jul 2026): a pilot on XLRS-Bench (ultra-high-resolution remote sensing) finds zoom-in helps easy and medium tasks but plateaus on hard ones. It saturates on global search, multi-region comparison, path planning and dispersed evidence (abstract) — [arXiv](https://arxiv.org/abs/2607.25993)
- **VTOS density tiers** (LVIS-Count): the Sparse tier (<10 objects) was excluded because "zero-shot detectors already saturate it". Direct VLMs undercount badly and specialists overcount (bias +27 to +43) (HTML) — [arXiv HTML](https://arxiv.org/html/2606.20728)
- **ToolGate** (arXiv 2606.03054, Jun 2026): in a ReAct agent with OCR, detection and segmentation tools on Qwen3-VL backbones, helpful and harmful calls occur at similar rates (11.8% vs 9.9%), and "most calls do not change the immediate forced-answer prediction" (abstract) — [arXiv](https://arxiv.org/abs/2606.03054)
- **MG-VQA** (arXiv 2608.17129, formerly "Probe"; v3 Oct 2026): in cluttered tabletop scenes, perception tools (pointing, segmentation, scene graphs) raise average success only from 36.2% (direct) to 37.6%, against a 32.7% image-blind chance level. Manipulation tools raise it to 56.8% (abstract) — [arXiv](https://arxiv.org/abs/2608.17129)
- **"Seeing Is Not Measuring"** (arXiv 2609.29073, Sep 2026), Qwen3.5-4B with 3D detection, metric depth and solvers on ReVSI-Bench:
  - Relative direction 25.9% → 73.4%
  - Relative distance 39.1% → 67.4%
  - Absolute distance (MRA) 0.46 → 0.74
  - Object size 0.58 → 0.61, but 0.97 with ground-truth boxes, so the detector is the bottleneck
  - [arXiv](https://arxiv.org/abs/2609.29073)
- **Model size**
  - MM-ToolSandBox (arXiv 2607.11818; 12 models from 4B open to frontier): a "planning-to-precision crossover". Smaller models fail at deciding what to do; larger ones fail at perceiving. 53% of failures are "incorrect information extraction from images" (abstract) — [arXiv](https://arxiv.org/abs/2607.11818)
  - MG-VQA: the strongest models "searched persistently and re-grounded after failed interactions", while weaker ones answered without enough evidence or stopped after failed actions (abstract) — [arXiv](https://arxiv.org/abs/2608.17129)
- **Resolution**
  - Chain-of-Focus evaluates V* at 8 resolutions from 224 px to 4K, for its own model only — [arXiv](https://arxiv.org/abs/2505.15436) (via agentic_tool_use.md).
  - VisualOverload's InternVL3 ablation found gains plateau above Full HD (via agentic_tool_use.md) — [arXiv HTML](https://arxiv.org/html/2509.25339)
- **PyVision** builds a taxonomy of self-generated tools and reports that "different benchmarks and domains elicit different patterns" (snippet only) — [arXiv](https://arxiv.org/pdf/2507.07998)
- **VACoT** (arXiv 2512.02361): on OCRBench and ChartQA, models usually answer without tools, and hard or adversarial samples trigger more tool use (snippet only) — [arXiv](https://arxiv.org/pdf/2512.02361)

### Inferences
- Across sources, tools help most on **localisable, small-evidence** questions (OCR, small objects, spatial relations between small items) and least on **colour/attribute discrimination, counting precision, and dispersed or global evidence**. This is consistent with VisualNeedle (colour least), VTC-Bench (Tier 2 mixed) and "Beyond Zooming" (dispersed evidence saturates).
- MME-RW-Lite is a benchmark where *native* Qwen3-VL zooming often hurts (−3.7 and −1.3). That makes it a good stress test for whether *specialist* tools or code fix what crop alone cannot.
- Density effects on counting are U-shaped in difficulty: sparse scenes saturate, and extreme density defeats both VLMs and detectors with opposite biases (VTOS). A density-binned study across interfaces is unexplored.

### Gaps
- I found no controlled per-category analysis of P1 vs P2 vs P3 with the same open model on dense natural scenes.
- I found no published analysis of tool benefit as a function of *target box area* or *object count* across interfaces. VisualNeedle reports box-area statistics but not gain by area; MED has no size analysis.
- No study found covering activity or pose questions with tools in crowded natural scenes (see Q3).

---

## Q3. Do specialist tools (Grounding-DINO, YOLO-World, OWLv2, SAM2, CountGD, pose estimators, PaddleOCR) still help modern native VLMs (Qwen3-VL / Qwen3.5 / Gemma 4)?

### Takeaway
The 2025–2026 evidence is mixed and very task-specific.
- **Clear wins:**
  - Metric 3D tools for a 4B VLM (direction +47.5 points).
  - Grounded-SAM2 vs VLM-direct for zero-shot segmentation.
  - Detector tools reduce Qwen3-VL's undercounting in dense scenes, but replace it with overcounting.
- **Weak or no wins:**
  - ReAct + Grounding-DINO counting is *worse* than Claude Sonnet 4.6 alone.
  - Perception tools barely help on cluttered tabletops (MG-VQA).
  - In a ReAct agent on Qwen3-VL, OCR/detection/segmentation calls are about as often harmful as helpful (ToolGate).

There are no controlled 2025–2026 results for pose-estimation tools or PaddleOCR-style OCR tools paired with Qwen3-VL, Qwen3.5 or Gemma 4 on dense VQA.

### Cited Findings
- **VTOS LVIS-Count** (dense counting, 10–100 objects). MAE, mIoU and bias:

  | Method | MAE | mIoU | Bias |
  |---|---|---|---|
  | Qwen3-VL-8B direct | 54.26 | 3.38 | −54.26 |
  | Qwen3-VL-32B direct | 48.32 | 9.98 | — |
  | Sonnet 4.6 count-only | 35.13 | — | — |
  | Grounding DINO (thr 0.10) | 52.76 | 32.19 | +27.44 |
  | CountGD vanilla | 43.01 | 36.13 | +32.05 |
  | ReAct (Sonnet + DINO) | 46.31 | — | — |
  | CodeAct (Sonnet + tools) | 39.38 | — | — |

  Specialists give much better *localisation* (mIoU) than VLM boxes, but not better counts than a strong VLM — [arXiv HTML](https://arxiv.org/html/2606.20728)
- **VTOS PlantSeg-OOD** (Dice): Grounded-SAM2 36.56 vs Sonnet 4.6 VLM-direct 30.45 vs GPT-4o-mini 17.30. VisProg (Sonnet) 36.33 — [arXiv HTML](https://arxiv.org/html/2606.20728)
- **"Seeing Is Not Measuring"**: specialist geometric tools give large gains to Qwen3.5-4B on direction and distance; object size stays detector-limited (0.61 real vs 0.97 with ground-truth boxes) — [arXiv](https://arxiv.org/abs/2609.29073)
- **ToolGate**: in a ReAct agent with OCR, detection and segmentation on Qwen3-VL, helpful calls are 11.8% and harmful calls 9.9%. Gating calls cuts tokens to 64–69% of the ungated baseline with accuracy preserved — [arXiv](https://arxiv.org/abs/2606.03054)
- **MG-VQA**: pointing, segmentation and scene-graph tools add only about 1.4 points on average across 8 VLMs (36.2 → 37.6) — [arXiv](https://arxiv.org/abs/2608.17129)
- **Qwen3-VL-Seg** (May 2026) reports that native segmentation is "often more accurate than decoupled MLLM + SAM setups" (snippet only) — [alphaXiv](https://www.alphaxiv.org/abs/2605.07141)
- **Qwen3-VL launch post**: says tool use "significantly improves" fine-grained perception, without dense-counting numbers (snippet only) — [Qwen blog](https://qwen.ai/blog?id=qwen3-vl)
- **CAPTURe** (ICCV 2025): feeding a counting model's output into a VLM prompt reduced counting error under occlusion (snippet only) — [CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Pothiraj_CAPTURE_Evaluating_Spatial_Reasoning_in_Vision_Language_Models_via_Occluded_ICCV_2025_paper.pdf)
- **GeCo2** (arXiv 2511.08048) reports that CountGD fails in dense scenes with small objects (snippet only) — [arXiv](https://arxiv.org/html/2511.08048v1)
- **Visual Sketchpad (2024)**: GPT-4o + DINO/SAM/Depth tools give large gains (V* +14.3, BLINK depth +12.1), but "failures often come from vision specialists" — [arXiv HTML](https://arxiv.org/html/2406.09403v3)
- **OCR-type evidence**
  - VisualNeedle: crop tools give the largest gain on OCR (+26.6 average) — [arXiv HTML](https://arxiv.org/html/2605.26380)
  - Code training: Thyme's OCRBench dropped (88.4 → 86.3), while CodeVision's orientation-perturbed OCRBench rose 52.2 → 75.4 for its 8B model (agentic_tool_use.md) — [paper note](https://en.papernotes.org/CVPR2026/vlm_reasoning/thinking_with_programming_vision_towards_a_unified_view_for_thinking_with_images/)
  - Neither is a dedicated OCR-engine tool.
- **Pose**: a targeted search found no 2025–2026 VLM-agent paper using a pose-estimation tool for crowded-scene activity QA. The nearest hits were BehaviorVLM (animal pose, Mar 2026) and CrowdPose (2019) — [arXiv 2603.12176](https://arxiv.org/html/2603.12176); [CrowdPose](https://arxiv.org/pdf/1812.00324)
- **Gemma 4**: Google lists "native support for function calling" (sizes E2B, E4B, 12B, 26B, 31B per the page). I found no paper evaluating Gemma 4 with vision tools on high-resolution VQA, apart from MG-VQA v1 mentioning Gemma-4 (46.4 → 62.4 with tools, search summary of v1; the v3 abstract does not name Gemma) — [DeepMind Gemma 4](https://deepmind.google/models/gemma/gemma-4/); [arXiv](https://arxiv.org/html/2608.17129v1)

### Inferences
- Detectors and segmenters now mostly add **localisation** (boxes, masks), not better final answers. For counting, the VLM is biased low and the detector high, so a P3 agent that *calibrates* the detector (threshold sweep, NMS, tiling) is where code should beat ReAct. VTOS's CodeAct > ReAct result is consistent with this.
- Pose and OCR-engine tools against native Qwen3-VL/Qwen3.5/Gemma 4 on dense scenes is an **open, cheap-to-test question**.

### Gaps
- No 2025–2026 controlled result for PaddleOCR (or another OCR engine) as a tool vs Qwen3-VL native OCR on small scene text in high-resolution images.
- No result for ViTPose/RTMPose tools on activity questions.
- No result for OWLv2 or YOLO-World vs Grounding-DINO as the agent's detector.
- The full VTOS paper may contain Qwen-agent rows; I read only the main table.

---

## Q4. Is faithfulness (does the answer causally depend on tool outputs?) and cost (tokens, latency, number of calls vs accuracy) studied systematically for fixed-tool vs code agents?

### Takeaway
Faithfulness of **crop/zoom** agents has been audited heavily in 2026. Nearly all of this work studies crop-only agents, plus Thyme as the lone code agent, and none compares interfaces with the same base model and tools:
- The Illusion audit: causal interventions.
- VisualNeedle: crop-black control.
- CounterCredit: needed/used calls.
- CodeV: a judge-scored "faithful crop" rate.
- TextCall: call-but-no-return.
- "Do Multimodal Agents Really Benefit": capability-set analysis.

Cost is reported only piecemeal:
- TextCall: latency −29–46%.
- ToolGate: tokens down to 64–69%.
- VTOS: LLM-call crossover.
- Illusion audit and "Do Multimodal Agents Really Benefit": higher or unchanged token cost.

No one gives an accuracy-vs-cost Pareto comparison of P1, P2 and P3, and no one scores tool-call regions of *code* agents against ground-truth boxes.

### Cited Findings
- **Illusion of Visual Tool-Use** (arXiv 2608.06270)
  - Two failure modes: CWL and "Looking Without Planning". Gains are concentrated in a "Calibrated minority".
  - CWL share on V*: DeepEyes 72.8%, Qwen3-VL-4B 58.6%, Qwen3-VL-8B 70.7%, Mini-o3 34.6%.
  - Calibrated share: DeepEyes 7.9%, Qwen3-VL-4B 24.1%, Qwen3-VL-8B 20.9%, Mini-o3 45.0%.
  - Replacing returned crops with random crops: Mini-o3 drops 50–64 points; DeepEyes and Thyme are near zero.
  - Thyme is not included in the step-level diagnostic. Only open models were tested. Tools: crop-and-zoom only — [arXiv HTML](https://arxiv.org/html/2608.06270); code: [OpenCausaLab/CauAudit](https://github.com/OpenCausaLab/CauAudit)
- **CodeV** (arXiv 2511.19661): faithfulness is judged by **GPT-4o**, which is asked whether any crop shows the queried content. It does *not* use ground-truth boxes or IoU. Faithful% / Acc:

  | Model | V* | HR4K | HR8K |
  |---|---|---|---|
  | DeepEyes-7B | 49.7 / 84.8 | 24.7 / 73.1 | 6.7 / 69.1 |
  | Pixel-Reasoner-7B | 34.1 / 79.6 | — | — |
  | Thyme-RL-7B | 7.0 / 82.7 | 1.0 / 76.8 | — |
  | CodeV-7B | 68.0 / 84.8 | 33.5 / 76.1 | 13.3 / 71.3 |

  Thyme calls tools in fewer than 10% of rollouts (HTML) — [arXiv HTML](https://arxiv.org/html/2511.19661)
- **VisualNeedle crop-black**: crop-black accuracy stays near text-only, and real crops add 3.77–44.00 points over black crops.
  - Localisation-vs-answer gap: Gemini 3.1 Pro localises the target in 77.78% of runs but answers 56.00% correctly; GPT-5.4 localises in 77.00% but answers 31.22%.
  - Post-coverage over-cropping: GPT-5.4 makes 9.75 crops after first coverage — [arXiv HTML](https://arxiv.org/html/2605.26380)
- **"When Should a VLM Look?" / CounterCredit** (arXiv 2609.22910, Sep 2026, under review)
  - On a cold-start checkpoint, only 10–12% of visual calls were both *needed* (crop branch vs answer-now) and *used* (real crop vs random same-size patch).
  - Released agents make spurious calls 36–87% of the time.
  - Its RL credit gives V* 89.5, HR4K 80.2 and HR8K 76.4 at about 1.8 calls per question. Applied to Qwen3-VL-8B it raises the average 75.4 → 80.8 (abstract) — [arXiv](https://arxiv.org/abs/2609.22910)
- **TextCall** ("Thinking With Tools, Not With Pixels", arXiv 2608.09682, Aug 2026)
  - "Call-but-no-return", which swaps the returned image for a text placeholder, matches or beats full thinking-with-images across LoRA, full fine-tuning and RL, with 29–46% lower latency.
  - Under RL, seeing returned images can make the model stop calling tools.
  - Base models and benchmarks are not named in the abstract — [arXiv](https://arxiv.org/abs/2608.09682)
- **Position paper** (ICML 2026; Yang, Zhu, Huang): interleaved images "do not play a significant role" in recent thinking-with-images gains, which come from an improved language-generation distribution. The authors call for ablations and "more informative visual tools" — [ICML](https://icml.cc/virtual/2026/poster/67100)
- **"Do Multimodal Agents Really Benefit from Tool Use?"** (arXiv 2606.02357): tool access "doesn't reliably reduce generated-token cost". The full loop does not consistently beat either the tool-call format alone or the returned result alone — [arXiv](https://arxiv.org/abs/2606.02357)
- **MED**: "Only accuracy is measured. Efficiency, tool-use traces... are not evaluated" (stated limitation) — [arXiv HTML](https://arxiv.org/html/2602.01334v2)
- **TreeBench / TreeVGR** (arXiv 2507.07999): uses box-level "traceable evidence". Localisation mIoU correlates with accuracy on perception questions but less on reasoning questions. This is about boxes *output in text*, not tool-call regions (snippet-level detail) — [arXiv](https://arxiv.org/pdf/2507.07999)
- **OpenVisTool** keeps only SFT trajectories whose tool observations "causally contribute" to the answer — [arXiv](https://arxiv.org/abs/2608.08557). **ToolVision** rewards tool use only where with-tool beats without-tool — [arXiv](https://arxiv.org/abs/2608.08907)
- **Cost numbers**
  - VTOS: ReAct ≤6 calls per image, CodeAct 1, no latency.
  - ToolGate: 64–69% tokens.
  - GazeVLM: thinking-with-images "can add nearly a thousand additional visual tokens" (snippet only) — [arXiv](https://arxiv.org/html/2605.07817v1)

### Inferences
- Faithfulness for P3 agents is basically unmeasured. Thyme is the only code agent audited, and it barely calls tools.
- Faithfulness for **specialist-tool outputs** (detector boxes, OCR strings, pose keypoints) is unmeasured. Every audit corrupts *images*, not structured text outputs.
- Ground-truth-box-based scoring of where an agent *looked* exists only for crop agents on closed models (VisualNeedle coverage). CodeV uses a GPT-4o judge instead.

### Gaps
- None of these audits reports wall-clock latency together with tokens and calls for both interfaces.
- The Illusion audit's token-cost ratios were not in the part of the HTML I read.

---

## Q5. Do thinking/reasoning modes interact with tool use on perception tasks? Does thinking replace or complement zooming?

### Takeaway
No controlled factorial study (thinking on/off × tools on/off × interface) was found. The indirect evidence conflicts:
- **Thinking is neutral or harmful on fine-grained perception:**
  - Qwen3-VL-Thinking reportedly trails Instruct on V* and HR8K, though partly as a truncation artifact.
  - Tool-use RL gains are mostly intrinsic.
  - Zoom benefits can be internalised without tools (ZwZ, Imagine-OPD, Thinking-Once).
- **Thinking helps:**
  - Extended thinking adds 10.6–15.4 points on HoloCount counting.
  - Zero-shot CoT doubled VisualOverload's hard-split accuracy while lowering the total.
- On tool use under thinking, VTC-Bench shows Thinking variants gain with the interface but lose with code at 8B.

### Cited Findings
- **Chain-of-Visual-Thought** (arXiv 2511.19418) reportedly finds Qwen3-VL-Thinking more than 5% below Qwen3-VL-Instruct on V* and HRBench-8K, attributing this to text CoT "misdirecting" perception (snippet only) — [arXiv](https://arxiv.org/html/2511.19418v2)
- **Cross-paper numbers** (different harnesses, snippet only):
  - DiG lists Qwen3-VL-8B-Thinking at V* 79.1 and HR8K 69.9 — [arXiv](https://arxiv.org/pdf/2512.12633)
  - Perceive-to-Reason lists Qwen3-VL-8B-Instruct at V* 83.8 and HR8K 70.1 — [arXiv](https://arxiv.org/pdf/2607.01191)
- **EPIC-Bench**: under a 4,096-token cap, small Thinking models truncate and underperform Instruct, and larger budgets reverse this (snippet only) — [arXiv](https://arxiv.org/pdf/2605.17070)
- **VTC-Bench**: Qwen3-VL-30B-A3B-Thinking +3.68 with the interface; Qwen3-VL-8B-Thinking −3.97 with code; Qwen3-VL-235B-Thinking +2.94 with the interface — [arXiv HTML](https://arxiv.org/html/2603.15030v1)
- **VisualNeedle**: Qwen3-VL-235B-A22B-Thinking is the weakest tool user (11.44% with tools) and localises the target in only 20.33% of runs, with premature stopping — [arXiv HTML](https://arxiv.org/html/2605.26380)
- **HoloCount** (arXiv 2607.06420): extended thinking adds 10.6–15.4 points on counting (via `dense_counting.md`) — [arXiv](https://arxiv.org/abs/2607.06420)
- **VisualOverload**: zero-shot CoT on InternVL3-38B lowered the total (67.6 → 65.5) but doubled the hard split (7.2 → 14.4) (via agentic_tool_use.md) — [arXiv HTML](https://arxiv.org/html/2509.25339)
- **Internalisation papers**
  - ZwZ (arXiv 2602.11858, ICML 2026) distils zoom into single-pass inference — [arXiv](https://arxiv.org/pdf/2602.11858)
  - Imagine-OPD ("Thinking Without Images", arXiv 2606.08719) learns to "imagine" zoomed views without tools — [arXiv](https://arxiv.org/abs/2606.08719)
  - Thinking-Once (arXiv 2607.27830) routes intermediate-layer evidence. Training-free; +9.9 V* on Qwen2.5-VL-7B; 97.2% less inference time than DeepScan — [arXiv](https://arxiv.org/abs/2607.27830)
- **"Diversity Over Frequency"** (arXiv 2606.00096, ICML 2026)
  - "Tool-use collapse": calls decline while accuracy rises.
  - Removing tools hurts, but incentivising more calls gives only marginal gains.
  - Tools act as "training-time scaffolding" — [arXiv](https://arxiv.org/abs/2606.00096)
- CodeVision trains on Qwen3-VL-8B/32B-**Thinking** bases (so thinking + code is the default there) and does not report Instruct counterparts — [paper note](https://en.papernotes.org/CVPR2026/vlm_reasoning/thinking_with_programming_vision_towards_a_unified_view_for_thinking_with_images/)

### Inferences
- A plausible hypothesis (untested in the literature) is that thinking **complements** tools on counting and multi-step reasoning, but **substitutes for or interferes with** them on single-glance attribute and OCR questions. Thinking may also cut down calls ("thinking replaces looking"), which would inflate CWL.
- Any thinking-vs-instruct comparison must control the token budget; EPIC-Bench shows truncation artifacts.

### Gaps
- No paper varies thinking mode with the same weights and the same tool harness on dense VQA.
- The Qwen3-VL technical report's own Instruct-vs-Thinking numbers on V* and HR-Bench were not checked.

---

## Q6. What failure taxonomies exist for visual agents (wrong region, tool error, misreading tool output, over-calling, under-calling)?

### Takeaway
Several partial taxonomies exist, each tied to one interface or setting:

| Source | Categories |
|---|---|
| Illusion audit | CWL / LWP / No-call / Calibrated |
| MED | Call gain/harm, Schema gain/harm |
| VisualNeedle | localisation vs answer gap, post-coverage over-cropping, premature stopping, gain/harm counts |
| CounterCredit | needed × used |
| ToolFailBench (text LLMs) | Tool-Skip / Result-Ignore / Output-Fabrication |
| MM-ToolSandBox | planning vs information-extraction errors |
| MG-VQA | answering without evidence, stopping after failure |
| Over/under-use papers | AdaTooler-V (over-use), PyVision-RL (interaction collapse), Diversity-over-Frequency (tool-use collapse), CodeV (Thyme rarely calls tools) |

None unifies these for both P2 and P3 with specialist tools, and none labels failures automatically against ground-truth evidence boxes.

### Cited Findings
- **Illusion audit**: CWL (observations have no causal effect), LWP (informative observations but an incoherent call schedule), No-call, and Calibrated. On V*, LWP is 0–9.9% and No-call 5.2–19.4% — [arXiv HTML](https://arxiv.org/html/2608.06270)
- **MED**: Call Gain (tool-free failure fixed after a call), Schema Gain, Call Harm (tool-free success broken after a call), and Schema Harm (lost without a call, from schema interference) — [arXiv HTML](https://arxiv.org/html/2602.01334v2)
- **VisualNeedle**:
  - localised-but-wrong runs
  - excess crops after coverage (GPT-5.4: 9.75)
  - weak localisation and premature stopping (Qwen3-VL: 20.33% localisation, 0.10 post-coverage crops)
  - harmful calls (GPT-5.4: 75 gains / 20 harms; Qwen3-VL: 26 gains / 17 harms)
  
  [arXiv HTML](https://arxiv.org/html/2605.26380)
- **CounterCredit**: a call is "needed" if the crop branch beats answer-now, and "used" if the real crop beats random patches; spurious calls are 36–87% — [arXiv](https://arxiv.org/abs/2609.22910)
- **ToolFailBench** (arXiv 2607.04686, text LLM agents): Tool-Skip, Result-Ignore and Output-Fabrication (snippet only) — [arXiv](https://arxiv.org/pdf/2607.04686)
- **MM-ToolSandBox**: 53% of failures are "incorrect information extraction from images" even with a correct workflow; smaller models fail at planning — [arXiv](https://arxiv.org/abs/2607.11818)
- **DeepTumorVQA** (medical): agents call fewer tools than the ground-truth trace and skip measurement steps, i.e. under-calling (snippet only) — [arXiv](https://arxiv.org/pdf/2605.09679)
- **ToolGate**: harmful calls where faulty tool output overwrites a correct visual belief; the authors call this "diagnostic rather than definitive" (snippet) — [arXiv](https://arxiv.org/abs/2606.03054)
- **MG-VQA**: weaker models "answered without enough evidence or stopped after failed actions" — [arXiv](https://arxiv.org/abs/2608.17129)
- **Visual Sketchpad**: failures come mostly from vision specialists or VQA errors, not planning — [arXiv HTML](https://arxiv.org/html/2406.09403v3)
- **Interaction and tool-use collapse**:
  - PyVision-RL (accumulative tool reward as the fix) — [arXiv](https://arxiv.org/abs/2602.20739)
  - Diversity-over-Frequency — [arXiv](https://arxiv.org/abs/2606.00096)
  - TextCall (seeing returned images can stop calling under RL) — [arXiv](https://arxiv.org/abs/2608.09682)
- **Over-use**: AdaTooler-V finds models call tools when unnecessary, which hurts accuracy — [arXiv](https://arxiv.org/abs/2512.16918)

### Inferences
Ground-truth boxes allow several error classes to be labelled automatically:
- **wrong region**: crop or detector query IoU with the ground-truth box below a threshold
- **tool error**: detector or OCR output disagrees with the ground truth
- **misread output**: tool output correct, answer wrong
- **over-calling**: calls after coverage, or calls on items solved without tools
- **under-calling**: no call, wrong answer, and an oracle crop fixes it
- **CWL**: answer unchanged under observation corruption

Benchmarks with public boxes make this possible: TreeBench, VisualNeedle (bbox in its JSONL), ZoomBench, and LVIS-derived counting.

### Gaps
- No taxonomy covers code-agent-specific failures: runtime errors, hallucinated APIs, coordinate-system bugs, silent no-op code. CodeVision mentions "constraint penalties" and error-log self-correction, but I found no published code-agent error breakdown for VQA.

---

## Q7. Which open small VLMs (2B–8B) can be prompted zero-shot into tool use, and which need SFT?

### Takeaway
- **Qwen3-VL-4B/8B** can use a crop/zoom tool zero-shot; they are audited as native "thinking-with-images" models. Their gains are modest (+4 to +7 on V*), and the tool schema itself can depress accuracy when unused.
- **Qwen3.5-4B** sequenced geometric tools zero-shot without a script.
- **Qwen2.5-VL-7B** suffers schema harm and generally needs RL (crop) or SFT cold start (code).
- **Zero-shot code-tool use** by Qwen3-VL is weak: VTC-Bench shows declines.
- **Gemma 4** advertises native function calling, but I found no vision-tool evaluation.
- **2B-class**: needs SFT and RL (OpenThinkIMG Qwen2-VL-2B: 29.56 zero-shot → 45.67 SFT → 59.39 RL).

### Cited Findings
- **Qwen3-VL-4B/8B** zero-shot crop tool use, as audited: V* +4.4 / +6.9 and VisualProbe +6.8 / +11.7, but MME-RW-Lite −1.3 / −3.7 — [arXiv HTML](https://arxiv.org/html/2608.06270)
- **MED**: Qwen3-VL-8B is "trained with" crop-and-zoom; Qwen2.5-VL-7B is not. Tool-free / schema-only / tool-available averages: Qwen3-VL 53.0 / 40.0 / 61.2; Qwen2.5-VL 48.4 / 42.6 / 42.2 — [arXiv HTML](https://arxiv.org/html/2602.01334v2)
- **CounterCredit** RL raises Qwen3-VL-8B from 75.4 to 80.8 on average — [arXiv](https://arxiv.org/abs/2609.22910)
- **VTC-Bench**: Qwen3-VL code-based scores 26.32–34.41 vs base 29.56–36.32 (mostly declines); Qwen3-VL-8B-Thinking −3.97 with code — [arXiv HTML](https://arxiv.org/html/2603.15030v1)
- **"Seeing Is Not Measuring"**: Qwen3.5-4B "sequences the tools well without a scripted recipe" — [arXiv](https://arxiv.org/abs/2609.29073)
- **DeepEyes**: crop tool use emerges from RL alone on Qwen2.5-VL-7B. **DeepEyesV2**: "direct reinforcement learning alone fails" for code + search and needs a cold start (via agentic_tool_use.md) — [DeepEyes](https://arxiv.org/abs/2505.14362); [DeepEyesV2](https://arxiv.org/abs/2511.05271)
- **ToolVision**: SFT on strong-teacher trajectories can teach a small student to copy call patterns it cannot exploit, because the teacher relies on perception the student lacks — [arXiv](https://arxiv.org/abs/2608.08907)
- **OpenVisTool**: SFT on 42K causally filtered trajectories improves 4B–27B backbones — [arXiv](https://arxiv.org/abs/2608.08557)
- **MM-ToolSandBox** includes a 4B open model; small models fail mainly at planning — [arXiv](https://arxiv.org/abs/2607.11818)
- **OpenThinkIMG** (Qwen2-VL-2B, charts): zero-shot 29.56, SFT 45.67, V-ToolRL 59.39 (via agentic_tool_use.md) — [GitHub](https://github.com/zhaochen0110/OpenThinkIMG)
- **Gemma 4**: "native support for function calling". One practitioner blog reports Gemma 4 31B "will sometimes answer from internal knowledge when a tool call would be more accurate", and an HN user reports tool-calling issues on 26B-A4B. These are anecdotal, not research — [DeepMind](https://deepmind.google/models/gemma/gemma-4/); [HN](https://news.ycombinator.com/item?id=47961107)
- **Self-Evolving CwI**: an unnamed open 27B gets 33% with a bare Python interpreter vs 9% tool-free — [arXiv](https://arxiv.org/abs/2608.11292)
- **CodeAct (text)**: JSON is "consistently weaker" for open-source models — [arXiv](https://arxiv.org/pdf/2402.01030)

### Inferences
- Qwen3-VL-8B-Instruct (plus its Thinking twin) is the natural zero-shot base for P2. P3 zero-shot at 8B is risky; plan a frontier API model as a second base so P3 is not confounded by format failures.
- Qwen3-VL-4B is the realistic SFT/RL target on 2× A6000.
- Whether Gemma 4 E4B/12B can drive vision tools zero-shot is untested. It is a cheap add-on finding.

### Gaps
- I found no published zero-shot P2/P3 numbers for Gemma 4, Qwen3.5 small sizes (except the 4B geometric-tools paper), InternVL3.5 or MiniCPM-V on dense VQA.
- The exact Qwen3.5 small-size lineup was not checked.

---

## Q8. Candidate novel contributions (feasible in about 7 weeks on 2× A6000 + API)

### Takeaway
The clearest open niche is a **controlled, per-category, cost- and faithfulness-instrumented comparison of P1 / P2 / P3** on labelled dense benchmarks:
- same open base model (plus one API model)
- same specialist tool backends
- ground-truth evidence boxes used to label *where* the agent looked and *why* it failed

Each 2026 audit covers one slice: Illusion and MED (crop only), VisualNeedle (closed models, crop), VTC-Bench (OpenCV operations), and VTOS (counting, one closed LLM). None covers this combination. Contributions C1, C2 and C6 share one set of inference runs, so they can be done together. C3 to C5 are cheap add-on ablations. C7 is optional and higher-risk.

### Cited Findings
Benchmarks with public per-question labels and the ground-truth evidence the contributions need (sizes and licences from `../Dense scene VQA training datasets/high_res_detail_vqa.md` unless stated):

| Benchmark | Size | Labels | Notes |
|---|---|---|---|
| TreeBench | 405 MC | one box per question | SA-1B dense images; subcategories Attributes, Material, Physical State, Object Retrieval, OCR, Perspective, Ordering, Contact/Occlusion, Containment, Comparison; Apache-2.0 — [HF](https://huggingface.co/datasets/HaochenWang/TreeBench) |
| VisualNeedle | 300 | answers + target boxes (`bbox` x1y1x2y2) in released JSONL | 5 categories; 7.46 MP mean; "provided in the supplementary material"; dataset licence unclear — [arXiv HTML](https://arxiv.org/html/2605.26380) |
| ZoomBench | 845 | boxes and crop views | 6 perceptual dimensions; Apache-2.0 — [HF](https://huggingface.co/datasets/inclusionAI/ZoomBench) |
| HR-Bench 4K/8K | FSP/FCP | answers only | licence file empty — [HF](https://huggingface.co/datasets/DreamMr/HR-Bench) |
| MME-RealWorld-Lite | 50 per task | answers, 43 tasks | OCR-in-the-wild, monitoring, remote sensing, autonomous driving — [HF](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lite) |
| VisualProbe test E/M/H | — | free-form answers | needs a judge — [HF](https://huggingface.co/datasets/Mini-o3/VisualProbe_Hard) |
| HoloCount | 2,480 | counts (CC BY 4.0) | high-density subset; via dense_counting.md — [HF](https://huggingface.co/datasets/MM-MVR/HoloCount) |
| CountQA | about 1,500 | counts | release status unconfirmed — [arXiv](https://arxiv.org/abs/2508.06585) |
| LVIS-derived counting | — | LVIS boxes give counts and boxes | as VTOS did — [arXiv HTML](https://arxiv.org/html/2606.20728) |

V* boxes: **UNVERIFIED** whether V*Bench ships target boxes.

### Inferences

Compute assumptions (estimates, not measured):
- Qwen3-VL-8B in bf16 is about 17 GB of weights, so it fits on one A6000 with vLLM.
- Tool backends fit on the second GPU: Grounding-DINO/OWLv2, SAM2, ViTPose/RTMPose, PaddleOCR, CountGD.
- One agentic episode takes roughly 10–40 s.
- About 3,000 questions × 3 pipelines × 2 thinking modes ≈ 18K episodes ≈ 50–200 GPU-hours. That is feasible in 2–3 weeks of wall time.

**C1. Controlled interface study: P1 vs P2 vs P3 with identical tools, per category and density**
- **Hypothesis.** Code agents beat ReAct only on questions that need *composition or calibration*: counting with threshold/NMS/tiling, multi-object comparison, measurement. ReAct equals or beats code on single-region lookups (OCR, attributes). For open 8B models, code's advantage shrinks or reverses because of format and execution errors. This follows from VTOS (CodeAct > ReAct on counting) and VTC-Bench (parity; Qwen declines with code).
- **Benchmarks and labels.** TreeBench, VisualNeedle and ZoomBench (categories + boxes), an HR-Bench 4K subset, MME-RW-Lite OCR/monitoring tasks, and HoloCount high-density / LVIS-derived counts for counting.
- **Compute.** About 100–200 GPU-hours inference, plus API for one frontier model on a 500–1,000-item subset.
- **Risk.** Medium. P3 with an 8B model may fail on format; mitigate with a few-shot harness and a frontier-model arm. Prompt-engineering asymmetry between P2 and P3 must be controlled: same tool docstrings, same turn and token budgets.
- **Closest prior and difference.**
  - VTC-Bench: OpenCV operations, not specialist tools; not dense scenes; no faithfulness.
  - VTOS: counting only, Sonnet only, 100 images.
  - VisualNeedle Appendix J: crop vs OpenCV, closed models, no code.
  - m&m's and CodeAct: text LLMs.

**C2. Ground-truth-box faithfulness and causal audit across interfaces, including specialist-tool outputs**
- **Hypothesis.** P3 agents have *lower* evidence-hit rates than P2 even when accuracy matches, because code often processes the whole image or skips crops. CodeV found Thyme <10% tool use and 1–7% faithful. Specialist-tool outputs (boxes, OCR strings) are used more causally than returned crops, because text survives in context. This would extend the TextCall "scaffold" finding.
- **Method.**
  - Instrument every crop, detector query and OCR region, and score IoU or coverage against the ground-truth box.
  - Interventions:
    - crop-black (VisualNeedle)
    - random-patch swap (CounterCredit)
    - call-but-no-return (TextCall)
    - corrupted structured outputs, e.g. shuffled detector boxes or perturbed OCR strings (new)
- **Benchmarks.** TreeBench, VisualNeedle, ZoomBench (boxes required).
- **Compute.** Small: re-running the C1 episodes with 3–4 interventions on about 1,000 items is about 30–60 GPU-hours.
- **Risk.** Low to medium. Mapping code-agent operations to image regions needs a wrapped image API that logs accessed pixels.
- **Closest prior and difference.**
  - Illusion audit: crop only, image corruption only.
  - CodeV: GPT-4o judge, not ground-truth boxes; crop agents.
  - VisualNeedle: coverage for crop only, closed models.
  - TextCall: training-time, crop/zoom, benchmarks not named.

**C3. Do specialist tools still help native-grounding VLMs? Leave-one-tool-out plus oracle and noisy tools**
- **Hypothesis.** For Qwen3-VL/Qwen3.5/Gemma 4, the only tools with positive marginal value on dense scenes are crop (OCR, small objects) and detector-for-counting. Pose and segmentation add little, and an OCR engine adds little over crop + native OCR. Oracle tools built from ground truth give an upper bound, and adding noise shows how much tool error the agent tolerates ("Seeing Is Not Measuring": object-size MRA 0.61 with a real detector vs 0.97 with ground-truth boxes).
- **Benchmarks.** TreeBench OCR and object items; VisualNeedle OCR and entity items; LVIS/HoloCount counting by density bin. For pose/activity, a small self-built set from COCO-keypoints or CrowdPose images with templated questions is needed. **No public dense-scene activity VQA with keypoints was found**, so this is a risk.
- **Compute.** About 7 tool configurations × about 1,000 items on P2 only, roughly 40–80 GPU-hours.
- **Risk.** Medium, because the pose/activity part needs data construction.
- **Closest prior and difference.**
  - ToolGate: learns a gate, no per-tool ablation.
  - VTOS: DINO/SAM only.
  - MG-VQA: tabletop.
  - "Seeing Is Not Measuring": 3D metric tools.

**C4. Thinking × tools factorial with matched token budgets**
- **Hypothesis.** Thinking substitutes for zoom on attribute and OCR questions (fewer calls at equal accuracy, more CWL) and complements tools on counting and comparison, consistent with HoloCount's +10.6–15.4 from thinking.
- **Design.** Qwen3-VL-8B Instruct vs Thinking (same size, same harness) × {P1, P2, P3}, with token caps matched or swept to avoid the EPIC-Bench truncation artifact.
- **Benchmarks.** As in C1.
- **Compute.** Roughly doubles C1; reuse the runs.
- **Risk.** Low.
- **Closest prior and difference.** No controlled study was found. Existing evidence is cross-paper (CoVT, DiG, Perceive-to-Reason) or limited to VTC-Bench Thinking variants without Instruct twins in the same table.

**C5. Resolution and density dose-response: where do tools start paying off?**
- **Hypothesis.** Tool benefit rises as the target-box area fraction falls and as object count rises, then saturates or reverses at extreme density, where detectors overcount and agents fail to plan. The crossover point differs by interface.
- **Design.** Downsample the same images (for example 1K/2K/4K/native), and bin items by ground-truth box-area fraction and object count.
- **Benchmarks.** VisualNeedle and TreeBench (box areas), LVIS counts.
- **Compute.** Low, about 20–40 GPU-hours.
- **Risk.** Low.
- **Closest prior and difference.**
  - Chain-of-Focus: 8 resolutions, one trained model, crop only.
  - VTOS: two density tiers, counting.
  - "Beyond Zooming": remote-sensing difficulty plateau.
  - VisualOverload: plateau above Full HD with no tools.

**C6. A unified, automatically labelled failure taxonomy and released trajectory set**
- **Hypothesis.** Different interfaces fail differently.
  - P2: mostly wrong-region and premature-stop.
  - P3: mostly execution or coordinate bugs, plus "global processing" without localisation.
  - Model size shifts failures from planning to perception (MM-ToolSandBox).
- **Labels.** Ground-truth boxes for wrong-region and under-calling; ground-truth answers plus tool outputs for tool-error vs misread; corruption invariance for CWL; tool-free outcomes for over-calling (MED-style harm).
- **Compute.** Uses C1 and C2 logs; mostly analysis, plus about 200 human-checked trajectories to validate the automatic labels.
- **Risk.** Low.
- **Closest prior and difference.** Illusion (CWL/LWP), VisualNeedle (coverage, post-coverage crops), MED (gain/harm), ToolFailBench (text LLMs). None covers code agents or specialist tools, or labels automatically from ground-truth boxes.

**C7 (optional). Interface-matched SFT for a small VLM: does the interface matter for learnability?**
- **Hypothesis.** With the *same* teacher, questions and tool backends, and trajectories rendered either as P2 JSON calls or P3 code, a 4B student learns P2 faster and more faithfully under a small budget. P3 generalises better to composition-heavy categories. This is the VLM analogue of CodeAct's open-model finding, which favoured code for text.
- **Data.** Synthesise trajectories with an API teacher on TreeVGR-RL / ZwZ / VisualProbe-train items (answers + boxes; agentic_tool_use.md lists licences). Filter by OpenVisTool-style causal utility.
- **Compute.** LoRA SFT of Qwen3-VL-4B on 3–5K trajectories per arm is about 10–20 GPU-hours. Short GRPO on 2× A6000 is possible only with LoRA, small groups and MC/count rewards (VTool-R1 reports convergence in 50–100 steps on far larger hardware; via agentic_tool_use.md). This is high risk within 7 weeks.
- **Risk.** High, from time and API cost for teacher trajectories.
- **Closest prior and difference.**
  - OpenVisTool: function-call only.
  - ToolVision, CodeV, CodeVision: code only.
  - DeepEyesV2: needed a code cold start.
  
  None trains matched P2 and P3 arms from identical supervision.

**C8 (cheap add-on). Accuracy–cost Pareto plus a trivial "call-only-if-unsure" gate**
- **Hypothesis.** Most of P2/P3's accuracy is reachable at a fraction of the cost by gating calls on P1 confidence (answer-logit margin or self-consistency). This would extend ToolGate's 64–69% token result to both interfaces and report latency.
- **Benchmarks.** As in C1.
- **Compute.** Negligible beyond C1.
- **Risk.** Low.
- **Closest prior and difference.** ToolGate (trained external gate, ReAct only), AdaTooler-V (RL), CounterCredit (RL credit). None gives a training-free cross-interface Pareto analysis.

**Novelty risks to watch.** Re-check arXiv in November 2026 for:
- VTC-Bench follow-ups
- a VisualNeedle v3 adding code or specialist tools
- a journal version of the Illusion audit extending to code agents
- the full ToolVision paper, whose tool set is unknown and might include detectors

Cite all of them as closest prior work: VTC-Bench, VTOS, VisualNeedle, the Illusion audit, MED, TextCall, "Do Multimodal Agents Really Benefit", CounterCredit, ToolGate, CodeV, OpenVisTool, ToolVision and m&m's.

### Gaps
- VisualNeedle's data release (supplementary vs HF) and its licence are unclear; confirm access before relying on it.
- I found no public dense-scene VQA benchmark with keypoint or activity labels for the pose-tool arm. It would need to be built from COCO-keypoints, CrowdPose, or HOI datasets (see `../Dense scene VQA training datasets/dense_scene_reasoning.md`).
- I did not verify that V*Bench and HR-Bench ship evidence boxes.
- GPU-hour figures are rough estimates, not measured.
