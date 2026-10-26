# Agentic / tool-use ("thinking with images") training datasets for VLMs

Scope: publicly released SFT trajectories and RL prompt sets with verifiable answers for VLMs that use tools (crop/zoom, grounding boxes, detection, OCR, Python). Checked October 2026. Availability, row counts and license tags were checked live against the Hugging Face (HF) Hub API, the HF dataset cards, and GitHub READMEs on 2026-10-15. Benchmark numbers come from arXiv papers or abstracts. Anything not checked on the actual page is marked **UNVERIFIED**.

Abbreviations:
- V* = V*Bench
- HR4K / HR8K = HR-Bench 4K / 8K
- MME-RW = MME-RealWorld
- QVL-7B = Qwen2.5-VL-7B-Instruct
- P2 = pipeline 2 (ReAct agent with fixed visual tools)
- P3 = pipeline 3 (Python-interpreter agent)

---

## Q1. Catalogue: name, venue/year, links, size, source images/resolution

### Takeaway
At least about 25 agentic or tool-use training releases can be downloaded today. Most of them come from 2025–2026, are built on Qwen2.5-VL-7B, and focus on crop/zoom visual search or Python "code-as-tool". The largest and cleanest RL prompt sets with verifiable answers are:
- DeepEyes-47k
- Thyme-RL (55K)
- PyVision-Image-RL (44.6K)
- CodeV-RL (44.6K)
- CodeDance-RL (60.8K)
- CodeVision-RL (35K)
- TreeVGR-RL-37K
- PixelReasoner-RL (15.4K)
- CoF-RL (8.9K)
- Mini-o3 VisualProbe

The richest SFT trajectory sets are:
- Thyme-SFT (code)
- DeepEyesV2_SFT (code + search)
- Mini-o3 cold-start (multi-turn zoom)
- CodeDance-SFT (34K, code)
- PyVision-Image-SFT (6.9K, code)
- OpenVisTool-42K (function-call crop, 2026)
- CoTA 293K/815K (specialist tools, older)

### Cited Findings

**A. Crop/zoom ("fixed tool") family**
- **DeepEyes** (ICLR 2026; arXiv 2505.14362)
  - HF `ChenShawn/DeepEyes-Datasets-47k` is ungated and tagged apache-2.0. It has three parquet files: `data_v0.8_visual_toolbox_v2` (2.2 GB), `data_0.1.2_visual_toolbox_v2` (0.99 GB) and `data_thinklite_reasoning_acc` (1.66 GB). Code: github.com/Visual-Agent/DeepEyes — [HF](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k); [GitHub](https://github.com/Visual-Agent/DeepEyes); [arXiv](https://arxiv.org/abs/2505.14362)
  - Composition: about 22k samples (47%) from the V* training set, derived from COCO2017; about 14k (30%) from ArxivQA charts; about 11k (23%) from ThinkLite-VL. The data was filtered by Qwen2.5-VL-7B difficulty (samples at 0% or 100% accuracy were removed) and converted to open-ended questions. The paper does not give the resolution of the training images — [arXiv HTML](https://arxiv.org/html/2505.14362)
  - A 4K-sample subset is mirrored as `Mini-o3/DeepEyes_train_4K` — [HF search via API](https://huggingface.co/datasets/Mini-o3/DeepEyes_train_4K)
- **DeepEyesV2** (ICLR 2026; arXiv 2511.05271)
  - HF `honglyhly/DeepEyesV2_SFT` holds a json folder plus `images.zip` (10.3 GB).
  - `honglyhly/DeepEyesV2_RL` holds perception_all_1–5, reason, search and vstar_test parquets, about 10.7 GB in total.
  - The repo is Apache-licensed. The paper text does not give sample counts; they appear only in a figure — [GitHub](https://github.com/Visual-Agent/DeepEyesV2); [HF SFT](https://huggingface.co/datasets/honglyhly/DeepEyesV2_SFT); [HF RL](https://huggingface.co/datasets/honglyhly/DeepEyesV2_RL); [arXiv](https://arxiv.org/abs/2511.05271)
  - Data sources:
    - Perception: V*, ArxivQA, Pixmo-Count, TallyQA, SeekWorld
    - Reasoning: ReVisual, MathCoder, Retool
    - Search: MMSearch-R1
    - RL also adds VGR, Chain-of-Focus and VLM-R3 data
    - SFT trajectories were synthesized with Gemini 2.5 Pro, GPT-4o and Claude Sonnet 4 — [arXiv HTML](https://arxiv.org/html/2511.05271)
  - The HF datasets-server reports only 191 rows for the RL default config, which looks like a partial index. Count the files locally — [datasets-server](https://datasets-server.huggingface.co/size?dataset=honglyhly/DeepEyesV2_RL)
- **Mini-o3 / Visual Probe** (ICLR 2026; arXiv 2509.07969)
  - HF `Mini-o3/VisualProbe_train` (datasets-server: 2,900 rows), `VisualProbe_Easy/Medium/Hard` (test), `Mini-o3-Coldstart-Dataset` (7,267 rows, 4.3 GB download) and `DeepEyes_train_4K`. None of them carries a license tag on HF. The repo is Apache-2.0 — [GitHub](https://github.com/Mini-o3/Mini-o3); [HF coldstart](https://huggingface.co/datasets/Mini-o3/Mini-o3-Coldstart-Dataset); [HF VisualProbe_train](https://huggingface.co/datasets/Mini-o3/VisualProbe_train)
  - The paper says VisualProbe has 4,000 training and 500 test QA pairs, built from high-resolution images with small targets and many distractors, but it does not name the image source. It describes the cold-start set as "about 6,000" multi-turn trajectories, made with 6 hand-written exemplars and kept only when the answer was correct. RL used VisualProbe-train plus 8,000 DeepEyes-47k samples — [arXiv HTML](https://arxiv.org/html/2509.07969)
  - Conflict: 2,900 rows on HF vs. 4,000 in the paper, and 7,267 vs. "about 6,000". Count locally.
- **Chain-of-Focus (CoF / Adaptive-CoF)** (arXiv 2505.15436)
  - HF `xintongzhang/CoF-SFT-Data-5.4k` (apache-2.0) has 5.4k samples: 2.4k with zoom-in and 3k text-only, generated by the authors' visual agent "across a variety of image resolutions".
  - `xintongzhang/CoF-RL-Data` (8,862 rows, `images.zip` 1.7 GB, no license tag) was released on 2025-11-15 — [HF SFT](https://huggingface.co/datasets/xintongzhang/CoF-SFT-Data-5.4k); [HF RL](https://huggingface.co/datasets/xintongzhang/CoF-RL-Data); [GitHub](https://github.com/xtong-zhang/Chain-of-Focus)
  - The paper calls its SFT set "MM-CoF, 3K samples", which conflicts with the 5.4k on HF. It evaluates V* across 8 resolutions from 224 to 4K — [arXiv](https://arxiv.org/abs/2505.15436)
- **Pixel Reasoner** (arXiv 2505.15966; TIGER-Lab)
  - `TIGER-Lab/PixelReasoner-SFT-Data` (apache-2.0) has 8,000+ trajectories: 6,000+ pixel-space trajectories synthesized with GPT-4o (single-pass and error-induced self-correction, images and video) and 2,000+ text trajectories rejection-sampled from Qwen2.5-VL.
  - `TIGER-Lab/PixelReasoner-RL-Data` (apache-2.0) has 15,402 queries "with verifiable answers" over infographics, visually rich scenes and video — [HF SFT](https://huggingface.co/datasets/TIGER-Lab/PixelReasoner-SFT-Data); [HF RL](https://huggingface.co/datasets/TIGER-Lab/PixelReasoner-RL-Data)
- **TreeVGR / TreeBench** (ICLR 2026; arXiv 2507.07999)
  - `HaochenWang/TreeVGR-SFT-35K` and `TreeVGR-RL-37K`, both apache-2.0. SFT is sourced from VGR. RL images come from V* and VisDrone — [GitHub](https://github.com/Haochen-Wang409/TreeVGR); [HF RL](https://huggingface.co/datasets/HaochenWang/TreeVGR-RL-37K); [HF SFT](https://huggingface.co/datasets/HaochenWang/TreeVGR-SFT-35K)
- **ViGoRL** (arXiv 2505.23678)
  - `gsarch/vigorl_datasets` (MIT) has four tarballs: spatial reasoning (SAT-2, 11.2 GB), visual search (built from Segment-Anything images + GPT-4o, 10.7 GB), web action (ICAL) and web grounding (OS-ATLAS, 12.6 GB) — [HF](https://huggingface.co/datasets/gsarch/vigorl_datasets)
- **MGPO** (ACL Findings 2026; arXiv 2507.05920)
  - Trains on 21K standard VQA samples with no grounding labels. Code lives on a verl fork. I found no MGPO dataset on HF — [GitHub](https://github.com/EvolvingLMMs-Lab/MGPO); [arXiv](https://arxiv.org/abs/2507.05920)
- **Active-o3** (ICML 2026)
  - The README checklist still lists "Release the dataset" and "Release the training scripts" as **not done**. Only weights (ModelScope) and a demo are out. The license is BSD-2 for academic use — [GitHub](https://github.com/aim-uofa/Active-o3)
- **VLM-R³** (NeurIPS 2025; arXiv 2505.16192)
  - The SFT ("VLIR") and RL data are on ModelScope (`LittleHenry/VLM-R3-sft-rl-v1`), not HF. The model is `lh-hyr/VLM-R3-7b-rl-v1` — [GitHub](https://github.com/0001Henry/VLM-R3)
- **GRIT** (arXiv 2505.15879)
  - `yfan1997/GRIT_data` holds annotations only (no images, no license tag). Training sets are tiny by design (`tallyqa_train_10`, `vsr_cot_train_10` = 20 samples). Models: GRIT-20-Qwen2.5-VL-3B and InternVL-2B — [HF](https://huggingface.co/datasets/yfan1997/GRIT_data); [GitHub](https://github.com/eric-ai-lab/GRIT)
- **Ground-R1** (arXiv 2505.20272)
  - `ZHZ2002/Ground-R1` contains a subset of Visual-CoT plus SFT jsons (Vanilla_SFT and Ground_SFT). About 8,000 RL samples — [HF](https://huggingface.co/datasets/ZHZ2002/Ground-R1); [GitHub](https://github.com/zzzhhzzz/Ground-R1)
- **VGR** (arXiv 2506.11991)
  - `BytedanceDouyinContent/VGR` (VGR-SFT) is tagged cc0-1.0 and sized 10K–100K. Images are **not** redistributed and must be pulled from LLaVA-NeXT-Data. Reasoning chains carry region annotations — [HF](https://huggingface.co/datasets/BytedanceDouyinContent/VGR)
- **Visual CoT** (NeurIPS 2024; arXiv 2403.16999)
  - `deepcs233/Visual-CoT` provides `viscot_363k.json` (question, answer and key-region bbox) plus `viscot_mixed_2m.json` — [HF](https://huggingface.co/datasets/deepcs233/Visual-CoT)
- **SEAL / V*** (CVPR 2024)
  - `craigwu/seal_vqa_data` holds instruction-tuning subsets (COCO-2014/2017 and GQA images). The project is MIT — [GitHub](https://github.com/penghao-wu/vstar)
- **ZoomEye** (EMNLP 2025)
  - Training-free tree search, so there is no training set. `omlab/zoom_eye_data` is small (<1K) — [arXiv](https://arxiv.org/abs/2411.16044); [HF](https://huggingface.co/datasets/omlab/zoom_eye_data)
- **AdaTooler-V** (ACL 2026 Findings; arXiv 2512.16918)
  - `AdaTooler-V/AdaTooler-V-train-data` (apache-2.0) contains `adatooler_v_sft.json` (AdaTooler-V-CoT-100k, CoT written by Qwen2.5-VL-72B) and `adatooler_v_rl_with_deltaS.json` (AdaTooler-V-300k: single-image, multi-image and video, with `solution` fields) — [HF](https://huggingface.co/datasets/AdaTooler-V/AdaTooler-V-train-data); [GitHub](https://github.com/CYWang735/AdaTooler-V)
- **OpenVisTool-42K** (arXiv 2608.08557, Aug 2026)
  - `LockOnN/OpenVisTool-42K` (license: "other") has 42,048 trajectories across Chart, GUI Grounding, Table, Web-to-HTML and Visual Search. They are in ms-swift agent format (`tool_call`/`tool_response` roles, e.g., a `crop` function), were synthesized with a Qwen3.5-Plus teacher, and were filtered for answer correctness and "tool-use gain". OpenVisTool-Bench is released alongside — [HF](https://huggingface.co/datasets/LockOnN/OpenVisTool-42K); [arXiv](https://arxiv.org/abs/2608.08557)
- **Simple o3** (arXiv 2508.12109)
  - The paper claims an open-source "TWI-Tools-146K" set (crop, zoom, reuse). **UNVERIFIED**: HF searches for "TWI-Tools" and "Simple-o3" returned nothing — [arXiv](https://arxiv.org/abs/2508.12109)
- **DeepSketcher** (CVPR 2026 Findings; arXiv 2509.25866)
  - Says it has 31k interleaved trajectories with tool calls and edited images. **UNVERIFIED**: no HF dataset found and the GitHub README is nearly empty — [arXiv](https://arxiv.org/abs/2509.25866); [GitHub](https://github.com/MiliLab/DeepSketcher)
- **Skywork-R1V4** (arXiv 2512.02395)
  - SFT-only 30B-A3B model trained on fewer than 30,000 trajectories (image manipulation + search). I found no public dataset — [arXiv](https://arxiv.org/abs/2512.02395)
- **ZoomRefine, Look-Back, Argus**
  - ZoomRefine: no arXiv or HF hit.
  - Look-Back (arXiv 2507.03019): only `ShuoY/Look-Back-eval` is on HF.
  - Argus (CVPR 2025; arXiv 2505.23766): no training set confirmed. The HF `tomg-group-umd/argus` repo is tied to a *different* arXiv ID (2506.07371) — [arXiv Look-Back](https://arxiv.org/abs/2507.03019); [arXiv Argus](https://arxiv.org/abs/2505.23766)

**B. Python / code-interpreter family**
- **Thyme** (arXiv 2508.11630; Kwai)
  - `Kwai-Keye/Thyme-SFT` (MIT, 100K–1M; splits `wo_thinking_thyme_single_round`, `2round`, `computation`).
  - `Kwai-Keye/Thyme-RL` (MIT, 55,173 rows, about 48 GB, images inline as base64) — [HF SFT](https://huggingface.co/datasets/Kwai-Keye/Thyme-SFT); [HF RL](https://huggingface.co/datasets/Kwai-Keye/Thyme-RL); [GitHub](https://github.com/yfzhang114/Thyme)
  - The paper describes "500K" SFT samples, but the listed subsets add up to about 217K: 100K no-code, 50K executable real-world QA, 28K crop, 14K rotation, about 10K contrast, 15K computation, plus multi-round. For RL, the authors manually collected high-resolution images (at least 2048 px on one side) with small targets (≤5% of the image area); the paper gives 30K in one place and 10K in another — [arXiv HTML](https://arxiv.org/html/2508.11630)
- **PyVision-RL** (arXiv 2602.20739, Feb 2026)
  - `Agents-X/PyVision-Image-SFT-Data`: 6,880 rows, 6.7 GB images, no license tag.
  - `Agents-X/PyVision-Image-RL-Data`: 44,643 rows, apache-2.0, 12.1 GB images. Fields: question, answer, ability, data_source (e.g., "vstar").
  - Video SFT/RL sets also exist — [HF RL](https://huggingface.co/datasets/Agents-X/PyVision-Image-RL-Data); [HF SFT](https://huggingface.co/datasets/Agents-X/PyVision-Image-SFT-Data); [GitHub](https://github.com/agents-x-project/PyVision-RL)
  - Image SFT: about 7K samples synthesized with GPT-4.1 from MMK12, GMAI-Reasoning, ChartQA, InfoVQA and MMPR. Image RL: about 44K visual-search samples from DeepEyes and Mini-o3, plus V-Thinker and WeMath reasoning data — [arXiv HTML](https://arxiv.org/html/2602.20739)
  - The earlier PyVision (arXiv 2507.07998) is training-free — [GitHub](https://github.com/agents-x-project/PyVision)
- **CodeV** (arXiv 2511.19661)
  - `RenlyH/CodeV-RL-Data`: apache-2.0, 44,560 rows, 1.58 GB parquet — [HF](https://huggingface.co/datasets/RenlyH/CodeV-RL-Data)
- **CodeDance** (CVPR 2026; arXiv 2512.17312)
  - `Peterwatwec/CodeDance-SFT`: 33,816 rows, 12.3 GB, executable multi-turn trajectories.
  - `Peterwatwec/CodeDance-RL`: 60,774 rows (README says 63k). Uses the `execute_python_code` tool with a preloaded `input_image`. No license tag on HF — [GitHub](https://github.com/CodeDance-VL/CodeDance); [HF SFT](https://huggingface.co/datasets/Peterwatwec/CodeDance-SFT)
- **CodeVision** ("Thinking with Programming Vision", CVPR 2026; arXiv 2512.03746)
  - `kkwok/CodeVision-SFT`: CC-BY-NC-4.0, ShareGPT format with `code_image_tool` calls, intermediate images listed. SFT was built with GPT-5-High.
  - `kkwok/CodeVision-RL`: CC-BY-SA-4.0, "35k" — [HF SFT](https://huggingface.co/datasets/kkwok/CodeVision-SFT); [HF RL](https://huggingface.co/datasets/kkwok/CodeVision-RL); [GitHub](https://github.com/ByteDance-BandAI/CodeVision)
- **V-Thinker** (arXiv 2511.04460)
  - `We-Math/V-Interaction-400K` (preview: 252K samples) and `V-Perception-40K` (37K). Code-rendered and synthetic, mostly math/diagram data. No license tag on the HF dataset; the repo badge says MIT — [GitHub](https://github.com/We-Math/V-Thinker); [HF](https://huggingface.co/datasets/We-Math/V-Interaction-400K)
- **VTool-R1** (ICLR 2026; arXiv 2505.19255)
  - Python visual-editing tools on ReFOCUS chart and table data. `VTOOL/Refocus_Chart` is on HF (10K–100K); a preprocessed parquet is on Google Drive. 3B, 7B and 32B weights are released — [GitHub](https://github.com/VTool-R1/VTool-R1)
- **Visual-ARFT / MAT** (arXiv 2505.14246)
  - `laolao77/MAT` (apache-2.0, 2,969 rows including the MAT-Training and MAT-Benchmark folders). Covers search and coding (crop, rotate) — [HF](https://huggingface.co/datasets/laolao77/MAT); [arXiv](https://arxiv.org/abs/2505.14246)
- **ViLaSR** (drawing operations: boxes and auxiliary lines)
  - `AntResearchNLP/ViLaSR-data`: ColdStart-33k, Reflective-Rejection-Sampling-8k, RL-40k. No license tag on the dataset; the SR_91k part is CC-BY-NC-4.0 (derived from SpaceR) — [HF](https://huggingface.co/datasets/AntResearchNLP/ViLaSR-data)
- **OpenThinkIMG / V-ToolRL** (arXiv 2505.08617)
  - `hitsmy/OpenThinkIMG-Chart-SFT-2942` (2,942 rows) and `OpenThinkIMG-Chart-RL-14501`, chart-only. Tools: SAM, Crop, Point, ZoominSubplot, SegmentRegionAroundPoint. Models are Qwen2-VL-2B — [GitHub](https://github.com/zhaochen0110/OpenThinkIMG); [HF](https://huggingface.co/datasets/hitsmy/OpenThinkIMG-Chart-SFT-2942)

**C. Specialist-tool / ReAct-style family (older or multi-tool)**
- **TACO / CoTA** (arXiv 2412.05479, now retitled "LATTE: Learning to Think with Vision Specialists")
  - `Salesforce/cota-llava` and `cota-mantis`: 293,105 and 815,582 rows, CC-BY-NC-4.0.
  - 15 actions: OCR, LocalizeObjects, GetObjects, EstimateRegionDepth/ObjectDepth, Crop, ZoomIn, QueryLanguageModel, image/text similarity, DetectFaces, QueryKnowledgeBase, Calculate, SolveMathEquation, and Terminate.
  - Generated by gpt-4o-2024-08-06 over Cauldron and Mantis-Instruct (COCO, AOKVQA, VG, …) — [HF](https://huggingface.co/datasets/Salesforce/cota-llava); [arXiv](https://arxiv.org/abs/2412.05479)
- **LLaVA-Plus** (2023)
  - `LLaVA-VL/llava-plus-data` (LLaVA-Plus-v1-117K): GPT-4-generated tool-use instructions, CC-BY-NC-4.0, must also follow OpenAI terms — [HF](https://huggingface.co/datasets/LLaVA-VL/llava-plus-data)
- **ToolVQA** (ICCV 2025)
  - `DietCoke4671/ToolVQA` (apache-2.0): 21,105 train and 2,550 test samples, 10 tools (ImageDescription, GoogleSearch, OCR, …) in AgentLego format, 2.78 steps per sample on average — [HF](https://huggingface.co/datasets/DietCoke4671/ToolVQA)
- **CogCoM**
  - `qijimrc/CoMDataset` (CC-BY-SA-4.0): 84K synthesized + 7K hand-labelled math chains of manipulations (GROUNDING, OCR, CROP_AND_ZOOMIN, …) — [GitHub](https://github.com/zai-org/CogCoM); [HF](https://huggingface.co/datasets/qijimrc/CoMDataset)
- **VisTA** (arXiv 2505.20289)
  - RL tool *selection* on ChartQA and Geometry3K. No new dataset; "additional datasets coming soon" — [GitHub](https://github.com/OoDBag/VisTA)
- **Point-RFT** (arXiv 2505.19702)
  - Point-CoT 71K (GPT-4o + Molmo points, Mammoth-VL images), on Google Drive only. Code is Apache-2.0 — [GitHub](https://github.com/kodenii/Point-RFT)
- **Rex-Thinker**
  - `IDEA-Research/HumanRef-CoT-45k` and `Mountchicken/RefCOCOg-RexThinker-20k` (apache-2.0): CoT for object referring/detection — [arXiv](https://arxiv.org/abs/2506.04034); [HF](https://huggingface.co/datasets/Mountchicken/RefCOCOg-RexThinker-20k)
- **Visual Sketchpad** (NeurIPS 2024)
  - Prompting framework (GPT-4o + gradio vision experts). Ships task data, not training data — [GitHub](https://github.com/Yushi-Hu/VisualSketchpad)
- **VisProg / ViperGPT / MM-ReAct**
  - Program-synthesis or prompting systems. I found no released training traces for them (I did not search these specifically).

**D. Other 2026 items that reuse existing data**
- These mostly train on DeepEyes or Mini-o3 data and are useful as recipes, not new datasets:
  - "Learning to Zoom Efficiently with a Contrastive Curriculum" (EMNLP 2026): zoom RL without warm-start SFT; adds a synthetic Muffin&Chihuahua grid set; code at UKPLab/emnlp2026-zoom-in — [arXiv](https://arxiv.org/abs/2609.03206)
  - NV-CoT (continuous bbox actions, follows DeepEyes data for RL) — [arXiv](https://arxiv.org/abs/2602.23959)
  - Self-Calling CoT (sCoT) — [arXiv](https://arxiv.org/abs/2512.08511)

### Inferences
- For dense-scene visual search, the core open pool is DeepEyes-47k (V*-train portion), VisualProbe, CoF-RL, PixelReasoner-RL, TreeVGR-RL and PyVision-Image-RL. These overlap heavily: PyVision-RL reuses DeepEyes and Mini-o3, and DeepEyesV2 reuses CoF, VGR and VLM-R3. Expect heavy duplication if you concatenate them, and deduplicate by image hash.
- Row counts on HF often disagree with the papers (Mini-o3, CoF, Thyme, CodeDance). Treat the numbers on the HF page as the real ones.

### Gaps
- The exact contents and counts of the DeepEyesV2 SFT/RL files are not in the paper text, and datasets-server indexing looks partial.
- I could not confirm the image resolution of most training sets beyond Thyme-RL (≥2048 px on one side) and the V*-derived sets.
- No public data found for Active-o3, MGPO, Skywork-R1V4, ZoomRefine, Simple o3 (TWI-Tools-146K) or DeepSketcher.
- Qwen3-VL / Qwen3.5 official agent training data: none found. The only Qwen3-family-related items are the OpenVisTool teacher (Qwen3.5-Plus) and CodeVision's Qwen3-VL recipe.

---

## Q2. Tool interface, trajectory format, and whether intermediate images are included

### Takeaway
There are three interface styles:
- **(a) Fixed JSON tool call with a bbox.** The crop is returned as a new image. Used by DeepEyes, Mini-o3, CoF, Pixel Reasoner, MGPO, ViGoRL multi-turn and OpenVisTool. This matches P2.
- **(b) Free-form Python in a sandbox.** stdout and rendered images come back. Used by Thyme, DeepEyesV2, PyVision-RL, CodeV, CodeDance, CodeVision and V-Thinker. This matches P3.
- **(c) Text-only grounded CoT with boxes or points and no execution.** Used by GRIT, VGR, Ground-R1, TreeVGR, Point-RFT and Visual CoT.

Most SFT sets in (a) and (b) store the intermediate images (crops or processed images) as extra image entries in each trajectory.

### Cited Findings
- **DeepEyes**: the `image_zoom_in_tool` takes `bbox_2d` [x1,y1,x2,y2] and an optional `label`. The cropped image is appended to the trajectory. Reward = accuracy + format + a tool bonus that applies only when the answer is correct. The data is RL-only (no SFT trajectories) — [arXiv HTML](https://arxiv.org/html/2505.14362)
- **DeepEyes training framework**: a verl plugin with async multi-turn rollouts. Each sample picks its tool via an `env_name` field, `max_turns` goes up to 50, and the "allowing dynamic multi-modal input in agent observations" feature is what makes returned images possible — [GitHub](https://github.com/Visual-Agent/DeepEyes)
- **Mini-o3**:
  - Actions are ground (zoom) with a normalized `bbox_2d` in [0,1] and `source` ∈ {original_image, observation_i}, so the model can crop from earlier crops, or a final answer.
  - Training is capped at 6 turns with 32K context; testing allows up to 32 turns.
  - An "over-turn masking" trick stops trajectories that hit the turn limit from being penalized — [arXiv HTML](https://arxiv.org/html/2509.07969)
  - The RL config uses `system_prompt="tool_crop"` and `answer_key=solution` — [GitHub](https://github.com/Mini-o3/Mini-o3)
- **Pixel Reasoner**: operations are zoom-in and select-frame. SFT is stored as Qwen message lists with image/video paths (unzip `images.zip`) — [HF SFT](https://huggingface.co/datasets/TIGER-Lab/PixelReasoner-SFT-Data)
- **MGPO**: the model predicts grounding coordinates and the framework crops automatically inside a multi-turn template. Reward is binary answer correctness only — [arXiv](https://arxiv.org/abs/2507.05920)
- **Thyme**:
  - Python goes in code blocks. The sandbox blocks dangerous operations, has a 10 s timeout, runs autopep8 and clamps crop coordinates.
  - Results return inside `<sandbox_output>` and are masked from the loss.
  - Answers use `<think>`/`<answer>`. Operations include crop, rotate, contrast and arithmetic — [arXiv HTML](https://arxiv.org/html/2508.11630)
  - Images in the HF dataset are base64 and need decoding — [GitHub](https://github.com/yfzhang114/Thyme)
- **DeepEyesV2**:
  - Jupyter-style sandbox with `image_1…image_n` preloaded and a 300 s timeout; `plt.show()` images and stdout are returned. Also `search` and `image_search` (SerpAPI) tools, one tool type per turn — [arXiv HTML](https://arxiv.org/html/2511.05271)
  - Sandbox server: ChenShawn/MultiModal-Jupyter-Sandbox (Docker recommended) — [GitHub](https://github.com/Visual-Agent/DeepEyesV2)
- **PyVision-RL**: code goes in code tags and results in interpreter tags; text and rendered images are appended. Turn limit is 4 in training and 30 in evaluation, with 32K context. An accumulative tool reward (0.1 per turn) prevents "interaction collapse" — [arXiv HTML](https://arxiv.org/html/2602.20739); [GitHub](https://github.com/agents-x-project/PyVision-RL)
- **CodeDance**: the RL prompt says "You may write python code… use the preloaded PIL Image object `input_image`". The tool is `execute_python_code`; operations cover crop, draw, count and plot — [GitHub](https://github.com/CodeDance-VL/CodeDance)
- **CodeVision**: ShareGPT format with a `code_image_tool` JSON call and a `tool` turn ("Here is the processed image…"). The `images` list includes each processed image (e.g., `sample28_0/1/2.png`). Metadata stores the transform, the answer and the number of steps — [HF](https://huggingface.co/datasets/kkwok/CodeVision-SFT)
- **CodeV (TAPO)**: visual tools are written as Python code, and the dense step-wise reward is computed from tool inputs and outputs. The paper argues that many agents get the final answer right while their crops are unfaithful — [arXiv](https://arxiv.org/abs/2511.19661)
- **OpenVisTool-42K**: ms-swift agent JSONL with `tools` schemas and `tool_call`/`tool_response` messages. Its `images` list matches every image token, including tool observations — [HF](https://huggingface.co/datasets/LockOnN/OpenVisTool-42K)
- **CoTA**: multi-turn chain-of-thought and actions over 15 specialist tools (detection, OCR, depth, crop/zoom, face detection, calculator, KB), ending with Terminate — [HF](https://huggingface.co/datasets/Salesforce/cota-llava)
- **ToolVQA**: each step logs tool name, thought, input and *text* output (e.g., an ImageDescription caption) — [HF](https://huggingface.co/datasets/DietCoke4671/ToolVQA)
- **Text-only grounded CoT**:
  - GRIT interleaves bbox coordinates in the reasoning without tool execution — [GitHub](https://github.com/eric-ai-lab/GRIT)
  - VGR "replays" the selected regions' features — [HF](https://huggingface.co/datasets/BytedanceDouyinContent/VGR)
  - Point-RFT uses points from Molmo — [GitHub](https://github.com/kodenii/Point-RFT)
- **Causal audits (caveat)**:
  - Across six models, returned crops often have no causal effect on the answer ("Calling Without Looking") — [arXiv](https://arxiv.org/abs/2608.06270)
  - VisualNeedle's "crop-black" test replaces returned crops with black images — [arXiv](https://arxiv.org/abs/2605.26380)

### Inferences
- For P2, the easiest formats to convert are DeepEyes and Mini-o3 (single `crop(bbox)` tool, returned image), OpenVisTool (function-calling schema) and CoTA (closest to a classic multi-tool ReAct).
- For P3, Thyme-SFT, DeepEyesV2_SFT, PyVision-Image-SFT, CodeDance-SFT and CodeVision-SFT already contain executable code with returned images. Their sandbox conventions differ (preloaded variable names, tags), so reformat them to your own executor's prompt before SFT.

### Gaps
- I did not check exact field-level schemas for DeepEyesV2_SFT, Mini-o3-Coldstart, PyVision-Image-SFT or CodeDance-SFT. Inspect a few rows after downloading.

---

## Q3. Are final answers public and verifiable (RL reward)? Licenses

### Takeaway
Almost every RL set listed includes ground-truth answers. Many expect an LLM-as-judge (Qwen2.5-72B) for free-form answers rather than exact match. Licenses range from permissive to non-commercial:
- Apache-2.0: DeepEyes, PixelReasoner, TreeVGR, PyVision-Image-RL, CodeV-RL, AdaTooler-V, CoF-SFT, ToolVQA, MAT
- MIT: Thyme, ViGoRL
- CC0 for annotations only: VGR
- CC-BY-SA: CodeVision-RL, CoMDataset
- CC-BY-NC: CoTA, LLaVA-Plus, CodeVision-SFT, Visual-CoT (per its card text), ViLaSR SR_91k part
- No license tag on HF: Mini-o3 sets, CoF-RL, CodeDance, PyVision-Image-SFT, OpenThinkIMG, V-Interaction, DeepEyesV2 datasets (the repo itself is Apache)

### Cited Findings
- PixelReasoner-RL: "15,402 training queries with verifiable answers"; fields are question, answer and qid — [HF](https://huggingface.co/datasets/TIGER-Lab/PixelReasoner-RL-Data)
- PyVision-Image-RL entries carry an `answer` (e.g., MC letter) and `data_source`. RL needs a served Qwen2.5-72B-Instruct judge — [HF](https://huggingface.co/datasets/Agents-X/PyVision-Image-RL-Data); [GitHub](https://github.com/agents-x-project/PyVision-RL)
- DeepEyes and DeepEyesV2 both require a vLLM-served Qwen2.5-72B-Instruct judge ("llm-as-a-judge verification"). They recommend ≥32 GPUs for 7B training and ≥1200 GB CPU RAM per node because of the high-resolution images — [GitHub DeepEyes](https://github.com/Visual-Agent/DeepEyes); [GitHub DeepEyesV2](https://github.com/Visual-Agent/DeepEyesV2)
- Mini-o3 RL uses `answer_key=solution`, multiple-choice rule rewards plus a GPT-based answer extraction — [GitHub](https://github.com/Mini-o3/Mini-o3)
- VTool-R1 uses a Qwen2.5-7B-Instruct judge, with exact-match as a fallback (with a noted performance cost). 3B training ran on 8×H100 and 7B on 16×H100, mostly converging within 50–100 steps — [GitHub](https://github.com/VTool-R1/VTool-R1)
- AdaTooler-V-300k entries contain `solution` and `problem_type` (multiple choice, etc.) — [HF](https://huggingface.co/datasets/AdaTooler-V/AdaTooler-V-train-data)
- Visual-CoT conflict: the HF tag says apache-2.0, but the card body says "Attribution-NonCommercial 4.0 International" — [HF](https://huggingface.co/datasets/deepcs233/Visual-CoT)
- VGR is CC0 "subject to any intellectual property rights… owned by Bytedance", and images come from LLaVA-NeXT under their own terms — [HF](https://huggingface.co/datasets/BytedanceDouyinContent/VGR)
- LLaVA-Plus is CC-BY-NC-4.0 and must follow OpenAI's terms — [HF](https://huggingface.co/datasets/LLaVA-VL/llava-plus-data)
- CoTA is CC-BY-NC-4.0; TACO models built with Llama 3 also fall under the Llama 3 license — [GitHub](https://github.com/SalesforceAIResearch/TACO)
- Thyme-SFT and Thyme-RL are tagged MIT — [HF](https://huggingface.co/datasets/Kwai-Keye/Thyme-RL)
- CodeVision-SFT is CC-BY-NC-4.0 and CodeVision-RL is CC-BY-SA-4.0 — [HF SFT](https://huggingface.co/datasets/kkwok/CodeVision-SFT); [HF RL](https://huggingface.co/datasets/kkwok/CodeVision-RL)
- OpenVisTool-42K license is "other" — [HF](https://huggingface.co/datasets/LockOnN/OpenVisTool-42K)
- MGPO trains from binary correctness only, with no grounding labels — [GitHub](https://github.com/EvolvingLMMs-Lab/MGPO)

### Inferences
- A course project (non-commercial academic use) can use all of these. Cite them and keep NC data out of any released commercial weights.
- To save compute, prefer RL subsets with multiple-choice or short numeric answers. These include PyVision-Image-RL's V*-style MC items, VisualProbe, CoF-RL and the counting items in DeepEyesV2 perception (Pixmo-Count, TallyQA). Rule-based rewards then work and you avoid hosting a 72B judge. A 7B or smaller judge (as in VTool-R1) is a tested alternative.

### Gaps
- I did not inspect the actual answer formats of DeepEyes-47k, CodeDance-RL, CodeV-RL or Thyme-RL, so I don't know what share is MC, numeric or free-form.
- Licenses of the underlying source images (SA-1B, COCO, web crawls) are not re-stated on most cards.

---

## Q4. Base models and reported gains (V*, HR-Bench, MME-RealWorld, …)

### Takeaway
Nearly every 2025–26 release is built on QVL-7B. Typical gains are +6 to +19 V*, +6 to +10 HR4K, +5 to +9 HR8K and +7 to +11 MME-RW. Baseline numbers for the *same* base model vary by 7 points across papers (V*: 71.2 / 75.5 / 76.4 / 78.5), so compare within a single table. Very few releases have 3B checkpoints: GRIT-3B, VTool-R1-3B, and OpenThinkIMG at 2B.

### Cited Findings
- **DeepEyes-7B vs QVL-7B (own paper)**:
  - V* 71.2→90.1, HR4K 68.8→75.1, HR8K 65.3→72.6, MME-RW-Lite 42.3→53.2
  - Also small gains on POPE and math — [arXiv HTML](https://arxiv.org/html/2505.14362)
- **Mini-o3 Table 1 (Avg@K, 7B)**:

  | Model | VisualProbe Hard / Medium / Easy | V* | HR4K | HR8K | MME-RW |
  |---|---|---|---|---|---|
  | Mini-o3 | 48.0 / 50.4 / 67.0 | 88.2 | 77.5 | 73.3 | 65.5 |
  | QVL-7B | 23.9 / 26.0 / 39.1 | 75.5 | 68.2 | 62.7 | 57.3 |
  | DeepEyes (re-evaluated) | 35.1 / 29.8 / 60.1 | 83.3 | 73.2 | 69.5 | 64.0 |
  | Pixel Reasoner | 28.8 / 29.6 / 58.4 | 86.3 | 74.0 | 66.9 | 64.4 |
  | Chain-of-Focus | — | 88.0 | — | — | — |

  — [arXiv HTML](https://arxiv.org/html/2509.07969)
- **Thyme-7B vs QVL-7B**:
  - V* 76.4→82.2, HR4K 68.8→77.0, HR8K 65.3→72.0
  - MME-RW overall 58.3→64.8 (perception 60.6→67.1, reasoning 38.6→48.4)
  - OCRBench 88.4→86.3 (a drop), BLINK 56.4→56.1 — [arXiv HTML](https://arxiv.org/html/2508.11630)
- **DeepEyesV2-7B vs QVL-7B**:
  - V* 78.5→81.8 (lower than DeepEyes' 85.6 in the same table), HR4K 71.6→77.9, HR8K 67.9→73.8
  - MME-RW 57.3→64.9, OCRBench 864→882 — [arXiv HTML](https://arxiv.org/html/2511.05271)
- **PyVision-RL Table 1 (QVL-7B base; V* / HR4K / HR8K)**:

  | Model | V* | HR4K | HR8K |
  |---|---|---|---|
  | QVL-7B | 78.5 | 71.6 | 67.9 |
  | **PyVision-Image** | **88.7** (avg@32) | 78.1 | 74.3 |
  | Mini-o3 | 88.2 | 77.5 | 73.3 |
  | DeepEyes | 85.6 | 75.1 | 72.6 |
  | CodeV | 84.8 | 76.1 | 71.3 |
  | CodeDance | 84.8 | 75.2 | 72.3 |
  | Pixel-Reasoner | 84.3 | 74.0 | 66.9 |
  | CodeVision | 83.7 | 75.6 | 72.2 |
  | Thyme | 82.2 | 77.0 | 72.0 |
  | DeepEyes-v2 | 81.8 | 77.9 | 73.8 |

  PyVision-Image also scored WeMath 34.6→47.7 and MathVerse 45.6→55.8. Training took 700 RL steps on 8×H100 — [arXiv HTML](https://arxiv.org/html/2602.20739)
- **Pixel Reasoner (own paper)**: V* 84%, TallyQA-Complex 74%, InfographicsVQA 84% — [arXiv](https://arxiv.org/abs/2505.15966)
- **ViGoRL**: 86.4% on V* with multi-turn zoom RL — [arXiv](https://arxiv.org/abs/2505.23678)
- **TreeVGR (QVL-7B init)**: V* +16.8, MME-RW +12.6, TreeBench +13.4 — [arXiv](https://arxiv.org/abs/2507.07999)
- **MGPO (QVL-7B, 21K samples)**: +5.4 MME-RW (in-distribution) and +5.2 V* (OOD) over GRPO — [arXiv](https://arxiv.org/abs/2507.05920)
- **AdaTooler-V-7B**: V* 89.8, +11.3 over base — [arXiv](https://arxiv.org/abs/2512.16918)
- **CoF**: beats prior VLMs by 5% on V* across 8 resolutions (224 px to 4K) — [arXiv](https://arxiv.org/abs/2505.15436)
- **VTool-R1-7B**: ReFOCUS-TableVQA 64.7→71.7 — [GitHub](https://github.com/VTool-R1/VTool-R1)
- **OpenThinkIMG (Qwen2-VL-2B)**: V-ToolRL 59.39 vs SFT 45.67 vs zero-shot 29.56 on charts; it beat TACO (30.50) and CogCoM (15.07) — [GitHub](https://github.com/zhaochen0110/OpenThinkIMG)
- **ZoomEye (training-free)**: InternVL2.5-8B gains 15.71 / 17.69 on HR-Bench — [arXiv](https://arxiv.org/abs/2411.16044)
- **Skywork-R1V4 (30B-A3B, SFT-only, <30K trajectories)**: MMSearch 66.1, FVQA 67.2 — [arXiv](https://arxiv.org/abs/2512.02395)
- **CodeVision**: reports results on both Qwen2.5-VL and Qwen3-VL and provides Qwen3-VL SFT/RL configs — [GitHub](https://github.com/ByteDance-BandAI/CodeVision)

### Inferences
- On the same evaluation harness (PyVision-RL's table), the best open 7B P3-style agent (PyVision-Image, 88.7 V*) and P2-style agent (Mini-o3, 88.2) are roughly tied on V*. Code agents gain more on math (WeMath, MathVerse).
- For VisualOverload's counting and OCR categories, relevant signals are Pixel Reasoner's TallyQA-Complex result and DeepEyesV2's counting training data. Note that Thyme's OCRBench drop suggests code-tool training can slightly hurt OCR.

### Gaps
- Few papers report QVL-3B numbers. I found no released 3B checkpoints for DeepEyes, Mini-o3, Thyme or PyVision-RL (I did not check exhaustively).
- MME-RW numbers are not comparable across papers (full vs Lite).

---

## Q5. Fit for pipelines 2 (fixed tools) and 3 (Python), SFT vs RL; results on VisualOverload

### Takeaway
No agentic or tool-use model has published VisualOverload results. The benchmark paper tests 37 plain VLMs (o3 without tools reaches 69.5% overall and 19.6% on hard), and QVL-3B/7B score 54.1 / 61.5 overall with only 4.8 / 9.6 on hard. Both pipelines therefore have a clean opportunity.

**P2 shortlist:**
- SFT: Mini-o3 Coldstart (7.3K multi-turn zoom), CoF-SFT (5.4K), PixelReasoner-SFT (≈8K), OpenVisTool-42K (Visual Search domain)
- RL: DeepEyes-47k, VisualProbe-train, CoF-RL, PixelReasoner-RL, TreeVGR-RL-37K

**P3 shortlist:**
- SFT: Thyme-SFT, PyVision-Image-SFT (6.9K), CodeDance-SFT (34K), CodeVision-SFT, DeepEyesV2_SFT
- RL: PyVision-Image-RL (44.6K), CodeV-RL (44.6K), CodeDance-RL (60.8K), Thyme-RL (55K), CodeVision-RL (35K)

**Multi-tool ReAct with detection/OCR/depth tools:** CoTA (293K, NC) is the main large set. ToolVQA (21K) is another but leans toward knowledge and search.

### Cited Findings
- **VisualOverload benchmark facts**:
  - 2,720 QA pairs, private ground truth, 150 public-domain painting scans downsampled to about 4K (3840×2160 pixel count), six task categories.
  - QVL-3B: 94.0 / 56.0 / 4.8 / 54.1 (easy / medium / hard / total). QVL-7B: 97.9 / 66.2 / 9.6 / 61.5.
  - No agentic, tool-use or zoom models were evaluated, and no test-time crop or zoom was tried. Specialized high-resolution models did *worse* than similar-size regular models. In an InternVL3 ablation, gains plateaued above Full HD.
  - Zero-shot CoT on InternVL3-38B lowered the total (67.6→65.5) but doubled the hard split (7.2→14.4) — [arXiv HTML](https://arxiv.org/html/2509.25339); [arXiv](https://arxiv.org/abs/2509.25339)
- The HF card lists answer types `choice` / `counting` / `ocr` and `difficulty` metadata; the license is CC-BY-SA-4.0 — [HF](https://huggingface.co/datasets/paulgavrikov/visualoverload)
- A web search for VisualOverload scores of DeepEyes or other "thinking-with-images" agents returned none. VisualNeedle (2026) only lists VisualOverload as a dense benchmark (2,720 Q, about 7.78 MP on average) — [VisualNeedle arXiv](https://arxiv.org/abs/2605.26380)
- **Direct RL without cold-start**:
  - DeepEyes reports tool use emerging from RL alone (no SFT) — [arXiv](https://arxiv.org/abs/2505.14362)
  - DeepEyesV2 reports that "direct reinforcement learning alone fails to induce robust tool-use behavior" for code + search and needs a cold start — [arXiv](https://arxiv.org/abs/2511.05271)
  - The contrastive-curriculum zoom paper (EMNLP 2026) learns zoom with no warm-start SFT, using an InfoNCE intrinsic reward — [arXiv](https://arxiv.org/abs/2609.03206)
- **Interaction collapse**: PyVision-RL finds that RL for code agents tends to reduce tool use, and fixes it with oversampling-filtering-ranking plus an accumulative tool reward — [arXiv](https://arxiv.org/abs/2602.20739)
- **Over-use**: AdaTooler-V finds open models call tools when they don't need to, which hurts accuracy, and adds a Tool-Benefit-Score-scaled reward — [arXiv](https://arxiv.org/abs/2512.16918)
- **Faithfulness**:
  - CodeV finds high final accuracy alongside low rates of faithful tool use — [arXiv](https://arxiv.org/abs/2511.19661)
  - The causal audit finds tool use often brings "marginal or negative gains over direct inference at substantially higher token cost" — [arXiv](https://arxiv.org/abs/2608.06270)
- **Training infrastructure**:
  - Mini-o3 uses LLaMA-Factory for SFT and verl for RL, on 8×8 GPUs — [GitHub](https://github.com/Mini-o3/Mini-o3)
  - AdaTooler-V uses verl-tool on 8×H100/A100-80GB — [GitHub](https://github.com/CYWang735/AdaTooler-V)
  - ViLaSR, CodeVision and CodeDance use LLaMA-Factory for SFT and EasyR1 or verl for RL — [GitHub ViLaSR](https://github.com/AntResearchNLP/ViLaSR); [GitHub CodeVision](https://github.com/ByteDance-BandAI/CodeVision)

### Inferences
- **P2 (fixed crop/zoom + maybe detect/OCR)**:
  - SFT on Mini-o3 Coldstart and/or CoF-SFT, then GRPO on VisualProbe-train + the DeepEyes V*-train subset + CoF-RL. This is the most direct match to VisualOverload's "find a small detail in a 4K dense scene" questions.
  - If P2 includes detection or OCR tools, CoTA is the only large trajectory set with those tool names, but it uses LLaVA-OV/Mantis formatting and is NC.
- **P3 (Python)**:
  - SFT on PyVision-Image-SFT (small, clean, GPT-4.1) or Thyme-SFT (large, MIT), then RL on PyVision-Image-RL (its data is DeepEyes/Mini-o3-derived, so it targets the same visual-search skills).
  - Thyme's computation split and DeepEyesV2's counting-derived data are the most relevant to VisualOverload counting.
- **SFT vs RL at 3B**: published evidence is thin. GRIT trained 3B with just 20 samples, and VTool-R1 trained 3B on 8×H100. A practical plan is SFT on a few thousand trajectories, then short GRPO (VTool-R1 reports convergence within 50–100 steps) with a small judge or rule-based MC/count rewards.
- **Evaluation design**: to tell whether tools actually help on VisualOverload, run a "crop-black" or observation-corruption control (VisualNeedle, causal audit). Note that VisualOverload's no-zoom QVL baselines are already given per difficulty, which is useful for P1.

### Gaps
- No published VisualOverload numbers exist for any agentic model (DeepEyes, Mini-o3, Thyme, PyVision, o3-with-tools). The leaderboard at paulgavrikov.github.io/visualoverload was not checked directly.
- No direct evidence on how well training on photos (COCO/SA-1B) transfers to paintings.

---

## Q6. Image-overlap risk with VisualOverload's public-domain paintings

### Takeaway
The risk looks low. VisualOverload's 150 images are Google Arts & Culture museum scans of public-domain paintings. The agentic training sets I checked draw mostly from:
- photos: COCO, SA-1B, GQA, VisDrone, Visual Genome
- charts and docs: ArxivQA, ChartQA, InfoVQA
- synthetic renders: V-Thinker, CodeVision charts

The highest residual risk is in web-crawled or loosely documented pools: Thyme-RL's manually collected high-resolution images, the DeepEyesV2 perception/search sets, Cauldron/Mantis-derived CoTA, and LLaVA-NeXT-derived VGR/TreeVGR-SFT. None of these was verified to exclude artworks.

### Cited Findings
- VisualOverload images are high-resolution scans of public-domain paintings from museum collections via Google Arts & Culture: 150 images at about 4K, with 28 images already below 4K but above Full HD — [arXiv HTML](https://arxiv.org/html/2509.25339)
- A copy is also on HF as `Voxel51/VisualOverload`, so the images are easy to fetch for hashing — [HF API search](https://huggingface.co/datasets/Voxel51/VisualOverload)
- DeepEyes sources: V* training (COCO2017), ArxivQA, ThinkLite-VL — [arXiv HTML](https://arxiv.org/html/2505.14362)
- TreeVGR-RL images come from V* and VisDrone; TreeBench uses SA-1B — [GitHub](https://github.com/Haochen-Wang409/TreeVGR)
- ViGoRL visual search is built on Segment-Anything images — [HF](https://huggingface.co/datasets/gsarch/vigorl_datasets)
- SEAL instruction data uses COCO and GQA — [GitHub](https://github.com/penghao-wu/vstar)
- Thyme RL uses manually collected high-resolution images (≥2048 px) plus V*, arXivQA and ThinkLite-VL; the specific sources are not named — [arXiv HTML](https://arxiv.org/html/2508.11630)
- DeepEyesV2 perception sources: V*, ArxivQA, Pixmo-Count, TallyQA, SeekWorld; search sources include MMSearch-R1 — [arXiv HTML](https://arxiv.org/html/2511.05271)
- CoTA sources: Cauldron and Mantis-Instruct (COCO, AOKVQA, ScienceQA, Visual Genome, …) — [HF](https://huggingface.co/datasets/Salesforce/cota-llava)
- VGR images come from LLaVA-NeXT-Data — [HF](https://huggingface.co/datasets/BytedanceDouyinContent/VGR)
- Mini-o3 VisualProbe images are described only as "high-resolution"; the source is not named in the paper — [arXiv HTML](https://arxiv.org/html/2509.07969)

### Inferences
- Before training, compute perceptual hashes (pHash/dHash, plus CLIP near-duplicate search at about 0.95 cosine) for all 150 VisualOverload images and check them against every downloaded training image. Prioritize Thyme-RL, DeepEyesV2, VisualProbe, CoTA and any LLaVA-NeXT-derived set. Pixmo (in DeepEyesV2) and the Cauldron mixtures are the most likely places for museum artwork to appear.
- The domain shift (photos → paintings) is a bigger concern than leakage. Consider adding unlabeled or self-made painting crops (from non-VisualOverload public-domain paintings) to the RL prompt pool, with programmatically verifiable questions such as counting with detector agreement. That is out of scope for these notes.

### Gaps
- I did not run actual image-overlap checks. Source lists for VisualProbe, Thyme-RL's manually collected images, and DeepEyesV2's SeekWorld/search images are not documented enough to rule out paintings.
