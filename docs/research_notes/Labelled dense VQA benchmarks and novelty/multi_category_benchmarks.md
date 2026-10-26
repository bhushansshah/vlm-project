# Multi-category dense / high-resolution VQA benchmarks with public answers (replacements for VisualOverload as an EVALUATION set)

Notes as of 2026-10-15. Project: "Studying the Performance of VLM Agents on VQA Tasks for Dense Datasets". It compares (1) a base VLM, (2) a ReAct agent with fixed tools (crop/zoom, detection, segmentation, pose, OCR) and (3) a VLM agent with a Python interpreter. It needs per-question correctness, per-category and per-difficulty accuracy, failure analysis, and analysis of how tool use relates to correctness.

Builds on `../Dense scene VQA training datasets/high_res_detail_vqa.md`, `dense_scene_reasoning.md` and `agentic_tool_use.md`. Facts carried over from those notes are marked **[N]**, with their original source link. They were not re-fetched.

Verification tags:
- **[V]**: checked this session against the primary page. That means the HF Hub API (license tag, file list), datasets-server `/size`, `/statistics` or `/first-rows`, the raw annotation file downloaded and counted, or the arXiv HTML.
- **[N]**: from earlier project notes.
- **[U]**: unverified.
- "HF license tag" is the dataset-card metadata. Image licenses from upstream sources may still apply.

Out of scope here (another researcher covers them): single-skill counting benchmarks (CountQA, HoloCount, CAPTURe, PairTally) and OCR-only benchmarks (OCRBench v2 etc.). They are mentioned only as pointers. Novelty/gap analysis is also out of scope.

---

## Q1. Catalogue: which multi-category dense / high-resolution benchmarks with public answers exist, and how do they rank for this project?

### Takeaway
No public-label benchmark reproduces VisualOverload (painted crowds at ~4K with six categories and three difficulty levels). A combination does cover it:
- **Core set.** TreeBench (10 categories, a box for every target, big headroom), MME-RealWorld-Lite (43 tasks, perception/reasoning split, includes monitoring crowd counting) and VisualProbe Easy/Medium/Hard (a built-in difficulty split, 4K).
- **Second tier.** ZoomBench (a box for every item), HR-Bench 4K/8K, FineRS-4k test (4K UAV, boxes plus masks, object-size tiers) and UR-Bench (the only public-label set with paintings: Chinese scrolls at gigapixel scale, with Micro/Regional/Global levels).
- **V\* is effectively saturated.** Recent 7–8B models and agents score 85–93%.

### Cited Findings

**A. Natural-scene, high-resolution, multi-category (best fit)**

- **TreeBench** (from "Traceable Evidence Enhanced Visual Grounded Reasoning", arXiv 2507.07999; ICLR 2026 per earlier notes [N])
  - Links: HF [HaochenWang/TreeBench](https://huggingface.co/datasets/HaochenWang/TreeBench), GitHub [Haochen-Wang409/TreeVGR](https://github.com/Haochen-Wang409/TreeVGR).
  - Size: 405 questions in a single split that HF calls `train` [V]. The images are 1K SA-1B images sampled "prioritizing images with dense objects" and filtered down to 647 [V] — [arXiv HTML](https://arxiv.org/html/2507.07999).
  - Resolution: Table 1 gives 2,152×1,615 [V]. Target instances occupy on average 3.05% of the image [V] — [arXiv HTML](https://arxiv.org/html/2507.07999).
- **MME-RealWorld** (ICLR 2025, arXiv 2408.13257)
  - Links: HF [yifanzhang114/MME-RealWorld](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld), [MME-RealWorld-Lite](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lite), GitHub [yfzhang114/MME-RealWorld](https://github.com/yfzhang114/MME-RealWorld).
  - Paper figures: 13,366 images averaging 2,000×1,500 and 29,429 QA. RS images exceed 5000×5000, and one monitoring item counts 133 vehicles [N] — [GitHub](https://github.com/yfzhang114/MME-RealWorld).
  - The released English annotation file `MME_RealWorld.json` has **23,609** items [V, downloaded and counted]. The repo also ships `MME_RealWorld_CN.json` [V] — [HF file list](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld).
  - The Lite version keeps "50 samples per task, or all if fewer than 50" and is supported by VLMEvalKit and lmms-eval [V] — [GitHub README](https://github.com/yfzhang114/MME-RealWorld).
- **VisualProbe** (Mini-o3, arXiv 2509.07969)
  - Size: the test sets are HF [Mini-o3/VisualProbe_Easy](https://huggingface.co/datasets/Mini-o3/VisualProbe_Easy) (141), [VisualProbe_Medium](https://huggingface.co/datasets/Mini-o3/VisualProbe_Medium) (268) and [VisualProbe_Hard](https://huggingface.co/datasets/Mini-o3/VisualProbe_Hard) (106), 515 in total. The split is named `validation` [V]. The paper says "500 pairs for testing" [N] — [arXiv](https://arxiv.org/html/2509.07969).
  - Images are "4K-resolution images with small targets, dense distractors" [N] — [arXiv 2607.00816](https://arxiv.org/html/2607.00816).
  - Sample questions: "What color is the ankle band on the feet of the seventh person in the dark blue checked robe…" and "What is the first word in the lower left corner of the text?" [V] — [HF first-rows](https://huggingface.co/datasets/Mini-o3/VisualProbe_Easy).
- **ZoomBench** (ICML 2026, "Zooming without Zooming", arXiv 2602.11858)
  - Links: HF [inclusionAI/ZoomBench](https://huggingface.co/datasets/inclusionAI/ZoomBench), GitHub [inclusionAI/Zooming-without-Zooming](https://github.com/inclusionAI/Zooming-without-Zooming).
  - Size: 845 test items [V].
  - Resolution: full-image width is 1,275–12,500 px (mean 2,173). The supplied evidence crops are 44–7,068 px wide (mean 762) [V, datasets-server statistics].
  - Six dimensions: Fine-Grained Counting, OCR, Color Attributes, Structural Attributes, Material Attributes, Object Identification [V] — [arXiv HTML](https://arxiv.org/html/2602.11858).
- **HR-Bench 4K / 8K** (DC² paper, arXiv 2408.15556; AAAI 2025 [U])
  - Links: HF [DreamMr/HR-Bench](https://huggingface.co/datasets/DreamMr/HR-Bench).
  - Size: 800 rows per version [V].
  - Rows repeat the same question with rotated options (`cycle_category` 0–3), i.e. a circular-evaluation layout [V, first-rows]. That implies about 200 distinct questions per version (inferred).
  - The 8K version averages 8K resolution, and the 4K version is cropped from it around the annotated objects [N] — [HF card](https://huggingface.co/datasets/DreamMr/HR-Bench).
- **FineRS-4k test** (NeurIPS 2025, arXiv 2510.21311)
  - Links: HF [Jiazuo98/Finers-4k-benchmark](https://huggingface.co/datasets/Jiazuo98/Finers-4k-benchmark).
  - Size: the test file holds **2,427 annotations on 1,132 images**. Of those, 2,376 items are 3840×2160 [V, downloaded `all_annotations_final_test_v5.json` and counted].
  - Content: UAV/drone 4K frames from YouTube and the authors' own footage [N] — [arXiv](https://arxiv.org/html/2510.21311v1).
- **UR-Bench** (arXiv 2601.08748, "Multi-Hop Reasoning over Ultra-High-Resolution Images")
  - Links: HF [UR-Bench/UR-Bench](https://huggingface.co/datasets/UR-Bench/UR-Bench).
  - Size: **528 questions** in `UR-Bench.jsonl`, plus about 98 images in `image/` [V, downloaded and counted]. Earlier notes said the release was unverified; it has since been released (HF last modified 2026-08-24) [V].
  - Content: the scroll paintings average 48,685×2,821 px (up to 718 MP). There are also satellite images and Google Street View panoramas [N] — [arXiv](https://arxiv.org/html/2601.08748v1).
- **VisualNeedle** (arXiv 2605.26380, v2 dated 30 Sep 2026)
  - Size: 300 open-ended questions on very dense images: urban street views, dense documents, shelves, bookcases and maps.
  - Density (Table 1): 7.46 MP mean, 122.79 DINO-X entities per image, 24.55 DINO-X categories per image and 352.17 SAM segments per image [V] — [arXiv HTML](https://arxiv.org/html/2605.26380).
  - Release: data and code are "provided in the supplementary material". No HF or GitHub link was found [V].
- **V\* Bench** (CVPR 2024)
  - Links: HF [craigwu/vstar_bench](https://huggingface.co/datasets/craigwu/vstar_bench).
  - Size: 191 SA-1B images, averaging 2,246×1,582 [N].

**B. Crowd / gigapixel / multi-domain high-resolution**

- **HRScene** (**ICCV 2025**, arXiv 2504.18406) [V venue] — [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_HRScene_How_Far_Are_VLMs_from_Effective_High-Resolution_Image_Understanding_ICCV_2025_paper.html); HF [Wenliang04/HRScene](https://huggingface.co/datasets/Wenliang04/HRScene)
  - Coverage: 25 real-world datasets plus 2 synthetic diagnostics. Resolutions run from 1,024² to 35,503×26,627.
  - Size: 7,073 samples, split into val 750, testmini 1,000 and test 5,323 (test labels hidden) [V] — [HF README](https://huggingface.co/datasets/Wenliang04/HRScene).
  - Relevant real-world configs, each with about 30 val + about 40 testmini public items [V, datasets-server /size]:
    - `realworld_PANDA` (gigapixel crowds; 30 val / 44 testmini / 226 test)
    - `realworld_Video_Monitoring`
    - `realworld_ArtBench`, `realworld_VStar_Bench`, `realworld_HR-Bench`, `realworld_Remote_Sensing`, `realworld_Autonomous_Driving`
- **COREVQA** (arXiv 2507.13405, 2025) — HF [COREVQA2025/COREVQA](https://huggingface.co/datasets/COREVQA2025/COREVQA)
  - Size: 5,608 image–statement pairs on CrowdHuman crowd photos (4,927 from train01, 681 from train02) [V].
  - The released `COREVQA.csv` has only three columns: `image_id, question, answer`, with 1,566 TRUE and 4,042 FALSE [V, downloaded and counted].
- **"Gigapixel Evaluation on Vision Language Models"** (UC eScholarship thesis)
  - Describes scenes of about 25,000×14,000 px with box-location, attribute-search and spatial-relation questions [U, search snippet only] — [eScholarship](https://escholarship.org/uc/item/7mt7612b).
  - I found no public dataset for it.

**C. Aerial / remote sensing, ultra-high resolution, multi-category**

- **LRS-VQA** (arXiv 2503.07588) — HF [ll-13/LRS-VQA](https://huggingface.co/datasets/ll-13/LRS-VQA)
  - Size: 7,333 QA [V].
  - Resolution: per-item `image_size` values such as 4000×4000 to 7000×7000. The paper range is 1,024–27,328 px [N].
- **XLRS-Bench** (CVPR 2025)
  - Full benchmark: 8,500² average, 45,942 annotations, 16 tasks [N] — [GitHub](https://github.com/AI9Stars/XLRS-Bench).
  - Released VQA: HF [initiacms/XLRS-Bench-lite](https://huggingface.co/datasets/initiacms/XLRS-Bench-lite) has only 420 VQA [V]. There are also grounding and caption repos (`XLRS-Bench_visual_grounding_en`, `_caption_en`) [V, HF search].
- **RSHR-Bench** (arXiv 2512.17319) — HF [RL-MIND/RSHR-Bench](https://huggingface.co/datasets/RL-MIND/RSHR-Bench)
  - Images: 5,329 full-scene images averaging 8,700×8,065, up to 29,200×27,620 [V] — [arXiv HTML](https://arxiv.org/html/2512.17319v1).
  - Size: HF lists 1,932 test rows [V]. The paper says 3,864 VQA tasks; the paper itself is inconsistent here (1,932 vs 3,864) [V].
- **GEOBench-VLM**: more than 10k instructions [N] — [HF](https://huggingface.co/datasets/aialliance/GEOBench-VLM).

**D. Dense documents, maps and charts (information-dense, not natural scenes)**

- **O3-Bench** (InSight-o3, **ICLR 2026**, arXiv 2512.18745) — HF [m-Just/O3-Bench](https://huggingface.co/datasets/m-Just/O3-Bench)
  - Size: 204 images (117 chart, 87 map) and 345 QA (163 chart, 182 map) [V] — [arXiv](https://arxiv.org/html/2512.18745).
  - Images reach e.g. 8000×5125 [V, first-rows].
- **SEED-Bench-2-Plus**: 2.3K MC on charts, maps and webs [N] — [HF](https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2-plus). This is OCR-heavy, so it is only a pointer.

**E. Art / cultural heritage with public labels**

- **UR-Bench Humanistic Scenes**: 180 questions on Chinese portrait and narrative scrolls [V] (see A).
- **HRScene `realworld_ArtBench`**: 30 val + 44 testmini public; 232 hidden [V].
- **VQArt-Bench** (ICCV 2025 Workshop AI4VA, arXiv 2510.12750) — [GitHub AlfaranoAndrea/VQArt-Bench](https://github.com/AlfaranoAndrea/VQArt-Bench)
  - Size and images: 14,463 four-option MC on Wikipedia artwork images [V] — [arXiv HTML](https://arxiv.org/html/2510.12750v1).
  - Not available: resolution and density figures are not stated, and I found no HF release [V, HF search returned nothing].
- **BLINK `Art_Style`**: 117 val / 117 test [V]. This is a style-matching task, not dense-scene QA.

**F. Agentic / tool-use benchmarks (relevant to the tool analysis, but not dense-scene)**

- **TIR-Bench** (arXiv 2511.01833) — HF [Agents-X/TIR-Bench](https://huggingface.co/datasets/Agents-X/TIR-Bench): 1,215 test items, 13 tasks that require image-processing tools (e.g. jigsaw, auxiliary lines) [V].
- **VisualToolBench** (Scale AI) — HF [ScaleAI/VisualToolBench](https://huggingface.co/datasets/ScaleAI/VisualToolBench): 1,204 items (603 single-turn, 601 multi-turn). It is rubric-graded and ships reference tool trajectories [V].
- **RealXBench** (DeepEyesV2) — HF [glowol/RealXBench](https://huggingface.co/datasets/glowol/RealXBench): 194 of 300 items released, flagged perception/search/reason. Many need web search [V].
- **Pix2Fact** (arXiv 2602.00593) — HF [pix2fact/Pix2FactBenchmark](https://huggingface.co/datasets/pix2fact/Pix2FactBenchmark): 1,000 4K+ images with a `bounding_box` and web-evidence fields. Every question also needs external knowledge [V].

**G. General, low-resolution perception sets (sanity checks only)**

- MMStar: 1,500 [V]. HF [Lin-Chen/MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar).
- CV-Bench: 2,638 (2D 1,438, 3D 1,200) [V]. HF [nyu-visionx/CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench).
- BLINK: per-task val/test of about 120–150 [V]. HF [BLINK-Benchmark/BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK).
- NaturalBench: 1,900 rows [V]. HF [BaiqiL/NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench).

**H. Checked and judged not applicable**

- **Ref-L4**: REC boxes only, 13,420 val / 31,921 test [V] — [HF](https://huggingface.co/datasets/JierunChen/Ref-L4).
- **Visual Haystacks**: multi-image retrieval, MIT [V, HF search] — [HF](https://huggingface.co/datasets/tsunghanwu/visual_haystacks).
- **DiCoBench** (ECCV 2026): multi-image, 765 MC [V, search snippet] — [ECCV](https://eccv.ecva.net/virtual/2026/poster/4297).
- **CapProbe** (arXiv 2608.11074): 346 images and 25,650 MC QA, but "will be released soon" [V] — [arXiv](https://arxiv.org/html/2608.11074).
- **DenseWorld**: HF `xcll/DenseWorld_Benchmark` exists (56,234 rows, archives named Adverse_Env, Smart_City, etc.) but has no card [V, HF API]. I could not determine its task or labels.
- **DynRefer**: a region-captioning method, not a VQA benchmark [U].
- **MagnifierBench** (OtterHD) and an "MME-HD" set: no HF dataset found under those names [V, HF search]. The `MME-HD-CN.tar.gz` file inside the MME-RealWorld repo is the CN image archive [V].
- Not found: a CrowdHuman- or PANDA-based VQA benchmark beyond COREVQA and the HRScene PANDA slice [V, negative search].

### Inferences

**Ranked catalogue for this project.** The ranking weighs:
- multi-category coverage with category labels in the data
- density
- public answers for every item
- a difficulty split
- evidence boxes
- headroom for Qwen3-VL-8B-class models
- cost
- license

| Rank | Benchmark | Why | Main weakness |
|---|---|---|---|
| 1 | **TreeBench** | 10 labelled categories, a box for every target, dense SA-1B scenes, 405 items (cheap), Apache-2.0. Qwen2.5-VL-7B 37.0, Qwen3-VL-8B 48.3, o3 54.8 | No difficulty label (use #targets or box area as a proxy); ~2K px only |
| 2 | **MME-RealWorld-Lite** (→ full for final runs) | 43 tasks over 5 domains with a Perception/Reasoning split; monitoring person/vehicle counting, attributes, orientation, intention; driving pedestrian-motion and intention; all answers public; Apache-2.0 | No boxes; RS and driving domains are far from paintings; MGPO trained on it |
| 3 | **VisualProbe Easy/Medium/Hard** | Built-in 3-level difficulty, 4K, small targets among distractors, open answers. Qwen3-VL-8B Hard 31.6 | No category field (needs manual tagging); no boxes; Mini-o3 trained on its train split; no license tag |
| 4 | **ZoomBench** | A box and crop for every item (best for "did the agent look at the right place"), MCQ plus numeric counting, Apache-2.0, very low base scores (Qwen3-VL-8B 37.9) | The 6 dimension labels are not in the released fields |
| 5 | **FineRS-4k test** | True 4K UAV, 1,132 images; box and mask for every item; object-size tiers (natural difficulty); MC + open + referring | Narrow categories (color/position/shape/other); "cc" license with no variant; aerial viewpoint |
| 6 | **HR-Bench 4K/8K** | Standard, cheap, 4K/8K, FSP vs FCP | Only 2 categories; license "other" with an empty LICENSE file; scores ~75–80 for agents |
| 7 | **UR-Bench** | Only public-label painting set (scrolls), 3 levels (Micro/Regional/Global), street view and satellite, Apache-2.0, 528 Q | Scroll questions are in Chinese; no published Qwen/agent numbers verified here |
| 8 | **LRS-VQA** | 8 categories, 7,333 items, `hbox`/`rbox` evidence, target-size bins (difficulty proxy) | Remote sensing; CC BY-NC; text-only shortcut rate 16–20% |
| 9 | **HRScene val+testmini** | 25 domains incl. PANDA crowds, Art, monitoring; 1,750 public | Only ~70 public items per domain; CC BY-NC-SA; test hidden (VisualOverload's problem again) |
| 10 | **O3-Bench** | Information-dense maps and charts, layout boxes, 6-way MC, multi-hop | Not natural scenes |
| 11 | **COREVQA** | Real crowds (CrowdHuman), large | Binary only, no categories, 72.1% majority baseline |
| 12 | **V\*** | Standard and cheap | Saturated (86–93) |
| — | VisualNeedle, CapProbe | Very dense and relevant | Not publicly downloadable (as of this check) |

- **Suggested evaluation suite.**
  - Main: TreeBench (405) + MME-RealWorld-Lite + VisualProbe (515) + ZoomBench (845). That is about 3.8k questions with full per-question GT, categories, a difficulty split and boxes for two of the four sets.
  - Optional domain-transfer slice: UR-Bench Humanistic Scenes (180) as the art/painting proxy, and HRScene PANDA/Monitoring/Art val+testmini as crowd/gigapixel probes.

### Gaps
- I could not find VisualNeedle or CapProbe data on HF or GitHub.
- HR-Bench's distinct question and image counts are not stated on the card; ~200 per version is inferred from the 4-way rotation.
- The DenseWorld_Benchmark purpose is unknown.
- I did not verify a dataset (not code) license or images for the eScholarship gigapixel benchmark.

---

## Q2. Category taxonomies, per-category counts, difficulty splits and answer formats

### Takeaway
Exact per-category counts are verifiable from the released files for TreeBench, MME-RealWorld, FineRS-4k, LRS-VQA, UR-Bench, V\* and XLRS-lite. Explicit difficulty splits exist only in:
- VisualProbe (Easy/Medium/Hard)
- UR-Bench (Micro/Regional/Global)
- FineRS-4k and LRS-VQA (object-size tiers or bins)

ZoomBench and VisualProbe do not ship a category field.

### Cited Findings

**TreeBench** [V, datasets-server statistics on `category`]:
- Counts:

  | Group | Category | Questions |
  |---|---|---|
  | Perception | Attributes | 29 |
  | Perception | Material | 13 |
  | Perception | Physical State | 23 |
  | Perception | Object Retrieval | 16 |
  | Perception | OCR | 68 |
  | Reasoning | Perspective Transform | 85 |
  | Reasoning | Ordering | 57 |
  | Reasoning | Contact and Occlusion | 41 |
  | Reasoning | Spatial Containment | 29 |
  | Reasoning | Comparison | 44 |

- Format: MC with 2–11 options (columns A–K). Answers are mostly A–D, with E/F for 7 items [V].
- Targets per question: 1 target in 41.5% (168) and 2 targets in 44.9% (182) [V] — [arXiv](https://arxiv.org/html/2507.07999).
- Difficulty: questions were removed if all of Qwen2.5-VL-72B, InternVL3-78B, GPT-4o and Gemini-2.5-Flash answered correctly. There is no released difficulty label [V] — [arXiv](https://arxiv.org/html/2507.07999).

**MME-RealWorld (English JSON, 23,609 items)** [V, counted]. Fields: `Question_id, Question Type ("Multiple Choice"), Image, Text, Answer choices, Ground truth, Task (Perception/Reasoning), Subtask, Category, Dataset`.
- **Autonomous Driving**
  - Perception: Objects_Identify 1,101, Object_Count 720, Attribute_Motion_MultiVehicles 823, Attribute_Motion_MultiPedestrians 493, Attribute_Motion_Pedestrian 164, Attribute_Motion_Vehicle 158, Attribute_Visual_TrafficSignal 201.
  - Reasoning: Prediction_Intention_Ego 304, Pedestrian 103, Vehicle 207; Relation_Interaction ×4 (101–201); Attention_TrafficSignal 217.
- **Monitoring**
  - Perception: person/counting 964 (+28), vehicle/counting 353 (+255), person colour 89, person orientation 19, vehicle colour 197, vehicle orientation 155, vehicle location 136.
  - Reasoning: calculate 300, intention 98, property 100.
- **OCR with Complex Context**
  - Perception: adver_and_product 1,558, book_map_poster 1,555, text_recog 1,198, license 852, phone_and_address 577.
  - Reasoning: Character Identification 250, Scene Understanding 250.
- **Remote Sensing** (Perception): color 1,255, count 1,226, position 1,257.
- **Diagram and Table**
  - Perception: table 4,018, diagram 1,415.
  - Reasoning: table 326, diagram 174.

**VisualProbe** [V]: fields `images, doc_id, problem, solution, data_source`. Answers are short free-form, 1–16 characters (Hard). The difficulty is the split itself. There is no category field.

**ZoomBench** [V]:
- `question_type`: mcq 621 and blank 224. The blank items have numeric answers (counts 1–18).
- Fields: `id, query, response, bbox, question_type, image, crop_image`.
- The six dimensions are named in the paper, but no per-item dimension field exists, and the per-dimension counts appear only in a figure [V] — [arXiv](https://arxiv.org/html/2602.11858).

**HR-Bench** [V]:
- Fields: `index, question, answer, category (single|cross), A–D, cycle_category, image`.
- FSP ("single") covers attribute recognition, OCR and visual prompting. FCP ("cross") covers maps, charts and spatial relations. Each is described as 100 samples [V] — [HF card](https://huggingface.co/datasets/DreamMr/HR-Bench).
- Third-party copies are named `hrbench_4k_single_400` and `hrbench_4k_cross_400`, which suggests 400/400 rows per version [V, HF search].

**FineRS-4k test** [V, counted]:
- Q-type: referring (segmentation from a text phrase) 939; reasoning (open answer) 744; option (MC) 744.
- `attribute`: colors 1,280, position 597, shape 261, others 289.
- Object size (`area_percent`): XXS <0.017% 944, XS 0.017–0.055% 964, S ≥0.055% 519. These tiers match the paper's definitions [N] — [arXiv](https://arxiv.org/html/2510.21311v1).

**UR-Bench** [V, counted]:
- Level: Micro 162, Regional 198, Global 168.
- Category: Humanistic Scenes 180 (Portrait Scrolls 64, Narrative Scrolls 116) and Natural Scenes 348 (Street-view 199, Satellite 149).
- Format: 4-option MC (A–D).
- The first (scroll) item is in Chinese ("请直接回答选项的字母").

**LRS-VQA** [V, statistics]:
- Categories: rural or urban 1,252, count 1,199, object status 1,000, reasoning 1,000, object category 987, object shape 885, object color 765, object background 245.
- Answers are short open-ended phrases ("Answer the question using a single word or phrase") [V].
- Difficulty proxies: `size_bin` gives the target size (32×32: 820, 64×64: 1,354, 128×128: 1,196, 256×256: 928, 512×512: 926, 1024×1024: 598, >1024: 312), and `abs_size` gives the image size (small 88, medium 2,079, large 2,118, giant 1,849).

**XLRS-Bench-lite** [V]: Object spatial relationship 220, Anomaly Detection and Interpretation 100, Environmental condition reasoning 100. MC A–D.

**RSHR-Bench** (paper Table 7) [V]:
- Tasks: COL 443, SHP 243, DET 228, OC 436, REL 444, OGD 430, RG 219, OCN 412, RCN 245, MRJC 140, OSJ 409, AD 246, FP 429.
- Formats: MC and open-ended.
- Filtering: text-only adversarial filtering with Qwen3-8B and Llama3-8B [V] — [arXiv](https://arxiv.org/html/2512.17319v1).
- HF annotation files include color, shape, detection, object_counting, regional_counting, relation, object_grounding, regional_grounding, anomaly_detection and multi-turn variants [V].

**V\*** [V]: direct_attributes 115 and relative_position 76. MC with 4 or 2 options.

**VisualNeedle** [V]:
- Categories: OCR 71, Color 73, Entity 64, Spatial Relationship 58, Occluded Object 34.
- Format: open-ended, scored by normalized exact match, then a Gemini 3.1 Pro judge.
- Difficulty: no levels. 297/300 targets cover ≤1% of the image — [arXiv](https://arxiv.org/html/2605.26380).

**VQArt-Bench** [V]:
- Dimensions: Instance Identity 2,031, Attribute 2,598, Location 2,100, Counting 1,710, Spatial Relation 2,067, Interaction 1,794, Visual-Inspired Reasoning 2,163.
- Format: 4-way MC. No difficulty split stated — [arXiv](https://arxiv.org/html/2510.12750v1).

**O3-Bench** [V]: subsets map and chart. 6-option MC with "No Right Choice". Each item has `explanation`, `target_layouts` and `layout_infos` (boxes).

**HRScene real-world** [V]: MC, answer format `<ans>ANSWER</ans>`, exact match. The leaderboard groups results as Art, Daily, Medical, Paper, Remote, Research, Sub-Img and Urban — [arXiv](https://arxiv.org/html/2504.18406v2); [HRScene card](https://huggingface.co/datasets/Wenliang04/HRScene).

**COREVQA** [V]:
- True/False statements (~30 words), with no category field.
- The paper gives only lexical shares: spatial terms 57.7%, clothing 39.0%, color 35.1%.
- Error categories reported for hard cases: action recognition 81.3%, detail oversight 78.1%, counting 60.8% — [arXiv](https://arxiv.org/html/2507.13405v1).

### Inferences

**Mapping to VisualOverload's six categories** (activity / attributes / counting / OCR / reasoning / scene):
- **Attributes**: TreeBench Attributes/Material/Physical State, ZoomBench color/structure/material, MME-RW monitoring colour/orientation, FineRS colors/shape, LRS-VQA color/shape/status.
- **Counting**: MME-RW Monitoring person/vehicle counting (~1,600) and AD Object_Count, ZoomBench blank items, LRS-VQA count, RSHR OCN/RCN.
- **OCR**: TreeBench OCR (68), MME-RW OCR (~6k), ZoomBench OCR, VisualNeedle OCR.
- **Reasoning/spatial**: TreeBench reasoning (256), HR-Bench FCP, V\* relative_position, UR-Bench Regional/Global.
- **Activity**: weakly covered. The best sources are MME-RW Autonomous Driving pedestrian motion/intention and Monitoring intention, plus COREVQA statements, which by its own error analysis are action-heavy.
- **Scene**: LRS-VQA rural/urban, XLRS environmental condition, UR-Bench Global level.

**Difficulty for benchmarks without levels.** Tiers can be derived as VisualOverload derived its own, from the share of a model panel answering correctly (VisualOverload: hard 0–20%, medium 20–90%, easy 90–100% [N] — [arXiv](https://arxiv.org/html/2509.25339)). The project's three systems plus 2–3 extra open models could serve as the panel. Alternatively, use the target box area relative to the image, which TreeBench, ZoomBench, FineRS and LRS-VQA all allow.

### Gaps
- ZoomBench per-dimension counts and per-item dimension labels: they appear only in a figure, not in the data.
- VisualProbe category composition: unknown and would need manual tagging.
- The MME-RealWorld JSON count (23,609 EN) differs slightly from the paper's 29,429 total minus 5,917 CN = 23,512. This is unexplained.
- MME-RealWorld-Lite exact size: 50 per task would give 1,997 if a "task" is each of the 41 Subtask×Category groups counted above. Not verified against the Lite zip.

---

## Q3. Are ground-truth answers public for every question, and are there evidence annotations (boxes/regions)?

### Takeaway
Every benchmark in the top tiers publishes all answers. The exceptions are HRScene test, which is hidden like VisualOverload, VisualNeedle and CapProbe, which are not yet downloadable, and RealXBench, of which only 194/300 items are public. Evidence annotations that allow a "did the agent look at the right place" analysis exist in:
- TreeBench (`target_instances` boxes)
- ZoomBench (`bbox` + `crop_image`)
- FineRS-4k (bbox + polygon mask + area_percent)
- LRS-VQA (`hbox`/`rbox`, except count items)
- O3-Bench (layout boxes + step-wise explanations)
- RSHR-Bench (annotator boxes)
- Pix2Fact (`bounding_box`)
- VisualNeedle (`bbox`, if released)

### Cited Findings
- **TreeBench**: `answer` plus `target_instances`, a list of [x1,y1,x2,y2] boxes, e.g. `[[86.0, 750.0, 919.0, 1249.0], [460.0, 1023.0, 499.0, 1070.0]]`. It is present for all 405 items (no NaN) [V, statistics and first-rows]. The paper scores box mIoU for grounded models, e.g. DeepEyes 30.0, Pixel-Reasoner 35.7, TreeVGR 44.0 [V] — [arXiv](https://arxiv.org/html/2507.07999).
- **ZoomBench**: `response` (GT) and a `bbox` for all 845 items, plus the evidence `crop_image` [V].
- **MME-RealWorld**: `Ground truth` is present for all 23,609 EN items. There are no box fields [V, counted].
- **VisualProbe**: `solution` is present for all items. There are no boxes [V].
- **HR-Bench**: `answer` is present. There are no boxes in the released fields. The card says relevant objects were "manually annotated" for cropping 8K→4K [V] — [HF card](https://huggingface.co/datasets/DreamMr/HR-Bench).
- **FineRS-4k test**: every annotation has `bbox`, `points` (polygon), `area`, `area_percent` and `position`. Answers `A` are present for option and reasoning items. Referring items have an empty `A` and a mask target [V, counted].
- **LRS-VQA**: `ground_truth` is present for all items. `hbox` and `rbox` are null for exactly 1,199 items, which equals the count category, so every non-count item has a box [V, statistics].
- **UR-Bench**: `label` is present for all 528 items. There are no box fields [V].
- **COREVQA**: TRUE/FALSE for all 5,608 items. There are no boxes [V].
- **O3-Bench**: `answer`, `explanation`, `target_layouts` and `layout_infos` with bboxes [V, first-rows].
- **HRScene**: the test labels "will not be publicly released"; a sample test row shows `answer: 'None'`. Val and testmini are for development [V] — [HF README](https://huggingface.co/datasets/Wenliang04/HRScene).
- **VisualNeedle**: the JSONL format includes a GT `bbox`, but the data is only in supplementary material [V] — [arXiv](https://arxiv.org/html/2605.26380).
- **RealXBench**: "only a subset of 194 samples is released" out of 300 [V] — [HF card](https://huggingface.co/datasets/glowol/RealXBench).
- **RSHR-Bench**: "Annotators drew bounding boxes around target objects" [V] — [arXiv](https://arxiv.org/html/2512.17319v1).
- **V\***: `label` is public for all 191 items [V]. The V\* paper's original JSON also has target-object boxes (as I recall) [U].

### Inferences
- For the tool-use analysis ("did the crop or detection cover the evidence?"), compute IoU or coverage between each crop or detection the agent made and the GT box.
  - **Usable on**: TreeBench, ZoomBench, FineRS-4k and LRS-VQA without extra work.
  - **No box**: on MME-RW, VisualProbe, HR-Bench and UR-Bench only the outcome can be analysed (tool called or not, and how often).
- The boxes also give a principled **difficulty axis** (target area share). That matches VisualOverload's intuition that small and crowded targets are hard.

### Gaps
- I did not confirm whether HRScene val/testmini parquet rows carry non-empty answers. pyarrow was not available to read the remote files, and datasets-server first-rows failed on the huge images. The card's framing (val = "fine-grained validation", testmini = "rapid model development") implies they do.
- The V\* bench box annotations in the GitHub JSON were not re-checked.

---

## Q4. Licenses

### Takeaway
Permissive (Apache-2.0 tag): TreeBench, ZoomBench, MME-RealWorld (full), UR-Bench, RSHR-Bench, TIR-Bench, CV-Bench, BLINK, NaturalBench and RealXBench. Non-commercial: LRS-VQA (CC BY-NC 4.0), XLRS-Bench (CC BY-NC-SA 4.0 plus image terms), HRScene (CC BY-NC-SA 4.0 per its LICENSE file [N]) and Ref-L4 (CC BY-NC). No license tag: VisualProbe, V\*, MMStar, COREVQA, O3-Bench, VisualToolBench, Pix2Fact and MME-RealWorld-Lite. HR-Bench is "other" with an empty LICENSE file. FineRS-4k is just "cc". All of them are fine for a course project; record the terms in the report.

### Cited Findings
- HF license tags read via the Hub API this session [V]:
  - **apache-2.0**: HaochenWang/TreeBench, inclusionAI/ZoomBench, yifanzhang114/MME-RealWorld, UR-Bench/UR-Bench, RL-MIND/RSHR-Bench, Agents-X/TIR-Bench, nyu-visionx/CV-Bench, BLINK-Benchmark/BLINK, BaiqiL/NaturalBench, glowol/RealXBench.
  - **cc-by-nc-4.0**: ll-13/LRS-VQA, JierunChen/Ref-L4.
  - **cc-by-nc-sa-4.0**: initiacms/XLRS-Bench-lite and the other XLRS repos.
  - **cc-by-4.0**: MM-MVR/HoloCount (counting pointer).
  - **other**: DreamMr/HR-Bench.
  - **cc**: Jiazuo98/Finers-4k-benchmark.
  - **none**: Mini-o3/VisualProbe_\*, craigwu/vstar_bench, Lin-Chen/MMStar, COREVQA2025/COREVQA, m-Just/O3-Bench, ScaleAI/VisualToolBench, pix2fact/Pix2FactBenchmark, yifanzhang114/MME-RealWorld-Lite, Wenliang04/HRScene (no tag, but a LICENSE file exists).
- HRScene's LICENSE file is CC BY-NC-SA 4.0 [N] — [HF LICENSE](https://huggingface.co/datasets/Wenliang04/HRScene).
- The HR-Bench LICENSE file is 0 bytes [N] — [HF](https://huggingface.co/datasets/DreamMr/HR-Bench/blob/main/LICENSE).
- XLRS images inherit DOTA/Google Earth academic-only terms and ISPRS no-redistribution terms [N] — [HF card](https://huggingface.co/datasets/initiacms/XLRS-Bench-lite).
- The Mini-o3 GitHub repo is Apache-2.0, which covers the code [N] — [GitHub](https://github.com/Mini-o3/Mini-o3). The V\* repo is MIT [N] — [GitHub](https://github.com/penghao-wu/vstar).
- COREVQA images are CrowdHuman, which has custom non-commercial terms [N]. The arXiv page's CC BY 4.0 applies to the paper, not the data [V] — [arXiv](https://arxiv.org/html/2507.13405v1).
- UR-Bench images include Google Street View panoramas [N] — [arXiv](https://arxiv.org/html/2601.08748v1). Google's terms may restrict redistribution even though the repo is tagged Apache-2.0 (inference).

### Inferences
- For a course project with no redistribution, all are usable. Do not re-host the images of LRS-VQA, XLRS, HRScene, COREVQA or UR-Bench street-view; link to the original sources instead.

### Gaps
- Licenses not verified: VisualProbe, V\* data (separate from code), O3-Bench, COREVQA, MMStar, VQArt-Bench and VisualNeedle.

---

## Q5. Published results for base open VLMs and tool-using / "thinking with images" agents; headroom and saturation

### Takeaway
- **V\* is saturated.** Qwen3-VL-8B scores 85.9–86.4, the agents 82–89, and ZwZ-4B 92.7.
- **HR-Bench and MME-RealWorld are mid-range**, at 70–84 and 58–70.
- **Large headroom** remains on:
  - TreeBench: Qwen3-VL-8B 48.3, best reported 54.8
  - VisualProbe-Hard: Qwen3-VL-8B 31.6, Mini-o3 48.0
  - ZoomBench: Qwen3-VL-8B 37.9, best 59.3
  - O3-Bench: Qwen3-VL-8B 43.6
  - VisualNeedle: tool-less models ≤18%
  - RSHR and LRS-VQA
- **Tool agents trained on Qwen2.5-VL-7B do not beat plain Qwen3-VL-8B** on V\*, HR-Bench or MME-RW. They can gain on VisualProbe, but on TreeBench gains are small or absent (DeepEyes 37.5 vs base 37.0).
- **Absolute numbers for the same model differ by up to 5 points between papers**, so the project must rerun its own baselines.

### Cited Findings

**Cross-benchmark tables (accuracy %)**

- **InSight-o3 Table** [V] — [arXiv 2512.18745](https://arxiv.org/html/2512.18745):

  | Model | V\* | HR-4K | TreeBench | VProbe-Hard | MME-RW-Lite | O3-Bench |
  |---|---|---|---|---|---|---|
  | Qwen2.5-VL-7B | 75.5 | 68.2 | 37.0 | 23.9 | 46.7 | 27.4 |
  | Qwen3-VL-8B | 86.4 | 78.9 | 48.3 | 31.6 | 53.0 | 43.6 |
  | Qwen3-VL-32B | 86.0 | 81.1 | 48.2 | 28.6 | 54.2 | 60.4 |
  | InternVL3.5-8B | 64.0 | 64.5 | 40.5 | 11.0 | 48.0 | 24.3 |
  | DeepEyes | 83.3 | 73.2 | 37.5 | 35.1 | – | 27.0 |
  | Mini-o3 | 88.2 | 77.5 | – | 48.0 | – | 29.1 |
  | o3 | 76.4 | 74.3 | 52.3 | 23.6 | 55.2 | 40.8 |
  | GPT-5-mini | 73.8 | 72.0 | 54.6 | 26.4 | 56.1 | 39.0 |
  | GPT-5-mini + InSight-o3 | 86.9 | 86.7 | 54.1 | 41.2 | 59.0 | 61.5 |
  | Gemini-2.5-Flash | 80.1 | 83.5 | 49.9 | 39.6 | 56.5 | 60.4 |

- **ZwZ Table 2** [V] — [arXiv 2602.11858](https://arxiv.org/html/2602.11858):

  | Model | ZoomBench | HR-4K | HR-8K | V\* | CV-Bench | MME-RW-en | MME-RW-cn |
  |---|---|---|---|---|---|---|---|
  | Qwen2.5-VL-7B | 42.49 | 71.62 | 67.88 | 78.53 | 75.34 | 60.80 | 58.30 |
  | Qwen3-VL-4B | 40.24 | 78.25 | 72.88 | 80.10 | 84.95 | 63.47 | 63.63 |
  | Qwen3-VL-8B | 37.87 | 78.88 | 74.63 | 86.39 | 85.44 | 65.96 | 66.67 |
  | Qwen3-VL-235B-A22B | 49.11 | 84.50 | 81.62 | 87.96 | 86.72 | 67.07 | 65.29 |
  | GPT-5.1 | 47.22 | 67.00 | 65.25 | 70.16 | 84.22 | 64.04 | 55.57 |
  | Gemini-3-Flash | 59.29 | 87.88 | 85.00 | 86.39 | 89.57 | 74.86 | 72.62 |
  | ZwZ-8B | 58.11 | 84.38 | 82.00 | 91.10 | 87.40 | 69.87 | 70.59 |
  | ZwZ-4B | 55.74 | 81.75 | 79.50 | 92.67 | 87.90 | 68.52 | 68.09 |

  ZwZ Table 4 adds the agents (V\* / HR-4K / HR-8K / MME-RW-en): DeepEyes 85.6 / 75.1 / 72.6 / 64.1, Thyme 82.2 / 77.0 / 72.0 / 64.8, Mini-o3 88.2 / 77.5 / 73.3 / 65.5.

- **CounterCredit Table 1** ("When Should a VLM Look?", arXiv 2609.22910) [V] — [arXiv](https://arxiv.org/html/2609.22910):

  | Model | V\* | HR-4K | HR-8K | MME-RW | MMStar | BLINK |
  |---|---|---|---|---|---|---|
  | Qwen2.5-VL-7B | 77.0 | 69.1 | 65.6 | 58.0 | 64.4 | 56.2 |
  | Qwen3-VL-8B | 85.9 | 79.2 | 74.3 | 60.7 | 70.5 | 68.9 |
  | InternVL3-8B | 70.2 | 70.4 | 69.1 | 61.7 | 68.2 | 55.6 |
  | DeepEyes | 84.8 | 73.4 | 70.9 | 64.3 | 65.1 | 57.4 |
  | Pixel-Reasoner | 83.8 | 70.6 | 67.2 | 64.5 | 64.8 | 56.9 |
  | Mini-o3 | 87.4 | 78.0 | 73.4 | 65.2 | 64.1 | 55.6 |
  | Thyme | 83.8 | 76.8 | 71.3 | 64.7 | 65.2 | 56.4 |
  | CodeV-RL | 84.8 | 76.3 | 71.5 | 64.6 | 67.2 | 58.3 |
  | DeepEyesV2 | 83.2 | 77.6 | 73.2 | 64.9 | 65.4 | 57.8 |
  | PyVision-RL | 86.9 | 77.3 | 73.8 | 62.4 | 65.7 | 57.9 |
  | GPT-5 | 72.3 | 75.6 | 73.8 | 68.9 | 73.4 | 69.7 |
  | Gemini-2.5-Pro | 79.1 | 84.2 | 81.1 | 58.7 | 73.4 | 73.6 |

  Note that Qwen3-VL-8B's MME-RW is 60.7 here versus 65.96 in ZwZ [V, both]: a harness or subset difference.

- **Mini-o3 Table 1** (Avg@K, 7B; VisualProbe Hard / Medium / Easy) [N] — [arXiv](https://arxiv.org/html/2509.07969):
  - Qwen2.5-VL-7B: 23.9 / 26.0 / 39.1
  - DeepEyes: 35.1 / 29.8 / 60.1
  - Pixel Reasoner: 28.8 / 29.6 / 58.4
  - Mini-o3: 48.0 / 50.4 / 67.0

  "Reliable Thinking with Images" (arXiv 2602.12916) reports a Qwen3-VL thinking base at 47.6 / 25.8 / 28.3 and DeepEyes at 60.1 / 29.8 / 35.1 (Easy / Medium / Hard order inferred from the matching DeepEyes numbers) [V, search snippet only] — [arXiv](https://arxiv.org/pdf/2602.12916).

- **PyVision-RL Table 1** (V\* / HR-4K / HR-8K) [N] — [arXiv](https://arxiv.org/html/2602.20739):
  - QVL-7B: 78.5 / 71.6 / 67.9
  - PyVision-Image: 88.7 / 78.1 / 74.3
  - CodeV: 84.8 / 76.1 / 71.3
  - CodeVision: 83.7 / 75.6 / 72.2
  - CodeDance: 84.8 / 75.2 / 72.3

**TreeBench**

- Full Table 2 [V] — [arXiv](https://arxiv.org/html/2507.07999). Overall (mIoU) for each model:
  - Qwen2.5-VL-7B 37.0, Qwen2.5-VL-72B 42.2
  - InternVL3-8B 38.8, InternVL3-78B 46.4
  - DeepEyes-7B 37.5 (mIoU 30.0), Pixel-Reasoner-7B 39.0 (35.7), TreeVGR-7B 50.4 (44.0)
  - GPT-4o 46.9, Gemini-2.5-Flash 45.9, Gemini-2.5-Pro 54.1, o3 54.8
- Per-category, Qwen2.5-VL-7B:

  | Attr | Material | Phys | ObjRet | OCR | PerspT | Order | Contact | Contain | Compare |
  |---|---|---|---|---|---|---|---|---|---|
  | 55.2 | 53.8 | 56.5 | 62.5 | 27.9 | 20.0 | 35.1 | 39.0 | 44.8 | 43.2 |

  I recovered the column order from the HF card's category list and checked it arithmetically: the weighted sum over the verified category counts reproduces 37.0.
- Perspective Transform is the hardest category for every model: 11.8–22.4, including o3 at 22.4 [V].

**Other benchmarks**

- **VisualNeedle** (w/ tools vs w/o tools) [V] — [arXiv](https://arxiv.org/html/2605.26380):
  - Gemini 3.1 Pro 56.00 vs 16.33
  - GPT-5.4 31.22 vs 14.33
  - Qwen3.5-Plus 22.11 vs 15.00
  - Qwen3-VL-235B-A22B-Thinking 11.44 vs 8.33
  - Humans: majority vote@3 63.0
- **RSHR-Bench** [V] — [arXiv](https://arxiv.org/html/2512.17319v1):
  - Qwen2.5-VL-7B MC perception average 25.2, reasoning average 21.8.
  - GPT-4o: perception 30.2, reasoning 50.1.
  - All models score below 50 on open-ended perception.
- **HRScene** testmini real-world [V] — [arXiv](https://arxiv.org/html/2504.18406v2):
  - Qwen2-VL-72B 62.03, Qwen2-VL-7B 55.39, GPT-4o 51.24
  - Human category average 64.72
  - No Qwen2.5-VL or agent results were found.
- **COREVQA** [V] — [arXiv](https://arxiv.org/html/2507.13405v1):
  - GPT-4.1 77.57, GPT-4o-mini 76.60, Qwen2.5-VL-72B 69.95, LLaVA-NeXT 39.98
  - The all-FALSE baseline is 72.1% (computed from 4,042/5,608).
- **VQArt-Bench**: Gemini 2.5 0.71, Kimi-VL 0.67, GPT-4o 0.64. Qwen and InternVL were not evaluated [V] — [arXiv](https://arxiv.org/html/2510.12750v1).

**Agent gains on their own benchmarks** [N] — [agentic_tool_use notes](https://arxiv.org/html/2505.14362):
- DeepEyes over its Qwen2.5-VL-7B base: V\* 71.2→90.1, HR4K 68.8→75.1, MME-RW-Lite 42.3→53.2.
- Thyme: MME-RW 58.3→64.8, but OCRBench 88.4→86.3 and BLINK 56.4→56.1 — [arXiv](https://arxiv.org/html/2508.11630).

**Tool-use versus correctness** (CounterCredit) [V] — [arXiv](https://arxiv.org/html/2609.22910):
- Tool calls per question (V\* / HR-4K / HR-8K):
  - DeepEyes 0.91 / 0.98 / 0.96
  - Mini-o3 2.50 / 2.46 / 2.60
  - Thyme 0.14 / 0.19 / 0.22
  - DeepEyesV2 1.23 / 1.27 / 1.32
  - PyVision-RL 2.87 / 3.30 / 3.60
- "Spurious" call rates (calls neither needed nor used):
  - DeepEyes 87.4 / 83.6 / 82.1%
  - Mini-o3 40.7 / 50.5 / 51.0%
  - PyVision-RL 54.6 / 56.4 / 63.9%
- Only 10–12% of visual calls were both needed and used at the cold-start checkpoint.
- Real calls changed the decoded answer about 16× as often as wasted calls.

### Inferences
- **Saturation.**
  - Drop V\* as a headline metric; keep it as a regression check.
  - HR-Bench and MME-RW are not saturated, but the base-vs-agent gap there is a few points and within cross-paper noise.
  - **TreeBench, VisualProbe (Medium/Hard), ZoomBench and O3-Bench give the largest headroom and the clearest spread between systems.**
- **Expected pattern for the project's three systems.** If the base model is Qwen3-VL-8B or Qwen3.5-class, fixed-tool ReAct and Python agents built on it should show gains mainly on VisualProbe, ZoomBench and VisualNeedle-style tiny-target items. They should show little gain or a loss on reasoning categories (TreeBench Perspective Transform/Ordering) and on low-resolution BLINK/MMStar. This is consistent with the tables above but is a hypothesis.
- The **spurious-call findings** suggest a metric for the project's tool-use analysis: the per-question rate at which (a) a tool was called and (b) its crop overlapped the GT box, cross-tabulated with correctness. This can be done on TreeBench, ZoomBench, FineRS and LRS-VQA.

### Gaps
- I found no published Qwen2.5-VL-32B, InternVL3.5 or Qwen3.5 (open 7–32B) results on TreeBench, ZoomBench, VisualProbe, UR-Bench, FineRS-4k test or LRS-VQA. Qwen3-VL-32B is covered only in the InSight-o3 table.
- PyVision, Thyme and Chain-of-Focus have no published TreeBench or VisualProbe numbers that I found. ZoomEye numbers on these newer sets were not found either.
- FineRS-4k test and UR-Bench result tables were not fetched this session.
- The VisualProbe column order for the "Reliable Thinking with Images" numbers is inferred.

---

## Q6. Evaluation cost: how many images and questions, and is a subset enough?

### Takeaway
The recommended core (TreeBench 405 + VisualProbe 515 + ZoomBench 845 + MME-RW-Lite ≈2k) is about 3.8k questions. With multi-turn agents that make 1–4 tool calls per question, that is affordable on one GPU. Expensive options:
- HR-Bench 8K (8K images; 800 rows/version because of 4× option rotation)
- full MME-RealWorld (23.6k EN)
- HRScene/UR-Bench gigapixel images (hundreds of MB to GB per parquet shard)
- RSHR-Bench (about 210 GiB of parquet [N])

### Cited Findings
- Sizes [V]:
  - TreeBench 405
  - VisualProbe 141/268/106
  - ZoomBench 845
  - HR-Bench 800 rows per version; the HF dataset is 9.82 GB in total [V, card viewer]
  - MME-RW 23,609 EN; Lite is 50 per task
  - UR-Bench 528 Q on ~98 images
  - FineRS test 2,427 annotations / 1,132 images (744 MC + 744 open)
  - LRS-VQA 7,333
  - COREVQA 5,608
  - O3-Bench 345
- HRScene PANDA alone is 21 test shards of about 150–220 MB each, plus val and testmini shards [V, HF tree].
- RSHR-Bench is 1,932 test rows [V]; 210 GiB of parquet [N] — [HF](https://huggingface.co/datasets/RL-MIND/RSHR-Bench).
- Agent cost signal: per-question latency of 2–23 s and 0.14–3.6 calls per question for the 7B agents [V] — [arXiv 2609.22910](https://arxiv.org/html/2609.22910).

### Inferences
- **Subsets.**
  - HR-Bench: evaluate only `cycle_category == 0` (~200 Q per version) for agent runs, or run full CircularEval for the base model only.
  - MME-RW: use Lite, optionally restricted to Monitoring + Autonomous Driving + OCR-in-the-wild (the dense, natural-scene parts).
  - FineRS: use the 1,488 MC and open items and skip the referring-segmentation items unless the segmentation tool is being evaluated.
  - LRS-VQA: stratify about 1k by category × size_bin.
- **Statistical power.** With 105–270 items per VisualProbe level and 13–85 per TreeBench category, per-category differences under about 8–10 points are not significant. Report bootstrap CIs, or pool categories into VisualOverload-like super-categories.

### Gaps
- No measured token or GPU-hour costs for these sets with Qwen3-VL at native resolution.

---

## Q7. Contamination: do agentic models' training sets overlap these benchmarks?

### Takeaway
- **Confirmed overlap or in-distribution training:**
  - MGPO trained on MME-RealWorld.
  - DeepEyesV2's RL repo ships the full V\* bench (`vstar_test.parquet`, 191 rows).
  - Mini-o3 trained on VisualProbe-train, so VisualProbe is in-distribution for it.
  - TreeVGR was co-designed with TreeBench.
  - FineRS models train on FineRS train.
- **Likely image-source overlap:** ZwZ training uses SA-1B, which is also the image source of V\* and TreeBench.
- HRScene re-packages V\* and HR-Bench as subsets.

### Cited Findings
- `honglyhly/DeepEyesV2_RL` contains `vstar_test.parquet` (113 MB). datasets-server exposes it as a `test` split with **191 rows**, the size of V\* Bench, next to the perception, reason and search RL parquets [V, HF tree + /size].
- MGPO post-trained Qwen2.5-VL-7B on 21K MME-RealWorld samples. A third-party repo `xinyu1205/mme_realworld_train_21690` exists [N] — [arXiv 2507.05920](https://arxiv.org/pdf/2507.05920).
- Mini-o3 RL uses the VisualProbe train split plus DeepEyes-47k [N] — [arXiv](https://arxiv.org/html/2509.07969).
- TreeVGR-RL-37K's parquet is named `vstar30k_visdrone6k_x1y1x2y2.parquet`, i.e. V\*-train (COCO) plus VisDrone [N] — [HF](https://huggingface.co/datasets/HaochenWang/TreeVGR-RL-37K). TreeBench uses SA-1B [V] — [arXiv](https://arxiv.org/html/2507.07999).
- ZwZ-RL-VQA sources include SA-1B, LAION, MetaCLIP, Visual Genome, CC12M and STPLS3D [N] — [HF](https://huggingface.co/datasets/inclusionAI/ZwZ-RL-VQA). V\* images come from SA-1B [N] — [arXiv](https://arxiv.org/html/2312.14135v2).
- DeepEyes, TreeVGR and Q-Zoom draw on V\*-train (COCO) [N], which is a different image pool from V\*-bench (SA-1B).
- HRScene includes `realworld_VStar_Bench` and `realworld_HR-Bench` configs [V].
- FineRS-4k train/val/test (8,956 / 749 / 2,427) are released together [V].

### Inferences
- If the project evaluates released agent checkpoints:
  - Exclude MME-RW for MGPO-based models.
  - Flag V\* for DeepEyesV2 (likely used for validation or checkpoint selection; whether it was used for training is unverified).
  - Treat Mini-o3's VisualProbe results as in-distribution.
- If the project builds its own agents on an untouched base model with no fine-tuning, contamination only matters through the base model's pretraining. Qwen3-VL's data is undisclosed.
- TreeBench, ZoomBench, UR-Bench, LRS-VQA and O3-Bench have no known training-set overlap with the agents named above. An SA-1B image-hash check against ZwZ data is advisable if ZwZ is evaluated.

### Gaps
- I did not verify whether `vstar_test.parquet` was used for training or only for validation in DeepEyesV2.
- I did not run any image-hash overlap check between SA-1B-based training pools (ZwZ, GranD, DCI) and TreeBench or V\*.
- Qwen3-VL and Qwen3.5 pretraining corpora are undisclosed, so benchmark leakage into base models cannot be ruled out.
