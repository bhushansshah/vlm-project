# Skill-specific evaluation benchmarks (with public per-question answers) for tool-using VLM agents on dense / cluttered scenes

Notes as of 2026-10-15, for the course project "Studying the Performance of VLM Agents on VQA Tasks for Dense Datasets". The project compares three systems:
1. A base VLM.
2. A ReAct agent with fixed tools: crop/zoom, a detector (YOLO / Grounding-DINO), a segmenter (SAM/SAM2), a pose model (MediaPipe) and OCR.
3. A Python-interpreter VLM agent.

Scope is evaluation only. Training-set facts are in `../Dense scene VQA training datasets/` (dense_counting.md, scene_text_ocr.md, dense_scene_reasoning.md) and are not repeated except where needed for evaluation use. Multi-category high-resolution benchmarks (V*, HR-Bench, MME-RealWorld, TreeBench, HRScene, VisualOverload, etc.) are out of scope.

Verification conventions:
- "HF API" means the license, files and schema were read this session from `huggingface.co/api/datasets/<id>` or the raw README.
- "Code license" means a GitHub or HF licence that may cover only code, not images.
- **UNVERIFIED** marks anything not confirmed from a primary page.

---

## (a) Dense counting in crowds and clutter: which benchmarks test detector/segmenter counting against VLM counting?

### Takeaway
Best first picks: **CountQA** (1,528 QA, CC BY 4.0, real cluttered photos, counts up to 400) and **HoloCount** (2,480 QA, CC BY 4.0, 20 diagnostic subsets with image sizes). Both are on HF with answers. Their published scores span Qwen2.5-VL, Qwen3-VL, Qwen3.5 and InternVL3.5.

Strong complements:
- **UNICBench image track**: 5,508 QA. Answers include instance points, and counts go up to 10,294.
- **CAPTURe**: occluded patterns. MIT licence. Has a CountGD baseline and an oracle-coordinates condition.
- **PairTally**: two categories per image. Has CountGD and Qwen2.5-VL baselines.
- **FSC-147 test, REC-8K and the LVLM-Count benchmarks**: these give detector and counting-model reference points (CountGD, GroundingREC, DAVE).

CountBench/CountBenchQA and PixMo-Count test are near ceiling (counts 2–10; Qwen2.5/3-VL score 80–95%), so they work as sanity checks, not tests of tools. The single highest-headroom slice is HoloCount's High-Density subset: Qwen2.5-VL and Qwen3-VL score 0–3%, Qwen3.5-27B scores 20%. However, those images are synthetic symbols.

### Cited Findings

**Summary table**

| Benchmark (venue/year) | Size | Density / resolution | Answer format | Answers / boxes public? | License | Key open-VLM scores | Tool / specialist baselines |
|---|---|---|---|---|---|---|---|
| CountQA (arXiv 2508.06585, 2025; Google Research) | 1,001 images, 1,528 QA | GT counts 0–400, mean 12.1, median 9; `is_focused` flag marks cluttered images | Freeform integer; EM, RA@5%, RA@10% | Yes: single `test` split with `answers`, `objects`, `categories`; no boxes | CC BY 4.0 (HF API) | Best of 15 MLLMs is Gemini 2.5 Pro at 42.93 EM. No Qwen2.5-VL or InternVL row (only Qwen-VL-Max at 30.45 EM). Gemma3-27B 16.10, LLaVA-7B 8.37 | None; CountGD/SAM mentioned only as future work |
| HoloCount (arXiv 2607.06420, 2026; Meituan) | 2,480 QA, 20 subsets, 1,481 concepts | Counts 0–149, mean 8.06, median 3. Dense subset is synthetic symbols on a blank canvas. Per-sample `image_height`/`image_width` fields | Freeform integer; exact match (plus MAE, RA for dense) | Yes: `question`, `answer`, `split` on HF | CC BY 4.0 (HF API) | Avg: Qwen2.5-VL-7B 52.9, -72B 61.6; Qwen3-VL-8B 58.1, -32B 62.7; InternVL3.5-8B 52.6, -38B 60.6; Qwen3.5-9B 71.4, -27B 75.9, -397B 76.9. High-Density: Qwen2.5-VL-7B 2.0, Qwen3-VL-8B 0.0, InternVL3.5-8B 4.0, Qwen3.5-27B 20.0, Gemini-3.1-Pro 49.0 | None; no human score |
| UNICBench image track (CVPR 2026; arXiv 2603.00595) | 5,300 images, 5,508 QA, 49 categories | Counts 0–10,294, mean 63.5. Tiers: Easy 1–10, Medium 11–100, Hard >100. Resolution 234×180 to 6736×4640, mean 804×660. Sources: FSC147, NWPU-MOC, CARPK, JHU-Crowd++, UCF-QNRF, ShanghaiTech, IOCfish5K, GWHD, Snapshot Serengeti, cells | Number only; success, MAE, MSE, hit-rate@100/90/80% | GT stores `gt_count` and `[x,y]` instance points; **public release location and licence not stated** in paper text read | "Adhere to original licenses"; **UNVERIFIED** | Qwen2.5-VL-7B: overall MAE 41.8, Hard MAE 222.3, H100 16.6. Qwen2.5-VL-72B: MAE 274.8 (outliers). InternVL3.5-38B: MAE 75.4. Qwen3-VL-30B-A3B-Instruct: MAE 45.7. Only three models exceed a 50% hit rate at 20% tolerance | No specialist counter numbers in the image table |
| CAPTURe (arXiv 2504.15485, 2025; UNC) | Real: 924 images / 92 types (HF card says 962). Synthetic: 1,250 | Patterned objects, black-box occluder. Real images share FSC-147 file names | Integer; sMAPE (lower is better) | Yes: `real_metadata.json` / `synthetic_metadata.json` on HF | MIT (HF API) | Occluded real sMAPE: Qwen2-VL 29.33, InternVL2 32.90, GPT-4o 14.75. Unoccluded real: Qwen2-VL 18.96. No Qwen2.5-VL | **CountGD**: 3.15% unoccluded → 10.34% occluded on the 149-image FSC-147 subset, beating all VLMs. VLM+CountGD hybrid is worse than CountGD alone. Oracle object coordinates cut Qwen2-VL from 29.33 to 9.62. Humans 3.79% |
| PairTally (arXiv 2509.13939, 2025) | 681 images, 2 categories each (54 categories / 98 subcategories) | High-resolution smartphone photos (pixel sizes not stated) | `<count>N</count>`; MAE, RMSE, NAE | Per-type counts and 3 exemplar boxes per image; data at github.com/bbvisual/PairTally_Benchmark | Dataset licence not stated (**UNVERIFIED**) | Qwen2.5-VL MAE 59.36 (training notes, citing CountGD++ supplement) | CountGD++ MAE 35.27, CountGD 46.67 (same source) |
| CountBench / CountBenchQA (ICCV 2023; PaliGemma) | 540 images originally; 491 surviving in CountBenchQA | Counts 2–10, LAION | Integer | Yes, test only | No licence on HF card (HF API: `license: None`) | Qwen2.5-VL-72B **93.6** (with a "detect then count" prompt). Qwen3-VL instruct: 2B 88.4, 4B 84.9, 8B 80.5, 32B 94.9, 235B 93.0. Qwen3-VL-8B thinking 91.5. Molmo table: Qwen2-VL-7B 76.5, -72B 80.4, GPT-4o 87.9, Molmo-72B 91.2 | — |
| PixMo-Count val/test (Molmo, 2024) | 540 val + 540 test | Counts 2–10, web images | Integer | Yes; train has points, val/test counts only | ODC-BY (HF API) | Molmo Table 1 "PixMo-Count" column: Qwen2-VL-7B 48.0, -72B 55.7, InternVL2-8B 43.9, GPT-4o 59.6, Molmo-7B-D 84.8, Molmo-72B 85.2 | — |
| FSC-147 test | 1,190 test images | 7–3,731 objects/image, mean ~56. HF mirror images ~384 px tall | Class-agnostic counting (exemplars and/or text) | Points + 3 exemplar boxes public | No dataset licence stated (training notes) | (via LVLM-Count) Qwen2-VL-72B-AWQ MAE 34.18; LVLM-Count(Qwen2-VL) 22.29; GPT-4o 25.57 | CountGD MAE 14.76; GroundingREC 10.12; DAVE-prm 14.90; TFOC 24.79 |
| LVLM-Count benchmarks (TMLR 2026; arXiv 2412.00686) | Emoji-Count 415 QA (82 classes, 1024×1024, 30–50 per category, up to 6 categories); PASCAL-VOC 102 Q; Penguin 100 images (0–213); TallyQA-Complex 149-Q sample | Dense, many look-alike classes | Integer; MAE/RMSE | Emoji-Count release/licence not confirmed (**UNVERIFIED**) | — | Emoji-Count MAE: Qwen2-VL-72B 78.05 → 24.43 with LVLM-Count; GPT-4o 23.57 → 16.57 | Emoji-Count: CountGD 137.93, DAVE 198.99 (detectors fail on fine-grained emoji classes). LVLM-Count itself is a GroundingDINO + SAM + tiling tool pipeline |
| REC-8K (CVPR 2024) | 8,011 images, 17,122 referring expressions | Crowd, traffic, FSC-147 and drone sources | Count for a referring expression ("people wearing red") | Points public as annotations only; images fetched from 9 sources | Repo MIT; dataset licence not stated | No VLM numbers found | GroundingDINO-based REC baselines in paper (numbers not retrieved) |
| TallyQA Test-Complex (AAAI 2019) | 15,598 Q / 14,051 VG images | Counts 0–15 | Integer | Yes (Apache-2.0 repo) | Apache-2.0 | PaliGemma 72.3 (as cited by Youtu-VL); Omni-SMoLA 77.1. A 2026 paper reports "Qwen2.5-VL" (an 8B-labelled checkpoint) TallyQA 0.800 (split not stated) | — |
| CrowdVLM-R1 test (2025) | Small held-out sets (sheep, GTAV heads, pedestrians, wheat, cars) + manatee OOD | Train counts 1–3,546 | Integer; MAE | HF `yeyimilk/CrowdVLM-R1-data` is a single zip, no licence (HF API) | None stated | Qwen2.5-VL-3B/7B SFT MAE 29.89 / 27.52; FGRPR 25.16 / 24.07 (training notes) | — |

**Per-benchmark details (with sources)**

**CountQA**
- HF `Jayant-Sravan/CountQA` has one `test` split: 1,001 rows (one per image) holding 1,528 QA. Fields are `questions`, `answers`, `objects`, `categories`, `is_focused` and `full_config`. Licence is CC BY 4.0. — [HF card](https://huggingface.co/datasets/Jayant-Sravan/CountQA)
- GT counts are 0–400 (mean 12.1, median 9, sd 16.6). Metrics are EM, RA@5% and RA@10%. Table 2 has 15 models: Gemini 2.5 Pro 42.93 EM, o4-mini 34.42, o3 34.16, Qwen-VL-Max 30.45, Gemma3-27B 16.10, LLaVA-7B 8.37. No tool or detector baselines; CountGD, LVLM-Count and SAM appear only in related and future work. — [CountQA arXiv HTML](https://arxiv.org/html/2508.06585)

**HoloCount**
- HF `MM-MVR/HoloCount`: 2,480 QA, CC BY 4.0. Fields are `question`, `answer`, `split`, `image_height` and `image_width`. Subset sizes:

  | Subset | QA |
  |---|---|
  | High-Density Enumeration | 100 |
  | Small-Scale Enumeration | 101 |
  | Partial Object Occlusion | 103 |
  | Aerial Perspective Shift | 109 |
  | Coordinate-Prompt Region Grounding | 84 |
  | Visual-Prompt Region Grounding | 102 |
  | Action & State Attribution | 102 |
  | Null-Target Prompting | 250 |
  | Chromatic Attribution | 202 |
  | Others | — |

  — [HF card](https://huggingface.co/datasets/MM-MVR/HoloCount)
- **Image sources by subset:**
  - Atomic, region and set subsets: LVIS val.
  - Small-scale and occlusion: Objects365 val.
  - Aerial: CA44.
  - Dense: synthetic symbols on a blank canvas.
  - Linguistic-prior subset: Gemini-generated images.

  Overall counts are 0–149 (mean 8.06, median 3). The metric is exact match. No CountGD, GroundingDINO, tool or human baselines. — [HoloCount arXiv HTML](https://arxiv.org/html/2607.06420v1)
- **Per-subset scores (accuracy, %):**

  | Model | Dense | Small | Occlusion | Coordinate-region | Aerial |
  |---|---|---|---|---|---|
  | Qwen2.5-VL-7B | 2.0 | 29.7 | 36.9 | 25.6 | 18.3 |
  | Qwen3-VL-8B | 0.0 | 32.7 | 43.7 | 66.7 | 22.0 |
  | InternVL3.5-8B | 4.0 | 30.7 | 32.0 | 44.9 | 13.8 |
  | Qwen3.5-9B | 3.0 | 50.5 | — | — | 77.1 |
  | Gemini-3.1-Pro | 49.0 | — | — | — | — |
  | GPT-5.5 | 20.0 | — | — | — | — |

  — [HoloCount arXiv HTML](https://arxiv.org/html/2607.06420v1)
- A third-party review flags that the dense subset is synthetic and may not transfer to real-world density. — [Pith review](https://pith.science/paper/2607.06420)

**UNICBench**
- CVPR 2026. 5,300 images / 5,508 QA (image track), plus text and audio tracks. 45 MLLMs evaluated. — [CVF page](https://openaccess.thecvf.com/content/CVPR2026/html/Rong_UNICBench_UNIfied_Counting_Benchmark_for_MLLM_CVPR_2026_paper.html); [arXiv 2603.00595](https://arxiv.org/abs/2603.00595)
- Image sources, tiers, resolution range, GT format (`gt_count` and `[x,y]` points) and Table 2 numbers are given in the summary table above. The paper says data "adhere to original licenses"; no release URL was found in the text read. — [arXiv HTML](https://arxiv.org/html/2603.00595v1)

**CAPTURe**
- HF `atinp/CAPTURe` is MIT. It holds `real_dataset.zip`, `synthetic_dataset.zip` and metadata JSONs. Real images use FSC-147 file names, so unoccluded originals can be fetched. Prompt: "Count the exact number of [object]… Assume the pattern continues behind any black box." — [HF card](https://huggingface.co/datasets/atinp/CAPTURe)
- Scores, the CountGD baseline, oracle-coordinate results and human scores are given in the summary table above. — [CAPTURe arXiv HTML](https://arxiv.org/html/2504.15485)

**PairTally**
- 681 smartphone images, each with two categories. Exemplar boxes, text labels and per-type counts. VLMs answer in `<count>N</count>`. Metrics MAE, RMSE, NAE. Code and data at bbvisual/PairTally_Benchmark. The page shows only the arXiv paper licence. — [PairTally arXiv HTML](https://arxiv.org/html/2509.13939)

**Qwen technical-report counting numbers**
- **Qwen2.5-VL (Table 7):** CountBench 93.6 for Qwen2.5-VL-72B with a "detect then count" prompt, versus Molmo-72B 91.2, GPT-4o 87.9 and InternVL2.5-78B 72.1. — [Qwen2.5-VL tech report](https://arxiv.org/pdf/2502.13923)
- **Qwen3-VL (Tables 2–4):** CountBench as listed in the summary table. — [Qwen3-VL tech report](https://arxiv.org/pdf/2511.21631)

**Molmo Table 1 (counting columns)**
- Columns are CountBenchQA / PixMo-Count:
  - GPT-4o-0513: 87.9 / 59.6
  - Qwen2-VL-7B: 76.5 / 48.0
  - Qwen2-VL-72B: 80.4 / 55.7
  - InternVL2-8B: 57.8 / 43.9
  - Molmo-72B: 91.2 / 85.2

  Molmo calls PixMo-Count "more challenging natural images than CountBenchQA". — [Molmo arXiv 2409.17146](https://arxiv.org/pdf/2409.17146)

**LVLM-Count**
- Pipeline:
  1. The LVLM extracts the target expression.
  2. GroundingDINO (low threshold) finds the area and boxes.
  3. SAM masks the objects.
  4. MeanShift and A\* cut the image into tiles without splitting objects.
  5. The LVLM counts each tile, and the tile counts are summed.

  Benchmarks and MAE numbers are in the summary table. — [LVLM-Count arXiv HTML](https://arxiv.org/html/2412.00686v4)
- TMLR 2026. — [mlanthology](https://mlanthology.org/tmlr/2026/qharabagh2026tmlr-lvlmcount/)

**Other**
- **COUNTS** (arXiv 2504.10158) is an OOD **detection and grounding** benchmark (222K samples, 1.2M boxes, 14 shifts; GPT-4o grounding 56.7%), not counting VQA. Paper licence CC BY-NC-ND. — [arXiv abs](https://arxiv.org/abs/2504.10158)
- **TallyQA-Complex:** PaliGemma 72.3%, Youtu-VL 74.4%, Omni-SMoLA 77.1%. — [Youtu-VL arXiv 2601.19798](https://arxiv.org/pdf/2601.19798)
- **"Qwen2.5-VL" from a 2026 spatial-encoder study:** TallyQA 0.800, CountBenchQA 0.891, VSR 89.1, CV-Bench-2D 0.754, MMVP 0.770 (model listed as "Qwen2.5-VL-8B", likely the 7B; splits not stated). — [Spatial Blindspot arXiv 2601.09954](https://arxiv.org/pdf/2601.09954)
- **Other 2026 counting sets seen but not examined:** SITUATE (spatially constrained counting, SCITEPRESS 2026) and the CVPR-W 2026 synthetic "Can VLMs Count?" benchmark. — [search result: SCITEPRESS](https://www.scitepress.org/publishedPapers/2026/144283/pdf/index.html); [CVPR-W 2026](https://openaccess.thecvf.com/content/CVPR2026W/COGVL/html/Sengupta_Can_Vision-Language_Models_Count_A_Synthetic_Benchmark_and_Analysis_of_CVPRW_2026_paper.html)

### Inferences
- **Where tools should help:**
  - **Dense, many-instance counting of one class** (FSC-147 test, UNICBench Medium/Hard, HoloCount Dense/Small/Aerial, CountQA non-focused). Detection or segmentation counting plus tiling should beat VLM-only counting by large margins: CountGD beats Qwen2-VL by about 2× MAE on FSC-147, and the VLM scores near 0% on HoloCount Dense.
  - **Fine-grained, look-alike or attribute-constrained counting** (Emoji-Count, PairTally intra-category, CountQA "red thumb tacks", HoloCount Chromatic/Material). Off-the-shelf detectors do *worse* than VLMs here (CountGD MAE 137.9 vs GPT-4o 23.6 on Emoji-Count). This is a good test of whether the Python agent learns to filter detector output with VLM judgement instead of trusting it blindly.
  - **Occlusion and amodal counting** (CAPTURe). Oracle coordinates help a lot, but CountGD+VLM hybrids underperformed CountGD alone. So naive tool fusion can hurt, which is a useful negative-result test.
- **Ceiling sets.** CountBench, CountBenchQA and PixMo-Count test are near ceiling for Qwen2.5/3-VL (80–95%). Use them as regression sanity checks, not headline results.
- **Cheap evaluation:**
  - CountQA (1,528 Q, ~620 MB), HoloCount (2,480 Q) and CAPTURe-real (~924 images) each run in well under an hour on one GPU for a 7B model, but agent runs cost 3–10× more calls.
  - A stratified 300–500-question subset fits a course budget: HoloCount Dense + Small + Occlusion + Coordinate-region + Aerial is about 500 Q, plus 200 CountQA `is_focused=false`.
  - UNICBench needs its release located first.
- **Metrics.** Report both exact match and a tolerance metric (RA@10% or MAE). Tools tend to cut large errors (MAE) more than they raise exact match at high counts.

### Gaps
- No Qwen2.5-VL / InternVL row on CountQA. The project would need to run them.
- UNICBench public download, licence and whether instance points ship: not found.
- PairTally per-image count range and dataset licence: not stated on the arXiv page.
- No published ReAct or Python-agent (tool-calling VLM) results on CountQA, HoloCount, UNICBench or PairTally were found. The only tool-pipeline evidence is LVLM-Count (GroundingDINO+SAM+tiling) and CAPTURe's CountGD hybrid.
- Whether the Molmo "PixMo-Count" column is the val or test split was not re-confirmed.
- REC-8K published baseline numbers were not retrieved this session.

---

## (b) Scene text / OCR in high-resolution natural scenes (OCR tool and zoom)

### Takeaway
All five of these have public answers and published Qwen2.5-VL/Qwen3-VL numbers, so tool gains can be compared:
- **TextVQA val**: 5,000 Q with answers and the original Rosetta OCR tokens. That gives a free "OCR-tool" condition.
- **OCRBench v2**: 10,000 Q, MIT, answers plus answer bounding boxes.
- **CC-OCR multi-scene track**: scene-text subsets IC15, TotalText, InverseText, HierText, zh_scene, zh_dense.
- **MTVQA test**: 6,778 QA, 9 languages, CC BY-NC.
- **SEED-Bench-2-Plus**: MC; charts, maps and webs.

**ST-VQA test answers are not public**; the lmms-lab copy has no answer column, so evaluation goes through RRC. **HierText test GT is public** (CC BY-SA), but it is a spotting benchmark, not VQA. None of these is truly "dense high-resolution natural scene" except HierText (more than 100 words per image, long side up to 1,600 px). Headroom is moderate: Qwen2.5-VL-7B TextVQA 84.9, OCRBench v2-en 56.3, MTVQA 29.2.

### Cited Findings

| Benchmark | Size | Images / text density | Format & metric | Answers / boxes public | License | Open-VLM scores | OCR / tool baselines |
|---|---|---|---|---|---|---|---|
| TextVQA val (CVPR 2019) | 5,000 Q (train 34,602 / val 5,000 / test 5,734) | OpenImages photos, moderate resolution | Freeform; VQA accuracy over 10 human answers | Val `answers` public; test via server. Rows include `ocr_tokens` and `image_classes` | CC BY 4.0 per TFDS/HF loader (training notes) | Qwen2.5-VL 3B / 7B / 72B: 79.3 / 84.9 / 83.5. Molmo table: Qwen2-VL-7B 84.3, -72B 85.5, GPT-4o 77.4 | The provided `ocr_tokens` act as an off-the-shelf OCR tool output |
| ST-VQA (ICCV 2019) | Test 4,070 Q (lmms-lab copy) | ICDAR, COCO-Text, VizWiz etc. | Freeform; ANLS | **Test answers not in lmms-lab copy** (no `answers` field), so RRC submission is needed | Licence not found (training notes) | — | — |
| OCRBench v2 (2025) | 10,000 Q, 30+ task types (EN + ZH) | Mixed scenes, documents, charts | Mixed (VQA, spotting, grounding, parsing) | Yes: `answers` list, plus `bbox`, `bbox_list`, `content` fields | MIT (HF API) | Qwen2.5-VL-7B en/zh 56.3/57.2; -72B 61.5/63.7. Qwen3-VL-8B Instruct en 65.4 / zh 61.2; -32B Instruct 67.4/59.2; -235B Instruct 67.1/61.8 | DeepEyesV2 (Qwen2.5-VL-7B + Python + search) OCRBench (v1) 864 → 882 |
| CC-OCR (ICCV 2025) | 7,058 images, 39 subsets, 4 tracks | Multi-scene track: IC15_500, TotalText_300, InverseText_500, Hieragent_100, zh_scene_450, ugc_laion_400, zh_dense_50, zh_vertical_100 (plus doc subsets) | Full-text recognition (F1 / edit-distance style) | Yes (TSV with GT per subset) | HF tag MIT; card says "source code is licensed under MIT", so the data licence is ambiguous (**UNVERIFIED**) | Qwen2.5-VL 3B / 7B / 72B overall 74.5 / 77.8 / 79.8. Qwen3-VL-8B Instruct 79.9; -235B Instruct 82.2 | Leaderboard includes OCR specialists: GOT 61.00, Florence 49.24, KOSMOS2.5 47.55, TextMonkey 56.88 on multi-scene |
| MTVQA test (ACL 2025 Findings) | 2,116 images / 6,778 QA, 9 languages (no EN/ZH) | Scene and document text | Freeform; accuracy | Yes: `qa_pairs` with answers in test rows | CC BY-NC 4.0 (HF API) | Qwen2.5-VL 3B / 7B / 72B: 24.8 / 29.2 / 31.7; GPT-4o 27.8 | — |
| SEED-Bench-2-Plus | ~2.3K MC | Charts, maps, webs (not natural scenes) | 4-way MC | Yes | CC BY-NC 4.0 (HF API) | Qwen2.5-VL-7B 70.4, -72B 73.0 | DeepEyesV2 70.5 (vs base 70.4): tools give no gain |
| HierText test (CVPR 2022) | 1,634 test images | >100 words/image, long side ≤1,600 px | Word/line/paragraph spotting (not VQA) | Test GT released Dec 2024 | CC BY-SA 4.0 (training notes) | Not reported as VQA | Standard spotting baselines (not retrieved) |

Sources:
- **TextVQA:**
  - Splits and schema (`answers`, `ocr_tokens`): [HF lmms-lab-encoder/textvqa API](https://huggingface.co/api/datasets/lmms-lab-encoder/textvqa)
  - Qwen2.5-VL TextVQA, CC-OCR, OCRBench_v2, SEED-2-Plus and MTVQA scores: [Qwen2.5-VL tech report Tables 3 and 5](https://arxiv.org/pdf/2502.13923)
  - Molmo TextVQA column: [Molmo arXiv](https://arxiv.org/pdf/2409.17146)
- **ST-VQA test** schema without answers: [HF lmms-lab-encoder/ST-VQA API](https://huggingface.co/api/datasets/lmms-lab-encoder/ST-VQA)
- **OCRBench v2** schema (answers, bbox, bbox_list, content), 10,000 test, MIT: [HF ling99/OCRBench_v2](https://huggingface.co/datasets/ling99/OCRBench_v2)
- **Qwen3-VL** OCRBench_v2 and CC-OCR: [Qwen3-VL tech report Tables 2–4](https://arxiv.org/pdf/2511.21631)
- **CC-OCR:**
  - Subset file list, tracks, leaderboard and licence wording: [HF wulipc/CC-OCR](https://huggingface.co/datasets/wulipc/CC-OCR)
  - Paper: [arXiv 2412.02210](https://arxiv.org/abs/2412.02210)
- **MTVQA** test rows contain `qa_pairs` with answers (checked via the datasets-server first-rows API); CC BY-NC 4.0: [HF ByteDance/MTVQA](https://huggingface.co/datasets/ByteDance/MTVQA)
- **SEED-Bench-2-Plus** licence: [HF AILab-CVC/SEED-Bench-2-plus](https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2-plus)
- **DeepEyesV2:**
  - Tools: Python code (crop, region marking, computation), image search and text search.
  - OCRBench 864 → 882, SEED-2-Plus 70.4 → 70.5 over the Qwen2.5-VL-7B base.
  - Thyme (code agent) OCRBench 863.
  - "OCR tasks add region marking and numerical computation."

  — [DeepEyesV2 arXiv HTML](https://arxiv.org/html/2511.05271v2)
- **LensVLM** (2026): a VLM with OCR and zoom tools on compressed document images (MMLongBench-Doc). At 15× compression, zoom reaches 41.9% vs 31.1% for the base. OCR also beats the base but trails zoom "because extracted text discards layout". — [LensVLM arXiv](https://arxiv.org/html/2605.07019v1)
- **HAMMR** (2024): a ReAct-style multimodal agent with OCR and crop tools for generic VQA. — [arXiv 2404.05465](https://arxiv.org/pdf/2404.05465)

### Inferences
- **TextVQA val is the cheapest clean OCR-tool test.**
  - Three conditions are possible: (i) base VLM, (ii) base VLM plus the dataset's own `ocr_tokens` (a "free" OCR tool), (iii) agent calling PaddleOCR/zoom.
  - All 5,000 Q have public answers. A 1,000-Q random subset gives roughly ±1.5-pt 95% CI.
- **OCRBench v2 is the best fit for grounding-aware OCR.** Its answer bboxes let you check whether the agent zoomed or cropped to the right region, not only whether the answer was right. Filter to the scene-text task types (e.g., text recognition / VQA in scenes) for a 1–2K subset.
- **CC-OCR multi-scene** (IC15, TotalText, InverseText, zh_dense, ugc_laion) is a natural-scene full-transcription test. It is where a dedicated OCR engine (PaddleOCR) is most likely to *match or beat* the VLM.
- **Expect small or negative tool gains on TextVQA and SEED-2-Plus.** Qwen2.5-VL-7B already scores 84.9 on TextVQA, and DeepEyesV2 gained only +0.1 on SEED-2-Plus. Gains are more likely on small text in large images (CC-OCR zh_dense, OCRBench v2 fine-grained recognition, HierText-derived questions). Multi-category high-res sets (handled by other researchers) remain the strongest "zoom" tests.
- **HierText as VQA.** No VQA version exists. Template "read the text in box (x1,y1,x2,y2)" or "what is the word right of X" questions from the public test GT would give a dense-text, high-res benchmark with exact answers. This needs a little construction work and is not ready-made.

### Gaps
- No published numbers for a PaddleOCR-equipped ReAct agent on TextVQA, ST-VQA, CC-OCR or OCRBench v2 were found.
- OCRBench v2's "private" split label status, and which task types correspond to natural scenes, were not re-checked.
- CC-OCR data licence (vs code licence) is unclear.
- Image resolution statistics for TextVQA, OCRBench v2 and CC-OCR scene subsets were not retrieved.
- 2025–2026 search found no new natural-scene small-text VQA benchmark with public answers. EgoTextVQA is video; Res-Bench is about resolution robustness.

---

## (c) Human activity, pose and interaction in crowded scenes (pose tool)

### Takeaway
**No ready-made pose-QA benchmark on crowded still images was found.** The best available evaluation sets for "pose/activity among many people" are:
- **COREVQA**: 5,608 true/false statements on CrowdHuman images, answers public, but no licence. Action recognition is the top failure mode.
- **CrossHOI-Bench** (CVPR 2026): 3,773 MC questions converted from HICO-DET test, V-COCO and SWiG-HOI, with HOI-detector baselines and a "target person box given" condition.
- **HumanRef** (ICCV 2025): 6,000 multi-person referring expressions, 8.6 persons per image, including interaction and position subsets, with DINO-X and "all person boxes" baselines.
- **HoloCount Action & State Attribution**: 102 counting questions about people doing X.
- **Human-MME**: about 20K human-centric QA with HICO-DET images, but mostly single-person.

CrowdPose and OCHuman have public keypoints but no VQA. Pose questions would have to be templated from keypoints (inference).

### Cited Findings

| Benchmark | Size | Crowd density | Format | Answers / boxes public | License | Open-VLM scores | Tool / specialist baselines |
|---|---|---|---|---|---|---|---|
| COREVQA (arXiv 2507.13405, 2025) | 5,608 image–statement pairs (27.9% true / 72.1% false) | CrowdHuman train01/02 crowded images | True/false entailment, ~30-word statements; accuracy / P / R / F1 | Yes: `COREVQA.csv` (`image_id`, `question`, `answer`) plus CrowdHuman zips on HF | **None stated** (HF API `license: None`); CrowdHuman terms apply | Qwen2.5-VL-72B 69.95; GPT-4.1 77.57; GPT-4o-mini 76.60; LLaVA-NeXT 39.98. Always "false" would score 72.1 | None |
| CrossHOI-Bench (CVPR 2026; arXiv 2508.18753) | 3,773 MC Q. Main: 1,274 HICO-DET test images. V-COCO: 647 Q / 499 images. SWiG-HOI: 1,852 Q / 1,851 images | 31.2% multi-person different-HOI images | 4 options, multiple may be correct; Macro/Instance/Micro-F1, EM | Released at github.com/ChelsieLei/CrossHOI-Bench (contents not checked) | Dataset licence not read (**UNVERIFIED**) | Setting 1 (VLM finds the person itself), Macro-F1 / EM: Qwen2.5-VL-7B 29.73 / 14.29; -32B 50.71 / 26.06; InternVL3-38B 38.04 / 20.33; InternVL3-8B 6.58; Qwen3-VL-30B ~0 recall. Setting 2 (target person box given): Qwen2.5-VL-7B 48.93 / 25.98; -32B 62.90 / 35.01; InternVL3-8B 49.88 | HOI detectors (Setting 1) Macro-F1: ADA-CM 43.02, HOLa 43.61, CMD-SE 47.49 |
| HumanRef (ICCV 2025, IDEA) | Benchmark: 6,000 expressions / 5,732 images (1,000 each: attribute, position, interaction, reasoning, celebrity, rejection) | 8.6 persons/image; images >1000×1000 (avg 1432×1074); 2.2 target boxes per expression | Multi-instance REC (output all matching persons); DensityF1, rejection score | Yes: `annotations.jsonl` with boxes/masks on HF | **IDEA License 1.0** (custom; HF LICENSE file) | DensityF1@0.5:0.95. Qwen2.5-VL-7B: interaction 0.532, position 0.528, attribute 0.544, reasoning 0.403, rejection 71/1000. InternVL2.5-8B: 0.313 / 0.243 / 0.271 / 0.189. Supplement: Qwen2.5-VL avg DF1 56.2 zero-shot → 81.4 with candidate person boxes as hints | RexSeek-7B (detector + LLM): ~0.81–0.84 per domain. DINO-X (open-set detector): ~0.18–0.23. "Baseline" returning all GT persons: 0.16–0.30 |
| Human-MME (arXiv 2509.26165, 2025) | 19,945 QA; 16,765 images (8,755 HICO-DET + 8,010 Pexels/Pixabay) | Mostly single-person; has Multi-Person Reasoning and HOI dimensions | Choice, short answer, bbox, ranking, judgment | Promised release (github.com/Yuan-Hou/Human-MME); not checked | Not stated (**UNVERIFIED**) | Avg: Qwen2.5-VL-7B 63.0, -72B 72.8; InternVL3.5-38B 67.6. BBox sub-score: Qwen2.5-VL-7B 31.7 | — |
| HoloCount "Action & State Attribution" | 102 Q | LVIS-type images | Count people/objects in a state | Yes | CC BY 4.0 | Qwen2.5-VL-7B 66.7; Qwen3-VL-8B 78.4; InternVL3.5-8B 64.7; Qwen3.5-27B 87.3 | — |
| LLaVA-Pose E-HPAUB (arXiv 2506.21317) | Pose and action understanding benchmark (size not retrieved) | Keypoint-annotated humans (COCO-style) | GPT-judged open answers (**UNVERIFIED**) | **UNVERIFIED** | **UNVERIFIED** | LLaVA-1.5-7B +33.2% after pose-instruction tuning (self-reported) | — |
| CrowdPose / OCHuman | ~20K images / ~4.7K images | Crowd index; heavy occlusion | Keypoints only, **no QA** | Keypoints public | Data licence unverified (training notes) | — | Pose estimators only |

Sources:
- **COREVQA:**
  - CSV columns and sample rows (e.g., "only two have both arms visibly bent at the elbows…" → FALSE); no licence: [HF COREVQA2025/COREVQA](https://huggingface.co/datasets/COREVQA2025/COREVQA)
  - Results and failure modes (action recognition 81.3%, detail oversight 78.1%, counting 60.8%, spatial 41.7%, negation 31.3% of difficult cases); no train/test split; no human score: [arXiv HTML](https://arxiv.org/html/2507.13405v1)
- **CrossHOI-Bench:**
  - Construction, settings and all numbers: [arXiv HTML v4](https://arxiv.org/html/2508.18753v4)
  - CVPR 2026 acceptance: [arXiv abs](https://arxiv.org/abs/2508.18753)
- **HumanRef:**
  - Statistics, splits and LICENSE (IDEA License 1.0): [HF IDEA-Research/HumanRef](https://huggingface.co/datasets/IDEA-Research/HumanRef)
  - Per-domain scores: [HF comparison.md](https://huggingface.co/datasets/IDEA-Research/HumanRef/blob/main/evaluation_results/compare_different_models/comparison.md)
  - 56.2 vs 82.3 and the 81.4 box-hint result: [ICCV 2025 supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Jiang_Referring_to_Any_ICCV_2025_supplemental.pdf); [arXiv 2503.08507](https://arxiv.org/abs/2503.08507)
- **Human-MME:** [arXiv HTML](https://arxiv.org/html/2509.26165)
- **HoloCount Action subset:** [arXiv HTML](https://arxiv.org/html/2607.06420v1)
- **LLaVA-Pose:** [arXiv 2506.21317](https://arxiv.org/html/2506.21317v1)
- **MMT-Bench** includes a keypoint-localisation MC task (a person given by a box), but it is not crowd-specific. — [arXiv 2404.16006](https://arxiv.org/pdf/2404.16006)

### Inferences
- **Where a pose tool should plausibly help:**
  - COREVQA statements about body configuration ("arms bent at the elbows", "crossed arms", "jumping while the rest stand").
  - HumanRef interaction and position subsets ("the person raising a hand").
  - CrossHOI Setting 1 vs Setting 2. The gap (Qwen2.5-VL-7B Macro-F1 29.7 → 48.9 when given the person box) measures exactly what a detector tool should recover.
- **HumanRef is the clearest detector-plus-filtering test.** Two published results set the range:
  - Candidate person boxes as hints lift Qwen2.5-VL from 56.2 to 81.4 DF1. That is the "perfect detector" upper bound for a ReAct agent.
  - DINO-X alone scores about 0.2. Detection without VLM filtering fails.
- **COREVQA caveats.**
  - The label skew (72% false) means always answering "false" scores 72.1, above Qwen2.5-VL-72B's 69.95. Report balanced accuracy or F1, not just accuracy.
  - The CSV is small (5,608 rows), so a 1,000-row stratified subset is cheap.
  - It has no licence, so it is fine for a course project but should not be redistributed.
- **Building pose QA from CrowdPose / OCHuman.** No VQA exists. Template questions from keypoints, e.g. "How many people have a wrist above their head?" or "Is the person at the far left sitting?" (use hip/knee angles). This gives exact answers and a pose-tool-centric test. It is construction work for the project; nothing like it is published.

### Gaps
- No still-image benchmark of pose QA in crowds with public answers was found. Video pose/motion QA (HumanMoveVQA, HanDyVQA) is out of scope.
- CrossHOI-Bench and Human-MME data release status and licences were not verified on GitHub.
- SWiG and HICO-DET test sets "converted to QA" exist only through CrossHOI-Bench (SWiG-HOI 1,852 Q; HICO-DET 1,274 images) and Human-MME (HICO-DET images). No other conversion was found.
- No published tool-agent (pose-model) results on any of these.

---

## (d) Attributes and spatial relations among many objects, including referring expressions (detection plus filtering logic)

### Takeaway
Candidates with public labels:
- **HumanRef**: multi-person, multi-instance; also listed under (c).
- **RefDrone** (CC BY 4.0): 17,900 expressions on VisDrone, 0–242 targets per expression, 31% small objects.
- **Ref-L4** (CC BY-NC): 45K long expressions on COCO and Objects365, val/test with boxes.
- **RefCOCOg val/test.**
- **CV-Bench 2D** (Apache-2.0): 2,638 MC questions covering count and relation on COCO/ADE20K.
- **VSR** (CC BY 4.0) and **What'sUp** (MIT): relations.

RefCOCO-family scores are saturated: Qwen2.5-VL-7B RefCOCOg-test 87.2, and Grounding-DINO 87.0. **Multi-target and no-target referring in crowded or drone scenes** (HumanRef, RefDrone, gRefCOCO) is where detector-plus-Python filtering has a clear opening, because VLMs output too few boxes.

### Cited Findings

| Benchmark | Size | Density | Format / metric | Labels public | License | Scores (open VLM / specialist) |
|---|---|---|---|---|---|---|
| HumanRef | 6,000 expressions | 8.6 persons/image, 2.2 targets/expression | Multi-box; DensityF1 | Yes | IDEA License 1.0 | See (c) |
| RefDrone (arXiv 2502.00392, 2025) | 17,900 expressions, 8,536 images, 63,679 instances, 10 categories; keeps VisDrone train/val/test | 11,362 multi-target and 847 no-target expressions; 0–242 targets; 31% small objects | REC / generalized REC | Yes: MDETR-format train/val/test JSON + test TSV on HF | CC BY 4.0 (HF API) | NGDINO (Number-GroundingDINO) proposed. No MLLM numbers retrieved |
| Ref-L4 (2024) | 45,341 annotations, 365 categories; val + test parquet | COCO + Objects365; instance sizes 30–3,767 px; avg 24.2 words | Box; Acc@IoU0.5/0.75/0.9, mAcc | Yes (boxes); reviewed RefCOCO/+/g labels included (error rates 14% / 24% / 5%) | CC BY-NC 4.0 (HF API) | CogVLM-Grounding Acc@0.5 81.70, mAcc 66.09; Qwen-VL-Chat 73.80; GPT-4V 9.91. No Qwen2.5/3-VL row in the paper |
| RefCOCOg val/test (2016) | ~5K test expressions (HF mirrors) | COCO, ≥2 same-class objects | Box; Acc@0.5 | Yes | Unverified | Qwen2.5-VL 3B / 7B / 72B test: 85.7 / 87.2 / 90.3. **Grounding DINO 87.0**. InternVL2.5-78B 92.2 |
| CV-Bench 2D (Cambrian-1, 2024) | 2,638 total (2D + 3D) | COCO / ADE20K | 4-way MC (count, relation) | Yes: `answer` field | Apache-2.0 (HF API) | "Qwen2.5-VL" 2D overall 0.754 (COCO 0.820 / ADE20K 0.689) per a 2026 study |
| VSR random / zero-shot (TACL 2023) | >10K pairs; train/dev/test JSONL | COCO | True/false spatial relation | Yes | CC BY 4.0 (HF API) | "Qwen2.5-VL" VSR 89.1 (split unstated) |
| What'sUp (EMNLP 2023) | Subset A 408 + Subset B 412 controlled photos; plus COCO-spatial and GQA-spatial | Low clutter (1–2 objects) | 4-way caption MC | Yes (repo) | Repo MIT (training notes) | Not retrieved |
| HoloCount Coordinate-/Visual-Prompt Region + Orientation subsets | 84 + 102 + 78 Q | LVIS | Count within a region | Yes | CC BY 4.0 | Coordinate-region: Qwen2.5-VL-7B 25.6, -72B 14.1, Qwen3-VL-8B 66.7, InternVL3.5-8B 44.9 |

Sources:
- **RefDrone:**
  - Statistics and NGDINO: [arXiv 2502.00392](https://arxiv.org/html/2502.00392v3)
  - HF files and CC BY 4.0 licence: [HF sunzc-sunny/RefDrone](https://huggingface.co/datasets/sunzc-sunny/RefDrone)
- **Ref-L4:**
  - Card (size, fields, CC BY-NC 4.0, RefCOCO error rates): [HF JierunChen/Ref-L4](https://huggingface.co/datasets/JierunChen/Ref-L4)
  - Scores: [Ref-L4 arXiv 2406.16866](https://arxiv.org/abs/2406.16866); [GitHub README](https://github.com/JierunChen/Ref-L4)
- **RefCOCOg and Grounding-DINO scores:** [Qwen2.5-VL tech report Table 6](https://arxiv.org/pdf/2502.13923)
- **CV-Bench** fields and licence: [HF nyu-visionx/CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench)
- **CV-Bench / VSR / MMVP numbers:** [Spatial Blindspot arXiv 2601.09954](https://arxiv.org/pdf/2601.09954)
- **VSR** licence and splits: [HF cambridgeltl/vsr_random](https://huggingface.co/datasets/cambridgeltl/vsr_random); [HF vsr_zeroshot](https://huggingface.co/datasets/cambridgeltl/vsr_zeroshot)
- **What'sUp** subset sizes: [What'sUp arXiv 2310.19785](https://arxiv.org/pdf/2310.19785)
- **HumanRef** finding that VLMs "output an insufficient number of bounding boxes" because they are trained on one-to-one referring: [arXiv 2503.08507](https://arxiv.org/html/2503.08507v1)

### Inferences
- **Best "detector + Python filter" tests** are HumanRef (reasoning subset, where Qwen2.5-VL-7B scores DF1 0.40) and RefDrone (multi-target, small objects). The Python agent can:
  1. Detect all persons or vehicles.
  2. Crop each one and ask the VLM an attribute or interaction question.
  3. Apply positional logic (sort by x, "second from left").
  4. Return all matches or none.

  Expected gains are largest on multi-target and rejection cases.
- **Saturated, so low value as headline results:** RefCOCO, RefCOCOg and VSR. Grounding-DINO already equals Qwen2.5-VL-7B on RefCOCOg. Use them as "do tools hurt on easy cases?" checks.
- **Ref-L4 test** is cheap to subset by size bucket. Its small-object bucket (CogVLM 75.1 Acc@0.5 vs 86.4 for medium) is the obvious zoom-tool slice.
- **Low clutter.** What'sUp and VSR are not dense. Keep them out of the core "dense" claim.

### Gaps
- No Qwen2.5-VL / Qwen3-VL numbers on RefDrone or Ref-L4 were found; the project would need to run them.
- RefDrone NGDINO and baseline numbers were not retrieved.
- gRefCOCO (multi/no-target COCO) was not examined this session.
- The "Qwen2.5-VL" VSR / CV-Bench numbers come from a third-party study that lists an "8B" checkpoint and does not state splits. Treat them as indicative.

---

## (e) Logical consistency and paired questions in cluttered images

### Takeaway
**COREVQA** is the only paired/entailment-style set found on genuinely crowded images, but it is single statements, not pairs. Paired-consistency benchmarks with public answers exist:
- **NaturalBench**: 1,900 groups × 2 images × 2 questions; Apache-2.0; group accuracy.
- **VisMin**: minimal-change image/caption pairs differing in object, attribute, **count** or **spatial relation**, on COCO.
- **ConMe** and **Winoground**: compositional hard negatives.

All are on ordinary photos (Flickr30k, DOCCI, COCO), not dense scenes. For a dense-scene consistency test, the practical options are:
- Pairing HoloCount Null-Target vs Atomic questions.
- Building opposite-polarity pairs from COREVQA or HumanRef annotations (inference).

### Cited Findings
- **NaturalBench (NeurIPS 2024):**
  - HF holds 1,900 rows. Each has `Image_0`, `Image_1`, `Question_0`, `Question_1` and all four GT answers (Yes/No or MC), plus question type and language (EN/ZH/HI).
  - Images come from Flickr30k, DOCCI and XM3600; questions were generated by GPT-4 and human-filtered.
  - Apache-2.0.
  - Also ships `dataset_split_for_finetuning.zip`.

  — [HF BaiqiL/NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench)
- NaturalBench reports Acc, Q-Acc, I-Acc and G-Acc. Human G-Acc 92.1, random 6.3; older open models score G-Acc of about 2–17. — [NaturalBench arXiv](https://arxiv.org/pdf/2410.14669)
- **VisMin (NeurIPS 2024):**
  - Two images and two captions; pick the correct matching.
  - Pairs differ only in object, attribute, count or spatial relation.
  - Built on COCO with a 4-step human verification.
  - VLMs are notably weak on spatial relations and counting.

  — [VisMin arXiv](https://arxiv.org/abs/2407.16772); [NeurIPS poster](https://neurips.cc/virtual/2024/poster/94495)
- **ConMe (NeurIPS 2024):** VLM-generated hard compositional questions, manually verified. Performance drops up to 33% relative to earlier compositional benchmarks. — [ConMe arXiv](https://arxiv.org/abs/2406.08164)
- **Winoground:** HF gated (auto-approval) with a `license_agreement.txt`. Ships per-model score files (CLIP, FLAVA, LXMERT, human). Size tag n<1K. — [HF facebook/winoground](https://huggingface.co/datasets/facebook/winoground)
- **COREVQA:** crowded true/false statements, many with negation and quantifiers ("not a single person…", "only two…"). Negation causes 31.3% of difficult cases. — [arXiv HTML](https://arxiv.org/html/2507.13405v1); [HF CSV](https://huggingface.co/datasets/COREVQA2025/COREVQA)
- **HoloCount Null-Target Prompting** (250 Q, asking for counts of absent objects) and **Linguistic Prior Conflict** (163 Q):

  | Model | Null-Target | Linguistic Prior |
  |---|---|---|
  | Qwen2.5-VL-7B | 95.6 | 20.9 |
  | Qwen3-VL-8B | 96.4 | 37.4 |
  | Gemini-3.1-Pro | 55.2 | — |

  Strong models over-count absent objects. — [HoloCount arXiv HTML](https://arxiv.org/html/2607.06420v1)
- **HumanRef rejection subset** (1,000 expressions with no matching person): Qwen2.5-VL-7B correctly rejects 71 of 1,000, InternVL2.5-8B 549, RexSeek 541. — [HF comparison.md](https://huggingface.co/datasets/IDEA-Research/HumanRef/blob/main/evaluation_results/compare_different_models/comparison.md)

### Inferences
- **For logical consistency in clutter**, the most informative cheap setups are:
  1. **HumanRef rejection vs positive subsets.** Does the agent hallucinate a person when none matches? Qwen2.5-VL-7B rejects only 7.1%.
  2. **COREVQA**, reporting F1 and balanced accuracy.
  3. **A derived pair set.** For each COREVQA statement, write its negation (or for HoloCount, "Are there more than N…?" vs "fewer than N…?") and score pair-level consistency. The project would build this; it does not exist as a published set.
- **NaturalBench and VisMin** are good "general consistency" controls with public answers and permissive licence (NaturalBench). They are not dense, so detection tools should help little. That makes them a useful control for showing tool gains are specific to clutter.
- **Python-agent advantage.** Count- and spatial-change pairs (VisMin count/spatial categories) are exactly the questions an agent can answer by detecting and comparing coordinates. This is a cheap, targeted check.

### Gaps
- No NaturalBench, ConMe or VisMin numbers for Qwen2.5-VL / Qwen3-VL / InternVL3 were found this session.
- VisMin and ConMe HF repository names and licences were not found (guessed repo IDs returned errors). **UNVERIFIED.**
- No benchmark with paired or opposite-polarity questions on dense or crowded images (other than VisualOverload, out of scope) was found.

---

## Cross-cutting: published tool and agent baselines, headroom, and cheap-subset plan

### Takeaway
Published evidence that tools help on these skill-specific sets is thin, and none of it comes from a ReAct or Python agent on the dense-counting or crowd sets. The available evidence:

| Source | Finding |
|---|---|
| LVLM-Count (GroundingDINO + SAM + tiling) | Cuts Qwen2-VL-72B MAE on FSC-147 from 34.2 to 22.3 and on Emoji-Count from 78.1 to 24.4 |
| CAPTURe | CountGD beats all VLMs under occlusion; VLM+CountGD hybrids are worse than CountGD alone |
| HumanRef | Person-box hints raise Qwen2.5-VL from 56.2 to 81.4 DF1, but DINO-X alone scores about 0.2 |
| CrossHOI | Giving the person box raises Qwen2.5-VL-7B Macro-F1 from 29.7 to 48.9 |
| DeepEyesV2 (Python + search) | OCRBench +18 (864 → 882) and SEED-2-Plus +0.1 over Qwen2.5-VL-7B |

So the project's comparisons on CountQA, HoloCount, COREVQA, HumanRef, RefDrone and CC-OCR-scene would be largely **new measurements**.

### Cited Findings
- **LVLM-Count:** [arXiv HTML](https://arxiv.org/html/2412.00686v4)
- **CAPTURe:** [arXiv HTML](https://arxiv.org/html/2504.15485)
- **HumanRef:** [ICCV 2025 supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Jiang_Referring_to_Any_ICCV_2025_supplemental.pdf); [HF comparison](https://huggingface.co/datasets/IDEA-Research/HumanRef/blob/main/evaluation_results/compare_different_models/comparison.md)
- **CrossHOI:** [arXiv HTML](https://arxiv.org/html/2508.18753v4)
- **DeepEyesV2 and Thyme:** [arXiv HTML](https://arxiv.org/html/2511.05271v2)
- **PyVision** ("MLLMs generate and execute Python code on the fly") reports perception and maths benchmarks, not the skill sets here. — [PyVision arXiv](https://arxiv.org/pdf/2507.07998)
- **Qwen2.5-VL** reaches CountBench 93.6 with a "detect then count" prompt, i.e. self-grounding acts like an internal detection tool. — [Qwen2.5-VL tech report](https://arxiv.org/pdf/2502.13923)

### Inferences
**Headroom ranking** (Qwen2.5-VL-7B / Qwen3-VL-8B base scores, lower = more room):

| Benchmark slice | Base score(s) | Headroom |
|---|---|---|
| HoloCount Dense | 2.0 / 0.0 | Very high (but synthetic) |
| UNICBench Hard (MAE 222 for Qwen2.5-VL-7B) | — | Very high |
| HumanRef reasoning | DF1 0.40 (Qwen2.5-VL-7B) | High |
| HumanRef rejection | 71/1000 (Qwen2.5-VL-7B) | High |
| CrossHOI Setting 1 | Macro-F1 29.7 (Qwen2.5-VL-7B) | High |
| HoloCount Small / Occlusion / Aerial | 18–44 | High |
| CountQA | Best overall 43 EM | High |
| MTVQA | 29.2 (Qwen2.5-VL-7B) | High (multilingual OCR) |
| OCRBench v2-en | 56.3 / 65.4 | Medium |
| COREVQA | Qwen2.5-VL-72B 70 vs 72 majority | Medium |
| CC-OCR | ~78–80 | Low–medium |
| TextVQA | ~85 | Low |
| RefCOCOg | ~87 | Low |
| CountBench | 80–94 | Low |

**Suggested cheap evaluation suite** (about 3–4K questions; all answers public on HF except CrossHOI, which is on GitHub):

| Skill | Set | Questions | Cost / notes |
|---|---|---|---|
| (a) | CountQA full | 1,528 | Cheap |
| (a) | HoloCount Dense + Small + Occlusion + Aerial + Coordinate-region | ~500 | Cheap |
| (a) | CAPTURe-real | 924 | Optional |
| (b) | TextVQA val random | 1,000 | Base / base+`ocr_tokens` / agent |
| (b) | CC-OCR multi-scene scene_text subsets | ~1,850 images | Or subsample 500 |
| (b) | OCRBench v2 scene-text tasks | 500–1,000 | Subset |
| (c) | COREVQA stratified | 1,000 | Report F1 |
| (c) | CrossHOI-Bench main | 1,274 images | Setting 1 vs Setting 2 is a built-in "tool oracle" |
| (d) | HumanRef | 6,000 (or 300 per domain = 1,800) | — |
| (d) | RefDrone test | 500-expression subset | — |
| (e) | HumanRef rejection | 1,000 | Or a subset |
| (e) | NaturalBench | 300 groups | Non-dense control |

**Licence notes for a course project** (evaluation only, no redistribution):
- Fine: CC BY (CountQA, HoloCount, RefDrone, VSR), MIT (CAPTURe, OCRBench v2), Apache (NaturalBench, CV-Bench).
- Usable for evaluation but restrictive: CC BY-NC (MTVQA, Ref-L4, SEED-2-Plus) and the custom IDEA licence (HumanRef).
- No stated licence: COREVQA, CountBenchQA. Do not redistribute.

### Gaps
- No published ReAct-with-fixed-tools or Python-agent numbers exist for CountQA, HoloCount, UNICBench, COREVQA, HumanRef, RefDrone, CrossHOI-Bench, CC-OCR or MTVQA (searched; none found).
- No published MediaPipe/pose-tool agent results on any VQA benchmark were found.
- Per-question image resolution statistics were not found for CountQA, OCRBench v2, CC-OCR or COREVQA. HoloCount and HumanRef ship image sizes or averages.
