# Dense-scene reasoning training datasets for VisualOverload (attributes, activities/HOI, spatial relations, compositional reasoning, scene classification, paired yes/no)

Notes as of 2026-10-15. Target models: Qwen2.5-VL 3B/7B. Out of scope (covered elsewhere): counting-specific, OCR-specific, art-specific, high-resolution-detail-specific and tool-use datasets.

Verification tags used in this file:
- **[V]**: checked this session against a primary page (HF dataset API/card, GitHub repo/LICENSE file, arXiv page, or by reading dataset rows).
- **[P]**: the paper is the primary source and the link goes to it. The figure came from the abstract or paper as I remember it and was **not re-fetched this session**, so re-check it before publishing.
- **[U]**: license not verified.
- A GitHub "repo license" covers the **code**. It is not necessarily the license for the annotations or images. Where only a code license was found, these notes say so.

---

## Q0. What exactly does VisualOverload test, and which training signals map to it? (target profile used for all "match" judgements below)

### Takeaway
VisualOverload has 150 paintings from Google Arts & Culture, each at about 4K pixels, and 2,720 questions. **Scene yes/no pairs make up 1,388 of the 2,720 (51%)**, and a pair counts only if both answers are right. Reasoning has 356 questions, activity 150 and attributes 149. That makes logically paired binary scene questions and consistency training the biggest lever by item count. Activity, attribute and reasoning are where Qwen2.5-VL-3B/7B score lowest.

### Cited Findings
- 2,720 QA pairs. Ground truth is private and scored by an online evaluator. Best of 37 models is o3 at 69.5% overall and 19.6% on the hard split [V] — [arXiv 2509.25339](https://arxiv.org/abs/2509.25339)
- Images are public-domain paintings from museum collections, made available through **Google Arts & Culture**. There are **150 images**, downsampled to the pixel count of 4K (3840x2160) with aspect ratio preserved. 28 images were already below 4K but all are above Full HD [V] — [arXiv HTML 2509.25339](https://arxiv.org/html/2509.25339)
- Per-category counts: activity 150, attributes 149, counting 559, OCR 118, reasoning 356, scene 1,388 [V] — [arXiv HTML](https://arxiv.org/html/2509.25339)
- Each yes/no question is paired with its logical opposite ("Is it day?" / "Is it night?"). A pair is correct only if both answers are correct [V] — [arXiv HTML](https://arxiv.org/html/2509.25339)
- Difficulty is set by the share of the 37 models that answer correctly: hard 0–20%, medium 20–90%, easy 90–100% [V] — [arXiv HTML](https://arxiv.org/html/2509.25339)
- Qwen2.5-VL accuracy (3B / 7B): activity 60.7 / 63.3, attributes 61.7 / 69.1, reasoning 43.9 / 49.8, scene 77.5 / 85.3, total 54.1 / 61.5, hard 4.8 / 9.6 [V] — [arXiv HTML](https://arxiv.org/html/2509.25339)
- The error analysis reports logical inconsistency. Frontier models are fairly consistent on scene questions (~83%) but much less so on reasoning (~61%). Some models are at or below chance consistency, which suggests shortcut use [V] (the averages sentence in the HTML is garbled, so treat the numbers as approximate) — [arXiv HTML](https://arxiv.org/html/2509.25339)
- The authors say the images are new, which they argue prevents pretraining leakage. No dev split is provided, and models are expected to be evaluated without fine-tuning on the benchmark [V] — [arXiv HTML](https://arxiv.org/html/2509.25339)
- HF release: `paulgavrikov/visualoverload`, a single `test` split, **CC BY-SA 4.0**. Images are based on CC0 public-domain artworks. Fields: question_id, image, question, question_type (choice / counting / ocr), options, difficulty, category [V] — [HF card](https://huggingface.co/datasets/paulgavrikov/visualoverload/blob/main/README.md)

### Inferences
- The images (but not the answers) are public on HF. The training pool can therefore be **de-duplicated against the 150 images directly**, for example with SSCD or pHash, the way FineVision decontaminated (Q5).
- For the in-scope categories, the leverage ranking by item count × headroom is: scene pairs (1,388 items, and consistency-sensitive), then reasoning (356, the weakest at 44–50%), then activity and attributes (~150 each, ~61–69%).
- All in-scope answers are multiple choice or yes/no. Almost any dataset whose labels can be turned into MC or binary questions can produce **verifiable RL rewards** (GRPO-style exact match).

### Gaps
- It is unclear whether "total" is a micro-average over items or a macro-average over categories. The fetched excerpt did not say, and the answer decides how much weight scene pairs carry.
- The paper does not give per-image counts of people or objects. "Many people / subplots" is qualitative.

---

## Q1. (a) Compositional, region-grounded VQA and referring expressions in cluttered scenes: per-dataset facts

### Takeaway
For VisualOverload's "the person in brown at the front of the table in the leftmost house" style of question, GQA (scene-graph compositional questions), Visual Genome QA/regions and the RefCOCO family / Ref-L4 (long referring expressions) are the closest real-photo sources. VSR, What'sUp and SpatialRGPT-OSD add spatial-relation supervision. All are on natural photos with far fewer people per image than a crowded painting. GQA, VSR and the RefCOCO family have short, verifiable answers suited to RL. OSD is very large but its labels are synthetic and noisy.

### Cited Findings

| Dataset (venue/year) | Size | Density / images | Task & answer format | Labels per split | License | Source |
|---|---|---|---|---|---|---|
| **GQA** (CVPR 2019) | ~22M questions over ~113K images [P] | Visual Genome images; questions generated from scene graphs (multi-hop relations, attributes, spatial) [P] | Open short answer (1–3 words), many yes/no; "balanced" subset [P] | train / val public; test-dev answers released for balanced; test via server [P] | CC BY 4.0 [U] | [arXiv 1902.09506](https://arxiv.org/abs/1902.09506) |
| **Visual Genome** (IJCV 2017) | ~108K images, ~1.7M QA, dense region descriptions, objects, attributes, relationships [P] | ~35 objects, 26 attributes, 21 relationships per image on average [P] | Region captions; 7W QA (free-form); scene graphs | All annotations public (no hidden test) [P] | **CC BY 4.0** [V] (HF card) | [arXiv 1602.07332](https://arxiv.org/abs/1602.07332); [HF ranjaykrishna/visual_genome](https://huggingface.co/datasets/ranjaykrishna/visual_genome) |
| **VQA v2** (CVPR 2017) | ~204K COCO images, ~1.1M questions [P] | COCO (~7 objects/img on average) | Open short answer; **complementary image pairs**: same question, two similar images, different answers [P] | train / val public; test-dev/std via server [P] | Annotations CC BY 4.0 [U] | [arXiv 1612.00837](https://arxiv.org/abs/1612.00837) |
| **VQA-CP v1/v2** (CVPR 2018) | Re-split of VQA v1/v2 with different answer priors per question type [P] | COCO | Same as VQA | train / test public [P] | Inherits VQA [U] | [arXiv 1712.00377](https://arxiv.org/abs/1712.00377) |
| **CLEVR** (CVPR 2017) | 100K rendered images, ~1M programmatic questions [P] | 3–10 synthetic objects | Short answer; programs available | train / val public; test answers hidden [P] | CC BY 4.0 [U] | [arXiv 1612.06890](https://arxiv.org/abs/1612.06890) |
| **Super-CLEVR** (CVPR 2023) | Synthetic vehicles; domain-shift VQA [P] | Synthetic | Short answer | public [P] | [U] | [arXiv 2212.00259](https://arxiv.org/abs/2212.00259) |
| **SugarCrepe** (NeurIPS 2023 D&B) | Hard-negative caption pairs on COCO (replace / swap / add of objects, attributes, relations) [P] | COCO val2017 images [V] | Image–text matching (choose the true caption: binary/MC) | Eval-only, all labels public | Code **MIT** [V]; data license [U] | [GitHub RAIVNLab/sugar-crepe](https://github.com/RAIVNLab/sugar-crepe); [arXiv 2306.14610](https://arxiv.org/abs/2306.14610) |
| **VSR** (TACL 2023) | >10K natural text–image pairs, 66 spatial relation types [P] | COCO images | True / false caption about a spatial relation | random and zero-shot splits, all public [V] | Data **CC BY 4.0** [V] (HF), code Apache-2.0 [V] | [HF cambridgeltl/vsr_random](https://huggingface.co/datasets/cambridgeltl/vsr_random); [GitHub](https://github.com/cambridgeltl/visual-spatial-reasoning) |
| **What'sUp** (EMNLP 2023) | Controlled_Images_A/B (photos where only the spatial relation changes), plus COCO-QA and VG-QA one/two-object spatial subsets [V] | Controlled tabletop photos; COCO val2017 [V] | 4-way caption MC (left/right/on/under) [V] | Eval-oriented, all public | Repo **MIT** [V] | [GitHub amitakamath/whatsup_vlms](https://github.com/amitakamath/whatsup_vlms); [arXiv 2310.19785](https://arxiv.org/abs/2310.19785) |
| **SpatialRGPT / Open Spatial Dataset (OSD)** (NeurIPS 2024) | ~8.7M spatial concepts grounded in ~5M regions of ~1M OpenImages images [P]; released as one 31.9 GB JSON (`result_10_depth_convs.json`) [V] | OpenImages; depth from metric-depth pipeline | Region-referenced (mask/box) QA: relative direction, distance, size; qualitative and quantitative [P] | train public; SpatialRGPT-Bench for eval [V] | Data **Apache-2.0** per README badge [V]; images: OpenImages terms | [GitHub AnjieCheng/SpatialRGPT](https://github.com/AnjieCheng/SpatialRGPT); [HF a8cheng/OpenSpatialDataset](https://huggingface.co/datasets/a8cheng/OpenSpatialDataset) |
| **SAT: Spatial Aptitude Training** (2024/25) | Synthetic 3D-rendered scenes; static and dynamic spatial QA. 2026 update added `SAT-v2` and `SAT_perspective` with rendered-view reasoning traces [V] | Synthetic indoor | MC / short | train public | **MIT** [V] (HF and repo) | [GitHub arijitray1993/SAT](https://github.com/arijitray1993/SAT); [HF array/SAT](https://huggingface.co/datasets/array/SAT) |
| **SPAR-7M** (2025) | ~7M spatial QA (HF tag 10M–100M rows) [V tag] | Indoor 3D scans | Spatial QA | public | **MIT** [V] | [HF jasonzhango/SPAR-7M](https://huggingface.co/datasets/jasonzhango/SPAR-7M) |
| **RefCOCO / RefCOCO+ / RefCOCOg** (ECCV 2016 / CVPR 2016) | RefCOCO ~142K expressions for ~50K objects in ~20K images; RefCOCO+ (no absolute-location words) ~141K; RefCOCOg ~95K longer expressions in ~27K images [P] | COCO multi-instance images (≥2 same-class objects) | Box/mask for an expression (REC); reversible to REG or MC | train / val / testA / testB public [P] | [U] (COCO images CC BY 4.0 for annotations; Flickr image terms) | [arXiv 1608.00272](https://arxiv.org/abs/1608.00272); [arXiv 1511.02283](https://arxiv.org/abs/1511.02283) |
| **Ref-L4** (2024, LMM-era REC benchmark) | ~45K expressions over ~9.7K images, 365 categories, average expression ~24 words [P] | COCO + Objects365 [P] | REC (box) | Eval benchmark, labels public | **CC BY-NC 4.0** [V] (HF); code MIT [V] | [HF JierunChen/Ref-L4](https://huggingface.co/datasets/JierunChen/Ref-L4); [arXiv 2406.16866](https://arxiv.org/abs/2406.16866) |
| **GRIT (Kosmos-2)** (2023) | Web-scale grounded image–text pairs (HF tag 10M–100M; paper ~91M images) [V tag / P] | LAION/COYO web images | Noun-phrase / referring-expression → box | train only | **MS-PL** [V] (HF) | [HF zzliang/GRIT](https://huggingface.co/datasets/zzliang/GRIT); [arXiv 2306.14824](https://arxiv.org/abs/2306.14824) |
| **GRIT (Ferret)** (ICLR 2024), name collision | ~1.1M grounded instruction samples (VG/RefCOCO/Flickr-derived) [P] | Real photos | Refer + ground, free-form | train | [U] | [arXiv 2310.07704](https://arxiv.org/abs/2310.07704) |
| **COREVQA** (2025) | 5,608 image–statement pairs; images from **CrowdHuman** train01/train02 [V] | Crowded indoor/outdoor groups of people [V] | True/false entailment statement (~30 words); 27.9% true / 72.1% false [V] | No train/test split described; hand-verified labels [V] | Dataset license not stated [V], [U] | [arXiv 2507.13405](https://arxiv.org/html/2507.13405v1); [HF COREVQA2025/COREVQA](https://huggingface.co/datasets/COREVQA2025/COREVQA) |
| **NLVR2** (ACL 2019) | ~107K sentence–image-pair examples [P] | Web photo pairs | True/false | train/dev/test-P public [P] | [U] | [arXiv 1811.00491](https://arxiv.org/abs/1811.00491) |
| **SpatialSense** (ICCV 2019) | Adversarially mined spatial-relation triples; 10,440 samples in FineVision [V] | Flickr / NYU images | Binary relation true/false [P] | public | [U] | [FineVision paper](https://arxiv.org/html/2510.17269) |

### Inferences
- **Most transferable for referring activity/attribute MC.** GQA ("What is the man to the left of the table holding?") and Visual Genome region-QA can be rewritten into VisualOverload's 4-option MC format, using distractors from the same scene graph so the answers stay verifiable. RefCOCOg and Ref-L4 give long relational referring expressions that can be paired with VG attributes and relationships for "person dressed in brown at the front of …" style questions.
- **Spatial data.** VSR and What'sUp are small, relation-pure and verifiable, which fits RL or small SFT add-ins. OSD is very large but auto-labelled, and its metric depth means little on painted scenes. Use only its qualitative-relation subset.
- **COREVQA** is the only true/false resource found here that is built on genuinely crowded images. It has no train split, so it is best held out as a **proxy dev set** for crowded-scene binary reasoning.
- Ref-L4 is CC BY-NC (research-only) and CrowdHuman-based COREVQA has no stated license. Both suit academic use only.

### Gaps
- The GQA, VQA v2, CLEVR and RefCOCO data licenses were not re-verified this session. CC BY 4.0 is the commonly reported license for the first three.
- OSD's exact counts and file schema were not re-read (the 31.9 GB single JSON blocked inspection).
- No published dataset was found that gives multi-hop referring questions on **images with more than 30 people** and has a training split. This is the core coverage gap for VisualOverload.

---

## Q2. (b) Dense / long captions usable for QA generation

### Takeaway
DOCCI (CC BY 4.0), PixMo-Cap (ODC-BY), DenseFusion-1M (CC BY 4.0), IIW (CC BY 4.0) and Localized Narratives are the permissively licensed long-caption sources. DCI and ShareGPT4V are non-commercial. GranD, DLC-SDP and Osprey give region-level dense descriptions. All are caption supervision rather than verifiable QA. They help VisualOverload mainly as **seeds for programmatically generated MC/yes-no questions**, or as SFT warm-up for "describe everything" behaviour.

### Cited Findings

| Dataset (venue/year) | Size | Image / density notes | Format | License | Source |
|---|---|---|---|---|---|
| **DCI: Densely Captioned Images** (CVPR 2024) | **7,805** SA-1B images [V] | Full description plus nested **submask** descriptions aligned to SAM masks [V]; paper reports >1,000 words per image on average [P] | Hierarchical mask-aligned text; CLIP-ready summaries and LLaMA-2 negatives [V] | Annotations **CC BY-NC 4.0** [V] (repo LICENSE); images need the SA-1B license (`sa_000138.tar`) [V] | [GitHub facebookresearch/DCI](https://github.com/facebookresearch/DCI); [arXiv 2312.08578](https://arxiv.org/abs/2312.08578) |
| **DOCCI** (ECCV 2024) | ~15K images, ~136-word human captions on average [P] | Photos taken by one researcher, designed for spatial relations, counts, text rendering [P] | Long caption | **CC BY 4.0** [V] (HF) | [HF google/docci](https://huggingface.co/datasets/google/docci); [arXiv 2404.19753](https://arxiv.org/abs/2404.19753) |
| **PixMo-Cap** (Molmo, CVPR 2025) | ~712K images / ~1.3M captions [P]; HF tag 100K–1M [V] | Web images; captions transcribed from spoken descriptions, long [P] | Long caption | **ODC-BY** [V] | [HF allenai/pixmo-cap](https://huggingface.co/datasets/allenai/pixmo-cap); [arXiv 2409.17146](https://arxiv.org/abs/2409.17146) |
| **PixMo-CapQA / AskModelAnything / Points** | QA generated from captions; human-written QA; pointing [P] | Web | QA / points | **ODC-BY** [V] (all three) | [HF pixmo-cap-qa](https://huggingface.co/datasets/allenai/pixmo-cap-qa); [HF pixmo-ask-model-anything](https://huggingface.co/datasets/allenai/pixmo-ask-model-anything); [HF pixmo-points](https://huggingface.co/datasets/allenai/pixmo-points) |
| **ShareGPT4V** (ECCV 2024) | 100K GPT-4V captions + 1.2M ShareCaptioner captions [P]. The 100K comes from COCO 50K, LCS 30K, SAM 20K, TextCaps 500, **WikiArt 500**, web landmarks/celebrities 1K [V] | Mixed | Long caption | **CC BY-NC 4.0** [V] | [HF Lin-Chen/ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V); [ECCV supp](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02629-supp.pdf) |
| **DenseFusion-1M** (NeurIPS 2024 D&B) | ~1M (1,058,751 in FineVision) [V] | LAION images; captions fused from multiple perception experts (detection, OCR, attributes) [P] | Hyper-detailed caption | **CC BY 4.0** [V] | [HF BAAI/DenseFusion-1M](https://huggingface.co/datasets/BAAI/DenseFusion-1M); [arXiv 2407.08303](https://arxiv.org/abs/2407.08303) |
| **Localized Narratives** (ECCV 2020) | ~849K images (COCO, Flickr30k, ADE20K, Open Images) [P] | Spoken narrative plus mouse trace per word [P] | Caption + trace grounding | CC BY 4.0 [U] | [arXiv 1912.03098](https://arxiv.org/abs/1912.03098) |
| **LNQA** (QA derived from Localized Narratives) | 302,780 in FineVision; ~302K in Cambrian [V] | — | QA | [U] | [FineVision paper](https://arxiv.org/html/2510.17269) |
| **Panoptic Narrative Grounding** (ICCV 2021) | COCO panoptic + Localized Narratives; every noun phrase grounded to panoptic masks [P] | COCO | Phrase → mask | [U] | [arXiv 2109.04988](https://arxiv.org/abs/2109.04988) |
| **IIW: ImageInWords** (ECCV 2024) | HF tag 1K–10K [V]; human-in-the-loop hyper-detailed descriptions [P] | Mixed | Long caption + object-level descriptions | **CC BY 4.0** [V] | [GitHub google/imageinwords](https://github.com/google/imageinwords); [HF](https://huggingface.co/datasets/google/imageinwords) |
| **GranD / GranD-f** (GLaMM, CVPR 2024) | Automated dense grounding over SA-1B (~11M images) [P]; GranD-f is the human-verified grounded-conversation subset [P] | SA-1B (dense, many objects) | Grounded captions with masks | HF card **Apache-2.0** [V]; SA-1B image terms apply | [HF MBZUAI/GranD](https://huggingface.co/datasets/MBZUAI/GranD); [arXiv 2311.03356](https://arxiv.org/abs/2311.03356) |
| **DLC-SDP (Describe Anything)** (ICCV 2025) | HF tag 100K–1M [V]; sources COCOStuff, LVIS, Mapillary, OpenImages, PACO, SAM, SA-V [V] | Region (mask) → detailed localized caption | Region caption | **NVIDIA Noncommercial License** [V] | [HF nvidia/describe-anything-dataset](https://huggingface.co/datasets/nvidia/describe-anything-dataset); [arXiv 2504.16072](https://arxiv.org/abs/2504.16072) |
| **Osprey-724K** (CVPR 2024) | ~724K mask-text instruction samples [P] | COCO/RefCOCO/VG-derived masks | Region QA / description | **CC BY-NC 4.0** [V] | [HF AntGroup-MI/Osprey-724K](https://huggingface.co/datasets/AntGroup-MI/Osprey-724K) |
| **AS-V2 (All-Seeing v2)** (ECCV 2024) | Relation conversation with grounded subject/predicate/object [P] | COCO-based | Grounded relation QA | **Apache-2.0** [V] | [HF OpenGVLab/AS-V2](https://huggingface.co/datasets/OpenGVLab/AS-V2) |
| **PLM-Image-Auto** (PerceptionLM, 2025) | HF tag 10M–100M [V] | Synthetic captions and QA | Caption / QA | **Llama 3.2 license** [V] | [HF facebook/PLM-Image-Auto](https://huggingface.co/datasets/facebook/PLM-Image-Auto) |

### Inferences
- **QA-generation recipe.** Use DOCCI, IIW, PixMo-Cap or DenseFusion captions to generate attribute / activity / spatial MC questions. Use an **LLM-free or rule-checked** generator (parse captions into entity–attribute–relation tuples) so the answers stay verifiable for RL. Mask-aligned DCI and GranD descriptions allow region-anchored referring questions, which is the closest thing to VisualOverload's "person at the front of the table" phrasing.
- **Commercial-safety tiers.** Permissive: DOCCI, IIW, DenseFusion (CC BY 4.0), PixMo (ODC-BY), AS-V2 and GranD cards (Apache-2.0, but SA-1B terms apply). Non-commercial: DCI, ShareGPT4V, Osprey, DLC-SDP, Ref-L4.

### Gaps
- PixMo-Cap's exact image-topic mix (whether it includes artworks or museum photos) was not verified. That matters for overlap risk.
- The Localized Narratives and Panoptic Narrative Grounding licenses were not re-verified.

---

## Q3. (c) Crowded human activity, HOI and situation data (plus attribute and scene-classification sources)

### Takeaway
HICO-DET (600 HOI classes), V-COCO, imSitu/SWiG (504 verbs with grounded roles), PSG (panoptic relations) and Visual Genome relationships give **verb and interaction labels with closed vocabularies**. Those convert easily into verifiable activity MC ("What is the person in brown doing?"). CrowdPose, OCHuman, CrowdHuman and PANDA give crowd density and occlusion but **no activity labels**, so on their own they are useful only for referring and attribute questions. VCR and Sherlock are the only large, labelled sources for **intent and inference** reasoning about people.

### Cited Findings

| Dataset (venue/year) | Size / density | Labels | Fit | License | Source |
|---|---|---|---|---|---|
| **HICO-DET** (WACV 2018) | ~47.8K images (~38K train / ~9.7K test); 600 HOI categories = 117 verbs × 80 COCO objects [P] | Human box + object box + verb, train and test public [P] | Activity MC; verifiable | Utility code MIT [V]; annotation/image license [U] (Flickr images, research use) | [arXiv 1702.05448](https://arxiv.org/abs/1702.05448); [GitHub fredzzhang/hicodet](https://github.com/fredzzhang/hicodet) |
| **V-COCO** (2015) | ~10K COCO images, 26 action classes with semantic roles [P] | Per-person action and role boxes, public [P] | Activity MC on multi-person COCO | Code MIT [V]; annotations [U] | [arXiv 1505.04474](https://arxiv.org/abs/1505.04474); [GitHub s-gupta/v-coco](https://github.com/s-gupta/v-coco) |
| **imSitu** (CVPR 2016) / **SWiG** (ECCV 2020) | ~126K images, 504 verbs, frame roles (agent, tool, place…); SWiG adds role bounding boxes [P]. JSONs in `SWiG_jsons`, images 512-px zip [V] | train / dev / test labels public [P] | "What is X doing / with what / where" MC; verifiable | Code **MIT** [V]; image rights [U] (web images) | [GitHub allenai/swig](https://github.com/allenai/swig); [arXiv 2003.12058](https://arxiv.org/abs/2003.12058) |
| **PSG: Panoptic Scene Graph** (ECCV 2022) | **49K** images overlapping COCO and Visual Genome; mask-to-mask relations [V]; 56 predicates, 133 classes [P] | train / val public | Relation MC ("who is holding / looking at …") | Code MIT [V]; annotations [U] | [GitHub Jingkang50/OpenPSG](https://github.com/Jingkang50/OpenPSG); [arXiv 2207.11247](https://arxiv.org/abs/2207.11247) |
| **Visual Genome relationships** | See Q1; ~21 relationships per image on average [P] | public | Relation / attribute MC | CC BY 4.0 [V] | [HF card](https://huggingface.co/datasets/ranjaykrishna/visual_genome) |
| **Action Genome** (CVPR 2020) | Video (Charades); spatio-temporal scene graphs of person–object relations [P] | public | Frame stills give attention / contact / spatial relations | [U] | [arXiv 1912.06992](https://arxiv.org/abs/1912.06992) |
| **CrowdPose** (CVPR 2019) | ~20K images, ~80K persons, with a crowd-index split [P] | Keypoints only | Crowd density; referring only | [U] | [arXiv 1812.00324](https://arxiv.org/abs/1812.00324) |
| **OCHuman** (CVPR 2019) | ~4.7K images, ~8.1K heavily occluded persons [P] | Boxes, keypoints, masks | Occlusion | API repo MIT [V]; data [U] | [GitHub OCHumanApi](https://github.com/liruilong940607/OCHumanApi); [arXiv 1803.10683](https://arxiv.org/abs/1803.10683) |
| **CrowdHuman** (2018) | ~15K train images with ~23 persons per image on average [P] | Full / visible / head boxes | Crowd density (also the COREVQA source [V]) | Non-commercial research [U] | [arXiv 1805.00123](https://arxiv.org/abs/1805.00123) |
| **PANDA** (CVPR 2020) | Gigapixel human-centric scenes (very large crowds) [P] | Boxes, tracks, some group/interaction labels [P] | Closest *real* analogue to "overload" density | Toolkit MIT [V]; data terms [U] | [GitHub PANDA-Toolkit](https://github.com/GigaVision/PANDA-Toolkit); [arXiv 2003.04852](https://arxiv.org/abs/2003.04852) |
| **VCR** (CVPR 2019) | ~290K MC questions from ~110K movie frames; Q→A and QA→R (rationale) [P] | Answers public for train/val; test via leaderboard [P] | **Intent / why / what-next** MC with person tags; verifiable | Research-only terms [U] | [arXiv 1811.10830](https://arxiv.org/abs/1811.10830) |
| **Sherlock** (ECCV 2022) | Abductive inferences on VG/VCR images [P]; included as a `sherlock` config in LLaVA-OneVision-1.5 instruct data [V] | public train | Intent and causal inference about a region | [U] | [arXiv 2202.04800](https://arxiv.org/abs/2202.04800); [HF LLaVA-OV-1.5 instruct](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data) |
| **FineBench** (CVPR 2026 workshop) | 199,420 MC QA over 64 long videos (~15 min each); person movement, interaction, object manipulation; reports weak multi-person spatial reasoning [V] | splits not stated [V] | Video; multi-person activity | arXiv paper CC BY 4.0; data [U] | [arXiv 2605.19846](https://arxiv.org/abs/2605.19846) |
| **VAW: Visual Attributes in the Wild** (CVPR 2021) | ~260K instances, 620 attributes with explicit positive and negative labels [P] | VG images | **Attribute yes/no and MC with hard negatives**; verifiable | [U] | [arXiv 2104.13921](https://arxiv.org/abs/2104.13921) |
| **Places365** (TPAMI 2017) | ~10M images, 365 scene categories [P] | public | Scene classification ("indoor/outdoor", "market", "harbor") | [U] | [arXiv 1610.02055](https://arxiv.org/abs/1610.02055) |
| **Vision-Flan scene/time tasks** | Contains `Dark-Zurich+time_classification`, `places365+Image_Classification`, `Places205+Image_env_classification`, `ImageNet-C+image_classification_weather`, `HICO+human_activity_detection`, `VQA_activity_recognition`, `VQA_scene_recognition`, `VQA_positional_reasoning`, `VQA_attribute`, `GQA`, `visualgenome_vqa` (1,000 each) [V] | public | Ready-made instruction form | Per-task licenses [V] | [HF Vision-Flan](https://huggingface.co/datasets/Vision-Flan/vision-flan_191-task_1k) |

### Inferences
- **Best activity sources.** SWiG/imSitu (verbs plus role nouns), HICO-DET and PSG. Convert them to "What is the person [attribute] [spatial anchor] doing?" by adding a VG/VAW attribute and a relative position computed from boxes. The answer stays a verb class, so it is exactly verifiable for RL.
- **Intent / path / occupancy reasoning** (VisualOverload "reasoning", the weakest category): VCR and Sherlock are the only large labelled sources. VCR's movie stills are multi-person, closer to painted genre scenes than COCO is.
- **Scene day/night etc.** Places365 and Dark-Zurich give single labels. To match VisualOverload's pair scoring, generate **both polarities** ("Is it day?" yes / "Is it night?" no) from one label and train with a pair-consistency objective (see Q4).
- **Crowd-only datasets** (CrowdPose, OCHuman, CrowdHuman, PANDA) need synthetic attribute and position questions. They are useful for teaching referring among many similar people, not activities.

### Gaps
- HICO-DET, imSitu, PSG, CrowdPose, CrowdHuman and PANDA dataset licenses were not verified this session. Only code licenses were.
- No still-image dataset with **group activity in crowds of more than 20 people** plus per-person verbs was found.

---

## Q4. (d) Logical-consistency and paired-question resources

### Takeaway
Paired and consistency resources exist at scale only for COCO-style photos: VQA v2 complementary pairs, VQA-LOL (logical compositions of yes/no questions), VQA-Introspect (perception sub-questions) and ConVQA (entailed QA sets). Smaller sets cover abstract scenes (Yin-Yang) and paired-image benchmarks (NaturalBench). None is built on dense scenes. The practical route is to **generate logically opposed pairs from any labelled dataset** and train or reward at the pair level, as in VQA-LOL and the CVPR 2023 logical-implication loss.

### Cited Findings
- **VQA v2 complementary pairs** (CVPR 2017): each question is paired with a similar image that yields a different answer [P] — [arXiv 1612.00837](https://arxiv.org/abs/1612.00837)
- **Yin and Yang: binary VQA** (CVPR 2016): balanced binary questions on abstract clip-art scenes, where each question has a complementary scene flipping the answer [P] — [arXiv 1511.05099](https://arxiv.org/abs/1511.05099)
- **ConVQA** (Ray et al., EMNLP-IJCNLP 2019, "Sunny and Dark Outside?!"): sets of logically consistent QA pairs about one observable fact, plus a Consistency Teacher Module that generates entailed questions and fine-tunes on consistent answers [V] — [paper page](https://cs.paperswithcode.com/paper/sunny-and-dark-outside-improving-answer); [arXiv 1909.04696](https://arxiv.org/abs/1909.04696)
- **VQA-Introspect** (CVPR 2020): ~200K perception sub-questions linked to reasoning questions. In 28.14% of cases where models get the reasoning question right, they fail the perception sub-question [V] — [arXiv 2001.06927](https://www.alphaxiv.org/abs/2001.06927.md); [MSR project](https://www.microsoft.com/en-us/research/project/vqa-introspect/?lang=zh-cn)
- **VQA-LOL** (ECCV 2020): VQA-Compose has 844,122 train and 437,861 val questions formed by composing pairs of yes/no VQA questions with negation, conjunction and disjunction [V] — [benchmark page](https://www.alphaxiv.org/benchmarks/arizona-state-university/vqa-lol); [arXiv 2002.08325](https://arxiv.org/abs/2002.08325)
- **Logical Implications for VQA Consistency** (CVPR 2023): a consistency loss over logically related question pairs, with the relations inferred by a language model; evaluated on VQA-Introspect and DME [V] — [arXiv 2303.09427](https://arxiv.org/pdf/2303.09427)
- **VQA-Rephrasings / cycle-consistency** (CVPR 2019): paraphrase-consistency test set over VQA v2 [P] — [arXiv 1902.05660](https://arxiv.org/abs/1902.05660)
- **NaturalBench** (NeurIPS 2024 D&B): paired images × paired questions with group scoring; HF license Apache-2.0 [V]. Benchmark, so hold it out — [HF BaiqiL/NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench); [arXiv 2410.14669](https://arxiv.org/abs/2410.14669)
- **COREVQA** (2025): crowded-scene true/false entailment, no train split [V] — [arXiv 2507.13405](https://arxiv.org/html/2507.13405v1)
- "**LogicVQA**" as a named dataset: no reliable source found [V, negative result].

### Inferences
- A **pairwise reward fits RL naturally.** Reward 1 only when both members of a generated opposite pair (day/night, indoor/outdoor, "is anyone sitting?"/"is everyone standing?") are answered correctly. This mirrors VisualOverload's scoring and directly attacks the inconsistency failure mode documented in Q0.
- VQA-LOL's negation and conjunction compositions and VQA v2 complementary pairs are the largest ready-made SFT sources. Because they are COCO-based, expect transfer of the *behaviour* (polarity consistency) rather than of domain knowledge.

### Gaps
- No dataset with paired yes/no questions on paintings or dense crowds was found.
- The ConVQA release size and license, and the VQA-Introspect license, were not verified.

---

## Q5. (e) Large VLM SFT mixtures: which subsets matter, and the art-image overlap risk

### Takeaway
Most of the relevant compositional data (GQA, VG, VQA v2, VSR, CLEVR, Visual7W, A-OKVQA, LNQA, Sherlock) is already bundled in Cambrian-10M, The Cauldron, LLaVA-OneVision(-1.5) and FineVision. **Important overlap finding:** the `vision_flan(filtered)` subset carried by LLaVA-OneVision and FineVision contains **WikiArt and SemArt paintings**, and the `sharegpt4v(knowledge)` subset contains WikiArt-named paintings. LLaVA-OV-1.5 also adds `wit` and `wikipedia_2m` (Wikimedia images). All of these should be **image-deduplicated against VisualOverload's 150 paintings** before training.

### Cited Findings
- **LLaVA-OneVision-Data**: Apache-2.0, 89 subsets [V]. Relevant subsets: `vsr`, `clevr`, `visual7w`, `aokvqa`, `tallyqa`, `sharegpt4v(coco/sam/llava/knowledge)`, `image_textualization(filtered)`, `vision_flan(filtered)`, `allava_instruct_*`, `lrv_normal`. **No GQA, VG or VQA v2 subset** [V] — [HF lmms-lab/LLaVA-OneVision-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data); [arXiv 2408.03326](https://arxiv.org/abs/2408.03326)
- **Art inside LLaVA-OV `vision_flan(filtered)`**: reading the `id` column of 2 of its 50 parquet shards found rows from `WIKIART+art_classification` (e.g. Rembrandt, Degas, van Gogh, Dürer, Gustave Doré files), `semart+image_{description,timeframe,technique,type,school}` (~194 rows in 2 shards), `PACS+art_painting_object_classification` (42) and `DOMAIN_NET+painting_image_classification` (34) [V, read directly this session] — [HF parquet](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data/tree/main)
- **Vision-Flan 191-task release**: 191 tasks × ~1,000 each, including `WIKIART+art_classification`, five `semart+*` tasks, `PACS+art_painting…` and `DOMAIN_NET+painting…` [V, metadata.json parsed]. Per-task licenses are on the project page [V] — [HF Vision-Flan](https://huggingface.co/datasets/Vision-Flan/vision-flan_191-task_1k); [arXiv 2402.11690](https://arxiv.org/abs/2402.11690)
- **Art inside `sharegpt4v(knowledge)`** (LLaVA-OV): ids include WikiArt-style slugs such as `william-hogarth_the-graham-children-1742`, `vittore-carpaccio_the-crucifixion-and-the-glorification-the-ten-thousand-martyrs-on-mount-ararat-1515` and `raphael_the-death-of-ananias…`, alongside celebrity and landmark images [V, read directly]. The ShareGPT4V 100K set includes 500 WikiArt images [V] — [ECCV supp](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02629-supp.pdf)
- **LLaVA-OneVision-1.5 Instruct Data** (mvp-lab, updated 2026-07): Apache-2.0, HF tag 10M–100M [V]. Configs include `gqa`, `vg`, `vsr`, `visual7w`, `aokvqa`, `sherlock`, `svit`, `tallyqa`, `cambrian`, `vision_flan`, `vflan`, `sharegpt4v`, `image_textualization`, `llava_cot_100k`, **`wit`**, **`wikipedia_2m`**, `viquae`, `idk` [V] — [HF](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data)
- **Cambrian-10M**: Apache-2.0 [V]; ~9.8M points. Includes GQA 72K, Visual Genome 86K, VQAv2 83K, RefCOCO 30K, LNQA 302K, CLEVR 350K, TallyQA 250K, Visual7W 14K, A-OKVQA 50K, Q-Instruct 400K, ALLaVA 700K. **No art or painting sources appear in the data figure** [V] — [arXiv 2406.16860](https://arxiv.org/html/2406.16860); [HF](https://huggingface.co/datasets/nyu-visionx/Cambrian-10M). A HF discussion notes that the JSON `source` fields (38 sources) don't match the paper (~68), which complicates subset filtering [V] — [HF discussion #9](https://huggingface.co/datasets/nyu-visionx/Cambrian-10M/discussions/9)
- **The Cauldron** (Idefics2): 50 subsets, including VQAv2 (82,772 images / 443,757 QA), VSR (2,157 / 3,354), CLEVR (70,000 / 699,989), CLEVR-Math, NLVR2 (50,426 / 86,373), Localized Narratives, Visual7W, A-OKVQA, TallyQA. No art subset seen. Licensing is per subset [V] — [HF the_cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron); [arXiv 2405.02246](https://arxiv.org/abs/2405.02246)
- **FineVision** (HF, 2025): 24M samples, 17M images, 89M turns, 9.5B answer tokens; 200+ sources in 185 subsets across 9 categories. Images were SSCD-embedded and checked against 66 lmms-eval benchmarks at cosine ≥0.95; contamination was 1.02% vs 2.15–3.05% for baselines, but the release was **not** filtered [V]. Relevant subsets: `vqav2` 82,772, `vsr` 2,157, `spatialsense` 10,440, `clevr` 70,000, `densefusion_1m` 1,058,751, `localized_narratives` 199,998, `lnqa` 302,780, `image_textualization(filtered)` 99,573, **`vision_flan(filtered)` 175,964**, **`sharegpt4v(knowledge)` 1,988**, `cambrian(filtered)_processed` 83,124. No GQA, VG, RefCOCO, DOCCI or PixMo subsets [V]. The HF card gives no license (API `license: None`) [V] — [arXiv 2510.17269](https://arxiv.org/html/2510.17269); [HF FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision)
- FineVision also has a subset named **`art` (5,603, under "Naive OCR")** [V]. Its category suggests the ICDAR **ArT** scene-text set rather than artwork, but this is **unverified**.
- **MAmmoTH-VL-Instruct-12M** (ACL 2025): Apache-2.0, HF tag 10M–100M [V]. A rewrite of existing open data into reasoning-style instructions [P] — [HF](https://huggingface.co/datasets/MAmmoTH-VL/MAmmoTH-VL-Instruct-12M)
- **Infinity-MM**: **CC BY-SA 4.0**, gated (auto-approve), HF tag 10M–100M [V] — [HF BAAI/Infinity-MM](https://huggingface.co/datasets/BAAI/Infinity-MM); [arXiv 2410.18558](https://arxiv.org/abs/2410.18558)
- **PixMo / Molmo data**: ODC-BY for every PixMo subset checked (cap, cap-qa, ask-model-anything, points, docs) [V] — [HF allenai](https://huggingface.co/allenai)

### Inferences
- **Most relevant subsets to extract** (deduplicated across mixtures):
  - GQA and VG: Cambrian or LLaVA-OV-1.5.
  - VSR, SpatialSense, Visual7W, A-OKVQA: Cauldron or FineVision.
  - Sherlock: LLaVA-OV-1.5.
  - LNQA and DenseFusion: FineVision.
  - VQAv2 and NLVR2: Cauldron.
  - Vision-Flan HICO, Places, Dark-Zurich and scene/activity tasks, with the art tasks **removed**.
- **Overlap risk ranking** (highest first):
  1. `vision_flan` / `vflan` WikiArt and SemArt tasks. SemArt is drawn from the Web Gallery of Art (European fine art, largely public domain), so the same artworks as Google Arts & Culture scans are plausible.
  2. `sharegpt4v(knowledge)` WikiArt images.
  3. `wit` / `wikipedia_2m` (Wikimedia Commons hosts many public-domain painting scans).
  4. Web-crawled caption sets (PixMo-Cap, DenseFusion/LAION, ALLaVA-LAION).
  5. COCO / VG / OpenImages / SA-1B based sets, which are effectively nil.

  Even an exact artwork match would be a different scan at lower resolution with no VisualOverload questions attached. The risk is image familiarity, not answer leakage, but a CVPR-grade paper should still report a dedup step.
- **Decontamination recipe.** SSCD embeddings (as in FineVision) or pHash over the 150 public VisualOverload images versus all training images. Also drop whole tasks named `WIKIART*`, `semart*`, `PACS+art_painting*`, `DOMAIN_NET+painting*` and `sharegpt4v(knowledge)`.

### Gaps
- Total WikiArt/SemArt row counts in LLaVA-OV `vision_flan(filtered)` were sampled from 2/50 shards only. Extrapolation suggests roughly 1K per task (consistent with the 1K-per-task design), but this is an estimate.
- It was not checked whether FineVision's `vision_flan(filtered)` keeps the same art tasks. It is likely inherited from LLaVA-OV, but unverified.
- The Infinity-MM, MAmmoTH-VL and PixMo-Cap image-source mixes were not audited for artwork.
- FineVision's dataset license is not stated on the card.

---

## Q6. Cross-cutting: match to VisualOverload, SFT vs RL fit, and a recommended shortlist

### Takeaway
No public training set reproduces VisualOverload's combination of 4K paintings, dense crowds, referring MC and pairwise yes/no. The best approach is **compositional transfer**:
- SFT on GQA, VG, SWiG/HICO-DET-derived referring activity and attribute MC, plus DOCCI/PixMo dense captions.
- Then RL with exact-match rewards on verifiable MC/yes-no built from closed-vocabulary labels (verbs, attributes, relations, scene classes), using a **pair-level consistency reward** for scene/reasoning opposites.

### Cited Findings
- VisualOverload's in-scope answers are all MC (A–D or A–B) or paired yes/no [V] — [HF card](https://huggingface.co/datasets/paulgavrikov/visualoverload/blob/main/README.md); [arXiv HTML](https://arxiv.org/html/2509.25339)
- Qwen2.5-VL-7B gaps to close: reasoning 49.8, activity 63.3, attributes 69.1, scene 85.3 [V] — [arXiv HTML](https://arxiv.org/html/2509.25339)
- Datasets with closed-vocabulary, exactly checkable labels (suitable for RL rewards):
  - GQA, VQA v2 yes/no, CLEVR [P] — [GQA](https://arxiv.org/abs/1902.09506)
  - VSR true/false [V] — [HF](https://huggingface.co/datasets/cambridgeltl/vsr_random)
  - What'sUp 4-way MC [V] — [GitHub](https://github.com/amitakamath/whatsup_vlms)
  - HICO-DET verbs [P] — [arXiv](https://arxiv.org/abs/1702.05448)
  - SWiG verbs and roles [P] — [arXiv](https://arxiv.org/abs/2003.12058)
  - VAW positive/negative attributes [P] — [arXiv](https://arxiv.org/abs/2104.13921)
  - PSG predicates [P] — [arXiv](https://arxiv.org/abs/2207.11247)
  - VCR MC [P] — [arXiv](https://arxiv.org/abs/1811.10830)
  - RefCOCO boxes (IoU reward) [P] — [arXiv](https://arxiv.org/abs/1608.00272)
- Caption datasets (DCI, DOCCI, PixMo-Cap, DenseFusion, IIW, LN) provide free text only. They need conversion before RL [V licenses as in Q2].

### Inferences
**Per-dataset fit summary** (match: H/M/L to VisualOverload in-scope categories; overlap: risk of sharing artworks with Google Arts & Culture paintings):

| Dataset | Best-matching VO category | Match | Overlap risk | SFT / RL |
|---|---|---|---|---|
| GQA | attribute / activity referring MC | H | nil (VG photos) | both; RL after MC conversion |
| Visual Genome QA + regions + relations | attribute / activity / referring | H | nil | both |
| VAW | attribute MC with negatives | H | nil | RL-ready |
| SWiG / imSitu | activity MC | H | low (web photos) | RL-ready |
| HICO-DET / V-COCO | activity MC | H | nil (Flickr/COCO) | RL-ready |
| PSG | activity / relation | M-H | nil | RL-ready |
| VCR / Sherlock | reasoning (intent / why) | H | nil (movie stills / VG) | VCR RL-ready; Sherlock SFT |
| RefCOCOg / Ref-L4 | referring disambiguation | M | nil | RL (IoU) or SFT |
| VSR / What'sUp / SpatialSense | spatial sub-skills in reasoning | M | nil | RL-ready |
| SpatialRGPT-OSD | spatial / distance reasoning | M-L (metric depth ≠ paintings) | nil | SFT (noisy) |
| VQA v2 pairs / VQA-LOL / ConVQA / VQA-Introspect | scene and reasoning consistency | M-H (behavioural) | nil | RL pair-reward |
| Places365 / Dark-Zurich / Vision-Flan scene tasks | scene yes/no pairs | M-H | nil | RL-ready (generate opposites) |
| DOCCI / IIW / DCI / PixMo-Cap / DenseFusion | QA-generation seed; dense description | M | low–medium (web: PixMo / LAION) | SFT |
| CrowdPose / OCHuman / CrowdHuman / PANDA | referring in crowds (density) | M (density only) | nil | needs synthetic QA |
| COREVQA | crowded true/false | M-H | nil | **hold-out proxy dev set** |
| LLaVA-OV / FineVision `vision_flan`, `sharegpt4v(knowledge)`; LLaVA-OV-1.5 `wit` | — | — | **HIGH (WikiArt / SemArt / Wikimedia paintings confirmed or plausible)** | filter before use |

- Suggested composition for Qwen2.5-VL-3B/7B (an inference, not tested):
  - **SFT.** About 300–500K examples, mixing GQA balanced, VG region-QA, SWiG/HICO-DET activity MC with attribute and spatial anchors, VAW attribute MC, VCR, Sherlock, VSR and What'sUp. Add 10–20% dense-caption data (DOCCI / PixMo-Cap) to keep the "inspect everything" behaviour, and retain general data (e.g. Cauldron/Cambrian subsets) to avoid forgetting.
  - **RL (GRPO).** On held-out MC/yes-no from the same label sources, with a **pair reward** for opposite scene and reasoning questions.
  - **Evaluation.** COREVQA and NaturalBench as held-out sanity checks before using the rate-limited VisualOverload server.
- All of the above use photos. The domain gap to paintings is uncovered by these datasets and falls to the art-specific researcher's scope.

### Gaps
- No published fine-tuning result on VisualOverload was found that shows which of these sources transfer. The paper evaluates zero-shot only [V].
- Per-image people counts for VisualOverload, and the macro vs micro definition of its total score, remain unknown.
- Dataset (not code) licenses for HICO-DET, SWiG/imSitu, PSG, CrowdPose, CrowdHuman, PANDA, VCR, Sherlock, VAW, Places365, GQA, RefCOCO and Localized Narratives were **not verified this session** and need checking before redistribution.
