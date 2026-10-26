# Dense / Crowded-Scene Counting Datasets for Fine-tuning Small VLMs toward VisualOverload

Scope: labelled counting datasets for SFT and RL of Qwen2.5-VL 3B/7B. The target is the VisualOverload counting split: 559 freeform questions on dense, roughly 4K-pixel public-domain paintings, mostly with counts of 0–80 and often limited to part of the scene. Research done 2026-10-15. Items marked **UNVERIFIED** could not be confirmed from a primary source during this pass.

---

## Target benchmark recap: what VisualOverload counting needs

### Takeaway
VisualOverload counting means dense paintings at about 4K resolution, freeform integer answers, and questions about the whole scene or one region. Models are accurate at low counts and undercount sharply as the true count rises. Training data therefore needs counts well above 10, high-resolution inputs and spatially constrained questions. Domain similarity to paintings is a bonus.

### Cited Findings
- VisualOverload has 2,720 QA pairs over 150 high-resolution scans of public-domain paintings taken from Google Arts & Culture. The artists died more than 100 years ago. Images are downsampled to about 4K total pixels with the aspect ratio kept; 28 images were already below 4K but all are above Full HD. — [VisualOverload arXiv v2](https://arxiv.org/html/2509.25339v2)
- 559 of the 2,720 questions are counting questions with freeform answers. They cover the whole scene or a spatial region and need mild reasoning, e.g. "How many roses are lying on the floor?" — [VisualOverload arXiv v2](https://arxiv.org/html/2509.25339v2)
- Ground truth is kept private. Only images and questions are released, scoring goes through a rate-limited evaluation server, and the leaderboard is opt-in. There is no training or dev split, and the authors say the tasks should be solvable without finetuning. — [VisualOverload arXiv v2](https://arxiv.org/html/2509.25339v2)
- Counting accuracy: Gemini 2.0 Flash 41.7% (best), o4-mini 38.3%, o3 36.7%, InternVL3-38B 35.4%, Qwen2.5-VL-72B 35.1%, PaliGemma 2 3B 20.4%. Models are accurate at low counts, but errors rise sharply with the true count and models tend to undercount. Some errors are refusals such as "too many objects to count". Allowing a 10% tolerance raises average accuracy by only 1.6 points. Small counts are the most common answers. — [VisualOverload arXiv v2](https://arxiv.org/html/2509.25339v2)
- Accepted at CVPR 2026. On the hardest split, the best model (o3) reaches 19.6% (19.8% on the CVPR page); overall accuracy is 69.5%. — [CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/36243); [arXiv 2509.25339](https://arxiv.org/pdf/2509.25339)

### Inferences
- Because the main failure is undercounting at medium-to-high counts, the most valuable training data has many examples with counts of about 10–80. TallyQA, CountBench and the PixMo-Count val/test splits cap at 10–15, so they add little in that range.
- With no VisualOverload dev split, any held-out tuning has to use proxy sets such as HoloCount's high-density subset, CountQA or PixMo-Count test.

### Gaps
- The exact count distribution of the 559 counting questions is not published. The 0–80 range comes from the assignment brief and was not verified in the paper.

---

## (a) Counting VQA datasets built for VLMs: size, labels, licence, fit

### Takeaway
**PixMo-Points (counting subset)** and **PixMo-Count** are the best ready-made VLM counting sets. They are ODC-BY, have per-object points and in PixMo-Points reach about 1.78k points per image, which suits both point-then-count SFT and verifiable RL. **TallyQA** (Apache-2.0, 288K questions) contributes attribute and relation constraints but caps at 15. **CountBench/CountBenchQA** (counts 2–10) and **CLEVR** are low-count or synthetic and suit evaluation or format warm-up, not dense counting.

### Cited Findings
**PixMo-Count (Ai2 / Molmo, 2024)**
- 36.9k train rows; 540 validation and 540 test rows. Train rows include per-object (x,y) points; val/test rows have only counts. — [HF allenai/pixmo-count](https://huggingface.co/datasets/allenai/pixmo-count)
- The count field allows 0–255, but val/test contain only counts of 2–10. There are 365 distinct object labels. — [HF allenai/pixmo-count](https://huggingface.co/datasets/allenai/pixmo-count)
- Labels were made by running the Detic detector on web images and then filtering. Val/test were human-verified, so train labels are machine-generated and filtered, not human. — [HF allenai/pixmo-count](https://huggingface.co/datasets/allenai/pixmo-count)
- Images are distributed as URLs, mostly Flickr, with SHA-256 hashes; they must be downloaded and some links may have died. Licence is ODC-BY-1.0, intended for research and educational use. — [HF allenai/pixmo-count](https://huggingface.co/datasets/allenai/pixmo-count)

**PixMo-Points (Ai2 / Molmo, 2024)**
- About 2.38M rows (2,376,222), each an image URL, a label and point coordinates. The collection method is either "counting" (high-frequency counting) or "pointing". — [HF allenai/pixmo-points](https://huggingface.co/datasets/allenai/pixmo-points)
- Points per image range from 0 to about 1,780. Labels can be referring expressions such as "Book on right bookshelf", so spatially constrained count questions can be built directly. — [HF allenai/pixmo-points](https://huggingface.co/datasets/allenai/pixmo-points)
- ODC-BY-1.0. Images come from many web domains via URL. Spawning.ai reports about 2.9% of elements opted out. — [HF allenai/pixmo-points](https://huggingface.co/datasets/allenai/pixmo-points)

**Molmo 2 data (Ai2, Dec 2025)**
- Ai2 released nine new open datasets with Molmo 2, totalling more than 9M multimodal examples across captioning, QA, grounding, tracking and multi-image reasoning. All models, data and evaluation tools are on GitHub and HF. — [Business Wire release](https://secure.businesswire.com/news/home/20251216910167/en/Ai2-Releases-Molmo-2-State-of-the-Art-Open-Multimodal-Family-for-Video-and-Multi-Image-Understanding)
- Molmo2 beats Qwen3-VL on video counting (35.5 vs 29.6). — [Molmo2 arXiv 2601.10611](https://arxiv.org/html/2601.10611v4)

**TallyQA (AAAI 2019)**
- 287,907 counting questions: 211,430 simple and 76,477 complex. For comparison, HowMany-QA has 106,356 (68,956 simple / 37,400 complex), VQA2 has 113,254 and TDIUC has 164,762. — [TallyQA paper, Table 1](https://arxiv.org/pdf/1810.12440)
- Splits: Train has 249,318 questions on 132,981 images (COCO + Visual Genome). Test-Simple has 22,991 questions on 18,411 images. Test-Complex has 15,598 human-written AMT questions on 14,051 images. Both test splits use only Visual Genome images, none of which appear in training. — [TallyQA paper, Table 2](https://arxiv.org/pdf/1810.12440)
- Imported questions begin "How many..." and have whole-number answers from **0 to 15**. Complex questions involve attributes and relationships (e.g. "How many men are wearing glasses?"), and zero-answer "absurd" questions were added deliberately. — [TallyQA paper](https://arxiv.org/pdf/1810.12440)
- The repository is Apache-2.0, ships QA JSON with an `issimple` flag, and bundles HowMany-QA.zip. Images must be fetched from COCO and VG. — [GitHub manoja328/TallyQA_dataset](https://github.com/manoja328/TallyQA_dataset)

**HowMany-QA (Trott et al., ICLR 2018)**
- Built from VQA2 and Visual Genome counting questions. TallyQA describes it as "the gold standard for open-ended counting", with 106,356 questions. — [TallyQA paper](https://arxiv.org/pdf/1810.12440)
- Distributed as `HowMany-QA.zip` in the TallyQA repository. — [GitHub manoja328/TallyQA_dataset](https://github.com/manoja328/TallyQA_dataset)

**CountBench / CountBenchQA**
- CountBench (Paiss et al., ICCV 2023): 540 LAION-400M images, 60 per number from 2 to 10, checked by hand. It is test-only. — [Teaching CLIP to Count to Ten](https://arxiv.org/pdf/2302.12066)
- At least 30 of the 540 image URLs were dead by 2024. — [CountCLIP replication](https://arxiv.org/pdf/2406.03586)
- CountBenchQA (vikhyatk HF) has 491 surviving images, each with a manual question and an answer from 2 to 10, test split only. No licence is stated on the card. It was introduced with PaliGemma. — [HF vikhyatk/CountBenchQA](https://huggingface.co/datasets/vikhyatk/CountBenchQA)

**CLEVR**
- 70,000 train images with 699,989 questions; 15,000 images each for val and test. Counting is one of several question families. CC BY 4.0. — [CLEVR official page](https://cs.stanford.edu/people/jcjohns/clevr)

**Synthetic counting benchmark (Sengupta et al., arXiv 2511.17722, v3 Apr 2026)**
- Synthetic images with counts of 0–50 in steps of 10, varying texture, colour and background. Evaluates Qwen2.5-VL, Kimi-VL-A3B and InternVL3-9B. — [arXiv 2511.17722](https://arxiv.org/pdf/2511.17722)

### Inferences
- PixMo-Points "counting" rows are the closest ready-made match for high-count, point-supervised data. Filtering to images with 10–100 points and converting labels to "How many X ...?" gives a large pool with points that can be checked in RL.
- TallyQA Test-Complex-style questions ("How many men wearing glasses") cover attribute and relation constraints that resemble VisualOverload phrasing. Because answers stop at 15, TallyQA should be a minority of the mix to avoid reinforcing a low-count prior.
- CountBenchQA and PixMo-Count val/test (counts 2–10) are better held out as sanity-check evaluations.

### Gaps
- Licence of CountBenchQA / CountBench data: none stated (**UNVERIFIED**). LAION images carry their original rights.
- The exact names, sizes and count ranges of the Molmo 2 pointing/counting datasets (for example whether an image-level "PixMo-Count v2" exists) were not confirmed. Only the press release and the arXiv mention were seen.
- Number of PixMo-Points "counting" rows (vs "pointing") and its count histogram: not shown on the card.
- No source was found for a dataset called "CountBench-QA train" or for a large image-level counting-VQA set with counts above 50 that has human labels.

---

## (b) Class-agnostic, few-shot, referring and multi-label counting datasets

### Takeaway
**FSC-147** (6,135 images, 147 classes, counts 7–3,731, mean 56, point labels plus three exemplar boxes) is the standard dense, varied-object counting set and fits the 10–80 range well. Its licence is unstated. **FSCD-LVIS**, **REC-8K** (referring expressions, which naturally give attribute- and location-constrained counts), **OmniCount-191** (multi-label, with ready VQA annotations, Open RAIL++-S) and **PairTally** (two-category, fine-grained) add complementary constraint types. **MixCount** and **CLOC** are 2026 additions.

### Cited Findings
**FSC-147 (CVPR 2021, "Learning To Count Everything")**
- 6,135 images, 147 categories, counts of 7–3,731 per image with a mean of 56. Every instance has a centre dot and each image has three exemplar boxes. — [FSC-147 paper](https://arxiv.org/pdf/2104.08391); [CAC survey](https://arxiv.org/pdf/2501.19184)
- Images are on Google Drive and density maps on archive.org. The code repo is MIT, and **no dataset licence is stated**. — [GitHub cvlab-stonybrook/LearningToCountEverything](https://github.com/cvlab-stonybrook/LearningToCountEverything)
- Known flaws: duplicate images and overlap between splits. FSC-133 was proposed as a corrected version. — [search summary of FSC-147 literature](https://arxiv.org/pdf/2205.10203)
- An HF mirror uses a folder named `images_384_VarV2`, which suggests images resized to about 384 px tall. This is inferred from the folder name only. — [HF isentropic/FSC147](https://huggingface.co/datasets/isentropic/FSC147/tree/main)

**FSC-147-D (CounTX, 2023)**
- Text descriptions for FSC-147 images, distributed as `FSC-147-D.json` in the MIT-licensed CounTX repository and used by CountGD for text prompts. — [GitHub niki-amini-naieni/CounTX](https://github.com/niki-amini-naieni/CounTX)
- The CountGD repo (MIT) trains on FSC-147 and releases a model, not new data. — [GitHub niki-amini-naieni/CountGD](https://github.com/niki-amini-naieni/CountGD)

**FSCD-147 and FSCD-LVIS (Counting-DETR, ECCV 2022)**
- FSCD-147 adds detection boxes to FSC-147. FSCD-LVIS is built from LVIS with 6,196 images and 377 classes; later papers report 6,195 images and 372 classes. — [Counting-DETR arXiv 2207.10988](https://arxiv.org/pdf/2207.10988)
- Code is at github.com/VinAIResearch/Counting-DETR. No dataset licence was found in search results. — [VinAI research page](https://research.vinai.io/publication/few-shot-object-counting-and-detection)

**REC-8K (CVPR 2024, Referring Expression Counting)**
- 8,011 images with 17,122 referring expressions and point annotations. Images come from FSC-147, JHU-Crowd, NWPU, VisDrone, DETRAC, CARPK, Mall and Crowd Surveillance, plus Pixabay and Unsplash. — [REC CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Dai_Referring_Expression_Counting_CVPR_2024_paper.pdf)
- The repo provides annotations only; images must be fetched from each source "due to the restriction of the original dataset license". The repo is MIT and no dataset licence is stated. — [GitHub sydai/referring-expression-counting](https://github.com/sydai/referring-expression-counting)
- One later paper reports 286,621 point annotations. — [arXiv 2510.24374](https://arxiv.org/pdf/2510.24374)
- REC-RL (arXiv 2605.16460) applies Gaussian- and range-based reward optimisation to REC. — [REC-RL](https://arxiv.org/pdf/2605.16460)

**OmniCount-191 (AAAI 2025)**
- 30,230 images, 191 categories, 1–160 instances per image (mean 10). Several classes appear per image. Annotations are points, boxes and **VQA**. Train and test categories do not overlap. — [OmniCount arXiv 2403.05435](https://arxiv.org/pdf/2403.05435)
- On HF (mirror danmoruye/OmniCount-191; original cvssp/OmniCount-191). Licence is Open RAIL++-S, which forbids monitoring individuals without safeguards. — [HF OmniCount-191](https://huggingface.co/datasets/danmoruye/OmniCount-191)

**PairTally (arXiv 2509.13939, 2025)**
- 681 high-resolution images, each containing two object categories, either distinct objects or close subtypes. 54 categories and 98 subcategories. Data and code at bbvisual/PairTally_Benchmark. — [PairTally arXiv](https://arxiv.org/pdf/2509.13939)
- On PairTally, CountGD++ scores MAE 35.27, CountGD 46.67 and Qwen2.5-VL 59.36. — [CountGD++ CVPR 2026 supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Amini-Naieni_CountGD_Generalized_Prompting_CVPR_2026_supplemental.pdf)
- CoCount (from CountEx, arXiv 2602.19432) has 97 pairs and 10,086 frames with explicit negative prompts. — [CountEx](https://arxiv.org/pdf/2602.19432)

**MixCount (arXiv 2605.18063, 2026)**
- A synthetic pipeline that generates images, fine-grained text and "pixel-perfect" counting labels for open-vocabulary mixed-object counting. It reduces MAE by 20.14% on FSC-147 and 18.3% on PairTally. The arXiv licence is CC BY 4.0. Project page: corentindumery.github.io/projects/mixcount.html. — [MixCount arXiv](https://arxiv.org/abs/2605.18063)

**CLOC / Count Anything (arXiv 2605.30846, 2026)**
- About 220K images, 619 categories and 15M instances, reorganised from public sources across general scenes, remote sensing, histopathology, microscopy, agriculture and microbiology. The model predicts points. The HF page shows no linked dataset, so release and licence are unconfirmed. — [HF paper page 2605.30846](https://huggingface.co/papers/2605.30846)

**Mind the Prompt (2024)**: a benchmark for prompt-based class-agnostic counting. — [arXiv 2409.15953](https://arxiv.org/pdf/2409.15953)

### Inferences
- FSC-147 is the best single source for medium-to-dense counts of varied object types (mean 56 sits in the middle of VisualOverload's range). Images with counts above about 150 should be filtered or capped, both to match the target range and because a VLM reading 384-px images cannot resolve them. VisualOverload is about 4K, so a resolution mismatch remains.
- REC-8K and OmniCount-191 are the best sources of constrained counting ("How many people wearing red...", "How many X on the left..."). REC-8K needs images collected from eight upstream datasets with mixed licences.
- Each FSC-147 image is labelled for one class only. Template questions must ask only about that class, because other objects in the image are uncounted, which would give wrong labels.

### Gaps
- **"CA-44"**: no dataset by this name was found. It may be a misremembered name.
- FSC-147, FSCD-LVIS and REC-8K dataset licences: **not stated / UNVERIFIED**.
- REC-8K splits and per-expression count distribution: not retrieved.
- MixCount size and count range: not on the abstract page.
- CLOC download availability: unconfirmed.

---

## (c) Crowd counting datasets with point annotations

### Takeaway
Crowd datasets give dense point labels but cover only one class (people), and most counts are far above 80 (UCF-QNRF 49–12,865; NWPU up to 20,033). Licences are mostly unstated or non-commercial. They are best used as a filtered minority (images or crops with fewer than about 100 heads), mainly to counter undercounting of crowds, which paintings contain often. JHU-Crowd++ (with weather and blur labels) and NWPU (points plus boxes) are the richest. NWPU test labels appear to be withheld.

### Cited Findings
- **ShanghaiTech** (CVPR 2016): 1,198 images, 330,165 heads. Part A has 482 dense internet images (300 train / 182 test); Part B has 716 Shanghai street images (400 / 316). Licence is listed as BSD-2-Clause only by a third-party catalogue (**UNVERIFIED**). — [hyper.ai listing](https://hyper.ai/en/datasets/14773); [Counting with Focus for Free](https://arxiv.org/pdf/1903.12206)
- **UCF-QNRF** (ECCV 2018): 1,535 images with 1,251,642 head annotations from Flickr, web search and Hajj footage. Counts are 49–12,865 per image. Split is 1,201 train / 334 test. High resolution (one listing shows 2013×2902). No licence found. — [Composition Loss paper](https://arxiv.org/pdf/1808.01050); [Relevant Region Prediction](https://arxiv.org/pdf/2005.09816); [JHU-CROWD++ paper](https://arxiv.org/pdf/2004.03597)
- **JHU-CROWD++** (TPAMI 2020): 4,372 images with 1.51M annotations, including image-level and head-level labels and weather and illumination degradations. Images come from Google, Bing and Flickr. Download is at crowd-counting.com. Mirrors list the licence as "Other", and the official terms were not found (**UNVERIFIED**). — [JHU-CROWD++ arXiv](https://arxiv.org/abs/2004.03597); [hyper.ai mirror](https://hyper.ai/en/datasets/17179)
- **NWPU-Crowd** (TPAMI 2020): 5,109 images with 2,133,375 annotated heads as points and boxes, counts of 0–20,033. Only 3,609 annotation files ship, and the test set is scored through crowdbenchmark.com, implying test labels are withheld. The code repo is MIT; no dataset licence was found. — [NWPU arXiv](https://arxiv.org/abs/2001.03360); [GitHub gjy3035/NWPU-Crowd-Sample-Code](https://github.com/gjy3035/NWPU-Crowd-Sample-Code)
- **CrowdHuman** (2018): 15,000 train / 4,370 val / 5,000 test images. Each person has head, visible-body and full-body boxes. The licence is custom (Megvii disclaimer) and the full terms were not found. — [CrowdHuman arXiv](https://arxiv.org/pdf/1805.00123); [HF mirror](https://huggingface.co/datasets/ShantyCam/crowdhuman)
- **PANDA** (CVPR 2020): gigapixel video covering about 1 km² with up to about 4k people per scene, 15,974.6k boxes and 111.8k attribute labels. The PANDA-Crowd subset has 45 images averaging about 2,714 persons each. Licence is CC BY-NC-SA 4.0. — [PANDA arXiv](https://arxiv.org/pdf/2003.04852); [GIFNet GitHub](https://github.com/THU-luvision/GIFNet)
- **CrowdVLM-R1 data** (HF yeyimilk/CrowdVLM-R1-data and -BiggerDS): 3,803 training images across sheep (aerial), GTAV heads, street pedestrians, wheat heads (GWHD) and cars (TRANCOS), plus a manatee OOD test set. Train counts range from 1 to 3,546; sheep, wheat and cars have means of 31–45. — [CrowdVLM-R1 arXiv v2](https://arxiv.org/html/2504.03724v2); [GitHub yeyimilk/CrowdVLM-R1](https://github.com/yeyimilk/CrowdVLM-R1)

### Inferences
- Tiling high-resolution crowd images (UCF-QNRF, JHU, NWPU) into crops and counting the points inside each crop gives exact labels in the 0–80 range. It also produces natural spatially constrained questions such as "How many people are in the top-left quadrant?". That makes these sets useful despite their huge per-image counts.
- Their head-point annotations transfer to "How many people / figures" questions in crowded genre paintings, but not to other object types.

### Gaps
- UCF-CC-50 (50 images) was not researched. It is too small to matter for SFT.
- Official licences for ShanghaiTech, UCF-QNRF, JHU-CROWD++ and NWPU: **UNVERIFIED**. Assume research-only until checked.
- Whether NWPU test labels have since been released: not confirmed.

---

## (d) Dense detection datasets whose boxes can generate count questions

### Takeaway
Exhaustively labelled detection sets (LVIS, Objects365, SKU-110K, CARPK, VisDrone, DOTA, CrowdHuman, DEArt) can be turned into "How many X [in region R / with attribute A]?" questions with exact answers, provided annotation is complete for the queried class. SKU-110K (about 147 boxes per image, one class) and VisDrone/DOTA (small dense objects) cover the high-count tail. LVIS and Objects365 give category diversity. Most carry non-commercial or custom licences.

### Cited Findings
- **SKU-110K** (CVPR 2019): 11,762 images of shelves with about 1.73M boxes. Split is 8,219/588/2,936 per Ultralytics, or 8,233/588/2,941 per HyperAI. Licence is academic and non-commercial (proprietary, by Trax) per Voxel51; other mirrors disagree. — [Voxel51 card](https://docs.voxel51.com/dataset_zoo/datasets_hf/sku110k_test.html); [Dataset Ninja](https://datasetninja.com/sku110k); [Ultralytics](https://docs.ultralytics.com/ko/datasets/detect/sku-110k)
- **CARPK** (ICCV 2017): nearly 90,000 cars in four parking lots, drone images at about 40 m, one box per car. National Taiwan University designates it public; exact licence not found. — [Visual counting survey](https://arxiv.org/pdf/2206.03033); [CARPK at TIB](https://service.tib.eu/ldmservice/dataset/carpk)
- **Objects365**: about 15.8 boxes per image versus about 1.1 for ImageNet; roughly 2M images and over 30M boxes. Sources conflict on the licence (non-commercial vs research-use). — [Multimodal DL survey](https://arxiv.org/pdf/2301.04856); [Ultralytics Objects365](https://docs.ultralytics.com/datasets/detect/objects365/)
- **LVIS**: 1,203 long-tail categories, about 2M masks, about 164K COCO images. Licence is "custom" per Dataset Ninja and "free for research" per Voxel51; CC BY 4.0 was not confirmed. — [LVIS CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gupta_LVIS_A_Dataset_for_Large_Vocabulary_Instance_Segmentation_CVPR_2019_paper.pdf); [Voxel51 LVIS](https://docs.voxel51.com/dataset_zoo/datasets_hf/lvis.html)
- **DOTA**: images and annotations are "for academic purposes only, commercial use prohibited", and Google Earth terms also apply. This comes from a secondary source (the torchgeo iSAID mirror). — [HF torchgeo/isaid](https://huggingface.co/datasets/torchgeo/isaid)
- **VisDrone**: licence metadata conflicts between CC BY-SA 3.0 and CC BY-NC-SA 3.0. — [Voxel51 VisDrone](https://docs.voxel51.com/dataset_zoo/datasets_hf/visdrone2019_det.html); [HF VisDrone mirror](https://huggingface.co/datasets/ObversaryStudiosOrg/VisDrone2019-DET)
- TreeVGR built 38K multiple-choice counting RL problems from VisDrone, keeping only samples with true counts of 5–10. — [TreeVGR arXiv 2507.07999](https://arxiv.org/pdf/2507.07999)

### Inferences
- LVIS uses federated (non-exhaustive) labelling: some categories are not exhaustively annotated in some images. Count questions should only be generated for categories marked exhaustive in that image, or labels will be wrong. This is general knowledge about LVIS design and was not re-verified here.
- Objects365 and LVIS boxes support spatial constraints (box-centre-in-region) and size-based constraints for free. Attribute constraints need an extra captioner or VLM to verify.
- Remote-sensing (DOTA/DIOR) and retail (SKU-110K) images look very different from paintings and are useful mainly for raising high-count robustness, not for domain match.

### Gaps
- DIOR stats and licence were not researched.
- Official licences for Objects365, LVIS, CARPK and VisDrone: **UNVERIFIED** (aggregator sources conflict).
- VSR (Visual Spatial Reasoning) is a true/false spatial-relation dataset, not a counting set. Not pursued.

---

## Art-domain counting/detection data and image-overlap risk with VisualOverload

### Takeaway
No public art-specific counting dataset was found. The closest art-domain sources are detection sets. **DEArt** (15k+ European paintings from the 12th–18th century, boxes for all instances of 69 classes, about 80% non-iconic) can generate count questions on paintings and is the best domain match. It also carries the **highest overlap risk**, since it covers the same public-domain European paintings that Google Arts & Culture hosts. **PeopleArt** (4,821 images, person class only) is a smaller option. Web-scraped sets (PixMo, CountBench/LAION, REC-8K's Pixabay/Unsplash, HoloCount) carry a small, incidental risk of containing reproductions of the same paintings.

### Cited Findings
- **DEArt** (ECCV-W 2022): over 15,000 images, about 80% non-iconic, with manual boxes for all instances of 69 classes and 12 poses for human-like figures. More than 50 classes are specific to cultural heritage. A 2025 extension in the Journal of Cultural Heritage adds automatically generated relationship labels. The arXiv page shows CC BY 4.0 for the paper; the dataset licence and download location were not found on that page. — [DEArt arXiv](https://arxiv.org/abs/2211.01226); [DEArt PDF](https://arxiv.org/pdf/2211.01226)
- **PeopleArt** (Westlake et al., 2016): 4,821 annotated artwork images, person class only. — [search summary citing DEArt related work](https://arxiv.org/pdf/2211.01226)
- The Museo del Prado is working with Sherpa.ai on AI counting of people and objects in its paintings. Its public survey covers scenes such as Codazzi's amphitheatre (about 1,827 people). No labelled dataset release was found. — [Artnet News](https://news.artnet.com/art-world/counting-prado-artificial-intelligence-2629953)
- VisualOverload images are public-domain paintings from museums worldwide, accessed through Google Arts & Culture. — [VisualOverload arXiv v2](https://arxiv.org/html/2509.25339v2)
- Crowd datasets contain paintings only by accident. One 50-image crowd set mentioned in a patent includes one painting and one abstract crowd image. — [USPTO patent 9946952](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9946952)

### Inferences
- **Overlap risk ranking** (inferred): DEArt and other European-painting sets are high risk. Web-URL sets (PixMo-Count/Points, CountBench/LAION, HoloCount's internet images, FSC-147's web images) are low but nonzero risk. Photo-only sets (COCO/VG/TallyQA, crowd sets, SKU-110K, CARPK, DOTA, VisDrone) are negligible risk. If DEArt is used, deduplicate by perceptual hash against the 150 released VisualOverload images, which are public even though answers are not.
- VisualOverload answers are private, so training on the same images would not leak labels directly. But same-image training would weaken any claim of generalisation in a paper, so deduplication is still advisable.

### Gaps
- DEArt licence, download host (Zenodo or similar) and image sources (Europeana / Wikimedia / museums): **UNVERIFIED**.
- IconArt, ArtDL and other art detection sets were not checked for completeness of instance labels.
- Whether PixMo or FSC-147 contain painting reproductions was not measured.

---

## Recent (2025–2026) counting benchmarks and RL work: SFT/RL fit and reward design

### Takeaway
Integer counts give a cheap verifiable reward, but **0/1 exact-match GRPO can underperform SFT**. Graded rewards work better: CrowdVLM-R1's fuzzy relative-error reward, or Perception-R1's Hungarian-matched point F1 on PixMo-Count. Pointing before counting helps generalisation. New 2025–2026 benchmarks (CountQA, HoloCount, UNICBench, CAPTURe, PairTally) are evaluation-only, but HoloCount's high-density subset and CountQA are good stand-ins for VisualOverload.

### Cited Findings
**RL and SFT methods**
- **CrowdVLM-R1** (Qwen2.5-VL 3B/7B): reward r = r_f + r_p, where r_p = 1.5 − |ŷ−y|/y if relative error < 0.5, else 0. Overall MAE: 3B SFT 29.89, 3B 0/1 GRPO 32.44, 3B FGRPR 25.16; 7B SFT 27.52, 7B 0/1 GRPO 38.81, 7B FGRPR 24.07. Standard 0/1 GRPO underperforms SFT. — [CrowdVLM-R1 arXiv v2](https://arxiv.org/html/2504.03724v2)
- **Perception-R1** (NeurIPS 2025): GRPO with an F1 counting reward after Hungarian matching of predicted points. +17.9% on PixMo-Count. In ablation, PixMo val scores 78.1 with the full method, 77.1 without reward matching and 60.2 without RL. The base model is Qwen2-VL-2B in the NeurIPS abstract and Qwen2.5-VL-3B in arXiv v1 (conflicting). — [Perception-R1 arXiv](https://arxiv.org/pdf/2504.07954); [NeurIPS abstract](https://papers.nips.cc/paper_files/paper/2025/hash/880ee6e23b6ee548ebcccafe0ef0f513-Abstract-Conference.html)
- **Waterloo explainable counting** (Qwen2.5-VL, LoRA + GRPO): trained on TallyQA simple questions augmented with object centroids, using format, count and spatial-precision rewards. Accuracy: 67.94% GRPO vs 59.93% SFT vs 34.84% base. Single-reward training led to reward exploitation. — [UWaterloo VSL](https://openjournals.uwaterloo.ca/index.php/vsl/article/view/7124)
- **"Getting to the Point"** (arXiv 2603.21746): point-then-count (output coordinates, then the count) beat direct counting and improved OOD generalisation on synthetic and real data. Points were correctly grounded in more than 94% of cases by F1, with spatial biases across image regions. — [arXiv 2603.21746](https://arxiv.org/abs/2603.21746)
- **TreeVGR**: 38K VisDrone multiple-choice counting RL items with counts of 5–10. — [arXiv 2507.07999](https://arxiv.org/pdf/2507.07999)
- **REC-RL** (arXiv 2605.16460): Gaussian- and range-based rewards for referring expression counting. — [REC-RL](https://arxiv.org/pdf/2605.16460)

**Evaluation-only benchmarks (2025–2026)**
- **CountQA** ("How Well Do MLLMs Count in the Wild?", arXiv 2508.06585): over 1,500 QA pairs on real images with high density, clutter and occlusion. Best of 15 MLLMs scores 42.9%, and accuracy falls as counts rise. The release was promised on acceptance; a later index page links an HF dataset (release status unconfirmed). — [CountQA arXiv](https://arxiv.org/abs/2508.06585); [HF papers search](https://huggingface.co/papers?q=Object+counting)
- **HoloCount** (arXiv 2607.06420): 2,480 QA pairs, 20 subsets and 1,481 concepts. Images come from the internet, detection datasets, specialised benchmarks and synthesis. Licence CC BY 4.0 on HF (single "train" split, intended as a benchmark). Counts reach about 149 in the dense and aerial subsets. In the high-density subset most models score below 10% and Qwen3.5-27B scores 20.0%. Extended thinking adds 10.6–15.4 points. — [HoloCount project](https://mm-mvr.github.io/HoloCount/); [HF MM-MVR/HoloCount](https://huggingface.co/datasets/MM-MVR/HoloCount); [arXiv 2607.06420](https://arxiv.org/html/2607.06420v1)
- **UNICBench** (CVPR 2026): 5,300 images with 5,508 QA, plus text and audio counting parts. — [search summary of CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Rong_UNICBench_UNIfied_Counting_Benchmark_for_MLLM_CVPR_2026_paper.pdf)
- **CAPTURe** (ICCV 2025): counting of occluded patterns. CAPTURe-real draws on FSC-147 and there is also a synthetic split. GPT-4o, InternVL2, Molmo and Qwen2-VL all degrade under occlusion. — [CAPTURe arXiv](https://arxiv.org/abs/2504.15485)
- **COUNTS** (arXiv 2504.10158): detectors and MLLMs under distribution shift. — [arXiv 2504.10158](https://arxiv.org/abs/2504.10158)
- **Object Counting Across Modalities** (arXiv 2608.23845): a 2026 survey of counting taxonomies and benchmarks, useful for further sweeps. — [arXiv 2608.23845](https://arxiv.org/pdf/2608.23845)

### Inferences
- **Recommended mix for VisualOverload** (inferred):
  1. Primary: PixMo-Points counting rows and PixMo-Count train, filtered toward 5–100 objects, as point-then-count SFT. Points also let RL use a Perception-R1-style matched-F1 reward on top of a count reward.
  2. FSC-147 / FSCD-LVIS for dense, varied objects (counts 10–80, after dropping test images and FSC-133 duplicates).
  3. REC-8K and OmniCount-191 for constrained counts; TallyQA complex questions for attribute and relation phrasing (low counts only, small share).
  4. Crowd-dataset crops for "how many people/figures" at 20–80.
  5. Optionally DEArt-derived counts for the painting domain, after deduplication against VisualOverload images.
- For RL, use a graded reward (relative-error or Gaussian) plus format, instead of exact match. Bias sampling toward counts of 15 or more to target the undercounting failure.
- Use HoloCount's high-density subset, CountQA and the PixMo-Count test split as held-out proxies. None of them shares images with VisualOverload as far as known.
- Licensing: only CLEVR (CC BY 4.0), TallyQA QA (Apache-2.0), PixMo (ODC-BY), HoloCount (CC BY 4.0) and OmniCount (Open RAIL++-S) have clearly stated permissive or open terms. Most crowd and detection sets are research-only or unknown, which is fine for academic ACL/CVPR work but should be noted in a paper's data statement.

### Gaps
- A search snippet claimed that MGPO (arXiv 2507.05920) trained Qwen2.5-VL-7B on FSC-147 and PixMo-Count, with GRPO at 81.0 vs SFT at 72.7. The arXiv abstract does not mention counting, so this is **UNVERIFIED** and should not be cited.
- UNICBench count ranges, sources and licence: the PDF returned HTTP 403.
- CountQA image count, count range and licence after release: not confirmed.
- No dataset or paper was found that fine-tunes on VisualOverload-like paintings and reports VisualOverload counting gains.
