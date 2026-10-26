# Scene-text / in-image OCR datasets for fine-tuning Qwen2.5-VL (3B/7B) on the VisualOverload OCR subset

Status: compiled 2026-10-15. Each claim below cites the page it was checked against. "Unverified" means I could not find the license or the fact on a primary page during this session. Where only third-party catalogues (hyper.ai, Dataset Ninja, HF mirrors) were available, that is stated.

## Q0. What the target (VisualOverload OCR) actually requires (context for every match judgement)

### Takeaway
The VisualOverload OCR subset has 118 freeform questions on public-domain paintings of about 4K pixel count. It is scored by exact match after aggressive normalization (no diacritics, no punctuation, no spaces, lowercase, Latin V→U and J→I). Edit distance is also reported. Accuracy correlates strongly with input resolution. So the useful training signal is verbatim reading of small, rare-word text in large cluttered images, plus picking out *which* text the question means. Broad OCR volume matters less.

### Cited Findings
- There are 118 freeform OCR questions out of 2,720 total. The text is in English, Latin, Chinese, Dutch and Greek. Examples include "What is the last name of the signature?" and "What does the word below the main character read?" — [VisualOverload arXiv HTML](https://arxiv.org/html/2509.25339)
- **Metric:** accuracy after answer extraction. Normalization removes diacritics, punctuation and spacing, lowercases, and maps V→U and J→I for Latin. Normalized Levenshtein edit distance is also reported. — [VisualOverload arXiv HTML](https://arxiv.org/html/2509.25339)
- **Reported OCR accuracy / NED:**

  | Model | Accuracy | NED |
  |---|---|---|
  | Qwen2.5-VL 72B | 72.9% | 0.16 |
  | Qwen2.5-VL 32B | 61.0% | 0.23 |
  | o3 | 61.0% | 0.24 |
  | InternVL3 38B | 45.8% | 0.33 |
  | LLaVA 1.5 7B | 3.4% | 0.76 |

  The prose names o4-mini as best, but Table 1 has Qwen2.5-VL 72B higher. — [VisualOverload arXiv HTML](https://arxiv.org/html/2509.25339)
- **Failure modes:**
  - Hallucinations and extracting irrelevant text.
  - Reading-order errors in side-by-side multi-line paragraphs and banners.
  - Autocorrection toward probable words, e.g. "accidunt" → "accident".
  - Incorrect answers have NED centred around 0.7.

  — [VisualOverload arXiv HTML](https://arxiv.org/html/2509.25339)
- **Other findings in the paper:**
  - OCR accuracy correlates strongly with input resolution (App. D).
  - Chain-of-thought raised InternVL3-38B OCR from 45.8% to 50.0% (App. F).
  - OCR questions form the most disconnected cluster in question-embedding space (App. G).

  — [VisualOverload arXiv HTML](https://arxiv.org/html/2509.25339)
- **Images:**
  - 150 public-domain painting scans from Google Arts & Culture, downloaded with dezoomify-rs in April–May 2025.
  - Downsampled to roughly 4K pixel count. 28 images below 4K were not upscaled, but all exceed Full HD.
  - Table 4 lists every artwork with metadata and links.

  — [VisualOverload arXiv HTML](https://arxiv.org/html/2509.25339)
- **Dataset release:**
  - The HF dataset is CC BY-SA 4.0, built on CC0 / public-domain artwork, with a single `test` split.
  - Ground-truth answers are private; the benchmark is leaderboard-evaluated.

  — [HF README](https://huggingface.co/datasets/paulgavrikov/visualoverload/blob/main/README.md); [arXiv PDF](https://arxiv.org/pdf/2509.25339)

### Inferences
- Rewards should copy VisualOverload's normalization exactly (strip diacritics, punctuation and spaces; lowercase; V→U, J→I). Otherwise RL optimizes a different target than the benchmark measures.
- Only 118 questions means about 1 question is about 0.85 points of OCR accuracy. Any fine-tuning gain needs bootstrap confidence intervals.

### Gaps
- The per-language breakdown of the 118 questions (how many Latin vs. Chinese vs. Greek) was not found in the parts of the paper I read (characters 100k–120k of the HTML were not read).

---

## Q1. Catalogue: name, venue/year, link, size, resolution, languages

### Takeaway
Labelled data with direct relevance is plentiful for English and Chinese *scene* text. Examples are TextOCR, HierText, CTW, Uber-Text, ST-VQA, TextVQA, LSVT, ArT and MLT-2019. For *historical, inscription and art* text there is only a thin layer of small, mostly non-VQA datasets: CATMuS Medieval, TRIDIS, Bessarion, EpiSignum, Ancient-Bench, MCCD, the VOC Dutch HTR ground truth and MapText. I found no public, labelled "text in paintings" or artist-signature OCR dataset.

### Cited Findings

#### (a) Scene-text VQA (freeform answers)

**TextVQA v0.5.1** (CVPR 2019)
- 45,336 questions on 28,408 OpenImages images.
- Splits: 34,602 train / 5,000 val / 5,734 test.
- License listed as CC BY 4.0 on the TFDS/HF loader, not on the authors' page.
- Sources: [TFDS catalog](https://www.tensorflow.org/datasets/community_catalog/huggingface/textvqa?hl=it); [HF loader script](https://huggingface.co/datasets/textvqa/blob/main/textvqa.py)

**ST-VQA** (ICCV 2019 / ICDAR 2019 competition)
- 23,038 images and 31,791 QA pairs.
- Images come from ICDAR 2013/2015, ImageNet, VizWiz, IIIT-STR, Visual Genome and COCO-Text.
- Answers are always grounded in text present in the image.
- License not found (unverified).
- Sources: [arXiv 1905.13648](https://arxiv.org/pdf/1905.13648); [arXiv 1907.00490](https://arxiv.org/pdf/1907.00490)

**EST-VQA** (CVPR 2020)
- The first bilingual (English + Chinese) scene-text VQA dataset.
- About 25,239 images and 28,158 questions, per a secondary CVC slide deck.
- Hosted at est-vqa.org. License unverified.
- Sources: [CVPR paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_On_the_General_Value_of_Evidence_and_Bilingual_Scene-Text_Visual_CVPR_2020_paper.pdf); [DAS2022 slides](https://iapr.org/archives/das2022/wp-content/uploads/OS-slides/48.pdf)

**MTVQA** (ACL 2025 Findings; ByteDance)
- 8,794 images: 6,678 train / 2,116 test.
- 21,829 train QA and 6,778 test QA, written by native annotators.
- 9 languages: KO, JA, IT, RU, DE, FR, TH, AR, VI. No English, Chinese, Latin, Greek or Dutch.
- License CC BY-NC-4.0, "Academic use only".
- Sources: [HF card](https://huggingface.co/datasets/ByteDance/MTVQA); [arXiv 2405.11985](https://arxiv.org/html/2405.11985v3)

**MUST-VQA** (2022)
- Multilingual scene-text VQA covering English, Catalan, Spanish and Chinese.
- Source: [arXiv 2209.06730](https://arxiv.org/pdf/2209.06730)

**OCR-VQA** (ICDAR 2019)
- 207,572 book-cover images and more than 1M template QA (title, author, year, genre).
- License unverified.
- Source: [LaTr paper summary](https://arxiv.org/pdf/2112.12494); [IITJ page](https://research.iitj.ac.in/publication/ocr-vqa-visual-question-answering-by-reading-text-in-images)

**TextCaps** (ECCV 2020)
- About 145k captions on about 28k TextVQA images.
- Captions, not VQA. License unverified.
- Source: [LaTr paper summary](https://arxiv.org/pdf/2112.12494)

**ViTextVQA**
- Vietnamese; more than 16,000 images and more than 50,000 QA.
- "Available for research purposes" via GitHub.
- Source: [arXiv 2404.10652](https://arxiv.org/html/2404.10652v3)

**ViOCRVQA**
- 28,282 Vietnamese book covers and 123,781 QA.
- Source: [arXiv 2404.18397](https://arxiv.org/html/2404.18397v1)

**ViSignVQA** (Dec 2025)
- Signboard-oriented Vietnamese VQA.
- Source: [arXiv 2512.22218](https://arxiv.org/pdf/2512.22218)

**SEA-Vision** (CVPR 2026)
- 11 Southeast Asian languages.
- 7,496 text-centric VQA pairs plus 15,234 document-parsing pages.
- Source: [arXiv 2603.15409](https://arxiv.org/abs/2603.15409); [CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/39901)

#### (b) Scene-text detection/recognition with transcriptions (convertible to "read the text at/near X" or full-transcription prompts)

**HierText** (Google; CVPR 2022)
- 11,639 Open Images images (8,281 train / 1,724 val / 1,634 test) and about 1.2M words, more than 100 words per image.
- Long side capped at 1600 px.
- Hierarchical paragraph → line → word labels with legibility, handwritten and vertical flags. Words within a line follow reading order; line order within a paragraph is not guaranteed.
- Test GT released in December 2024.
- License CC BY-SA 4.0.
- Source: [GitHub](https://github.com/google-research-datasets/hiertext)

**TextOCR** (Meta; CVPR 2021)
- 28,134 TextVQA images and about 903k words, about 32 words per image.
- Splits: 21,778 train / 3,124 val / 3,232 test.
- License CC BY 4.0, per the HF mirror and hyper.ai; the official page did not render.
- Sources: [arXiv 2105.05486](https://arxiv.org/pdf/2105.05486); [HF mirror](https://huggingface.co/datasets/yunusserhat/TextOCR-Dataset); [hyper.ai](https://hyper.ai/en/datasets/20131)

**COCO-Text** (2016)
- 63,686 MS-COCO images.
- The paper reports 173,589 text regions; the official page lists 145,859 instances. Counts conflict.
- Annotations are CC BY 4.0; a v2 release exists.
- Sources: [arXiv 1601.07140](https://arxiv.org/pdf/1601.07140); [Cornell page](https://vision.cornell.edu/se3/coco-text-2/)

**Uber-Text** (CVPR 2017 workshop)
- 117,969 Bing Streetside images from 6 US cities and 571,534 line-level polygons.
- Split into 1K and 4K subsets — one of the few native high-resolution sets.
- 197 GB. CC BY-SA 4.0.
- Sources: [Uber-Text index](https://s3-us-west-2.amazonaws.com/uber-common-public/ubertext/index.html); [Uber research page](https://www.uber.com/blog/research/uber-text-a-large-scale-dataset-for-optical-character-recognition-from-street-level-imagery); [real-data survey 2007.03098](https://arxiv.org/pdf/2007.03098)

**CTW — Chinese Text in the Wild** (Tsinghua/Tencent; JCST 2019)
- 32,285 Tencent Street View images and 1,018,402 characters (3,850 distinct).
- Per-character boxes plus attributes: occluded, complex background, distorted, 3D-raised, **wordart**, **handwritten**.
- License listed only as "Other" (unverified).
- Sources: [JCST](https://jcst.ict.ac.cn/en/article/pdf/preview/10.1007/s11390-019-1923-y.pdf); [ar5iv 1803.00085](https://ar5iv.arxiv.org/html/1803.00085); [hyper.ai](https://hyper.ai/datasets/20542)

**ICDAR2019-LSVT** (Chinese street view)
- About 450k images: 50k fully annotated (30k train / 20k test) and 400k weakly annotated (text only).
- **Test labels not released**; evaluation via rrc.cvc.uab.es ch=16.
- A recognition variant has 210k labelled crops.
- Source: [PaddleOCR dataset list](https://www.paddleocr.ai/latest/datasets/datasets.html)

**ICDAR2017 RCTW-17**
- More than 12,000 Chinese images: street, posters, menus, screenshots.
- Source: [PaddleOCR dataset list](https://www.paddleocr.ai/latest/datasets/datasets.html)

**ICDAR2019-ArT** (arbitrary-shaped)
- 10,166 images (5,603 train / 4,563 test).
- Built from Total-Text plus SCUT-CTW1500 (3,055 images) plus 7,111 new images.
- Total-Text was re-annotated to add Chinese.
- Sources: [arXiv 1909.07145](https://arxiv.org/pdf/1909.07145); [PaddleOCR list](https://www.paddleocr.ai/latest/datasets/datasets.html)

**Total-Text** (ICDAR 2017 / IJDAR 2020)
- 1,555 images and 9,330 words (horizontal, multi-oriented, curved).
- BSD-3-Clause, per HF and Dataset Ninja mirrors; the LICENSE file was not seen.
- Sources: [GitHub](https://github.com/cs-chan/Total-Text-Dataset); [HF mirror](https://www.huggingface.co/datasets/Voxel51/Total-Text-Dataset/blob/main/README.md)

**SCUT-CTW1500**
- 1,000 train / 500 test and 10,751 boxes (3,530 curved), line-level.
- License "Non-Commercial" per hyper.ai (unverified).
- Sources: [arXiv 1712.02170](https://arxiv.org/pdf/1712.02170); [hyper.ai](https://hyper.ai/en/datasets/16943)

**ICDAR2019 MLT**
- 20,000 real images, 2,000 per language: Arabic, Bangla, Chinese, Devanagari, English, French, German, Italian, Japanese, Korean.
- 10k train / 10k test, plus 277k synthetic images.
- "Freely available for academic and research purposes".
- Sources: [arXiv 1907.00945](https://arxiv.org/pdf/1907.00945); [CBDAR slides](https://cbdar2019.univ-lr.fr/wp-content/uploads/2019/11/CBDAR2019_RRC-MLT-2019_CBDAR.pdf); [IIIT index](https://ilocr.iiit.ac.in/icpr_2024_wirindic/dataset.html)

**Union14M** (ICCV 2023)
- Union14M-L is 4M labelled word crops from 17 public datasets (3.2M train); Union14M-U is 10M unlabelled.
- The Union14M-Benchmark includes **Contextless** (779 images, "no semantic meaning, out-of-dictionary words") and **Artistic** subsets.
- License inherits from the source datasets (unverified).
- Sources: [arXiv 2307.08723](https://arxiv.org/pdf/2307.08723); [OpenOCR-Data HF](https://huggingface.co/datasets/topdu/OpenOCR-Data)

**WATER-S / WATER-R** (ECCV 2026)
- WordArt-oriented scene text recognition.
- WATER-S: 2M synthetic images (SynthWordArt plus Qwen3-VL prompt mining with Z-Image synthesis).
- WATER-R: real data reorganized from existing STR sets.
- Data on GitHub YesianRohn/WATER; dataset license not stated.
- Source: [arXiv 2606.24484](https://arxiv.org/abs/2606.24484)

**ICDAR 2024 MapText** (historical maps)
- Rumsey collection: 936 map images; annotations CC BY-NC-SA 4.0.
- French Land Registers (IGN): 145 maps; license not seen; the test record lists annotations as N/A.
- Curved, rotated, spaced-out historical lettering — the closest public analogue to painted banners and cartouches.
- Sources: [Zenodo Rumsey](https://zenodo.org/records/10987299); [paper](https://weinman.cs.grinnell.edu/pubs/li24icdar.pdf); [Zenodo IGN](https://zenodo.org/records/10732281)

**Movie-Poster artistic-text dataset** (2024)
- Artistic-style text detection. Found incidentally; not examined.
- Source: [arXiv 2406.16307](https://arxiv.org/pdf/2406.16307)

#### (c) Historical, handwritten, inscription and calligraphy (Latin, Greek, Dutch, Chinese)

**CATMuS Medieval** (ICDAR 2024)
- More than 200 manuscripts and incunabula, more than 160k lines and 5M characters, 8th–16th century.
- 10 languages including **Latin** and **Middle Dutch**.
- Line image plus transcription; `gen_split` gives a 90/5/5 split by manuscript.
- CC BY 4.0.
- Sources: [HF card](https://huggingface.co/datasets/CATMuS/medieval-segmentation/blob/e813bec31fba823f185ce5d44445035299a84d13/README.md); [HAL](https://hal-enc.archives-ouvertes.fr/hal-04453952v1); [ICDAR24 PDF](https://iris.cnr.it/retrieve/58f33c23-bc7b-4e15-a3d0-e3f7275c1551/ICDAR24___CATMUS_Medieval-1.pdf)

**CATMuS Modern (McCATMuS)**
- More than 170 manuscripts in 8 languages.
- Sources: [HF commit](https://huggingface.co/datasets/CATMuS/modern/commit/3ffb4227602b0c5ccf9fbc560838611bd3a53d19.diff); [Zenodo](https://zenodo.org/records/13788177)

**TRIDIS** (2025)
- Medieval and early-modern documentary HTR plus NER, 12th–17th century.
- Latin, Old French, Middle High German, Old Spanish.
- Source: [arXiv 2503.22714](https://arxiv.org/pdf/2503.22714)

**VOC / notarial deeds ground truth** (Dutch National Archives + Noord-Hollands Archief)
- About 6,000 transcribed scans of 17th–18th c. Dutch East India Company (VOC) and 19th c. notarial deeds.
- JPG plus PAGE XML (earlier versions: ALTO word-level).
- Also about 3M pages of HTR output, which is noisy machine labels and should not be used as GT.
- License not confirmed in snippets (unverified).
- Sources: [Zenodo 4159268](https://zenodo.org/records/4159268); [Zenodo 3906480](https://zenodo.org/records/3906480)

**Bessarion** (DAS 2024)
- 122 images of 25 Byzantine-era Medieval Greek donative inscriptions.
- Line-, word- and character-level polygon plus text annotations.
- Images resized to at most 1024 px width for baselines; 80/20 split; CER/WER baselines.
- GitHub Archaeocomputers/Bessarion. License not stated in the paper (unverified).
- Source: [DAS2024 PDF](https://www.cs.uoi.gr/~sfikas/DAS2024-bessarion.pdf)

**EpiSignum** (Udine)
- Late-Antique Latin inscriptions with *symbol* detection annotations, not full transcription.
- Source: [PDF](https://ailab.uniud.it/wp-content/uploads/2025/11/EpiSignum_dataset-1.pdf); GitHub AI4CH-UniUD/EpiSignum

**LED — Latin Epigraphic Dataset** (DeepMind, 2025, Aeneas)
- 176,861 inscriptions; only about 5% have usable images.
- Published in google-deepmind/predictingthepast. License unverified.
- Source: [hyper.ai](https://hyper.ai/en/datasets/42127)

**LIST** (Latin Inscriptions in Space and Time)
- 528,484 inscriptions of text and metadata (EDH plus EDCS). Text only, no images.
- Source: [Zenodo 7870085](https://zenodo.org/records/7870085)

**Synthetic multi-script text-line dataset** (Zenodo, Feb 2025; Romanello/AjMC)
- 6.2M synthetic lines of polytonic Greek mixed with Latin-script text (FR, EN, DE, IT, Latin, Ancient Greek), including degraded lines.
- 1.3 GB sample public; full set on request. CC BY 4.0.
- Source: [Zenodo 14840349](https://zenodo.org/records/14840349)

**Pogretra and GT4HistComment**
- Ancient-Greek typeface training data and 19th-c. classical-commentary OCR ground truth.
- Source: [arXiv 2110.06817](https://arxiv.org/abs/2110.06817v1)

**Patrologia Graeca Corpus** (2026)
- Open OCR resource for 19th-c. polytonic Greek editions; CER 1.05%.
- Source: [arXiv 2603.09470](https://arxiv.org/pdf/2603.09470)

**IAM Handwriting** (v3.0)
- 657 writers, 1,539 pages, 13,353 lines, 115,320 words; English.
- Free for **non-commercial research**; registration required.
- Source: [FKI IAM page](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database); [download terms](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database)

**Ancient-Bench** (EMNLP 2026; SCUT)
- 2,700 images of ancient Chinese artifact text in 9 media (incl. oracle bone, bronze, silk, steles, calligraphy) and 7 script forms over about 3,000 years.
- On GitHub SCUT-DLVCLab/Ancient_Bench. Annotation release and license unclear.
- Source: [arXiv 2608.27169](https://arxiv.org/abs/2608.27169)

**MCCD** (ICDAR 2025)
- 329,715 isolated Chinese calligraphy character images in 7,765 classes, with script-style, dynasty and calligrapher labels.
- GitHub SCUT-DLVCLab/MCCD.
- Source: [arXiv 2507.06948](https://arxiv.org/html/2507.06948v1)

**HCSU** (2026)
- 39,307 calligraphy character images, including a 3,240-image stele-rubbing domain.
- Style understanding, not reading.
- Source: [arXiv 2607.04147](https://arxiv.org/pdf/2607.04147)

#### (d) Synthetic and large-scale text-rich corpora

**SynthText** (CVPR 2016)
- 800k images and about 8M synthetic words. License unverified.
- Source: [hyper.ai](https://hyper.ai/en/datasets/4917)

**SynthDoG** (Donut)
- English, Chinese, Japanese and Korean document-style synthetic sets, about 0.5M each.
- MIT per a third-party wiki only; the HF card was not checked.
- An Apache-2.0 copy sits inside LLaVA-OneVision-Mid-Data.
- Synthdog-Multilingual-100 (WueNLP) covers 100 languages.
- Sources: [HF synthdog-en](https://huggingface.co/datasets/naver-clova-ix/synthdog-en/tree/c08d8463ef5db9c8945f5a7fa565084fbbce2ec1); [aiwiki](https://aiwiki.ai/wiki/donut_model); [Synthdog-Multilingual-100](https://huggingface.co/datasets/WueNLP/Synthdog-Multilingual-100)

**TextAtlas5M** (2025)
- 5M dense-text images (synthetic plus real subsets such as Cover Book and PPT2Details).
- License not confirmed.
- Source: [arXiv 2502.07870](https://arxiv.org/pdf/2502.07870)

**MARIO-10M**
- 10,061,720 LAION, TMDB-poster and Open-Library-cover images with OCR annotations (pseudo-labels from TextDiffuser).
- Source: [typography review 2407.14774](https://arxiv.org/pdf/2407.14774)

#### (e) Benchmarks often asked about (labels and licensing)

**OCRBench v2** (2025)
- The HF `ling99/OCRBench_v2` dataset is a 10,000-row **test** split with answers included (multiple accepted variants), 30 task types, MIT license.
- The private set (about 1,500 images) was "released" 2025-06-21 for a quarterly leaderboard; label release is unclear.
- Sources: [HF](https://huggingface.co/datasets/ling99/OCRBench_v2); [GitHub](https://github.com/yuliang-liu/multimodalocr); [arXiv 2501.00321](https://arxiv.org/pdf/2501.00321)

**CC-OCR** (ICCV 2025)
- 7,058 annotated images in 39 subsets across 4 tracks: multi-scene, multilingual, document parsing, KIE.
- License not found.
- Sources: [arXiv 2412.02210](https://arxiv.org/html/2412.02210v3); [ICCV](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_CC-OCR_A_Comprehensive_and_Challenging_OCR_Benchmark_for_Evaluating_Large_ICCV_2025_paper.html)

**SEED-Bench-2-Plus**
- 2.3K **multiple-choice** questions over charts, maps and webs. CC BY-NC 4.0.
- Source: [HF card](https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2-plus)

**PM4Bench** (2025)
- Parallel 10-language multimodal benchmark, plus a paper doing OCR-centric GRPO on fully synthetic, label-free OCR data.
- Release of the training data not confirmed.
- Source: [arXiv 2503.18484](https://arxiv.org/abs/2503.18484)

### Inferences
- **Highest-value real data** for the VisualOverload OCR failure modes:
  - HierText: dense text, more than 100 words per image, reading-order hierarchy, CC BY-SA, all splits labelled.
  - TextOCR: CC BY 4.0, about 32 words per image.
  - Uber-Text 4K: native high resolution, CC BY-SA.
  - CTW: high-density Chinese with wordart and handwriting attributes.
  - ArT / Total-Text: curved text, close to banners.
  - MapText Rumsey: historical curved lettering, non-commercial.
- **For Latin, Greek and Dutch**, the only sizeable labelled sources are line-level manuscript/print HTR (CATMuS, TRIDIS, VOC GT) and synthetic Greek/Latin lines (Zenodo 14840349). These are document crops, not scene text. They teach verbatim non-dictionary transcription but not localisation in a cluttered 4K scene. They can be pasted or composited into painting-like scenes to produce a localisation task.
- **Anti-autocorrection signal:** Union14M Contextless (out-of-dictionary words), HTR corpora with abbreviations and medieval spellings, and inscriptions (Bessarion's "systematic misspellings") all reward exact glyph copying rather than language-model priors.

### Gaps
- No public, labelled dataset of text in paintings, artist signatures or painted banners was found. Searches for "text in paintings/artworks" and "artist signature dataset" returned only artist-identification or object-detection art datasets (DEArt, iMet, Painting91).
- Not verified this session (for these I have no checked source, only prior knowledge):
  - ICDAR2013, ICDAR2015 and ICDAR2017-MLT sizes and terms.
  - ReCTS (ICDAR 2019 signboards) size and terms.
  - The READ / ICDAR2017 HTR competition sets.
  - GT4HistOCR.
  - AnyWord-3M details.
  - LLaVAR / LLaVA-OneVision OCR subsets.
  - Transkribus public collections.
- Licenses for ST-VQA, EST-VQA, OCR-VQA, TextCaps, CTW, ArT, LSVT, SynthText, CC-OCR, Union14M, Bessarion, VOC GT and Ancient-Bench could not be confirmed from primary pages.

---

## Q2. Task / answer format per dataset

### Takeaway
Only a few sets are natively freeform VQA with verbatim string answers: TextVQA, ST-VQA, EST-VQA, MTVQA, OCR-VQA, ViTextVQA and SEA-Vision. Most others are polygon + transcription and need templating into questions. SEED-Bench-2-Plus is multiple-choice, so it is unsuitable for freeform OCR SFT.

### Cited Findings
- **Freeform VQA:**
  - ST-VQA answers are always grounded in in-image text — [arXiv 1907.00490](https://arxiv.org/pdf/1907.00490)
  - MTVQA rows are image plus a list of `qa_pairs` — [HF](https://huggingface.co/datasets/ByteDance/MTVQA)
  - OCR-VQA uses template questions about title, author and year — [LaTr](https://arxiv.org/pdf/2112.12494)
  - OCRBench v2 rows carry multiple accepted answer variants — [HF](https://huggingface.co/datasets/ling99/OCRBench_v2)
- **Hierarchical paragraph/line/word transcription with reading order inside lines:** HierText — [GitHub](https://github.com/google-research-datasets/hiertext)
- **Word or line polygon + transcription:**
  - TextOCR — [arXiv](https://arxiv.org/pdf/2105.05486)
  - Total-Text — [GitHub](https://github.com/cs-chan/Total-Text-Dataset)
  - ArT — [arXiv](https://arxiv.org/pdf/1909.07145)
  - Uber-Text (line-level, free-hand polygons) — [index](https://s3-us-west-2.amazonaws.com/uber-common-public/ubertext/index.html)
  - Bessarion (line, word and char) — [PDF](https://www.cs.uoi.gr/~sfikas/DAS2024-bessarion.pdf)
- **Per-character labels plus attributes:** CTW — [JCST](https://jcst.ict.ac.cn/en/article/pdf/preview/10.1007/s11390-019-1923-y.pdf)
- **Line image → text (HTR):**
  - CATMuS (`im`, `text`) — [HF](https://huggingface.co/datasets/CATMuS/medieval-segmentation/blob/e813bec31fba823f185ce5d44445035299a84d13/README.md)
  - VOC GT (PAGE XML) — [Zenodo](https://zenodo.org/records/4159268)
- **Multiple-choice:** SEED-Bench-2-Plus — [HF](https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2-plus)

### Inferences
- Box/polygon datasets support VisualOverload-style templates such as:
  - "What is written on the banner at the top?"
  - "What is the last word of the line below X?"
  - "Transcribe the text in region (x1,y1,x2,y2)"
- Partial-text questions ("last name of the signature") can be built from word-level labels within a line: take the last word, the first word, or the number.

### Gaps
- TextVQA's answer schema (10 human answers per question) is from prior knowledge; it was not re-verified this session.

---

## Q3. Label availability per split and license

### Takeaway
Labelled train splits with permissive licenses (CC BY / CC BY-SA / MIT / BSD):
- HierText (all splits, CC BY-SA 4.0)
- TextOCR (CC BY 4.0)
- COCO-Text (CC BY 4.0)
- Uber-Text (CC BY-SA 4.0)
- Total-Text (BSD-3, mirror-reported)
- CATMuS (CC BY 4.0)
- Synthetic Greek/Latin lines (CC BY 4.0)

Non-commercial or academic-only:
- MTVQA (CC BY-NC)
- IAM (non-commercial research)
- MapText Rumsey (CC BY-NC-SA)
- SCUT-CTW1500 ("Non-Commercial", third-party)
- SEED-Bench-2-Plus (CC BY-NC)

Hidden test labels: LSVT test; the VisualOverload answers themselves.

### Cited Findings
- HierText test GT was released in December 2024 (`gt/test.jsonl.gz`); CC BY-SA 4.0 — [GitHub](https://github.com/google-research-datasets/hiertext)
- LSVT test labels are not open-sourced; results must be submitted to RRC — [PaddleOCR](https://www.paddleocr.ai/latest/datasets/datasets.html)
- MTVQA train QA are visible; test-label status is not stated on the card; CC BY-NC 4.0 — [HF](https://huggingface.co/datasets/ByteDance/MTVQA)
- OCRBench v2 HF test split includes answers; MIT — [HF](https://huggingface.co/datasets/ling99/OCRBench_v2)
- Uber-Text is CC BY-SA 4.0 with privacy blurring of faces and plates — [index](https://s3-us-west-2.amazonaws.com/uber-common-public/ubertext/index.html)
- COCO-Text annotations are CC BY 4.0; images come from MS-COCO — [Cornell](https://vision.cornell.edu/se3/coco-text-2/)
- IAM is free for non-commercial research only — [FKI](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database)
- MapText Rumsey annotations are CC BY-NC-SA 4.0 — [Zenodo](https://zenodo.org/records/10987299)
- The Greek/Latin synthetic set is CC BY 4.0; only a sample is public, the full set on request — [Zenodo](https://zenodo.org/records/14840349)
- CATMuS Medieval is CC BY — [HF card](https://huggingface.co/datasets/CATMuS/medieval-segmentation/blob/e813bec31fba823f185ce5d44445035299a84d13/README.md)
- MLT-2019 is "freely available for academic and research purposes" — [IIIT index](https://ilocr.iiit.ac.in/icpr_2024_wirindic/dataset.html)

### Inferences
- For an academic ACL 2027 paper, non-commercial licenses are acceptable for training. However, the released fine-tuned checkpoints should state which NC data was used.

### Gaps
- TextVQA's license was confirmed only via TFDS/HF loader metadata, not textvqa.org (the page fetch failed).
- The Total-Text BSD license was not seen in the repo itself.

---

## Q4. Match to VisualOverload OCR (small text in dense scenes, art/historical text, verbatim reading)

### Takeaway
No dataset matches on all three axes. The best partial matches:
- **Density, small text and reading order:** HierText, TextOCR, Uber-Text 4K, CTW.
- **Historical or curved lettering in images:** MapText, ArT, Total-Text.
- **Verbatim rare-word / Latin / Greek / Dutch reading:** CATMuS, TRIDIS, VOC GT, Bessarion, synthetic polytonic Greek/Latin, Union14M Contextless.
- **Chinese:** CTW, LSVT, ArT-Chinese, MCCD/Ancient-Bench (calligraphy).

### Cited Findings
- HierText has more than 100 words per image, is the densest public scene-text set, and has line/paragraph reading-order structure — [GitHub](https://github.com/google-research-datasets/hiertext)
- Uber-Text provides a 4K subset — [index](https://s3-us-west-2.amazonaws.com/uber-common-public/ubertext/index.html)
- CTW characters are tagged wordart / handwritten / 3D-raised / distorted, which parallels painted or carved lettering — [JCST](https://jcst.ict.ac.cn/en/article/pdf/preview/10.1007/s11390-019-1923-y.pdf)
- Union14M Contextless targets out-of-dictionary words — [arXiv 2307.08723](https://arxiv.org/pdf/2307.08723)
- Bessarion inscriptions feature natural wear and systematic misspellings — [PDF](https://www.cs.uoi.gr/~sfikas/DAS2024-bessarion.pdf)
- CATMuS includes Latin and Middle Dutch manuscripts — [HF card](https://huggingface.co/datasets/CATMuS/medieval-segmentation/blob/e813bec31fba823f185ce5d44445035299a84d13/README.md)
- VisualOverload accuracy is resolution-limited — [arXiv HTML](https://arxiv.org/html/2509.25339)

### Inferences
- Suggested mixture for 3B/7B SFT:
  1. HierText + TextOCR, converted to localized-read and partial-read questions at native resolution, with max_pixels raised.
  2. CTW and ArT for Chinese and curved text.
  3. CATMuS / VOC / synthetic Greek-Latin lines composited onto painting-like backgrounds for Latin, Greek and Dutch verbatim copying.
  4. A small slice of Union14M Contextless / Artistic.
- Reading-order supervision: use HierText paragraph → line ordering. Note that line order within a paragraph is not guaranteed, so multi-line answers need sorting by geometry.

### Gaps
- No evidence found on how much any of these sets has improved painting-OCR specifically; there is no published transfer study.

---

## Q5. Image-overlap risk with VisualOverload

### Takeaway
Risk is very low for photo-based scene-text sets (Open Images, COCO, Bing/Tencent street view, book covers) and for HTR/inscription corpora. It is non-trivial for web-scraped corpora: LAION-derived MARIO-10M, TextAtlas5M real subsets, and any Wikimedia- or Google-Arts-derived art data. The 150 source artworks are listed in VisualOverload Table 4, so perceptual-hash dedup is straightforward.

### Cited Findings
- VisualOverload images come from Google Arts & Culture, public domain, collected April–May 2025, with Table 4 listing each artwork and link — [arXiv HTML](https://arxiv.org/html/2509.25339)
- Image sources by dataset:
  - HierText and TextOCR/TextVQA: Open Images — [HierText](https://github.com/google-research-datasets/hiertext); [hyper.ai TextOCR](https://hyper.ai/en/datasets/20131)
  - Uber-Text: Bing Streetside — [index](https://s3-us-west-2.amazonaws.com/uber-common-public/ubertext/index.html)
  - CTW: Tencent Street View — [JCST](https://jcst.ict.ac.cn/en/article/pdf/preview/10.1007/s11390-019-1923-y.pdf)
  - ST-VQA: ICDAR, ImageNet, VizWiz, VG, COCO-Text — [arXiv](https://arxiv.org/pdf/1905.13648)
  - MARIO-10M: LAION, TMDB, Open Library — [review](https://arxiv.org/pdf/2407.14774)

### Inferences
- Open Images and ImageNet are Flickr-sourced and may contain incidental museum photos of famous paintings. The probability that one of the specific 150 high-resolution scans is present is low but non-zero. Run pHash/CLIP near-duplicate filtering against the 150 images (and crops of them) before any training.
- LAION-based sets (MARIO-10M) are the highest-risk class, because LAION contains many Google Arts & Culture and Wikimedia painting reproductions.

### Gaps
- No dataset in this list reports deduplication against VisualOverload; overlap was not measured.

---

## Q6. Fit for SFT and RL (exact-match / edit-distance rewards)

### Takeaway
All the transcription-labelled sets support verifiable rewards: normalized exact match plus (1 − NED). Recent OCR RL work uses exactly this. Edit-distance rewards can be gamed by "semantic correction", which is the same autocorrection failure VisualOverload penalizes. So a strict exact-match term (with VisualOverload normalization) should dominate, with NED only as a shaping term.

### Cited Findings
- Recent RL systems using edit-distance-type OCR rewards:
  - Leum-VL (2026) uses an edit-distance reward for regular and handwritten OCR in its RL mix — [arXiv 2603.20354](https://arxiv.org/pdf/2603.20354)
  - MinerU2.5-Pro uses edit distance as a text-recognition reward and filters samples that are too easy or too hard by reward — [arXiv 2604.04771](https://arxiv.org/pdf/2604.04771)
  - A self-speculative OCR paper uses 1 − NED — [arXiv 2609.26638](https://arxiv.org/pdf/2609.26638)
- TextPecker argues string-similarity rewards are biased: OCR models and MLLMs "hallucinate corrections" for flawed text, which inflates scores — [arXiv 2602.20903](https://arxiv.org/pdf/2602.20903)
- "Seeing is Believing?" proposes degradation-based rule rewards with GRPO to reduce OCR hallucination — [arXiv 2506.20168](https://arxiv.org/pdf/2506.20168)
- An OCR-centric GRPO study found full and LoRA SFT degraded multilingual understanding, whereas reward-constrained GRPO on synthetic, label-free OCR data improved it — [arXiv 2503.18484](https://arxiv.org/pdf/2503.18484)
- olmOCR 2 (RLVR with binary unit tests) and LightOnOCR (GRPO, 28 rollouts) are alternative verifiable-reward designs — as summarized in search results citing [LightOnOCR arXiv 2601.14251](https://arxiv.org/pdf/2601.14251)

### Inferences
- **SFT:** use HierText/TextOCR/CTW/ArT templated QA plus HTR lines. Keep answers verbatim (no case or diacritic normalization in targets) so the model learns to copy.
- **RL (GRPO):** use
  - reward = EM_norm (VisualOverload normalization) + λ·(1 − NED) + a format term (answer only, no extra text — this addresses the "not replying with only the text" failure).
  - Seed prompts with rare and non-dictionary words from Union14M Contextless, CATMuS Latin and synthetic Greek, so autocorrection is penalized.
  - Filter out prompts with all-correct or all-wrong rollouts.
- Small labelled art-like sets (Bessarion, MapText, Ancient-Bench) are better as held-out dev sets than as training data, given their size.

### Gaps
- No public RL dataset specifically for scene/art OCR was found; the 2025–2026 OCR-RL work is document- or multilingual-synthetic-focused.
- The olmOCR 2 details came from a search summary; the primary paper was not fetched.
