# Art / Painting Datasets with Published Labels for VisualOverload Fine-Tuning (with Contamination Risk)

Scope: art-domain datasets (VQA, captions, region/dense descriptions, boxes, pose, emotion/style/iconography) that could be used to SFT/RL small open VLMs (Qwen2.5-VL 3B/7B) for VisualOverload. Also covers public-domain museum image sources for building synthetic QA. Research done 2026-10-15. General counting/OCR/high-res datasets are out of scope (other notes cover them).

Verification legend: **[V]** means checked against a primary or near-primary page in this session. **[V-2nd]** means checked only via a secondary or aggregator page. **[UNVERIFIED]** means from background knowledge, not confirmed this session. Do not treat [UNVERIFIED] items as fact without checking.

---

## Q1. What exactly is in the VisualOverload test set? (needed to judge contamination)

### Takeaway
VisualOverload is 150 public-domain artworks from Google Arts & Culture, downsampled to about 4K pixels, with 2,720 hand-written, knowledge-free QAs. Scene questions dominate (1,388), then counting (559). The full artwork list is published in Table 4 of the arXiv paper. It is dominated by Bruegel paintings and Bruegel-designed prints, Avercamp ice scenes, Aertsen/Beuckelaer market scenes, Jan Steen, and other dense Northern Renaissance/Baroque genre or history scenes, plus several canonical, heavily reproduced masterpieces (e.g., The Night Watch, A Sunday on La Grande Jatte, Hunters in the Snow). That makes overlap with WikiArt, Web Gallery of Art, and museum open-access scrapes very likely.

### Cited Findings
- HF dataset `paulgavrikov/visualoverload`: 2,720 QA pairs, single `test` split. Categories: activity, attributes, counting, ocr, reasoning, scene. Question types: choice (A–D), counting (free-form), ocr (free-form). Difficulty: easy/medium/hard. Dataset license is CC BY-SA 4.0, and the images are described as based on public-domain (CC0) artwork. Ground truth is private, and scoring goes through an evaluation server. Fields: `question_id, image, question, question_type, options, difficulty, category, default_prompt`. Most images match the 4K pixel count (3840×2160) in various aspect ratios, with widths of 922–7,440 px. [V] — [HF dataset card](https://huggingface.co/datasets/paulgavrikov/visualoverload)
- Paper: arXiv 2509.25339, CVPR 2026. The best of 37 models (o3) scored 19.6% on the hardest split and 69.5% overall. Failure modes are counting, OCR, and logical inconsistency. [V] — [arXiv abs](https://arxiv.org/abs/2509.25339)
- Sourcing quote: the paintings were "curated from collections held by museums around the world and made available through Google Arts & Culture". The datasheet calls them "a subset of public domain artworks hosted on" that platform, "filtered to display visually complex and dense scenes". Images were "downsampled … to match the nearest number of pixels in 4K, while preserving their original aspect ratios". 28 images were below 4K and were not downsampled, but all are above Full HD. [V] — [arXiv HTML v3](https://arxiv.org/html/2509.25339v3)
- Per-category counts: activity 150, attributes 149, counting 559, OCR 118, reasoning 356, scene 1,388 (about 18 Q/image). Six annotators, who are the authors, with no crowdworkers. Images were collected April–May 2025 and annotated May–Aug 2025. Gemini 2.5 Pro was used to probe language bias. The datasheet restricts use to academic research and education. [V] — [arXiv HTML v3](https://arxiv.org/html/2509.25339v3)
- **Table 4 "List of artworks"** (random order, each linked to Google Arts & Culture) contains 150 entries. The list below was extracted from the arXiv HTML in two reads. Small transcription errors by the fetch tool are possible, so re-extract programmatically before using it as a blocklist. [V] — [arXiv HTML v3](https://arxiv.org/html/2509.25339v3)
  - **Pieter Bruegel the Elder (paintings):** Peasant Wedding; Children's Games; Hunters in the Snow; The Dutch (Netherlandish) Proverbs; Landscape with the Fall of Icarus; The Census at Bethlehem; Massacre of the Innocents; The Fall of the Rebel Angels; The Sermon of Saint John the Baptist; The Adoration of the Magi; Winter Landscape with Skaters and Bird Trap; The Dirty Bride / Marriage of Mopsus and Nisa.
  - **Bruegel-designed prints** (engraved by van der Heyden, Galle, Cock, Hogenberg, etc.): The Big Fish Eat the Little Fish; The Kermis at Hoboken; Seven Deadly Sins (Anger, Avarice, Sloth/Desidia, Luxuria, Pride/La soberbia, Gluttony); Virtues (Prudence, Temperance, Fortitude, Faith, Hope, Charity, Justice, Patience); The Alchemist; The Witch of Malleghem; The Last Judgment; The Descent of Christ into Limbo; The Resurrection; Wise and Foolish Virgins; The Fat Kitchen (1563 print); The Festival of Fools (after Bruegel).
  - **Hendrick Avercamp (about 9):** Enjoying the Ice; Enjoying the Ice near a Town; Ice-Skating in a Village; Skating Scene; Winter Landscape; Frolicking on a Frozen Canal in a Town; A Scene on the Ice; Frozen River with Skaters; Winter Landscape with Ice Skaters (ca. 1608). Also Aert van der Neer's Winter Landscape with Kolf Players; an anonymous Winter Scene on a Frozen Canal; Frans Huys's Ice Skating before the Gate of Saint George.
  - **Jan Steen:** The Dancing Couple; The Worship of the Golden Calf; The Feast of St Nicholas; Villagers Merrymaking Outside an Inn; Beware of Luxury ("In Weelde Siet Toe").
  - **Pieter Aertsen / Beuckelaer:** The Fat Kitchen. An Allegory; Market Scene (1550 and 1569); Christ with Mary and Martha; A Meat Stall with the Holy Family Giving Alms; The Egg Dance; The Adoration of the Magi; Wing of an Altarpiece; Beuckelaer's Woman Selling Vegetables.
  - **Hieronymus Bosch:** Ecce Homo (1500). Joos van Craesbeeck's Temptation of St Anthony is also Bosch-like.
  - **Canonical masterpieces with very high web/reproduction frequency:** Rembrandt, The Night Watch; Seurat, A Sunday on La Grande Jatte (listed with creator "Unknown"); Brullov, The Last Day of Pompeii; Dürer, Feast of Rose Garlands; Goya, El Entierro de la Sardina; Ivanov, The Apparition of Christ to the People; Makovsky, A Boyar Wedding Feast; Hayez, Pope Urban II Preaching the First Crusade; Giulio Romano, Chamber of the Giants ceiling; Memling (2); Cranach, Crucifixion; Altdorfer; Lucas van Leyden; Cornelis van Haarlem; Honthorst; Heemskerck (2); Ferdinand Bol (2); Dou, The Quack; Teniers, The Surgeon; Ostade; Molenaer; Dirck Hals (2); Vermeyen, The Spanish Brothel; Wtewael; Hans Bol; Carl Bloch, In a Roman Osteria; Blunck, Danish artists at the Osteria La Gensola; Van der Ast still lifes (3), Claesz, Fyt, Roesen and Duffield still lifes; Charles M. Russell, The Medicine Man; "The Great Picture" (attrib. Jan van Belcamp).
  - **Non-European / less-common items:** Chinese export paintings (Canton factories, Whampoa, dragon boats, porcelain handscroll in two halves); Caroline Le Souef (3, Australian Aboriginal scenes); Thomas Baines (2); Edward Roper; James Taylor (Sydney panorama); Catalan works (Mas i Fondevila, Massó i Falp, Vayreda, Clavé, Pere Mates, Borrassà altarpiece); Estevão Silva (2 "Untitled" 1887); a Greek St George (1798); Herrlein's Ljubljana; Psyche tapestry/painting items (Atelier de Paris, 1650).
- Only one search (Q4) checked whether VisualOverload's images overlap with any training corpus: I found no contamination or overlap analysis on the arXiv abstract page. [V] — [arXiv abs](https://arxiv.org/abs/2509.25339)

### Inferences
- Hogarth and Frith do **not** appear in the extracted Table 4. Bruegel (paintings plus about 25 prints), Avercamp, Aertsen and Steen make up about a third of the benchmark. Contamination filtering should key on these artists first.
- Many entries have multiple physical versions or copies. Bruegel the Younger copied the Census, Massacre, Proverbs and Kermis compositions, and the Bruegel engravings exist as many impressions in different museums. Pixel-level dedup (pHash) will miss these, while semantic or composition-level matching (CLIP/SSCD) will catch them. Exact-copy matching is not enough here.
- Several test images are probably also distributed by CC0 museum APIs as near-identical scans: e.g., The Night Watch, Avercamp ca. 1608 and Steen's Feast of St Nicholas (Rijksmuseum), Steen's Dancing Couple (NGA Washington), and Bruegel prints held by the Met/NGA. This is inferred from known holdings and not checked per item. Building synthetic QA from museum open-access dumps without dedup would therefore very likely re-introduce test images.
- Because the scene category alone is 51% of the questions, an image-memorization effect (knowing what a famous painting depicts) could inflate scene/activity accuracy even without answer leakage.

### Gaps
- I did not obtain the Table 4 holding museums per item (only GA&C links), so per-item mapping to Rijksmuseum/Met/NGA object IDs was not done.
- The extraction of 150 titles went through an LLM-based fetch tool. Re-parse Table 4 from the LaTeX/HTML source for an exact blocklist.

---

## Q2. Catalogue of art datasets with published labels (per-dataset facts)

### Takeaway
There is **no public art dataset that matches VisualOverload's format**: knowledge-free, dense-scene VQA on high-resolution crowded paintings with counting, OCR, attributes and activity. The closest sources of supervision are: (a) the "visual" half of AQUA (about 32k visual QAs, but low-res, generated from WGA comments); (b) box/pose datasets on paintings (DEArt, Materials In Paintings, IconArt, PeopleArt, PoPArt, Human-Art, Oxford VGG Paintings), which can be converted to counting/grounding QA; (c) visual-content captions (DEArt Zooniverse captions, ArtCap, PaintingForm, MMArt narrative perspective). Most WikiArt-derived sets (ArtEmis, ArtELingo, MMArt, ArtBench, PoPArt, PeopleArt) carry WikiArt copyright ambiguity and NC-style terms.

### Cited Findings

#### A. VQA / instruction / reasoning datasets

| Dataset | Venue/Year | Size | Image source | Annotation | License | Status |
|---|---|---|---|---|---|---|
| **AQUA** (Art QUestion Answering) | ECCV-W (VISART) 2020 | Train 69,812 QA (29,568 visual / 40,244 knowledge); val 5,124 (1,507 / 3,617); test 4,912 (1,270 / 3,642) | SemArt (Web Gallery of Art) | Auto-generated QA from paintings and WGA comments, crowd-verified for grammar, answerability and correctness | Repo shows MIT. Images inherit SemArt terms | [V] stats — [ar5iv](https://ar5iv.labs.arxiv.org/html/2008.12520); [arXiv](https://arxiv.org/abs/2008.12520); [GitHub noagarcia/ArtVQA](https://github.com/noagarcia/ArtVQA) |
| **ExpArt** ("Explain Artworks") | ACL 2024 (short) | Train set: 7,704 images, 18,613 instructions | English Wikipedia articles on artworks | Explanation generation with and without title (knowledge-heavy) | Not found; hosted on HF under `naist-nlp` | [V-2nd] — [arXiv 2403.00068](https://arxiv.org/abs/2403.00068v1); [ACL listing](https://preview.aclanthology.org/json-schema/2024.acl-short.65.pdf). Cross-lingual follow-up in 10 languages (Findings NAACL 2025) — [arXiv 2409.01584](https://arxiv.org/html/2409.01584v1) |
| **PaintingForm** (GalleryGPT) | ACM MM 2024 (oral) | About 19k paintings, about 50k formal-analysis paragraphs | Not stated on repo | GPT-4/Gemini-generated formal analysis (composition, color, light), prompted **not** to mention title/artist | Repo is Apache-2.0. Dataset license not stated | [V] — [arXiv 2408.00491](https://arxiv.org/abs/2408.00491v1); [GitHub](https://github.com/steven640pixel/GalleryGPT); HF `steven16/Painting-Form` |
| **VISCOUNTH** | ACM TOMM 2023 | About 500K Italian cultural assets, 6.5M QA (IT/EN) | ArCo Italian cultural-heritage KG (catalog photos, not only paintings) | Template QA needing image plus text description (mostly knowledge/metadata) | Not found | [V-2nd] — [USiena PDF](https://usiena-air.unisi.it/bitstream/11365/1230154/1/3590773.pdf); [Crossref](https://api.crossref.org/works/10.1145%2F3590773) |
| **ArtSeek / WikiFragments** | arXiv 2507.21917 (2025) | Wikipedia-scale image-text fragments | Wikipedia/Wikimedia | Retrieval + agentic VQA with Qwen2.5-VL; classifier for artist, genre, style, media, tags | Not checked | [V-2nd] — [arXiv 2507.21917](https://arxiv.org/html/2507.21917v1) |
| **ArtCoT-QA** (from A-MAR) | ICMR 2026 | Not obtained | Not obtained | Diagnostic benchmark with step-wise reasoning and evidence annotations (visual + metadata + external knowledge) | Not checked | [V-2nd] — [arXiv 2604.19689](https://arxiv.org/pdf/2604.19689) |
| **ArTest** | arXiv 2409.03521 (2024; v2 Sep 2025) | Curated test set of pivotal paintings | Not obtained | Style/author/period classification (knowledge) | Not checked | [V-2nd] — [arXiv](https://arxiv.org/abs/2409.03521v2) |
| **CArtBench** | arXiv 2604.11632 (2026) | Not obtained | Chinese art | Understanding, interpretation, authenticity (knowledge) | Not checked | [V-2nd, title only] — [arXiv](https://arxiv.org/pdf/2604.11632) |
| **ArtQuest / ArtQuest-Type** | Used as zero-shot eval by GalleryGPT | Not obtained | — | Art VQA | — | [V-2nd mention] — [GalleryGPT arXiv](https://arxiv.org/abs/2408.00491v1) |
| **LLaVA-docent** | Computers & Education: AI, about 2024 | GPT-4-generated art appreciation dialogue data | Not obtained | Instruction dialogues | Not checked | [V-2nd] — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666920X24001000) |
| **Vision-Flan** (art subset) | ACL 2024 | 187 tasks overall; includes painting artist/genre/style tasks | Mixed | Instruction-formatted classification | — | [V-2nd] — [arXiv 2402.11690](https://arxiv.org/pdf/2402.11690) |

#### B. Caption / description datasets

| Dataset | Venue/Year | Size | Source | Annotation | License | Ref |
|---|---|---|---|---|---|---|
| **SemArt** | ECCV-W 2018 | 21,383 images | Web Gallery of Art [source per background knowledge, UNVERIFIED this session] | WGA art-historical comment, plus title, author, type, school, timeframe | Not confirmed (project page 404 at time of check) | [V-2nd] size — [georgeblck list](https://github.com/georgeblck/art-datasets); [arXiv 1810.09617](https://arxiv.org/pdf/1810.09617) |
| **Artpedia** | ICIAP 2019 | 2,930 paintings, 28,212 sentences (visual vs contextual sentences) | Wikipedia | Captions split into visual and contextual | Paper only | [V-2nd] — [georgeblck list](https://github.com/georgeblck/art-datasets) |
| **ArtCap** | IEEE (about 2022) | 3,606 paintings × 5 descriptions | Not obtained | Human captions of visual content | Not obtained | [V-2nd] — [IEEE 9965360](https://ieeexplore.ieee.org/document/9965360/); [ResearchGate](https://www.researchgate.net/publication/365834277_ArtCap_A_Dataset_for_Image_Captioning_of_Fine_Art_Paintings) |
| **DEArt Zooniverse descriptions** | B2SHARE 2023; LM4UC 2025 paper | 7,471 DEArt images, typically 4–5 crowd descriptions each | DEArt (European art, 12th–18th c.) | Crowd-written scene descriptions (visual, not style). Paper reports tuning MLLMs on them reduced hallucination | **CC BY 4.0** (descriptions) | [V] — [B2SHARE record](https://b2share.eudat.eu/records/e5bmq-qxh78); paper: [ACL Anthology 2025.lm4uc-1.6](https://preview.aclanthology.org/setup/2025.lm4uc-1.6) (findings via search snippet) |
| **MMArt** | arXiv 2608.10706 (Aug 2026) | 74,234 WikiArt paintings | WikiArt | Four perspectives (narrative, formal, emotional, historical) plus unified caption, from specialized VLMs or human annotators | Not checked | [V-2nd] — [arXiv](https://arxiv.org/pdf/2608.10706); [project](https://shuaiwang97.github.io/MMArt/) |
| **Iconographic captions (Iconclass)** | 2021 | Derived from Iconclass AI test set | Arkyves | Iconclass code → caption (knowledge/iconography) | — | [V-2nd] — [arXiv 2102.03942](https://arxiv.org/pdf/2102.03942) |
| **CognArtive** | arXiv 2502.04353 (2025) | Not listed | LLM-generated art analyses | Generated descriptions | Not listed | [V-2nd] — [georgeblck list](https://github.com/georgeblck/art-datasets) |
| **CANVAS** | arXiv 2606.09846 (2026) | Not obtained | — | Narrative/visual-audio art captioning | — | [title only] — [arXiv](https://arxiv.org/pdf/2606.09846) |
| **panopstor/greatpainters** | HF | About 2,500 classical painters' works (per search snippet) | Web | CogVLM captions including artist name | Listed CC0 (per snippet) | [V-2nd] — [HF](https://huggingface.co/datasets/panopstor/greatpainters) |
| **Mitsua/art-museums-pd-440k** | HF | Viewer shows about 294k rows (name says 440k) | AIC IIIF visible in sample. The search snippet says Smithsonian, Met, Cleveland and AIC | Short metadata-derived captions (no VLM captions) | Dataset tagged CC BY 4.0; per-image CC0 | [V] — [HF](https://huggingface.co/datasets/Mitsua/art-museums-pd-440k) |
| **Public Domain 12M (PD12M)** | arXiv 2410.23144 | About 12M PD/CC0 images | Museums + Wikimedia etc. | Florence-2-large synthetic detailed captions | PD/CC0 images | [V-2nd] — [arXiv](https://arxiv.org/pdf/2410.23144) |

#### C. Object detection / pose / dense region labels on paintings (most useful for counting-style supervision)

| Dataset | Year | Size | Source | Labels | License | Ref |
|---|---|---|---|---|---|---|
| **DEArt** | ECCV-W (VISART) 2022; J. Cultural Heritage 2025 | 15,000+ images (about 80% non-iconic) | European paintings, 12th–18th c. | Boxes for **all instances** of 69 classes, plus 12 pose classes on human-like boxes | Original dataset license **not confirmed** (Zenodo 6984525; fetch failed). A WikiArt-extended derivative (ECCV-W AI4DH 2024) is CC BY-NC-SA | [V-2nd] — [arXiv 2211.01226](https://arxiv.org/pdf/2211.01226); [Zenodo](https://zenodo.org/records/6984525); [AI4DH GitLab](https://gitlab.liris.cnrs.fr/anr-aaa/eccv-ai4dh-2024/-/blob/67ad7e2b9c42c50793197763efc32b556a5bccf1/README.md) |
| **Materials In Paintings (MIP)** | 2021 (TU Delft) | 19,325 paintings, 227,810 boxes | Museum images. Rights "belong to the original gallery and/or art institution" | Human and automatic boxes with material labels (fabric, metal, skin, sky, animals, flora, food, glass, liquid, paper, wood, stone) | No license; users told to check the source institution | [V] — [MIP site](https://materialsinpaintings.tudelft.nl/); size [V-2nd] — [georgeblck list](https://github.com/georgeblck/art-datasets) |
| **IconArt v1/v2** | ECCV-W (VISART) 2018 | 5,955 images (2,978 train / 2,977 test). 1,480 test images with boxes. v1 has 7 classes; v2 adds beard, capital, turban (10) | Wikimedia Commons, 11th–20th c. | Image labels plus test-set boxes (iconographic) | CC BY 4.0 per secondary mirror (verify on Zenodo) | [V-2nd] — [arXiv 1810.02569](https://arxiv.org/pdf/1810.02569); [Zenodo](https://zenodo.org/records/4737435); [HF NGonthier/IconArt](https://huggingface.co/datasets/NGonthier/IconArt) |
| **People-Art** | ECCV-W 2016 | 43 depiction styles, 41 from WikiArt | WikiArt | Person boxes | Not listed | [V-2nd] — [arXiv 1610.08871](https://arxiv.org/abs/1610.08871) |
| **PoPArt** (Poses of People in Art) | 2023 | 2,454 images, 10,749 figures, up to 17 keypoints, 22 styles | WikiArt metadata | COCO-format boxes plus keypoints, train/val/test | Described as "openly licensed" (exact license not confirmed) | [V] — [arXiv 2301.05124](https://arxiv.org/pdf/2301.05124); [Zenodo 7516230](https://zenodo.org/records/7516230) |
| **Human-Art** | CVPR 2023 | About 50k images, 123k+ person instances; 5 natural + 15 artificial scenarios | Paintings, cartoons, sculpture, etc. | Boxes, 2D keypoints (17 per MMPose; 21 per another source), self-contact, text descriptions | Access by application (per list); license not confirmed | [V-2nd] — [arXiv 2303.02760](https://arxiv.org/abs/2303.02760v1); [CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Ju_Human-Art_A_Versatile_Human-Centric_Dataset_Bridging_Natural_and_Artificial_Scenes_CVPR_2023_paper.pdf); [georgeblck list](https://github.com/georgeblck/art-datasets) |
| **Oxford VGG Paintings** | 2014 | 210k+ paintings (10k annotated) | Art UK | Crowdsourced object tags (VOC-style classes) | Not listed | [V-2nd] — [VGG](https://www.robots.ox.ac.uk/~vgg/data/paintings/) |
| **ArtDL 2.0** | 2021 | 42,479 images | Renaissance art | 19 Iconclass saint classes (plus boxes in 2.0 per Milani et al.) | Not listed | [V-2nd] — [artdl.org](https://artdl.org/); [PMC Milani 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9410216/) |
| **UCF_CC_50** (photo crowd set) | CVPR 2013 | 50 images, about 94–4,543 people each; one is a painting | Mixed | Dot counts | — | [V-2nd] — [USPTO patent describing set](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9946952); [UCF CRCV](https://www.crcv.ucf.edu/projects/crowdCounting/index.php) |
| **"Counting the Prado"** | 2025? project | Prado paintings | Museo del Prado + Sherpa.ai | AI counts of people/objects in paintings | Unknown whether data is released | [V-2nd, headline only] — [Artnet](https://news.artnet.com/art-world/counting-prado-artificial-intelligence-2629953) |

#### D. Emotion / style / iconography / large image corpora (weak relevance to knowledge-free VQA)

| Dataset | Size | Source | Labels | License | Ref |
|---|---|---|---|---|---|
| **ArtEmis** (CVPR 2021) | About 455K annotations on 80K artworks (v1 arXiv: 439K / 81K) | WikiArt | 8 emotions plus free-text explanations | "ArtEmis Terms of Use" (form acceptance required); code MIT. A secondary listing says CC BY-NC-SA 3.0 [unverified] | [V] — [artemisdataset.org](https://www.artemisdataset.org/); [hyper.ai](https://hyper.ai/en/datasets/14861) |
| **ArtEmis v2** (2022) | 260K contrastive instances | WikiArt | Contrastive emotion captions | Not listed | [V-2nd] — [georgeblck list](https://github.com/georgeblck/art-datasets) |
| **ArtELingo / ArtELingo-28** | About 1.24M / about 200K annotations | WikiArt | Multilingual emotion captions | Not listed | [V-2nd] — [georgeblck list](https://github.com/georgeblck/art-datasets) |
| **WikiArt Emotions** (2018) | 4,105 images | WikiArt | 20 emotion categories | Not listed | [V-2nd] — [georgeblck list](https://github.com/georgeblck/art-datasets) |
| **Iconclass AI test set** | 87,749 images (one paper used 86,530 valid) | Arkyves | Iconclass notations | CC0 per HF mirror `biglam/brill_iconclass`; official page says "freely available" | [V-2nd] — [iconclass.org/testset](https://iconclass.org/testset/); [HF mirror](https://huggingface.co/datasets/biglam/brill_iconclass/blob/1c44bf464035180fe923b8837652ac79536214cf/README.md?code=true) |
| **WikiArt (HF huggan/wikiart; matrixglitch/wikiart-215k)** | About 81k (huggan) / 215k | WikiArt | Artist, style, genre (+ title/year) | WikiArt images are not uniformly PD; no clear license | [V-2nd] — [georgeblck list](https://github.com/georgeblck/art-datasets) |
| **Abstract4D** (arXiv 2608.28339) | — | Abstract art | — | — | [title only] — [arXiv](https://arxiv.org/pdf/2608.28339) |
| **ARTeFACT** (2024) | 445 images, 11k+ annotations | Paintings, frescoes, manuscripts | Pixel-level damage segmentation | AFL-3.0 | [V-2nd] — [HF](https://huggingface.co/datasets/danielaivanova/damaged-media) |

### Inferences
- The only datasets with **instance-level annotations of many objects per painting** are DEArt (69 classes, all instances), MIP (about 12 boxes/painting on average: 227,810 / 19,325), PoPArt, People-Art and Human-Art (people), and IconArt (test-set boxes only). These are the natural raw material for converting to counting, attribute or grounding QA. That is also exactly why they are the **highest contamination risk** (see Q4): any of them containing Children's Games or Hunters in the Snow effectively leaks counting labels.
- AQUA's visual questions (about 32k total) are the only sizeable human-verified, painting-specific visual QA, but they are short, generated from SemArt and WGA images (typically web-resolution), and not dense-scene-focused. Expect limited transfer to 4K counting/OCR.
- Instruction datasets from 2024–2026 (ExpArt, PaintingForm, MMArt, ArtSeek, ArtCoT-QA, VISCOUNTH) are mostly art-history knowledge or formal analysis. They train the opposite skill from VisualOverload's knowledge-free perception, and could even encourage answering from recognized-painting priors.

### Gaps
- Not verified this session [UNVERIFIED, background knowledge only]: **OmniArt** (Strezoski & Worring 2018; about 1.3M+ artworks aggregated from Rijksmuseum, Met, WGA; metadata labels). **Art500k** (Mao et al. 2017; about 550k images, metadata labels). **NoisyArt** (Del Chiaro et al. 2019; webly-supervised artwork instance recognition, about 3k artworks). **ArtBench-10** (Liao et al. 2022; 60k images, 10 styles, from WikiArt/Ukiyo-e). **ArtGPT-4** (2023; a MiniGPT-4 adapter model, not a released labelled dataset as far as I recall). **ArtRAG** (2025; retrieval-augmented art description, no training set as far as I recall). **ClassArch** (Greek vase-painting pose dataset). **KALE**: I could not identify an art dataset by this name. Poetry-art datasets (e.g., painting–poem pairs) were not found or verified.
- Licenses for SemArt, AQUA images, DEArt (original), PoPArt (exact CC variant), Human-Art, ExpArt, PaintingForm, MMArt and VISCOUNTH were not confirmed from primary pages.
- I did not find any public **dense/region-caption dataset on paintings** (a Visual Genome-style dataset for art). The nearest are DEArt boxes plus DEArt crowd captions (separate, not region-linked).
- I did not find any public **crowd-counting dataset of paintings** (dot annotations). "Counting the Prado" data release status is unknown.

---

## Q3. How well does each dataset match VisualOverload's question types?

### Takeaway
Match is weak overall. VisualOverload's skills (scene 51%, counting 21%, reasoning 13%, activity 5.5%, attributes 5.5%, OCR 4%) are mostly served by **converting detection/pose labels to counting/grounding QA** and by **visual (not historical) captions**. Knowledge-based art QA (AQUA knowledge split, ExpArt, VISCOUNTH, ArtSeek, ArTest) does not match. No art dataset targets OCR in paintings (inscriptions, banners, signage in prints).

### Cited Findings
- VisualOverload questions are knowledge-free: "simple, knowledge-free visual skills in crowded scenes", with category counts as listed in Q1. — [arXiv HTML v3](https://arxiv.org/html/2509.25339v3); [arXiv abs](https://arxiv.org/abs/2509.25339)
- AQUA splits visual (painting-based) and knowledge (comment-based) questions, and the knowledge split is the majority (40,244 of 69,812 train). — [ar5iv AQUA](https://ar5iv.labs.arxiv.org/html/2008.12520)
- PaintingForm deliberately avoids titles and artists and focuses on composition, color, light and shadow, so it is visual rather than knowledge-based, but formal analysis rather than counting or attributes. — [GalleryGPT arXiv](https://arxiv.org/abs/2408.00491v1)
- DEArt crowd descriptions describe the visual scene rather than style or context. — [B2SHARE](https://b2share.eudat.eu/records/e5bmq-qxh78); [LM4UC 2025](https://preview.aclanthology.org/setup/2025.lm4uc-1.6)
- DEArt annotates all instances of 69 classes plus 12 pose classes. — [arXiv 2211.01226](https://arxiv.org/pdf/2211.01226)
- MMArt finds that narrative descriptions are the most discriminative perspective in retrieval (R@1 44.0%) and formal descriptions the least (7.8%). — [arXiv 2608.10706](https://arxiv.org/pdf/2608.10706)

### Inferences
Match levels by VisualOverload category:

| VisualOverload category | Best art-domain source | Match |
|---|---|---|
| Counting (559) | DEArt boxes (69 classes), MIP boxes, PoPArt/People-Art/Human-Art persons → templated "how many X" QA | Medium (boxes are sparse vs. VO crowds; counts need verification of completeness) |
| Scene (1,388) | DEArt captions, ArtCap, MMArt narrative, AQUA visual QA | Medium-low |
| Activity (150) | DEArt pose classes (12), Human-Art, PoPArt keypoints | Low-medium |
| Attributes/color (149) | MIP material labels; DEArt captions | Low |
| Reasoning (356) | None art-specific | Very low |
| OCR (118) | None. Bruegel prints carry Latin/Dutch inscriptions, but no public OCR labels on art were found | None |

- Best fit for SFT: DEArt (+ its CC BY 4.0 captions), PoPArt, IconArt, and MIP, converted into grounded counting/attribute QA **after** removing VisualOverload overlaps. AQUA visual subset is a low-cost add-on.
- Fit for RL (GRPO with verifiable rewards): counting QAs derived from complete instance annotations (DEArt "all instances") give a verifiable integer reward. Caption datasets are not suited to verifiable RL.

### Gaps
- I did not measure how complete DEArt/MIP instance annotations are in crowded scenes. If annotators skipped small figures, derived counts would be noisy.

---

## Q4. Contamination risk and how to de-duplicate

### Takeaway
Contamination risk is **high** for any WikiArt-, WGA-, Wikimedia- or museum-open-access-derived corpus. The 150 test works include the most reproduced dense genre paintings (Bruegel's Hunters in the Snow, Children's Games, Proverbs, Census; The Night Watch; La Grande Jatte; Avercamp ice scenes; Steen). The most dangerous are **instance-annotated** sets (DEArt, MIP, PoPArt, People-Art, Human-Art, IconArt, ArtDL), because boxes on a test painting turn into near-direct counting answers. Mitigation: (1) metadata blocklist on artist+title from Table 4; (2) embedding near-duplicate filtering against the 150 HF images (CLIP/SigLIP/DINOv2 plus SSCD copy-detection descriptors); (3) crop-aware local-feature matching for details and partial crops; (4) composition-level exclusion for copies and print impressions.

### Cited Findings
- VisualOverload images are public on HF (paulgavrikov/visualoverload), so a reference set for dedup exists; only answers are private. — [HF card](https://huggingface.co/datasets/paulgavrikov/visualoverload)
- Test set content (Bruegel paintings and prints, Avercamp, Aertsen, Steen, Bosch Ecce Homo, Night Watch, Grande Jatte, etc.) — [arXiv HTML Table 4](https://arxiv.org/html/2509.25339v3)
- DEArt covers European paintings of the 12th–18th centuries, the same period and region as most of the test set. — [arXiv 2211.01226](https://arxiv.org/pdf/2211.01226)
- PoPArt and People-Art draw on WikiArt (WikiArt metadata per annotation; 41 of 43 People-Art styles from WikiArt). — [PoPArt arXiv](https://arxiv.org/pdf/2301.05124); [People-Art arXiv](https://arxiv.org/pdf/1610.08871)
- ArtEmis, ArtELingo, MMArt and WikiArt HF dumps are WikiArt-based. — [artemisdataset.org](https://www.artemisdataset.org/); [MMArt](https://arxiv.org/pdf/2608.10706); [georgeblck list](https://github.com/georgeblck/art-datasets)
- IconArt draws from Wikimedia Commons, 11th–20th c. — [arXiv 1810.02569](https://arxiv.org/pdf/1810.02569)
- PD12M includes public-domain museum images and filtered living-artist mentions; it is a broad PD corpus likely to contain famous PD paintings. — [arXiv 2410.23144](https://arxiv.org/pdf/2410.23144)
- The Mitsua PD set draws from museum CC0 open access, with AIC visible in the sample. — [HF](https://huggingface.co/datasets/Mitsua/art-museums-pd-440k)

### Inferences
Contamination risk ranking:

| Risk | Datasets | Why |
|---|---|---|
| **Very high** (labels leak) | DEArt (+ captions), MIP, PoPArt, People-Art, Human-Art, IconArt, ArtDL, Oxford VGG Paintings | Instance boxes/poses on European old-master paintings. If they include Bruegel, Avercamp or Steen works, counting labels are effectively leaked |
| **High** (image + semantic leak) | AQUA/SemArt (WGA has extensive Bruegel/Steen/Avercamp coverage plus detail crops), WikiArt family (ArtEmis, ArtELingo, MMArt, ArtBench, WikiArt dumps), ExpArt/ArtSeek (Wikipedia: every famous test painting has an article), PaintingForm (source not stated), Iconclass test set | Same images with scene-level descriptions and captions |
| **High if unfiltered** | Museum open-access dumps (Rijksmuseum, Met, NGA, AIC, Cleveland, Smithsonian), PD12M, Mitsua PD, Wikimedia Commons | Same or near-identical scans of test items (Night Watch, Avercamps, Steens, Bruegel prints), but no labels, so the risk is limited to whatever synthetic QA we generate on them |
| **Low** | VISCOUNTH (Italian catalog photos), CArtBench (Chinese art), Human-Art non-painting subsets, ARTeFACT | Different domain or region |

Recommended dedup pipeline. These are suggestions, not tested; tune the thresholds on held-out positives.
1. **Metadata filter:** fuzzy-match (artist, title, date) against the 150-row Table 4 blocklist. Also block **all** Bruegel the Elder and Younger compositions with the same titles, and all impressions of the listed Bruegel engravings (Seven Deadly Sins, Virtues, Big Fish, Kermis at Hoboken, Alchemist, Witch of Malleghem, Last Judgment, etc.). For a conservative SFT set, optionally drop entire artists (Bruegel, Avercamp, Aertsen, Steen).
2. **Global near-duplicate filter:** pHash/dHash (imagehash) at multiple resizes, plus CLIP/SigLIP or DINOv2 embeddings, flagging cosine similarity above about 0.90 against the 150 test images for manual review. SSCD (Meta's self-supervised copy-detection descriptor) is designed for copy and transformation robustness.
3. **Crop/detail filter:** WGA, SemArt and museum sites host **detail crops** of famous paintings. Run tiled embedding search (e.g., 3×3 and 5×5 tiles of each test image vs. each candidate) or local-feature matching (SIFT/ORB/SuperPoint + LightGlue/LoFTR with RANSAC inlier count) to catch crops a global hash misses.
4. **Copies and versions:** composition-level CLIP similarity plus the metadata blocklist covers Brueghel-the-Younger copies and multiple print impressions with different paper tone or scan.
5. **Text-side check:** for caption/QA datasets, also grep text for test titles ("Hunters in the Snow", "Netherlandish Proverbs", "Night Watch", "Grande Jatte") to drop entries that reference a test work even when the image differs.
6. Report the dedup procedure and the number of removed items in the paper, since reviewers of an ACL-style submission will ask.

### Gaps
- I did not run an actual overlap check (e.g., how many of the 150 are in DEArt, SemArt or WikiArt). That requires downloading the datasets and running the pipeline above. Given the artist overlap, I expect non-zero overlap in DEArt, SemArt/AQUA and WikiArt, but this is **not verified**.
- No published contamination analysis for VisualOverload was found.

---

## Q5. Fitness for SFT/RL vs. as an image source for synthetic QA; public-domain museum APIs

### Takeaway
For VisualOverload, the most promising path is **not** reusing art-history labels. It is (a) a small, deduplicated SFT mix of converted DEArt/PoPArt/MIP/IconArt boxes (counting, grounding, pose → activity), the AQUA visual split and the CC BY 4.0 DEArt captions, and (b) **synthetic dense-scene QA generated on high-resolution CC0 museum images** (Rijksmuseum, Met, NGA, AIC, Cleveland, SMK, Smithsonian, Yale Center for British Art, Finnish National Gallery, Paris Musées, Getty, Walters). These must be deduplicated against the 150 test works and ideally restricted to crowded genre, history and market scenes by **other** artists (e.g., Teniers, Ostade, Brouwer, Vinckboons, Sebastian Vrancx, Lucas van Valckenborch, Brueghel the Younger's non-test compositions, Esaias van de Velde, Adriaen van de Venne, Hogarth and Frith, which are not in the test set). Google Arts & Culture itself does not offer a bulk open-access API, so source directly from the museums.

### Cited Findings
Museum open-access sources (licenses as listed by the curated art-datasets list; "[V-2nd]" unless noted):

| Source | Size | License | Link |
|---|---|---|---|
| Rijksmuseum | 800k+ objects | CC0 (public-domain works) | [data.rijksmuseum.nl](https://data.rijksmuseum.nl/) |
| The Met Open Access | 470k+ artworks | CC0 | [GitHub metmuseum/openaccess](https://github.com/metmuseum/openaccess) |
| National Gallery of Art (DC) | 130k+ | CC0 | [GitHub NGA opendata](https://github.com/NationalGalleryOfArt/opendata) |
| Art Institute of Chicago | not listed | CC0 | (API at api.artic.edu; IIIF) |
| Smithsonian Open Access | 11M+ records, 2.8M+ images | CC0 | [AWS registry](https://registry.opendata.aws/smithsonian-open-access/) |
| Cleveland Museum of Art | 61k+ | CC0 | [GitHub](https://github.com/ClevelandMuseumArt/openaccess) |
| SMK (Denmark) | 88k+ | CC0 | [API](https://api.smk.dk/api/v1/docs) |
| Paris Musées | 150k+ images | CC0 | [site](http://parismuseescollections.paris.fr/en) |
| Finnish National Gallery | 36k+ | CC0 | [GitHub](https://github.com/hugovk/finnishnationalgallery) |
| Yale Center for British Art | about 70k IIIF images | Public domain | [site](https://britishart.yale.edu/collections-data-sharing) |
| Getty Museum | 30k+ high-res | No known restrictions | [site](https://www.getty.edu/projects/open-data-apis/) |
| Walters, Cooper Hewitt, Whitney, Minneapolis, Carnegie, Nationalmuseum Sweden | various | CC0 | [georgeblck list](https://github.com/georgeblck/art-datasets) |
| Museo del Prado | about 40k | Non-commercial | same list |
| National Gallery London | about 2,300 paintings | CC BY-NC-ND 4.0 | same list |
| Tate | about 38k | CC BY-NC-ND 3.0 | same list |
| V&A, Louvre, Harvard, British Museum, MoMA | various | Varying / non-commercial | same list |
| Europeana / Wikimedia Commons | — | Per-item rights statements | [UNVERIFIED in this session] |

All rows above are from the [georgeblck/art-datasets list](https://github.com/georgeblck/art-datasets) (curated GitHub list, noted April 2026 for some dead links).

- Precedents for synthetic captions on PD art at scale: PD12M (Florence-2 captions on about 12M PD images) — [arXiv 2410.23144](https://arxiv.org/pdf/2410.23144). Mitsua PD (metadata captions, CC0 museum images) — [HF](https://huggingface.co/datasets/Mitsua/art-museums-pd-440k).
- PaintingForm shows LLM-generated art annotations that avoid title/artist mentions can be used for instruction tuning (GalleryGPT on ShareGPT4V-7B with LoRA). — [arXiv 2408.00491](https://arxiv.org/abs/2408.00491v1)
- DEArt crowd captions reportedly reduced hallucination when used to tune MLLMs (per Anthology snippet). — [LM4UC 2025](https://preview.aclanthology.org/setup/2025.lm4uc-1.6)

### Inferences
- **SFT/RL suitability:**
  - *Directly SFT-able (after dedup):* AQUA visual QA, DEArt captions (CC BY 4.0, the cleanest license), PaintingForm (formal analysis, visual), ArtCap.
  - *Convertible to verifiable QA for RL (GRPO):* DEArt (count per class, presence, pose → activity MCQ), PoPArt/People-Art/Human-Art (person counts, pose), MIP (material/attribute MCQ, counts per material), IconArt (presence/grounding).
  - *Avoid or keep tiny:* knowledge-heavy sets (AQUA knowledge split, ExpArt, VISCOUNTH, ArtSeek, ArtEmis), which push the model toward art-historical recall that VisualOverload does not reward.
- **Synthetic QA recipe on museum CC0 images:** Pull high-resolution (≥4K) scans via IIIF from Rijksmuseum, NGA, Met, AIC, Cleveland and SMK, then filter for dense scenes (e.g., detector person count ≥ 30, or high object-proposal density). Generate QA with a strong teacher VLM working on tiles, plus detector- or segmenter-verified counts. Mirror VisualOverload's category mix (scene, counting, reasoning, activity, attribute, OCR on inscriptions in prints). Then dedup against the 150 test images (Q4 pipeline). Prints (Met, Rijksmuseum, NGA hold many Bruegel-school engravings) are a good OCR source for Latin/Dutch captions, but the test-listed impressions must be excluded.
- License hygiene: museum CC0 images plus self-generated QA give a clean, releasable training set. WikiArt-derived and NC-licensed sets (ArtEmis terms, DEArt derivative CC BY-NC-SA, NG London, Tate, Prado) are fine for academic research but not for redistribution of images.

### Gaps
- Not verified this session: AIC, Europeana and Wikimedia Commons license specifics; per-API rate limits and maximum IIIF resolution per museum.
- I did not confirm that Google Arts & Culture lacks a bulk API. That claim is background knowledge [UNVERIFIED]; GA&C serves tiled zoomable images.
- I found no published art-domain synthetic dense-scene VQA dataset (2024–2026) that mirrors VisualOverload's categories. It appears to be an open gap and an opportunity.
