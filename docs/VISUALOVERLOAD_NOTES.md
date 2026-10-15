# VisualOverload: what matters for this project

Source: the CVPR 2026 paper, the HF dataset card (`paulgavrikov/visualoverload`) and the official
repo (cloned to `third_party/visualoverload/`).

## Data

- 2,720 questions over 150 images. Most images have the pixel count of 4K (3840×2160) and keep
  their original aspect ratio. All are at least 1080p.
- HF fields: `question_id`, `image` (PIL), `question`, `question_type` (`choice` | `counting` |
  `ocr`), `options`, `difficulty`, `category`, `default_prompt`.
- Questions per category: activity 150 · attributes 149 · counting 559 · OCR 118 · reasoning 356 ·
  scene 1388.
- Questions per difficulty: easy 986 · medium 1304 · hard 430. The difficulty labels come from how
  the paper's 37 models did on each question (hard means ≤20% of them got it right).

## Scoring

- The ground truth is private, and there is no dev split. The only way to get a score is the
  [evaluation server](https://huggingface.co/spaces/paulgavrikov/visualoverload-submit). It needs
  a HuggingFace login and is **rate-limited per user per day**.
- The submission is a JSON list of `{"question_id": ..., "response": ...}`. For `choice`
  questions the response is the option letter. For counting and OCR it is the free-form answer.
  (One line in the dataset card says free-form answers should be a letter. That is a typo.)
- The server extracts answers itself:
  - choice: finds the option letter or the option label
  - counting: takes the last integer, whether written as a numeral or a word
  - OCR: removes diacritics, punctuation and spaces, lowercases, and maps V→U and J→I
- Yes/no questions come in **logical-opposite pairs**, and a pair counts as correct only if both
  answers are correct. Scene and reasoning questions are mostly these pairs. Consistency within a
  pair can be measured without labels.
- `default_prompt` reproduces the paper's prompts. Use it for the base pipeline so our numbers can
  be compared with the paper's.

## Reference numbers (paper Table 1, total / hard)

| Model | Total | Hard | Counting | OCR | Reasoning |
|-------|------:|-----:|---------:|----:|----------:|
| Qwen2.5-VL 3B | 54.1 | 4.8 | 25.9 | 49.2 | 43.9 |
| Qwen2.5-VL 7B | 61.5 | 9.6 | 34.9 | 55.9 | 49.8 |
| InternVL3 8B | 63.9 | 7.9 | 32.2 | 42.4 | 59.0 |
| Qwen2.5-VL 32B | 63.6 | 12.4 | 30.8 | 61.0 | 61.5 |
| InternVL3 38B | 67.6 | 7.2 | 35.4 | 45.8 | 69.8 |
| o3 | 69.5 | 19.6 | 36.7 | 61.0 | 75.1 |
| Consistent chance | 27.2 | 4.7 | 0 | 0 | 42.5 |

Weak spots the paper reports: counting (models undercount, especially large counts), OCR (models
"autocorrect" toward likelier words and read text in the wrong order), and pair consistency on
reasoning questions. These are the categories where tools such as detection, crop and OCR should
help most.

## Design consequences (to discuss as a team)

1. **We have no labels to develop against.** Tuning agent prompts and tools by submitting to the
   server will hit the rate limit. Options:
   - (a) hand-label a small dev subset ourselves, e.g. ~150 counting, OCR and attribute questions
   - (b) use pair consistency as a label-free signal
   - (c) develop on a labelled high-resolution benchmark (V\*Bench, HR-Bench) and submit only
     final runs
2. **The SFT extension needs correct answers.** The proposal keeps only trajectories that reach the
   correct answer, which VisualOverload cannot supply. Trajectories would have to come from another
   labelled dataset, or from our dev labels in (1a).
3. **Resolution is a confound.** A crop tool partly works by showing the model more pixels. Fix the
   base model's input resolution (e.g. Qwen2.5-VL `max_pixels`) and report it. Also consider a
   control with no agent: tile the image and answer in one pass. That separates "agent reasoning"
   from "more pixels".
