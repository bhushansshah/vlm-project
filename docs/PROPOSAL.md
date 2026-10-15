# Studying the Performance of VLM Agents on VQA Tasks for Dense Datasets

Bhushan Shah (bhsshah@cs.stonybrook.edu) · Md. Tanvir Arafat (marafat@cs.stonybrook.edu) ·
Tanish Malekar (tmalekar@cs.stonybrook.edu)

Text version of the submitted proposal PDF, kept here so it can be searched and diffed.

## 1. Problem

VisualOverload [1] tests VLMs on dense, high-resolution scenes: 2,720 QA pairs over 150
public-domain paintings, in six categories (activity, attributes, counting, OCR, reasoning, scene).
The best model it evaluates scores 19.6% on the hard split and 69.5% overall. The benchmark only
measures direct image→answer inference. It does not test whether models do better when they can
use external visual tools.

**Research question:** How well do VLM agents with access to tools perform on dense-scene VQA,
compared with direct VLM inference?

## 2. Approach

1. **Base VLM**: `A = VLM(I, Q)`. This is the non-agentic baseline.
2. **Tool-augmented VLM agent**: a ReAct-style [7] loop that interleaves reasoning and tool calls.
   The candidate tools are crop/zoom, object detection, segmentation, human pose estimation and OCR.
3. **Python-enabled VLM agent**: the VLM writes and runs Python, so it can combine operations.
   Example: for "how many people wearing red?", detect people, filter them by clothing colour,
   then count. This tests whether a programmatic interface beats a fixed tool set.
4. **Optional: SFT, then RL**: fine-tune a small VLM on tool-use trajectories from frontier VLMs,
   keeping only trajectories that reach the correct answer. RL on top of SFT is a further option.

## 3. Resources

- Dataset: VisualOverload. The images and questions are public. Ground truth is private and
  scored by the official server.
- Compute: SBU AI3 cluster.
- Libraries: PyTorch, HF Transformers/Datasets, OpenCV, Pillow, torchvision, Ultralytics
  (detection), SAM/SAM2 (segmentation), MediaPipe (pose), NumPy, Pydantic (tool schemas).

## 4. Evaluation

- Metric: accuracy under the VisualOverload protocol, reported overall, per category and per
  difficulty level.
- Comparisons: base vs. tool agent vs. Python agent, plus SFT and SFT+RL if those get done.
- Success: accuracy goes up when tools are available.
- Further analysis:
  - whether tools help more on hard questions
  - which categories benefit
  - whether the Python agent beats the fixed tool set
  - how often each tool is used, and how tool use relates to correct answers

## 5. Timeline

| Dates | Milestone |
|-------|-----------|
| Sep 24 – Oct 2 | Setup: dataset, eval pipeline, VLM infra, literature review |
| Oct 3 – Oct 16 | Base VLM evaluation |
| Oct 17 – Oct 30 | Visual tool integration (crop/resize, detection, segmentation, pose, OCR) |
| Oct 31 – Nov 13 | ReAct tool agent and evaluation |
| Nov 14 – Nov 23 | Python-interpreter agent and evaluation |
| Nov 24 – Nov 30 | Comparison across categories and difficulty; tool-usage and failure analysis |
| Dec 1 – Dec 5 | Final experiments, error analysis, optional SFT/RL |
| Dec 6 – Dec 10 | Final report and presentation |

## References

1. Gavrikov et al., VisualOverload: Probing Visual Understanding of VLMs in Really Dense Scenes, CVPR 2026.
2. Antol et al., VQA: Visual Question Answering, ICCV 2015.
3. Goyal et al., Making the V in VQA Matter, CVPR 2017.
4. Campbell et al., Understanding the Limits of VLMs Through the Lens of the Binding Problem, NeurIPS 2024.
5. Paiss et al., Teaching CLIP to Count to Ten, ICCV 2023.
6. Chen et al., Are We on the Right Way for Evaluating Large Vision-Language Models?, NeurIPS 2024.
7. Yao et al., ReAct: Synergizing Reasoning and Acting in Language Models, ICLR 2023.
8. Koh et al., VisualWebArena, ACL 2024.
9. Yang et al., Magma: A Foundation Model for Multimodal AI Agents, CVPR 2025.
10. Fu et al., BLINK: Multimodal LLMs Can See but Not Perceive, ECCV 2025.
