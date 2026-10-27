# TreeBench: tool agent vs base model

Accuracy in %. Fixed = right with the agent, wrong without; broken = the reverse. Tuning set: 40 questions used to develop the agent (reported separately).

## Qwen3.8-27B (think low): base -> TreeBench agent tb2

Base: `treebench__openrouter__qwen_qwen3.8-27b__default__think-low`  
Agent: `treebench__openrouter__qwen_qwen3.8-27b__default__agent-tools-tb2__think-low`  
Questions answered by both: 405

| questions | n | base | agent | change | fixed | broken | McNemar p |
|---|---|---|---|---|---|---|---|
| all | 405 | 59.8 | 68.9 | +9.1 | 66 | 29 | 0.000186 |
| without tuning set | 365 | 61.6 | 69.3 | +7.7 | 57 | 29 | 0.00335 |

| category | n | base | agent | fixed | broken |
|---|---|---|---|---|---|
| **Perception** | 149 | 73.2 | 81.2 | 15 | 3 |
| **Reasoning** | 256 | 52.0 | 61.7 | 51 | 26 |
| Attributes | 29 | 62.1 | 79.3 | 6 | 1 |
| Material | 13 | 69.2 | 69.2 | 0 | 0 |
| OCR | 68 | 83.8 | 92.6 | 6 | 0 |
| Object Retrieval | 16 | 75.0 | 81.2 | 2 | 1 |
| Physical State | 23 | 56.5 | 56.5 | 1 | 1 |
| Comparison | 44 | 47.7 | 54.5 | 8 | 5 |
| Contact and Occlusion | 41 | 53.7 | 56.1 | 5 | 4 |
| Ordering | 57 | 42.1 | 52.6 | 11 | 5 |
| Perspective Transform | 85 | 48.2 | 64.7 | 25 | 11 |
| Spatial Containment | 29 | 86.2 | 89.7 | 2 | 1 |

Tool calls per question: mean 2.35; no tool used on 3 of 405 questions.

| tool | questions using it | agent accuracy when used | base accuracy on those questions |
|---|---|---|---|
| zoom | 363 | 69.1 | 60.1 |
| viewpoint | 80 | 63.8 | 48.8 |
| orient | 76 | 63.2 | 50.0 |
| grid | 47 | 55.3 | 38.3 |
| order | 24 | 62.5 | 58.3 |
| colour | 20 | 75.0 | 50.0 |
| distance | 14 | 35.7 | 28.6 |
| depth | 12 | 58.3 | 50.0 |
| enhance | 12 | 83.3 | 50.0 |
| detect | 10 | 60.0 | 40.0 |

| tool calls | questions | agent accuracy | base accuracy |
|---|---|---|---|
| 0 | 3 | 100.0 | 100.0 |
| 1 | 133 | 82.0 | 78.9 |
| 2 | 112 | 65.2 | 58.9 |
| 3 | 77 | 62.3 | 45.5 |
| 4+ | 80 | 57.5 | 41.2 |

Perspective questions: 85; viewpoint used on 80 (agent 63.8%, base on the same 48.8%); not used on 5 (agent 80.0%).

Evidence: boxes cited in 405 of 405 answers; TreeBench mIoU 34.8 (34.8 when cited); a tool call covered a target on 85.7% of questions.
Accuracy when a tool looked at a target: 68.9% (347); when none did: 69.0% (58).

| | input tokens / question | output tokens / question | cost (provider-reported) |
|---|---|---|---|
| base | 3,434 | 549 | – |
| agent | 38,050 | 1,373 | $2.33 |

## Muse Glimmer 30B (think low): base -> TreeBench agent tb2

Base: `treebench__openrouter__meta_muse-glimmer-30b__default`  
Agent: `treebench__openrouter__meta_muse-glimmer-30b__default__agent-tools-tb2`  
Questions answered by both: 405

| questions | n | base | agent | change | fixed | broken | McNemar p |
|---|---|---|---|---|---|---|---|
| all | 405 | 54.3 | 65.2 | +10.9 | 69 | 25 | 6.34e-06 |
| without tuning set | 365 | 53.4 | 65.8 | +12.3 | 65 | 20 | 1.03e-06 |

| category | n | base | agent | fixed | broken |
|---|---|---|---|---|---|
| **Perception** | 149 | 74.5 | 81.9 | 15 | 4 |
| **Reasoning** | 256 | 42.6 | 55.5 | 54 | 21 |
| Attributes | 29 | 55.2 | 69.0 | 5 | 1 |
| Material | 13 | 61.5 | 69.2 | 1 | 0 |
| OCR | 68 | 86.8 | 89.7 | 5 | 3 |
| Object Retrieval | 16 | 81.2 | 93.8 | 2 | 0 |
| Physical State | 23 | 65.2 | 73.9 | 2 | 0 |
| Comparison | 44 | 47.7 | 54.5 | 7 | 4 |
| Contact and Occlusion | 41 | 56.1 | 48.8 | 1 | 4 |
| Ordering | 57 | 45.6 | 45.6 | 8 | 8 |
| Perspective Transform | 85 | 16.5 | 52.9 | 35 | 4 |
| Spatial Containment | 29 | 86.2 | 93.1 | 3 | 1 |

Tool calls per question: mean 3.12; no tool used on 17 of 405 questions.

| tool | questions using it | agent accuracy when used | base accuracy on those questions |
|---|---|---|---|
| zoom | 265 | 69.4 | 60.8 |
| detect | 194 | 55.7 | 38.7 |
| grid | 85 | 68.2 | 60.0 |
| viewpoint | 74 | 55.4 | 17.6 |
| orient | 71 | 56.3 | 16.9 |
| order | 43 | 41.9 | 46.5 |
| depth | 32 | 59.4 | 43.8 |
| enhance | 21 | 52.4 | 52.4 |
| distance | 18 | 44.4 | 44.4 |
| colour | 16 | 75.0 | 62.5 |
| detect.arguments | 3 | 66.7 | 100.0 |

| tool calls | questions | agent accuracy | base accuracy |
|---|---|---|---|
| 0 | 17 | 58.8 | 52.9 |
| 1 | 130 | 78.5 | 74.6 |
| 2 | 40 | 62.5 | 47.5 |
| 3 | 38 | 63.2 | 39.5 |
| 4+ | 180 | 57.2 | 44.4 |

Perspective questions: 85; viewpoint used on 74 (agent 55.4%, base on the same 17.6%); not used on 11 (agent 36.4%).

Evidence: boxes cited in 405 of 405 answers; TreeBench mIoU 42.8 (42.8 when cited); a tool call covered a target on 74.6% of questions.
Accuracy when a tool looked at a target: 65.9% (302); when none did: 63.1% (103).

| | input tokens / question | output tokens / question | cost (provider-reported) |
|---|---|---|---|
| base | 4,151 | 201 | – |
| agent | 53,931 | 780 | $2.12 |

## Gemma 4 26B-A4B (no think): base -> TreeBench agent tb2

Base: `treebench__openrouter__google_gemma-4-26b-a4b-it__default`  
Agent: `treebench__openrouter__google_gemma-4-26b-a4b-it__default__agent-tools-tb2`  
Questions answered by both: 405

| questions | n | base | agent | change | fixed | broken | McNemar p |
|---|---|---|---|---|---|---|---|
| all | 405 | 43.7 | 62.0 | +18.3 | 105 | 31 | 1.33e-10 |
| without tuning set | 365 | 42.7 | 61.6 | +18.9 | 99 | 30 | 8.37e-10 |

| category | n | base | agent | fixed | broken |
|---|---|---|---|---|---|
| **Perception** | 149 | 57.0 | 79.9 | 42 | 8 |
| **Reasoning** | 256 | 35.9 | 51.6 | 63 | 23 |
| Attributes | 29 | 51.7 | 75.9 | 8 | 1 |
| Material | 13 | 53.8 | 61.5 | 1 | 0 |
| OCR | 68 | 50.0 | 86.8 | 30 | 5 |
| Object Retrieval | 16 | 75.0 | 75.0 | 0 | 0 |
| Physical State | 23 | 73.9 | 78.3 | 3 | 2 |
| Comparison | 44 | 45.5 | 50.0 | 8 | 6 |
| Contact and Occlusion | 41 | 43.9 | 53.7 | 5 | 1 |
| Ordering | 57 | 28.1 | 38.6 | 14 | 8 |
| Perspective Transform | 85 | 18.8 | 49.4 | 34 | 8 |
| Spatial Containment | 29 | 75.9 | 82.8 | 2 | 0 |

Tool calls per question: mean 2.72; no tool used on 4 of 405 questions.

| tool | questions using it | agent accuracy when used | base accuracy on those questions |
|---|---|---|---|
| detect | 357 | 60.5 | 41.2 |
| zoom | 287 | 63.8 | 47.0 |
| orient | 77 | 49.4 | 18.2 |
| viewpoint | 58 | 58.6 | 19.0 |
| order | 40 | 52.5 | 40.0 |
| grid | 22 | 36.4 | 22.7 |
| depth | 21 | 38.1 | 23.8 |
| enhance | 19 | 57.9 | 57.9 |
| distance | 13 | 30.8 | 46.2 |
| colour | 9 | 44.4 | 22.2 |

| tool calls | questions | agent accuracy | base accuracy |
|---|---|---|---|
| 0 | 4 | 75.0 | 100.0 |
| 1 | 55 | 78.2 | 67.3 |
| 2 | 151 | 69.5 | 53.0 |
| 3 | 90 | 52.2 | 31.1 |
| 4+ | 105 | 50.5 | 26.7 |

Perspective questions: 85; viewpoint used on 58 (agent 58.6%, base on the same 19.0%); not used on 27 (agent 29.6%).

Evidence: boxes cited in 404 of 405 answers; TreeBench mIoU 39.4 (39.5 when cited); a tool call covered a target on 77.8% of questions.
Accuracy when a tool looked at a target: 59.0% (315); when none did: 72.2% (90).

| | input tokens / question | output tokens / question | cost (provider-reported) |
|---|---|---|---|
| base | 350 | 15 | – |
| agent | 30,025 | 640 | $0.72 |

## Qwen3.8-27B (think low): base -> Bhushan v2 agent

Base: `treebench__openrouter__qwen_qwen3.8-27b__default__think-low`  
Agent: `treebench__openrouter__qwen_qwen3.8-27b__default__agent-tools-v2__think-low`  
Questions answered by both: 405

| questions | n | base | agent | change | fixed | broken | McNemar p |
|---|---|---|---|---|---|---|---|
| all | 405 | 59.8 | 65.7 | +5.9 | 50 | 26 | 0.00791 |
| without tuning set | 365 | 61.6 | 66.6 | +4.9 | 44 | 26 | 0.0414 |

| category | n | base | agent | fixed | broken |
|---|---|---|---|---|---|
| **Perception** | 149 | 73.2 | 79.2 | 14 | 5 |
| **Reasoning** | 256 | 52.0 | 57.8 | 36 | 21 |
| Attributes | 29 | 62.1 | 72.4 | 4 | 1 |
| Material | 13 | 69.2 | 61.5 | 0 | 1 |
| OCR | 68 | 83.8 | 91.2 | 6 | 1 |
| Object Retrieval | 16 | 75.0 | 81.2 | 2 | 1 |
| Physical State | 23 | 56.5 | 60.9 | 2 | 1 |
| Comparison | 44 | 47.7 | 56.8 | 7 | 3 |
| Contact and Occlusion | 41 | 53.7 | 58.5 | 4 | 2 |
| Ordering | 57 | 42.1 | 49.1 | 9 | 5 |
| Perspective Transform | 85 | 48.2 | 54.1 | 14 | 9 |
| Spatial Containment | 29 | 86.2 | 86.2 | 2 | 2 |

Tool calls per question: mean 1.87; no tool used on 5 of 405 questions.

| tool | questions using it | agent accuracy when used | base accuracy on those questions |
|---|---|---|---|
| zoom | 394 | 65.2 | 59.4 |
| grid | 34 | 44.1 | 38.2 |
| depth | 14 | 64.3 | 50.0 |
| detect | 4 | 50.0 | 50.0 |
| ocr | 1 | 0.0 | 0.0 |

| tool calls | questions | agent accuracy | base accuracy |
|---|---|---|---|
| 0 | 5 | 100.0 | 100.0 |
| 1 | 207 | 72.9 | 71.0 |
| 2 | 104 | 63.5 | 52.9 |
| 3 | 45 | 57.8 | 53.3 |
| 4+ | 44 | 40.9 | 25.0 |

Perspective questions: 85; viewpoint used on 0 (agent –%, base on the same –%); not used on 85 (agent 54.1%).

Evidence: boxes cited in 0 of 405 answers; TreeBench mIoU 0.0 (0.0 when cited); a tool call covered a target on 84.4% of questions.
Accuracy when a tool looked at a target: 65.5% (342); when none did: 66.7% (63).

| | input tokens / question | output tokens / question | cost (provider-reported) |
|---|---|---|---|
| base | 3,434 | 549 | – |
| agent | 20,532 | 1,247 | $1.78 |

## Muse Glimmer 30B: base -> Bhushan v2 agent

Base: `treebench__openrouter__meta_muse-glimmer-30b__default`  
Agent: `treebench__openrouter__meta_muse-glimmer-30b__default__agent-tools-v2`  
Questions answered by both: 405

| questions | n | base | agent | change | fixed | broken | McNemar p |
|---|---|---|---|---|---|---|---|
| all | 405 | 54.3 | 56.8 | +2.5 | 42 | 32 | 0.295 |
| without tuning set | 365 | 53.4 | 56.2 | +2.7 | 39 | 29 | 0.275 |

| category | n | base | agent | fixed | broken |
|---|---|---|---|---|---|
| **Perception** | 149 | 74.5 | 76.5 | 13 | 10 |
| **Reasoning** | 256 | 42.6 | 45.3 | 29 | 22 |
| Attributes | 29 | 55.2 | 65.5 | 5 | 2 |
| Material | 13 | 61.5 | 61.5 | 0 | 0 |
| OCR | 68 | 86.8 | 85.3 | 3 | 4 |
| Object Retrieval | 16 | 81.2 | 75.0 | 2 | 3 |
| Physical State | 23 | 65.2 | 73.9 | 3 | 1 |
| Comparison | 44 | 47.7 | 50.0 | 8 | 7 |
| Contact and Occlusion | 41 | 56.1 | 51.2 | 3 | 5 |
| Ordering | 57 | 45.6 | 49.1 | 9 | 7 |
| Perspective Transform | 85 | 16.5 | 22.4 | 7 | 2 |
| Spatial Containment | 29 | 86.2 | 89.7 | 2 | 1 |

Tool calls per question: mean 1.81; no tool used on 81 of 405 questions.

| tool | questions using it | agent accuracy when used | base accuracy on those questions |
|---|---|---|---|
| zoom | 272 | 61.8 | 59.2 |
| detect | 99 | 42.4 | 39.4 |
| depth | 57 | 35.1 | 42.1 |
| grid | 33 | 39.4 | 51.5 |
| ocr | 5 | 40.0 | 20.0 |

| tool calls | questions | agent accuracy | base accuracy |
|---|---|---|---|
| 0 | 81 | 48.1 | 45.7 |
| 1 | 183 | 71.6 | 66.7 |
| 2 | 40 | 40.0 | 30.0 |
| 3 | 24 | 33.3 | 45.8 |
| 4+ | 77 | 46.8 | 49.4 |

Perspective questions: 85; viewpoint used on 0 (agent –%, base on the same –%); not used on 85 (agent 22.4%).

Evidence: boxes cited in 0 of 405 answers; TreeBench mIoU 0.0 (0.0 when cited); a tool call covered a target on 64.0% of questions.
Accuracy when a tool looked at a target: 59.8% (259); when none did: 51.4% (146).

| | input tokens / question | output tokens / question | cost (provider-reported) |
|---|---|---|---|
| base | 4,151 | 201 | – |
| agent | 23,596 | 622 | $1.31 |

## Gemma 4 26B-A4B: base -> Bhushan v2 agent

Base: `treebench__openrouter__google_gemma-4-26b-a4b-it__default`  
Agent: `treebench__openrouter__google_gemma-4-26b-a4b-it__default__agent-tools-v2`  
Questions answered by both: 405

| questions | n | base | agent | change | fixed | broken | McNemar p |
|---|---|---|---|---|---|---|---|
| all | 405 | 43.7 | 51.9 | +8.1 | 72 | 39 | 0.00224 |
| without tuning set | 365 | 42.7 | 50.4 | +7.7 | 66 | 38 | 0.0078 |

| category | n | base | agent | fixed | broken |
|---|---|---|---|---|---|
| **Perception** | 149 | 57.0 | 76.5 | 40 | 11 |
| **Reasoning** | 256 | 35.9 | 37.5 | 32 | 28 |
| Attributes | 29 | 51.7 | 58.6 | 4 | 2 |
| Material | 13 | 53.8 | 53.8 | 0 | 0 |
| OCR | 68 | 50.0 | 86.8 | 31 | 6 |
| Object Retrieval | 16 | 75.0 | 68.8 | 0 | 1 |
| Physical State | 23 | 73.9 | 87.0 | 5 | 2 |
| Comparison | 44 | 45.5 | 36.4 | 5 | 9 |
| Contact and Occlusion | 41 | 43.9 | 58.5 | 7 | 1 |
| Ordering | 57 | 28.1 | 28.1 | 9 | 9 |
| Perspective Transform | 85 | 18.8 | 21.2 | 10 | 8 |
| Spatial Containment | 29 | 75.9 | 75.9 | 1 | 1 |

Tool calls per question: mean 2.80; no tool used on 1 of 405 questions.

| tool | questions using it | agent accuracy when used | base accuracy on those questions |
|---|---|---|---|
| detect | 360 | 49.4 | 42.8 |
| zoom | 344 | 54.7 | 45.1 |
| grid | 71 | 54.9 | 39.4 |
| depth | 34 | 26.5 | 32.4 |
| ocr | 24 | 58.3 | 33.3 |

| tool calls | questions | agent accuracy | base accuracy |
|---|---|---|---|
| 0 | 1 | 0.0 | 0.0 |
| 1 | 53 | 50.9 | 47.2 |
| 2 | 161 | 60.2 | 51.6 |
| 3 | 73 | 52.1 | 46.6 |
| 4+ | 117 | 41.0 | 29.9 |

Perspective questions: 85; viewpoint used on 0 (agent –%, base on the same –%); not used on 85 (agent 21.2%).

Evidence: boxes cited in 0 of 405 answers; TreeBench mIoU 0.0 (0.0 when cited); a tool call covered a target on 76.5% of questions.
Accuracy when a tool looked at a target: 51.9% (310); when none did: 51.6% (95).

| | input tokens / question | output tokens / question | cost (provider-reported) |
|---|---|---|---|
| base | 350 | 15 | – |
| agent | 15,398 | 488 | $0.40 |

## Qwen: v2 agent -> tb2 agent

Base: `treebench__openrouter__qwen_qwen3.8-27b__default__agent-tools-v2__think-low`  
Agent: `treebench__openrouter__qwen_qwen3.8-27b__default__agent-tools-tb2__think-low`  
Questions answered by both: 405

| questions | n | base | agent | change | fixed | broken | McNemar p |
|---|---|---|---|---|---|---|---|
| all | 405 | 65.7 | 68.9 | +3.2 | 44 | 31 | 0.165 |
| without tuning set | 365 | 66.6 | 69.3 | +2.7 | 40 | 30 | 0.282 |

| category | n | base | agent | fixed | broken |
|---|---|---|---|---|---|
| **Perception** | 149 | 79.2 | 81.2 | 6 | 3 |
| **Reasoning** | 256 | 57.8 | 61.7 | 38 | 28 |
| Attributes | 29 | 72.4 | 79.3 | 3 | 1 |
| Material | 13 | 61.5 | 69.2 | 1 | 0 |
| OCR | 68 | 91.2 | 92.6 | 2 | 1 |
| Object Retrieval | 16 | 81.2 | 81.2 | 0 | 0 |
| Physical State | 23 | 60.9 | 56.5 | 0 | 1 |
| Comparison | 44 | 56.8 | 54.5 | 5 | 6 |
| Contact and Occlusion | 41 | 58.5 | 56.1 | 3 | 4 |
| Ordering | 57 | 49.1 | 52.6 | 4 | 2 |
| Perspective Transform | 85 | 54.1 | 64.7 | 25 | 16 |
| Spatial Containment | 29 | 86.2 | 89.7 | 1 | 0 |

Tool calls per question: mean 2.35; no tool used on 3 of 405 questions.

| tool | questions using it | agent accuracy when used | base accuracy on those questions |
|---|---|---|---|
| zoom | 363 | 69.1 | 66.1 |
| viewpoint | 80 | 63.8 | 55.0 |
| orient | 76 | 63.2 | 55.3 |
| grid | 47 | 55.3 | 48.9 |
| order | 24 | 62.5 | 58.3 |
| colour | 20 | 75.0 | 70.0 |
| distance | 14 | 35.7 | 50.0 |
| depth | 12 | 58.3 | 58.3 |
| enhance | 12 | 83.3 | 75.0 |
| detect | 10 | 60.0 | 50.0 |

| tool calls | questions | agent accuracy | base accuracy |
|---|---|---|---|
| 0 | 3 | 100.0 | 100.0 |
| 1 | 133 | 82.0 | 78.2 |
| 2 | 112 | 65.2 | 67.0 |
| 3 | 77 | 62.3 | 53.2 |
| 4+ | 80 | 57.5 | 53.8 |

Perspective questions: 85; viewpoint used on 80 (agent 63.8%, base on the same 55.0%); not used on 5 (agent 80.0%).

Evidence: boxes cited in 405 of 405 answers; TreeBench mIoU 34.8 (34.8 when cited); a tool call covered a target on 85.7% of questions.
Accuracy when a tool looked at a target: 68.9% (347); when none did: 69.0% (58).

| | input tokens / question | output tokens / question | cost (provider-reported) |
|---|---|---|---|
| base | 20,532 | 1,247 | $1.78 |
| agent | 38,050 | 1,373 | $2.33 |

## Muse: v2 agent -> tb2 agent

Base: `treebench__openrouter__meta_muse-glimmer-30b__default__agent-tools-v2`  
Agent: `treebench__openrouter__meta_muse-glimmer-30b__default__agent-tools-tb2`  
Questions answered by both: 405

| questions | n | base | agent | change | fixed | broken | McNemar p |
|---|---|---|---|---|---|---|---|
| all | 405 | 56.8 | 65.2 | +8.4 | 64 | 30 | 0.000588 |
| without tuning set | 365 | 56.2 | 65.8 | +9.6 | 60 | 25 | 0.000187 |

| category | n | base | agent | fixed | broken |
|---|---|---|---|---|---|
| **Perception** | 149 | 76.5 | 81.9 | 12 | 4 |
| **Reasoning** | 256 | 45.3 | 55.5 | 52 | 26 |
| Attributes | 29 | 65.5 | 69.0 | 2 | 1 |
| Material | 13 | 61.5 | 69.2 | 1 | 0 |
| OCR | 68 | 85.3 | 89.7 | 5 | 2 |
| Object Retrieval | 16 | 75.0 | 93.8 | 3 | 0 |
| Physical State | 23 | 73.9 | 73.9 | 1 | 1 |
| Comparison | 44 | 50.0 | 54.5 | 9 | 7 |
| Contact and Occlusion | 41 | 51.2 | 48.8 | 2 | 3 |
| Ordering | 57 | 49.1 | 45.6 | 5 | 7 |
| Perspective Transform | 85 | 22.4 | 52.9 | 35 | 9 |
| Spatial Containment | 29 | 89.7 | 93.1 | 1 | 0 |

Tool calls per question: mean 3.12; no tool used on 17 of 405 questions.

| tool | questions using it | agent accuracy when used | base accuracy on those questions |
|---|---|---|---|
| zoom | 265 | 69.4 | 63.8 |
| detect | 194 | 55.7 | 43.8 |
| grid | 85 | 68.2 | 54.1 |
| viewpoint | 74 | 55.4 | 21.6 |
| orient | 71 | 56.3 | 21.1 |
| order | 43 | 41.9 | 53.5 |
| depth | 32 | 59.4 | 50.0 |
| enhance | 21 | 52.4 | 57.1 |
| distance | 18 | 44.4 | 38.9 |
| colour | 16 | 75.0 | 68.8 |
| detect.arguments | 3 | 66.7 | 100.0 |

| tool calls | questions | agent accuracy | base accuracy |
|---|---|---|---|
| 0 | 17 | 58.8 | 64.7 |
| 1 | 130 | 78.5 | 75.4 |
| 2 | 40 | 62.5 | 52.5 |
| 3 | 38 | 63.2 | 44.7 |
| 4+ | 180 | 57.2 | 46.1 |

Perspective questions: 85; viewpoint used on 74 (agent 55.4%, base on the same 21.6%); not used on 11 (agent 36.4%).

Evidence: boxes cited in 405 of 405 answers; TreeBench mIoU 42.8 (42.8 when cited); a tool call covered a target on 74.6% of questions.
Accuracy when a tool looked at a target: 65.9% (302); when none did: 63.1% (103).

| | input tokens / question | output tokens / question | cost (provider-reported) |
|---|---|---|---|
| base | 23,596 | 622 | $1.31 |
| agent | 53,931 | 780 | $2.12 |

## Gemma: v2 agent -> tb2 agent

Base: `treebench__openrouter__google_gemma-4-26b-a4b-it__default__agent-tools-v2`  
Agent: `treebench__openrouter__google_gemma-4-26b-a4b-it__default__agent-tools-tb2`  
Questions answered by both: 405

| questions | n | base | agent | change | fixed | broken | McNemar p |
|---|---|---|---|---|---|---|---|
| all | 405 | 51.9 | 62.0 | +10.1 | 70 | 29 | 4.61e-05 |
| without tuning set | 365 | 50.4 | 61.6 | +11.2 | 66 | 25 | 2.03e-05 |

| category | n | base | agent | fixed | broken |
|---|---|---|---|---|---|
| **Perception** | 149 | 76.5 | 79.9 | 13 | 8 |
| **Reasoning** | 256 | 37.5 | 51.6 | 57 | 21 |
| Attributes | 29 | 58.6 | 75.9 | 5 | 0 |
| Material | 13 | 53.8 | 61.5 | 1 | 0 |
| OCR | 68 | 86.8 | 86.8 | 5 | 5 |
| Object Retrieval | 16 | 68.8 | 75.0 | 1 | 0 |
| Physical State | 23 | 87.0 | 78.3 | 1 | 3 |
| Comparison | 44 | 36.4 | 50.0 | 9 | 3 |
| Contact and Occlusion | 41 | 58.5 | 53.7 | 2 | 4 |
| Ordering | 57 | 28.1 | 38.6 | 11 | 5 |
| Perspective Transform | 85 | 21.2 | 49.4 | 33 | 9 |
| Spatial Containment | 29 | 75.9 | 82.8 | 2 | 0 |

Tool calls per question: mean 2.72; no tool used on 4 of 405 questions.

| tool | questions using it | agent accuracy when used | base accuracy on those questions |
|---|---|---|---|
| detect | 357 | 60.5 | 50.1 |
| zoom | 287 | 63.8 | 57.1 |
| orient | 77 | 49.4 | 18.2 |
| viewpoint | 58 | 58.6 | 19.0 |
| order | 40 | 52.5 | 45.0 |
| grid | 22 | 36.4 | 18.2 |
| depth | 21 | 38.1 | 23.8 |
| enhance | 19 | 57.9 | 63.2 |
| distance | 13 | 30.8 | 38.5 |
| colour | 9 | 44.4 | 66.7 |

| tool calls | questions | agent accuracy | base accuracy |
|---|---|---|---|
| 0 | 4 | 75.0 | 75.0 |
| 1 | 55 | 78.2 | 72.7 |
| 2 | 151 | 69.5 | 65.6 |
| 3 | 90 | 52.2 | 40.0 |
| 4+ | 105 | 50.5 | 30.5 |

Perspective questions: 85; viewpoint used on 58 (agent 58.6%, base on the same 19.0%); not used on 27 (agent 29.6%).

Evidence: boxes cited in 404 of 405 answers; TreeBench mIoU 39.4 (39.5 when cited); a tool call covered a target on 77.8% of questions.
Accuracy when a tool looked at a target: 59.0% (315); when none did: 72.2% (90).

| | input tokens / question | output tokens / question | cost (provider-reported) |
|---|---|---|---|
| base | 15,398 | 488 | $0.40 |
| agent | 30,025 | 640 | $0.72 |

## A1 Qwen: tb2 -> tb2 without orient+viewpoint (perspective)

Base: `treebench__openrouter__qwen_qwen3.8-27b__default__agent-tools-tb2__think-low`  
Agent: `treebench__openrouter__qwen_qwen3.8-27b__default__agent-tools-zoom+grid+enhance+detect+order+depth+distance+colour-tb2__think-low`  
Questions answered by both: 85

| questions | n | base | agent | change | fixed | broken | McNemar p |
|---|---|---|---|---|---|---|---|
| all | 85 | 64.7 | 56.5 | -8.2 | 14 | 21 | 0.311 |
| without tuning set | 81 | 64.2 | 58.0 | -6.2 | 14 | 19 | 0.487 |

| category | n | base | agent | fixed | broken |
|---|---|---|---|---|---|
| **Reasoning** | 85 | 64.7 | 56.5 | 14 | 21 |
| Perspective Transform | 85 | 64.7 | 56.5 | 14 | 21 |

Tool calls per question: mean 2.19; no tool used on 1 of 85 questions.

| tool | questions using it | agent accuracy when used | base accuracy on those questions |
|---|---|---|---|
| zoom | 83 | 56.6 | 66.3 |
| grid | 11 | 45.5 | 72.7 |
| depth | 9 | 88.9 | 77.8 |
| detect | 3 | 66.7 | 33.3 |

| tool calls | questions | agent accuracy | base accuracy |
|---|---|---|---|
| 0 | 1 | 0.0 | 0.0 |
| 1 | 35 | 62.9 | 65.7 |
| 2 | 18 | 50.0 | 61.1 |
| 3 | 17 | 70.6 | 82.4 |
| 4+ | 14 | 35.7 | 50.0 |

Perspective questions: 85; viewpoint used on 0 (agent –%, base on the same –%); not used on 85 (agent 56.5%).

Evidence: boxes cited in 85 of 85 answers; TreeBench mIoU 47.7 (47.7 when cited); a tool call covered a target on 92.9% of questions.
Accuracy when a tool looked at a target: 59.5% (79); when none did: 16.7% (6).

| | input tokens / question | output tokens / question | cost (provider-reported) |
|---|---|---|---|
| base | 47,156 | 1,637 | $0.64 |
| agent | 30,890 | 2,174 | $0.60 |

## A2 Qwen: tb2 -> tb2 without order (ordering)

Base: `treebench__openrouter__qwen_qwen3.8-27b__default__agent-tools-tb2__think-low`  
Agent: `treebench__openrouter__qwen_qwen3.8-27b__default__agent-tools-zoom+grid+enhance+detect+depth+distance+colour+orient+viewpoint-tb2__think-low`  
Questions answered by both: 57

| questions | n | base | agent | change | fixed | broken | McNemar p |
|---|---|---|---|---|---|---|---|
| all | 57 | 52.6 | 43.9 | -8.8 | 2 | 7 | 0.18 |
| without tuning set | 53 | 54.7 | 43.4 | -11.3 | 1 | 7 | 0.0703 |

| category | n | base | agent | fixed | broken |
|---|---|---|---|---|---|
| **Reasoning** | 57 | 52.6 | 43.9 | 2 | 7 |
| Ordering | 57 | 52.6 | 43.9 | 2 | 7 |

Tool calls per question: mean 2.88; no tool used on 0 of 57 questions.

| tool | questions using it | agent accuracy when used | base accuracy on those questions |
|---|---|---|---|
| zoom | 56 | 44.6 | 51.8 |
| grid | 15 | 20.0 | 26.7 |
| detect | 8 | 37.5 | 37.5 |
| depth | 4 | 50.0 | 50.0 |
| colour | 2 | 50.0 | 50.0 |
| enhance | 2 | 0.0 | 50.0 |
| distance | 1 | 0.0 | 100.0 |

| tool calls | questions | agent accuracy | base accuracy |
|---|---|---|---|
| 1 | 9 | 55.6 | 77.8 |
| 2 | 17 | 47.1 | 47.1 |
| 3 | 13 | 46.2 | 46.2 |
| 4+ | 18 | 33.3 | 50.0 |

Evidence: boxes cited in 57 of 57 answers; TreeBench mIoU 47.7 (47.7 when cited); a tool call covered a target on 89.5% of questions.
Accuracy when a tool looked at a target: 47.1% (51); when none did: 16.7% (6).

| | input tokens / question | output tokens / question | cost (provider-reported) |
|---|---|---|---|
| base | 47,026 | 2,153 | $0.44 |
| agent | 43,160 | 2,248 | $0.49 |

## A3 Qwen: tb2 -> zoom only (perception)

Base: `treebench__openrouter__qwen_qwen3.8-27b__default__agent-tools-tb2__think-low`  
Agent: `treebench__openrouter__qwen_qwen3.8-27b__default__agent-tools-zoom-tb2__think-low`  
Questions answered by both: 149

| questions | n | base | agent | change | fixed | broken | McNemar p |
|---|---|---|---|---|---|---|---|
| all | 149 | 81.2 | 79.2 | -2.0 | 3 | 6 | 0.508 |
| without tuning set | 129 | 82.9 | 82.2 | -0.8 | 3 | 4 | 1 |

| category | n | base | agent | fixed | broken |
|---|---|---|---|---|---|
| **Perception** | 149 | 81.2 | 79.2 | 3 | 6 |
| Attributes | 29 | 79.3 | 69.0 | 1 | 4 |
| Material | 13 | 69.2 | 69.2 | 0 | 0 |
| OCR | 68 | 92.6 | 89.7 | 0 | 2 |
| Object Retrieval | 16 | 81.2 | 87.5 | 1 | 0 |
| Physical State | 23 | 56.5 | 60.9 | 1 | 0 |

Tool calls per question: mean 1.58; no tool used on 2 of 149 questions.

| tool | questions using it | agent accuracy when used | base accuracy on those questions |
|---|---|---|---|
| zoom | 147 | 78.9 | 81.0 |

| tool calls | questions | agent accuracy | base accuracy |
|---|---|---|---|
| 0 | 2 | 100.0 | 100.0 |
| 1 | 96 | 86.5 | 86.5 |
| 2 | 31 | 61.3 | 71.0 |
| 3 | 9 | 66.7 | 77.8 |
| 4+ | 11 | 72.7 | 63.6 |

Evidence: boxes cited in 148 of 149 answers; TreeBench mIoU 27.9 (28.1 when cited); a tool call covered a target on 77.2% of questions.
Accuracy when a tool looked at a target: 81.7% (115); when none did: 70.6% (34).

| | input tokens / question | output tokens / question | cost (provider-reported) |
|---|---|---|---|
| base | 31,112 | 835 | $0.61 |
| agent | 13,556 | 751 | $0.46 |

