# Evidence-Gated Dual Subspace (EGDS)

**Status.** Theory + method design for a primarily *training-free* intervention on tool-using VLMs for dense / trick-heavy VQA (VistaQA-style).  
**Scope.** Open-weight VLMs with residual-stream access (local). Hosted API models are behavioral baselines only.  
**Non-contamination.** Offline subspace construction uses COCO / FP-RefCOCO-style / SA-1B-style sources. Do **not** use VistaQA, TreeBench, or MME-RealWorld for subspace fitting or steering-vector mining.

---

## 1. Motivation

### 1.1 What we observed on VistaQA tool agents

On VistaQA with a ReAct-style tool agent (zoom / grid / detect / OCR / depth), medium-reasoning traces show a failure mixture that is *not* uniform:

| Failure family | Approx. share of failures | Typical symptom | Tool interaction |
|---|---|---|---|
| **premise_accepted** | ~19% (trick-only) | Model answers as if a queried absent object / relation / attribute exists | Tools often *hurt* (esp. Gemma): empty or noisy search is still verbalized into a committed answer |
| **miscount** | ~16% (mostly normal; ~72% undercount) | Wrong cardinality of visible instances | Tools help sometimes but single wrong detect/OCR reading locks the trajectory |
| **perception_error** | ~9% | Color / identity / pips / clock / fine attributes | Needs better pixels; internal edit alone is weak |

Trick items are ~27% of VistaQA and carry that weight in GROVE. Abstention / empty-evidence behavior is therefore a high-leverage lever, not a niche safety add-on.

### 1.2 The conceptual gap tools do not close by themselves

External tools attack **missing pixels** (tiny objects, clutter). They do not automatically attack:

1. **Language-prior object invention** when the question presupposes an entity.
2. **The commitment to answer** when visual/tool evidence is insufficient.
3. **Misreading of negative tool results** (“no boxes” interpreted as “I looked, so the answer is still X”).

In other words: tools change the *context*, but the model’s *internal decision geometry* can still prefer fluent, premise-consistent answers. That is the gap EGDS targets.

### 1.3 Why a theory is needed (not another prompt)

Prompting “abstain if unsure” is cheap but empirically weak when:

- the question is phrased as if the object exists (false premise),
- multi-step tool CoT amplifies textual priors (HalluSegBench observation on CoT/RL segmenters),
- a single confident wrong tool reading is invisible to intra-trajectory uncertainty (TTSP).

EGDS therefore combines three ideas that separately have strong empirical support:

- **low-rank hallucination directions** in hidden states (Nullu, CIPHER),
- **linearly encoded but poorly expressed abstention** (TRAPSBench, Dr. No),
- **evidence-first gating from detectors / existence checks** (Woodpecker, SESAME cascade).

---

## 2. What we know from previous methods

This section is the literature premise bank. EGDS does not invent these claims; it *composes* them.

### 2.1 Feature-level / subspace hallucination suppression

#### Nullu (Yang et al., CVPR 2025)

- **Idea.** Construct paired inputs with the *same image* and *different text*: truthful caption vs GPT-perturbed hallucinated caption.
- **Math sketch.** At layer \(\ell\), mean-pool token embeddings for hallucinated and truthful runs into matrices \(X^+_\ell, X^-_\ell\). Form
  \[
  E_\ell = X^+_\ell - X^-_\ell \in \mathbb{R}^{N \times d}.
  \]
  SVD: \(E_\ell = U_\ell \Sigma_\ell V_\ell^\top\). Top-\(k\) right singular vectors \(V_{\ell,k}\) span **HalluSpace**.
- **Intervention.** Orthogonalize MLP weights (or equivalently project features) into the null space of HalluSpace:
  \[
  W^{\text{ed}} = \big(I - V_{\ell,k} V_{\ell,k}^\top\big) W^{\text{org}}.
  \]
- **Interpretation.** HalluSpace often encodes **LLM priors** that drive object hallucination when vision is weak.
- **Limitation for us.** Text-side counterfactuals; captioning-oriented; no tool channel; no explicit abstention direction.

#### CIPHER (Dastmalchi et al., arXiv:2603.10470)

- **Idea.** Target *vision-induced* hallucination by counterfactual **images**, not counterfactual captions.
- **Offline data (OHC-25K).** From MSCOCO image–caption pairs \((I_i, C_i)\):
  1. GPT perturbs \(C_i \to \tilde C_i\) (add / swap objects).
  2. Partial forward diffusion on \(z_0 = \mathcal{E}(I_i)\) to timestep \(t_h\) (paper uses \(t_h = 0.5T\)):
     \[
     \tilde z_{t_h} = \sqrt{\bar\alpha_{t_h}}\, z_0 + \sqrt{1-\bar\alpha_{t_h}}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I).
     \]
  3. Reverse diffusion conditioned on \(\tilde C_i\) yields \(\tilde I_{i,j}\) (typically \(B=5\) seeds).
  4. Pair **hallucinated image with true caption**: \((\tilde I_{i,j}, C_i)\). This creates semantic conflict while preserving layout.
- **Subspace estimation.** For layer \(\ell\),
  \[
  h^{(i)}_\ell = \frac{1}{N}\sum_{k=1}^{N} h^{(i)}_{\ell,k},\quad
  \tilde h^{(i)}_\ell = \frac{1}{B}\sum_{j=1}^{B}\tilde h^{(i,j)}_\ell,
  \]
  \[
  \delta^{(i)}_\ell = \tilde h^{(i)}_\ell - h^{(i)}_\ell,\quad
  \Delta_\ell = [\delta^{(1)}_\ell;\ldots;\delta^{(M)}_\ell] = U_\ell\Sigma_\ell V_\ell^\top.
  \]
  Keep top \(r\) right singular vectors \(V_{\ell,r}\).
- **Inference nullification.** At each decode step \(k\),
  \[
  h^{\text{clean}}_{\ell,k} = h^{\text{test}}_{\ell,k} - \sum_{j=1}^{r}\langle h^{\text{test}}_{\ell,k}, v_{\ell,j}\rangle\, v_{\ell,j}
  = P_\ell h^{\text{test}}_{\ell,k},\quad P_\ell = I - V_{\ell,r}V_{\ell,r}^\top.
  \]
- **Empirical note.** Beats Nullu on CHAIR across LLaVA-1.5 / MiniGPT-4 / mPLUG-Owl2 while preserving BLEU; projection typically on upper layers (e.g. 16–32), rank model-dependent (\(r\in\{8,32,64\}\)).
- **Limitation for us.** Single-pass captioning; no VQA false premises; no tools; always-on projection can harm answerable dense VQA if the subspace overlaps useful features.

#### Related decoding-time methods (context, not EGDS core)

| Method | Mechanism | Cost | Relevance |
|---|---|---|---|
| **VCD** | Contrast logits from original vs *distorted* image | 2× forward | Vision-prior debiasing at tokens |
| **ICD** | Contrast original vs *instruction-disturbed* prompts | 2× | Language-side contrast |
| **DoLa** | Contrast early vs late layers | moderate | Layer-wise factuality |
| **OPERA** | Attention / beam constraints | moderate | Decoding regularity |
| **HALC** | Adaptive decoding with visual checks | higher | Mix of decode + vision |

EGDS prefers **feature-level** edits (CIPHER/Nullu family) because they are single-pass at inference once the basis is cached, and they compose cleanly with an evidence gate.

### 2.2 Abstention is encoded but not expressed

#### TRAPSBench (void / control steering)

- **Finding.** Across many VLMs, spontaneous abstention under missing visual evidence is poor.
- **Critical result.** Linear probes decode answerability / “void” from hidden states at high AUROC (reported up to ~0.91), yet models still answer.
- **Causal test.** Define a void direction at layer \(\ell\):
  \[
  v_\ell = \frac{\bar h^{\text{void}}_\ell - \bar h^{\text{control}}_\ell}{\|\bar h^{\text{void}}_\ell - \bar h^{\text{control}}_\ell\|}.
  \]
  Steer during generation:
  \[
  h^{(t)}_\ell \leftarrow h^{(t)}_\ell + \alpha_{\text{eff}}\, v_\ell.
  \]
  Positive \(\alpha\) can induce abstention; negative can force confabulation. Visual uncertainty is harder to express than textual impossibility (~4× gap in some analyses).
- **Implication for EGDS.** Probing / readout-only interpretability can look “useless for accuracy” even when the right direction exists. You need **causal steering**, and typically a way to open an **output gate**.

#### Dr. No (SAE feature steering for abstention)

- Mines sparse features associated with abstention vs commitment.
- Steers a small set of features, often in the **prefill** pass.
- **Input-dependent gate** is essential: always-on abstention steering over-abstains on answerable items; gated steering improves the answer/abstain trade-off.
- **Implication for EGDS.** The scalar evidence score \(e\) is not optional polish; it is part of the theory.

### 2.3 External evidence gating (tools as verifiers, not oracles)

#### Woodpecker (post-hoc correction)

Pipeline:

1. Extract key concepts from the model’s text.
2. Formulate existence / count / attribute questions.
3. Validate with open-set detector + short-form VQA.
4. Build visual claims (“there are \(N\) X” / “there is no X”).
5. Rewrite the answer.

Shows large POPE gains; precision↑ can trade recall↓. Position/relation verification is weaker than object existence.

#### SESAME cascade

- Segmentation LMMs forget “see” (existence) under fine-tuning.
- **Training-free cascade:** ask an MLLM “is X present?” first; only then segment.
- Large FP-RefCOCO “See” gains vs segment-everything baselines.
- **Implication.** Existence should be a **gate**, not a side comment inside free-form CoT.

#### HalluSegBench / RobustSeg

- Vision-driven hallucinations dominate label-driven ones.
- Multi-step reasoning can amplify text priors.
- Counterfactual negatives (object removed/replaced) with empty-mask targets restore abstention.

#### Reliable-V-RL

- Accuracy-only RL **destroys** abstention quickly.
- Explicit unanswerable rewards are required if one later trains.
- EGDS is training-free, but this warns against “just GRPO on accuracy” as a substitute for the theory.

### 2.4 Tool-agent reliability literature (why tools can hurt)

| Source | Relevant claim |
|---|---|
| **Pixel Reasoner** | Models hallucinate that a tool succeeded and ignore error / empty results |
| **MAPO** | Plan says “zoom on motorcycle” but crop is irrelevant; needs label–crop consistency checks |
| **TTSP** | Confident hallucination is invisible to intra-trajectory entropy; cross-trajectory consensus + evidence ledger help; negative knowledge (“searched, not found”) should be first-class |
| **SRICE** | Calibrate segmentation/detection and choose pathways by uncertainty |
| **VistaQA paper** | Hybrid “VLM + segmenter” beats dedicated reason-seg; abstention calibration is a cheap GROVE lever |

### 2.5 Relation to prior internal-interp experiments in this project

The project’s J-space / J-lens plan (`notes/jspace-vqa-experiment-plan.md`) hypothesized that verbalizable workspace contents are causal for trick / wrong-intermediate failures. Early experiments saw limited accuracy gains. Under the literature above, that is **expected** if interventions were:

- ungated (always on),
- readout-only (no causal steer),
- applied at the wrong time (final answer only, not post-tool),
- built from the wrong contrast (generic captioning HalluSpace, not void / false-premise),
- or run without a tool-derived evidence score.

EGDS is designed as a *narrower* interpretability-for-performance bet that absorbs those failure modes of experimentation.

---

## 3. How previous methods are incorporated

| Prior method | What we take | What we discard / change | Where it appears in EGDS |
|---|---|---|---|
| **CIPHER** | Vision counterfactuals → \(\delta=\tilde h-h\) → SVD HalluSpace → null projection \(P^H\) | Always-on projection; caption-only eval mindset | Offline Stage A; Inference Stage 3 when \(e\) low–mid |
| **Nullu** | Text-side HalluSpace as optional second bank; LLM-prior interpretation | Weight editing as primary (we prefer activation edit for tool timing) | Optional Offline Stage A′ |
| **TRAPSBench** | Void direction \(v_{\text{void}}\); causal steer; representation–expression gap | Video-physics-specific benchmark construction | Offline Stage B; Inference Stage 3 |
| **Dr. No** | Input-dependent gate; abstain vs commit features | SAE requirement (we start with difference-of-means / SVD; SAE optional upgrade) | Gate \(\alpha(e)\) |
| **Woodpecker / SESAME** | Detector/existence as evidence; see-before-commit | Full post-hoc rewrite as the *only* fix | Evidence score \(e\) from tools (Stage 2) |
| **VCD** | Distorted-vision contrast as optional logit-level assist on high-\(e\) perception | Using VCD as the main method | Optional Stage 4 for miscount/perception |
| **TTSP** | Negative knowledge; conflict → lower evidence; don’t trust one tool read | Full multi-round expensive ledger as mandatory | \(e\) penalties; optional multi-sample extension |
| **FP-RefCOCO / SESAME data** | False-premise pairs for VoidSpace mining | Training a segmenter | Offline Stage B data |
| **Reliable-V-RL** | Don’t optimize accuracy alone later | Not used in training-free EGDS core | Future RL compatibility note |

**Incorporation principle.**  
CIPHER removes “invented object” directions; TRAPS/Dr.No add “should abstain” directions; Woodpecker/SESAME decide *when* each applies via tools. EGDS is that composition with explicit math and timing relative to the tool loop.

---

## 4. The theory (informal statement)

### 4.1 Informal theory

> In dense VQA, especially under false premises and tool use, errors are dominated by a **prior-driven commitment** along low-rank residual directions, while the model **already encodes** a usable “evidence insufficient / void” signal that is **under-expressed** at the output. External tools primarily change the available evidence; they do not automatically open the abstention pathway. Therefore, tool outputs should be reduced to an **evidence score** \(e\), and hidden states should be edited as a **gated dual operation**: (i) amplify the void direction when \(e\) is low, and (ii) project out vision-hallucination directions that support inventing entities. When \(e\) is high, leave the representation nearly untouched so counting and fine perception are not damaged.

### 4.2 Formal theory sketch

Let \(h_\ell \in \mathbb{R}^d\) be the residual stream at layer \(\ell\) (token position as specified later). Assume, locally, next-token logits are approximately affine in \(h_\ell\):

\[
z = W_{\text{U}}\, h_\ell + b.
\]

Decompose the pre-edit state as

\[
h_\ell = h^\star_\ell + a\, u_{\text{hall}} + b\, u_{\text{void}} + \varepsilon,
\]

where:

- \(h^\star_\ell\) is the “task-useful” component (including genuine visual binding),
- \(u_{\text{hall}}\) spans directions that increase probability of **unsupported object / attribute claims** (CIPHER/Nullu HalluSpace),
- \(u_{\text{void}}\) spans directions that increase probability of **abstention / not-present** (TRAPS void),
- \(\varepsilon\) is residual unstructured noise.

**Claim 1 (low-rank structure).** Empirically, the systematic shift between clean and hallucinated vision (or truthful vs hallucinated text) concentrates in a low-rank subspace, so \(u_{\text{hall}}\) can be estimated by SVD of difference matrices.

**Claim 2 (representation–expression gap).** On false-premise / insufficient-evidence inputs, \(b\) (void coefficient) is often already informative to a linear probe, but the decoding pathway under standard prompts behaves as if an effective gate sets \(b_{\text{eff}}\approx 0\) (anti-abstention). Causal addition of \(+\alpha v_{\text{void}}\) increases expressed abstention.

**Claim 3 (tools as evidence, not answers).** A tool observation \(o\) should be treated as a measurement that updates a scalar evidence estimate \(e(I,Q,o)\in[0,1]\). Empty detections, conflicts, and failed zooms should decrease \(e\); consistent multi-tool support should increase \(e\).

**Claim 4 (gated optimality).** The intervention

\[
h \leftarrow P^H\big(h + \alpha(e)\, v_{\text{void}}\big),
\quad
\alpha(e)=\alpha_{\max}\,\sigma\big(\gamma(\tau-e)\big),
\quad
P^H=I-V^H(V^H)^\top
\]

approximately implements

\[
\log p(y\mid I,Q,o)
\;\approx\;
\log p_{\text{base}}(y\mid I,Q,o)
- \lambda(e)\, R_{\text{hall}}(y)
+ \mu(e)\, R_{\text{void}}(y),
\]

with \(\lambda(e),\mu(e)\) large iff \(e\) is small — i.e., only in the regime where language priors dominate and abstention is the correct behavior.

**Claim 5 (why always-on fails).** If \(\alpha\equiv\alpha_{\max}\) or \(P^H\) is always applied, components of \(h^\star\) that overlap HalluSpace / anti-overlap void get damaged on answerable items, canceling gains. This predicts null or negative net accuracy in ungated interp experiments.

---

## 5. Premises (explicit checklist)

EGDS stands only if these premises hold. Each is falsifiable.

### P1 — Low-rank HalluSpace (CIPHER/Nullu)

Difference vectors between hallucinated-vision (or hallucinated-text) and clean runs at some layers \(\ell\in\mathcal{L}_H\) are approximately low-rank: the top \(r\ll d\) singular vectors explain most systematic variance relevant to object invention.

**Falsifier.** Singular spectrum of \(\Delta^H_\ell\) is nearly flat; projection with any small \(r\) either does nothing or destroys BLEU/VQA broadly.

### P2 — Linear void direction (TRAPS)

There exists a layer \(\ell^\star\) and direction \(v_{\text{void}}\) such that:

- a probe on \(h_{\ell^\star}\) separates answerable vs false-premise / insufficient-evidence,
- \(+\alpha v_{\text{void}}\) causally increases abstention rate on held-out false-premise pairs,
- \(-\alpha v_{\text{void}}\) decreases it.

**Falsifier.** Probe AUROC ~0.5; steering does not move abstention vs random directions of equal norm.

### P3 — Evidence score is predictive

A cheap score \(e\) built from detector counts / emptiness / cross-tool agreement correlates with whether the question is answerable from the image (and with whether the agent’s final answer will be a premise-accepting hallucination).

**Falsifier.** \(e\) is uncorrelated with trick vs normal labels and with premise_accepted errors.

### P4 — Gating is necessary

Gated EGDS beats both (a) always-on CIPHER projection and (b) always-on void steering on the joint objective: trick accuracy ↑ without large normal-item accuracy ↓.

**Falsifier.** Always-on matches or beats gated EGDS on both splits.

### P5 — Timing: post-tool beats answer-only

Applying the edit immediately after tool observations (and during subsequent tokens) beats applying it only on the final answer span.

**Falsifier.** Answer-only intervention matches post-tool intervention.

### P6 — Non-contamination / transfer

Subspaces mined off VistaQA still improve VistaQA trick behavior (transfer). Training or mining on VistaQA itself is disallowed for the scientific claim of this method.

**Falsifier.** Only in-domain mining works; off-domain mining gives ≈0.

### P7 — Tools without EGDS are insufficient for tricks

On a matched budget, tool agent alone still shows high premise_accepted; EGDS+tools reduces it more than tools+prompted abstention.

**Falsifier.** Prompted abstention with tools matches EGDS.

---

## 6. Method overview

EGDS has four stages:

1. **Offline Stage A — HalluSpace bank** (CIPHER-style vision counterfactuals; optional Nullu text bank).
2. **Offline Stage B — VoidSpace bank** (false-premise / insufficient-evidence contrasts).
3. **Offline Stage C — (Optional) ToolMisread bank** (empty-tool context, affirm vs abstain).
4. **Online inference — Tool loop + evidence score + gated dual edit** (+ optional VCD assist).

No model weights need to be trained for the core method. Offline stages only run frozen forward passes to collect activations and compute SVD / mean differences. Online stage hooks residual streams.

```
                    ┌─────────────────────────────────────────┐
                    │ Offline (once per model)                │
                    │  A: CIPHER pairs → V^H_ℓ                │
                    │  B: FP pairs     → v_void_ℓ             │
                    │  C: empty-tool   → v_mis_ℓ  (optional)  │
                    └───────────────────┬─────────────────────┘
                                        │ cache basis bank
                                        ▼
   Image I, Question Q
            │
            ▼
   Prefill VLM (optional light gate on prefill)
            │
            ▼
   ┌──────────────────── Tool Agent Loop ────────────────────┐
   │  propose tool → execute → observation o_t               │
   │  compute e_t = EvidenceScore(I,Q,{o_≤t})                │
   │  while generating next thought/answer tokens:           │
   │      h ← P^H ( h + α(e_t) v_void + β(e_t) v_mis )       │
   └─────────────────────────────────────────────────────────┘
            │
            ▼
   Final answer (+ empty evidence on abstain)
```

---

## 7. Stage A — HalluSpace construction (detailed)

### 7.1 Goal

Estimate, per layer \(\ell\), an orthonormal basis \(V^H_{\ell,r}\in\mathbb{R}^{d\times r}\) for directions that shift representations toward **vision-induced object hallucination**.

### 7.2 Data construction (CIPHER-faithful)

**Source.** \(M\) image–caption pairs from MSCOCO train (CIPHER used \(M=5000\)). Do not use course eval benchmarks.

For each \(i=1,\ldots,M\):

1. Let \((I_i, C_i)\) be the clean pair.
2. Generate perturbed caption \(\tilde C_i\) with an LLM by adding or swapping objects / attributes that are plausible co-occurrences but false for \(I_i\) (Nullu/LURE-style factors: co-occurrence, uncertainty, position).
3. Encode \(z_0=\mathcal{E}(I_i)\) with a latent diffusion encoder.
4. For seeds \(j=1,\ldots,B\) (e.g. \(B=5\)):
   - sample \(\epsilon_j\sim\mathcal{N}(0,I)\),
   - form \(\tilde z_{t_h}\) as in CIPHER Eq. (2) with \(t_h=\lfloor 0.5T\rfloor\),
   - reverse-denoise conditioned on \(\tilde C_i\) with CFG scale ~7.5,
   - decode \(\tilde I_{i,j}=\mathcal{D}(\tilde z_0)\).
5. Store clean pair \((I_i,C_i)\) and counterfactual pairs \((\tilde I_{i,j}, C_i)\) (true caption, false image content).

**Design note.** Pairing \(\tilde I\) with **true** \(C\) is intentional: the model sees a caption that does *not* match the image, inducing the internal “describe / bind unsupported objects” shift CIPHER wants.

### 7.3 Activation collection

Let \(f_\theta\) be the frozen LVLM. Choose:

- layers \(\mathcal{L}_H\) (start with upper half, e.g. last 50% of LLM layers; CIPHER used 16–32 on LLaVA-scale),
- token pooling set \(\mathcal{T}\): caption / answer tokens (exclude pure image-patch tokens for the primary bank; optionally also pool last prompt token).

For each sample \(i\) and layer \(\ell\in\mathcal{L}_H\):

\[
h^{(i)}_\ell
=
\frac{1}{|\mathcal{T}|}
\sum_{k\in\mathcal{T}}
h_{\ell,k}(I_i, C_i),
\]

\[
\tilde h^{(i,j)}_\ell
=
\frac{1}{|\mathcal{T}|}
\sum_{k\in\mathcal{T}}
h_{\ell,k}(\tilde I_{i,j}, C_i),
\quad
\tilde h^{(i)}_\ell
=
\frac{1}{B}\sum_{j=1}^{B}\tilde h^{(i,j)}_\ell.
\]

Difference:

\[
\delta^{H,(i)}_\ell = \tilde h^{(i)}_\ell - h^{(i)}_\ell.
\]

Stack:

\[
\Delta^H_\ell
=
\begin{bmatrix}
\delta^{H,(1)}_\ell \\
\vdots \\
\delta^{H,(M)}_\ell
\end{bmatrix}
\in \mathbb{R}^{M\times d}.
\]

### 7.4 SVD and basis bank

\[
\Delta^H_\ell = U^H_\ell\, \Sigma^H_\ell\, (V^H_\ell)^\top.
\]

Let \(V^H_{\ell,r}\) be the first \(r\) columns of \(V^H_\ell\) (right singular vectors). Store:

- \(V^H_{\ell,r}\),
- singular values \(\sigma_1\geq\ldots\geq\sigma_r\) (for diagnostics),
- optional mean \(\bar\delta^H_\ell\) if a rank-1 mean-difference baseline is desired.

**Rank selection.** Grid-search \(r\) on a small **held-out COCO / POPE-like** validation set (not VistaQA): maximize hallucination reduction subject to ≤ small drop in a capability proxy (e.g. VQAv2 short subset or caption BLEU). CIPHER used \(r=8\) (LLaVA), \(32\) (mPLUG), \(64\) (MiniGPT-4).

Projection matrix:

\[
P^H_\ell = I_d - V^H_{\ell,r}(V^H_{\ell,r})^\top.
\]

### 7.5 Optional Stage A′ — Nullu text HalluSpace

Same images \(I_i\), captions \(C_i\) vs \(\tilde C_i\):

\[
\delta^{T,(i)}_\ell = h_\ell(I_i,\tilde C_i) - h_\ell(I_i,C_i).
\]

Build \(V^T_{\ell,r_T}\) analogously. At inference one may use:

\[
P^{HT}_\ell = P^H_\ell P^T_\ell
\quad\text{or}\quad
P^{HT}_\ell = I - [V^H;V^T]_{\text{orth}}[\ldots]^\top
\]

after orthonormalizing the concatenated bases (QR / Gram–Schmidt). Start **without** A′; add if trick gains plateau and CHAIR-style object invention remains.

---

## 8. Stage B — VoidSpace construction (detailed)

### 8.1 Goal

Estimate direction(s) that move the model toward **expressing** abstention / “not present” / empty evidence when the queried entity is absent or the question is unanswerable from the image.

### 8.2 Data construction

**Preferred public source.** FP-RefCOCO / FP-RefCOCO+ / FP-RefCOCOg (SESAME) with COCO images, plus matched positive RefCOCO expressions on the same images.

For each image \(I\):

- \(Q^+\) — referring / VQA question about an entity that **is** present (answerable),
- \(Q^-\) — context-aware false-premise question about an entity that **is not** present (or invalid relation/attribute).

Optional second family (attribute / relation false premises) to cover VistaQA’s non-object trick types: synthesize via LLM under constraints that the contradiction is visually checkable.

**Size.** Start with several thousand pairs (order \(10^3\)–\(10^4\)). Balance so Stage B is not tiny relative to Stage A.

### 8.3 Activation collection

Use a fixed prompt template that allows abstention, e.g.:

> Answer the question about the image. If the asked object, attribute, or relation is not supported by the image, say that it is not present / cannot be determined.

Collect hidden states at layers \(\mathcal{L}_V\) (search around mid-to-late layers; TRAPS used a mid-late layer such as 20 on an 8B-class model — retune per model).

Pooling options (record which you use):

1. **Last prompt token** (prefill-only void signal) — closest to Dr. No prefill steering.
2. **Mean over generated answer tokens** under teacher-forced “not present” vs concrete answer (requires constructing target strings).
3. **Mean over first \(K\) generated tokens** with free generation (noisier).

Recommended default for mining:

\[
h^+_\ell = h_{\ell,\text{last-prompt}}(I,Q^+),\quad
h^-_\ell = h_{\ell,\text{last-prompt}}(I,Q^-).
\]

### 8.4 Direction estimation

**Rank-1 (default).**

\[
\bar h^+_\ell = \frac{1}{N}\sum_i h^{+,(i)}_\ell,\quad
\bar h^-_\ell = \frac{1}{N}\sum_i h^{-,(i)}_\ell,
\]

\[
v^{\text{void}}_\ell
=
\frac{\bar h^-_\ell - \bar h^+_\ell}{\|\bar h^-_\ell - \bar h^+_\ell\|_2}.
\]

(Here \(Q^-\) is the void / false-premise condition, so \(\bar h^- - \bar h^+\) points toward void.)

**Low-rank (optional).** Stack \(\delta^{V,(i)}_\ell = h^{-,(i)}_\ell - h^{+,(i)}_\ell\), SVD, keep top \(r_V\) vectors as \(V^V_{\ell,r_V}\). Steering can then use the top vector or a learned combination.

### 8.5 Causal validation before trusting the bank

On a held-out FP set (still not VistaQA):

1. **Probe.** Logistic regression / linear probe on \(h_\ell\) → AUROC for void vs control. Expect high if P2 holds.
2. **Steer sweep.** For \(\alpha\in\{0,1,2,5,10\}\) (scale relative to \(\|h\|\) or fixed after normalizing \(v\)):
   - measure abstention rate, false abstention on answerable items, answer accuracy on answerable items.
3. **Random-direction control.** Same norms, random unit vectors; require void direction to dominate.

Only promote layers that pass (2)–(3) into the online bank.

---

## 9. Stage C — Optional ToolMisreadSpace (detailed)

### 9.1 Motivation

Pixel Reasoner and VistaQA tool traces show a distinct failure: tool returns empty / low-confidence evidence, yet the model **affirms** the queried object. This may not coincide exactly with VoidSpace mined from clean VQA prompts without tool messages.

### 9.2 Data construction

For pairs \((I,Q^-)\) (false premise) and optionally \((I,Q^+)\):

1. Run a detector / zoom stub that returns a **canonical empty observation** string \(o_\emptyset\) (e.g. `detect: []`, `ocr: ""`).
2. Force two continuations (teacher forcing or rejection sampling):
   - \(y_{\text{affirm}}\): answers as if the object exists,
   - \(y_{\text{abstain}}\): correctly says not present.
3. Collect hidden states during those continuations (mean over answer tokens) at layers \(\mathcal{L}_M\).

### 9.3 Direction

\[
v^{\text{mis}}_\ell
=
\frac{\bar h^{\text{affirm}|o_\emptyset}_\ell - \bar h^{\text{abstain}|o_\emptyset}_\ell}{\|\ldots\|_2}.
\]

Online, we want to **subtract** the affirm-under-empty direction when \(e\) is low:

\[
h \leftarrow h - \beta(e)\, v^{\text{mis}}_\ell
\]

with \(\beta(e)\) large when \(e\) small (same shape as \(\alpha(e)\)).

**Note.** If Stage B already captures this, Stage C will be redundant (cosine similarity of \(v^{\text{mis}}\) and \(v^{\text{void}}\) high). Keep C only if ablation shows additive gains.

---

## 10. Online Stage — Evidence score (detailed)

### 10.1 Role of \(e\)

\(e_t\in[0,1]\) is the probability-like estimate that **sufficient visual support exists** for answering the question’s presuppositions after observations \(o_{\le t}\).

EGDS does **not** require \(e\) to be a calibrated probability; it requires monotonicity: lower \(e\) ↔ more need for void push / hallu projection.

### 10.2 Inputs available in the tool agent

From the existing agent-tools stack (zoom, grid, detect, OCR, depth), define features:

| Feature | Definition (example) | Effect on \(e\) |
|---|---|---|
| \(s_{\text{det}}\) | max detector confidence for the queried noun phrase(s) | ↑ |
| \(n_{\text{det}}\) | number of boxes above threshold \(\tau_{\text{box}}\) | ↑ if expected count ≥1 |
| \(\mathbf{1}_\emptyset\) | no boxes / empty OCR / failed crop | ↓ |
| \(c_{\text{agree}}\) | agreement between two independent tools (e.g. detect count vs SAM mask count; OCR string vs VLM crop read) | ↑ if agree, ↓ if conflict |
| \(s_{\text{clip}}\) | optional CLIP score of “a photo of {object}” vs image/crop | ↑ |
| \(u_{\text{arg}}\) | optional entropy on tool-argument tokens (if logprobs available) | ↓ if high |

### 10.3 Default parametric form

Let \(x\) be a linear score:

\[
x_t
=
w_1 s_{\text{det}}
+ w_2 \tanh(n_{\text{det}})
+ w_3 s_{\text{clip}}
+ w_4 c_{\text{agree}}
- w_5 \mathbf{1}_\emptyset
- w_6 \mathbf{1}_{\text{conflict}}
- w_7 u_{\text{arg}}.
\]

\[
e_t = \sigma(x_t) = \frac{1}{1+e^{-x_t}}.
\]

**Initialization.** Start with hand-set weights emphasizing emptiness and detector score (Woodpecker spirit), then optionally fit \(w\) by logistic regression on held-out FP vs answerable COCO/FP-RefCOCO labels using only tool features (still no VistaQA).

### 10.4 Special case: counting questions

For “how many” questions, replace “existence” semantics:

\[
e^{\text{count}}_t
=
\sigma\big(
w_c\big(1 - \tfrac{|n_{\text{det}} - n_{\text{verbal}}|}{n_{\text{det}}+1}\big)
- w_\emptyset\mathbf{1}_\emptyset
\big),
\]

where \(n_{\text{verbal}}\) is the model’s currently proposed count if any. Low agreement → lower \(e\) → stronger hallu projection / less confident commitment (may trigger recount tools rather than abstain, depending on policy).

### 10.5 Mapping \(e\) to intervention strengths

\[
\alpha(e) = \alpha_{\max}\,\sigma\big(\gamma(\tau - e)\big),
\quad
\beta(e) = \beta_{\max}\,\sigma\big(\gamma(\tau - e)\big),
\quad
\lambda_H(e) = \lambda_{\max}\,\sigma\big(\gamma(\tau - e)\big).
\]

Interpretation:

- high \(e\) → \(\alpha,\beta,\lambda_H\approx 0\) (almost no edit),
- low \(e\) → full void push and full HalluSpace projection.

Soft projection (recommended vs hard always \(P^H\)):

\[
P^H(e) = I - \lambda_H(e)\, V^H(V^H)^\top.
\]

When \(\lambda_H(e)=1\), this recovers CIPHER’s hard nullification; when \(0\), identity.

---

## 11. Online Stage — Dual subspace intervention (detailed math)

### 11.1 Hook points

At each autoregressive step \(k\) after tool observation \(o_t\) has been written into context, for each layer \(\ell\in\mathcal{L}_{\text{edit}}\):

\[
h_{\ell,k}
\;\leftarrow\;
P^H_\ell(e_t)
\Big(
h_{\ell,k}
+ \alpha(e_t)\, v^{\text{void}}_\ell
- \beta(e_t)\, v^{\text{mis}}_\ell
\Big).
\]

Expanding soft projection:

\[
\begin{aligned}
h'
&=
h + \alpha(e)\, v^{\text{void}} - \beta(e)\, v^{\text{mis}}, \\
h^{\text{edit}}
&=
h' - \lambda_H(e)\, V^H(V^H)^\top h'.
\end{aligned}
\]

### 11.2 Order of operations

**Preferred order:** steer first, then project.

Reason: void steering may have a small component inside HalluSpace; projecting after removes that contamination so abstention is not implemented by inventing a different object. Ablate the reverse order.

### 11.3 Which tokens / which layers

| Choice | Recommendation | Rationale |
|---|---|---|
| Layers | Mid–late LLM layers that passed Stage B causal tests ∩ Stage A singular energy | TRAPS + CIPHER |
| Tokens | All tokens generated *after* the latest tool message; optionally also last prompt token at prefill | Timing premise P5 |
| Prefill | Optional weak \(\alpha_{\text{pre}}(e_0)\) with \(e_0\) from CLIP-only / no tools | Dr. No-style |
| Image tokens | Do **not** project vision encoder outputs in v1 | Avoid destroying perception features |

### 11.4 Decoding policy after edit

Keep the agent’s normal decoding (temperature / greedy / beam as in baseline). EGDS is not a contrastive decoder; it changes \(h\) before the LM head.

Optional hybrid (only if needed for perception/count):

**VCD-lite on high-\(e\) crops.** When \(e_t\) is high and the question is count/attribute:

\[
\log p_{\text{VCD}}(y)
\propto
(1+\eta)\log p(y\mid I_{\text{crop}}) - \eta\log p(y\mid I_{\text{blurred}}).
\]

Do **not** run VCD when \(e_t\) is low (distortion can worsen false-premise behavior).

### 11.5 Abstention → evidence output

For VistaQA-like evaluation requiring masks / evidence:

- if the generated text is classified as abstention (regex / small judge / leading “not present”), emit **empty evidence** (empty mask set),
- do not call SAM on invented boxes after abstention (SESAME cascade lesson).

---

## 12. How the tool agent loop uses EGDS

### 12.1 Pseudocode

```text
function EGDS_Agent(I, Q, bases, tools):
    e ← EvidenceScore_CLIP_only(I, Q)          # optional weak prefill e
    PrefillWithOptionalSteer(I, Q, e, bases)

    memory ← []
    for step = 1..max_steps:
        action ← DecodeNextAction(with_hooks=True, e=e, bases=bases)
        if action is FINAL_ANSWER:
            return action.answer, EvidenceFrom(action, e)

        o ← Execute(tools, action)
        memory.append(o)
        e ← EvidenceScore(I, Q, memory)

        # critical: subsequent tokens see updated e
        if e < τ_empty and action was SEARCH_LIKE:
            # optional behavioral assist aligned with theory
            AppendSystemNote("Tool returned no supporting evidence for the queried entity.")

    return timeout_answer
```

### 12.2 Interaction with known tool failure modes

| Failure | EGDS response |
|---|---|
| Empty detect on trick question | \(e\downarrow\) → \(\alpha\uparrow,\lambda_H\uparrow\) → abstain path |
| Detector hallucinates low-confidence boxes | threshold + \(\mathbf{1}_{\text{conflict}}\) with second tool → \(e\downarrow\) |
| Model ignores empty tool text | Stage C \(v^{\text{mis}}\) subtraction + system note |
| Single wrong crop | optional TTSP-lite: second independent crop; conflict lowers \(e\) |
| Undercount | high \(e\) path; VCD-lite / recount tools; do not void-steer |

### 12.3 What EGDS is *not*

- Not a replacement for zoom on tiny objects (perception still needs pixels).
- Not an RL algorithm (compatible later with Reliable-V-style negatives).
- Not a guarantee on API models without activation access (use Woodpecker-style rewrite + empty-evidence policy as behavioral cousin).

---

## 13. End-to-end algorithm summary

### Offline (once)

1. Build OHC-style set \(\{(I_i,C_i,\{\tilde I_{i,j}\})\}\).
2. Compute \(\Delta^H_\ell\), SVD → \(V^H_{\ell,r}\).
3. Build FP pairs \(\{(I_i,Q^+_i,Q^-_i)\}\).
4. Compute \(v^{\text{void}}_\ell\); validate with probe + causal steer.
5. Optionally compute \(v^{\text{mis}}_\ell\).
6. Cache `{V^H, v_void, v_mis, layers, r, α_max, τ, γ}` per model.

### Online (per question)

1. Run tool agent as usual.
2. After each tool result, update \(e_t\).
3. For each subsequent generated token, apply
   \[
   h \leftarrow P^H(e)\big(h + \alpha(e)v_{\text{void}} - \beta(e)v_{\text{mis}}\big).
   \]
4. On abstention text, emit empty evidence.

---

## 14. Experimental protocol (to test the theory)

### 14.1 Models

- Primary local: Qwen2.5-VL-3B/7B (or Qwen3-VL small), residual hooks.
- Scale-up: larger open VLM on H100 with pixel cap.
- Behavioral only: existing OpenRouter Muse/Gemma/Qwen agent runs.

### 14.2 Data splits

| Split | Use |
|---|---|
| COCO + FP-RefCOCO train | Stages A/B mining |
| Held-out FP / POPE-like | Hyperparameters \(r,\alpha_{\max},\tau,\gamma\) |
| VistaQA full / trick–normal split | **Eval only** |
| TreeBench / MME-RW | Transfer eval only (optional) |

### 14.3 Conditions

1. Base VLM (no tools)
2. Tool agent (current)
3. Tool agent + abstain prompt
4. Always-on CIPHER (\(P^H\) only)
5. Always-on void steer
6. Gate only (Woodpecker-style rewrite / SESAME see-gate, no activation edit)
7. **EGDS full**
8. EGDS w/o Stage C
9. EGDS w/o HalluSpace (void+gate only)
10. Random-direction gated control

### 14.4 Metrics

- VistaQA text accuracy / GROVE if masks available; **always** report trick vs normal separately.
- Abstention precision/recall on trick items.
- Break/fix counts vs tool agent (Beacon-style).
- Normal-item accuracy drop (must stay small).
- Optional: CHAIR on COCO captions to confirm HalluSpace quality.

### 14.5 Statistical plan

- Paired McNemar on item correctness for primary comparisons.
- Multiple seeds for decoding if stochastic.
- Pre-register primary metric: **trick-subset accuracy (or GROVE)** with constraint that normal-subset drop ≤ δ (e.g. 1–2 abs points) unless overall GROVE still rises.

---

## 15. Why this might work (argument map)

1. **CIPHER** shows vision hallucination is a removable low-rank feature shift.  
2. **TRAPS/Dr.No** show abstention is present in features but gated at output; causal steer works; ungated steer harms.  
3. **Woodpecker/SESAME** show external existence evidence is a reliable gate signal.  
4. **Our traces** show premise_accepted + empty-tool misreads as dominant trick failures; tools alone do not fix them.  
5. Therefore **gated composition** \(P^H(e)(h+\alpha(e)v_{\text{void}})\) is the minimal method that respects all four: remove invention directions, add abstention directions, only when tools say evidence is weak.

---

## 16. Risks, failure modes, and mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| HalluSpace overlaps useful features | Normal accuracy drops | Soft \(\lambda_H(e)\); lower \(r\); layer selection |
| Over-abstention | High trick score, collapsed normal | Raise \(\tau\); fit gate on FP+positives; Dr. No-style feature gate |
| Detector bias | Systematic wrong \(e\) | Dual-tool agreement; calibrate threshold |
| Wrong layer | No causal effect | Layer sweep with random controls |
| Multi-token abstention phrases | Steering helps “not” but not full phrase | Steer for several early answer tokens; template-friendly abstain string |
| API models | Cannot hook \(h\) | Behavioral cousin: Woodpecker rewrite + empty evidence |
| Contamination | Inflated VistaQA numbers | Strict mining source ban |

---

## 17. Future extensions (out of core theory)

1. **SAE upgrade** of VoidSpace (true Dr. No features) once a VLM SAE exists for the model.
2. **J-lens readout** of whether \(v_{\text{void}}\) is verbalizable as “not present” and whether tool negatives enter the workspace (connects to `notes/jspace-vqa-experiment-plan.md`).
3. **RL fine-tuning** that *freezes* or regularizes against destroying \(v_{\text{void}}\) (Reliable-V-RL compatibility): accuracy reward + abstention reward, with EGDS as an inference prior or as a teacher for DPO pairs.
4. **TTSP ledger** integration: Confirmed / Conflict / Not-found entries directly set \(e\).
5. **Count-specific subspace** if undercount remains after EGDS (separate \(\delta\) for under- vs over-count contrasts).

---

## 18. References (primary)

1. Dastmalchi et al. — *Fighting Hallucinations with Counterfactuals: CIPHER* (arXiv:2603.10470).  
2. Yang et al. — *Nullu: HalluSpace Projection* (CVPR 2025).  
3. Leng et al. — *Visual Contrastive Decoding (VCD)* (CVPR 2024).  
4. TRAPSBench — VLMs encode but fail to express epistemic restraint (void steering).  
5. Dr. No — Feature steering for abstention in VLMs.  
6. Yin et al. — *Woodpecker* (hallucination correction with detectors).  
7. Wu et al. — *SESAME / FP-RefCOCO* (see–say–segment; false premises).  
8. HalluSegBench / RobustSeg — vision-driven hallucination; counterfactual negatives.  
9. Reliable-V-RL — accuracy-only RL destroys abstention.  
10. TTSP / Pixel Reasoner / MAPO / SRICE — tool reliability, empty-tool misreads, consensus.  
11. VistaQA — free-form VQA + empty-evidence trick items; GROVE.  
12. Project notes — VistaQA agent failure taxonomy; `notes/jspace-vqa-experiment-plan.md`.

---

## 19. One-page recap

**Know.** Hallucination ≈ low-rank residual shift (CIPHER/Nullu). Abstention ≈ linear direction that is under-expressed (TRAPS/Dr.No). Tools ≈ evidence sensors that models often ignore (Woodpecker/SESAME/Pixel Reasoner). VistaQA hurts most on premise acceptance under weak evidence.

**Theory.** Edit internals as a function of tool evidence: when evidence is weak, push void and nullify hallu directions; when strong, leave the model alone.

**Method.** Offline SVD/mean-diff banks + online gated dual residual edit inside the tool loop.

**Premise to defend.** Gating + dual subspace + post-tool timing are jointly necessary; any one alone is insufficient.
