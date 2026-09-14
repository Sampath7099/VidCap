# Project: VidCap-Adaptive — Efficient Video Captioning via Learned Frame Selection

## Context
This supersedes the from-scratch VidCap plan. Full reasoning trail (kept for continuity, also logged in [project_requirements_and_constraints.md](/home/sampath/Desktop/project_requirements_and_constraints.md)):
1. OCR+RAG (from scratch) → felt like "just training an OCR," thin vision/language intersection.
2. Video captioning, fully from scratch (ViT + MAE pretraining + temporal transformer + own mini-LM) → genuinely deep, but realistically incapable of matching industry-standard caption quality on Kaggle-scale compute, which matters because this project targets **ML-role placement interviews**, where a working, competitive result carries real weight.
3. Realization: using pretrained backbones is how the field actually works (LLaVA's vision encoder and LLM are both frozen pretrained models; its contribution is a small trained connector). "Gluing models together" is not a lesser form of ML work when the glue — the connector, the adaptation method, the training recipe, the evaluation — is non-trivial and yours.
4. Final direction (this document): keep pretrained backbones frozen for competitive quality, but build a genuine, evaluable **contribution on top**: nearly every video captioning pipeline (research and industry) defaults to naive uniform/fixed-interval frame sampling. This project builds and rigorously evaluates a **learned frame-relevance selector** that picks the most informative frames under a fixed budget instead — a real, named gap (efficient frame selection under compute/latency constraints), decisively measurable, and motivated by the same compute constraints you actually have.

Two hard, non-negotiable conditions carried through every revision:
1. The end result must be genuinely **usable and viable** — feed it a real video clip, get a real caption.
2. The project must reach **real depth** in vision, language, and their intersection, and now also produce a **decisive, honestly-measured result**, not just working code.

Compute constraints (two Kaggle accounts, T4/P100, 12h session cap, mandatory resumable checkpointing) are unchanged and documented in the requirements file linked above. Time constraint remains set aside per your instruction.

## The story
Real video captioning deployment — accessibility tools describing what a phone camera sees, content moderation, edge/mobile inference — is frame-budget-constrained: you can't afford to run a vision encoder over every frame. The standard answer is naive uniform sampling (every Nth frame), regardless of what's actually in those frames. This project asks: *if you have to pick K frames, can you pick the K frames that actually matter, and does it help?* — and answers it with a trained selector, compared head-to-head against uniform sampling and a cheap classical heuristic, across multiple frame budgets, validated against real human-annotated frame-importance ground truth (TVSum/SumMe) and stress-tested on long videos with temporally sparse events (ActivityNet Captions) where naive uniform sampling is expected to visibly fail.

## The from-scratch ledger

**Reused, deliberately (this is where competitive quality comes from):**
- A frozen pretrained vision encoder (CLIP ViT, e.g. ViT-B/32) — applied per-frame, never fine-tuned. Frozen so its embeddings can be **precomputed and cached once**, making every downstream training stage cheap.
- A frozen (LoRA-adapted, not fully fine-tuned) small pretrained LLM decoder (e.g. GPT-2-small/distilGPT2, or a small open instruction model if quality demands it).
- PyTorch as the tensor/autograd substrate; a video-decoding library for frame extraction — infrastructure, not ML content.

**Built from scratch (this is the actual contribution — every one of these must be independently defensible in an interview):**
- **The frame-relevance scorer** — the headline component. A lightweight network trained on cached CLIP frame embeddings to predict how caption-relevant a frame is, supervised by that frame's CLIP similarity to the clip's ground-truth captions (text is a training signal only — never available at inference, so there's no chicken-and-egg problem).
- **A cross-attention resampler** (Flamingo-style: a small set of learnable query tokens attend over the selected frames' embeddings), plus the mean-pool and temporal-transformer variants used as ablation baselines against it.
- **LoRA**, implemented from raw tensor ops (low-rank update matrices applied to the frozen LLM's attention/MLP projections) — not imported from `peft`.
- The **multimodal fusion/attention-masking scheme** bridging selected-frame tokens into the LLM's input sequence.
- **Beam search and greedy decoding**, hand-written for inference.
- The **staged training recipe** (what's frozen, what's trained, in what order) and every loss function involved (captioning cross-entropy, the scorer's supervised relevance loss).
- The full **evaluation harness**: quality-vs-frame-budget curves, all ablations, the frame-selection visualizations.
- **(Optional ablation, reusing prior work)**: swap the pretrained LLM decoder for your own LMA-coursework mini-LM under the same recipe, and compare — a genuine, low-cost way to put your own prior from-scratch model to use without gating the main pipeline's competitiveness on it.
- **(Stretch)**: robustness-to-degraded-video training (motion blur/low-res/compression augmentation), evaluated on a deliberately degraded test set.

## Architecture

### Stage 0 — Frame pool & caching
- Extract a denser candidate pool of frames per clip (e.g. ~1 fps) at dataset-build time — this is the pool the selector chooses *from*, distinct from the small budget it ultimately picks.
- Run the frozen CLIP vision encoder once over every candidate frame; cache the embeddings. All later training operates on these cached embeddings, not raw pixels — this is what keeps every subsequent stage cheap enough for a T4.

### Frame selection (the headline contribution)
- **Frame-relevance scorer**: a small MLP/transformer taking a frame's cached CLIP embedding, outputting a relevance score. Trained with a supervised target derived from that frame's CLIP similarity to the clip's ground-truth caption(s) — visual input only at inference, so it needs no caption to run.
- **Baselines it's compared against**: (a) uniform/fixed-interval sampling — the de facto industry default; (b) a classical motion-magnitude heuristic (frame-to-frame pixel difference) — cheap, no training required.
- Selection is hard top-K by score (train and test) — deliberately avoiding RL/Gumbel-softmax differentiable selection, which is the standard way this kind of idea burns time on instability instead of producing a result.

### Fusion & language
- **Connector**: cross-attention resampler (primary) vs. mean-pool and temporal-transformer (ablation baselines) — takes the selected frames' embeddings, outputs a fixed-size set of video tokens.
- **Projector**: small MLP into the LLM's embedding dimension (if needed).
- **Decoder**: frozen pretrained small LLM, adapted via from-scratch LoRA; fusion via a prefix-attention mask (video tokens fully visible, text tokens causally masked among themselves, all attending back to the video prefix).
- **Decoding**: hand-written greedy and beam search.

### Training recipe
- **Stage A**: train the frame-relevance scorer on cached embeddings (fast — no vision/LLM forward passes needed).
- **Stage B**: freeze vision encoder and LLM; train the connector + projector on (selected-frames, caption) pairs with captioning cross-entropy loss.
- **Stage C**: unfreeze LoRA adapters (+ keep connector trainable); joint fine-tune for quality.

## Dataset
- **MSR-VTT** (~10,000 clips, ~20 captions each, public, established benchmark) — primary training and standard captioning-quality evaluation.
- Dense candidate-frame extraction (Stage 0) done once per clip, cached as CLIP embeddings.
- **TVSum + SumMe** — public video-summarization datasets with real **human-annotated frame-importance ground truth**. Used to directly validate the frame-relevance scorer (and the CLIP-similarity proxy signal it's trained on) against actual human judgment, not just a self-referential proxy. These datasets are specifically built around non-uniform importance, so they're also where uniform sampling can be shown to be provably suboptimal against real annotations.
- **ActivityNet Captions** (subset) — long videos with temporally localized/sparse events, public and established. Replaces a self-curated clip set as the stress-test evaluation where naive uniform sampling is expected to visibly fail and the learned selector's benefit should show up most clearly.
- A small real, non-benchmark video holdout (short and long, self-sourced) for the final usability check only — never scored/benchmarked, never touched during training.

## Metrics
- **Scorer correctness**: correlation between predicted relevance and true CLIP-caption similarity on held-out frames (proxy check), **plus** correlation against real human-annotated importance on TVSum/SumMe (ground-truth check) — the second is the stronger claim and should be reported separately from the first.
- **Headline result — quality-vs-frame-budget curves**: BLEU-4/METEOR/ROUGE-L/CIDEr at multiple budgets (e.g. 2/4/8/16 frames) for uniform sampling vs. motion heuristic vs. learned scorer, on both MSR-VTT and ActivityNet Captions. Success looks like: learned selector matches uniform sampling's quality at a lower budget, or beats it at equal budget — report honestly if the effect is small on MSR-VTT's short clips and larger on ActivityNet's longer, sparser-event videos, that's a legitimate, expected finding, not a failure.
- **Connector ablation**: mean-pool vs. temporal transformer vs. cross-attention resampler.
- **LoRA ablation**: frozen decoder vs. LoRA at a few ranks vs. full fine-tune (if feasible) — quality vs. trainable-parameter-count tradeoff.
- **(Optional) decoder ablation**: pretrained LLM vs. your LMA mini-LM as the decoder, same recipe.
- **Efficiency**: wall-clock training speedup from embedding caching, reported as a number.
- **Qualitative**: visualize which frames the selector picks vs. uniform sampling on a handful of real clips — a strong, cheap README figure.
- **Usability check**: sane captions on the real, non-benchmark holdout — the hard gate, non-negotiable.
- **(Stretch)**: robustness-to-degradation before/after comparison.

## Revised execution order (supersedes the phase numbering below)

The original order put the frame-relevance scorer first. Reordered so the **vanilla captioner with
uniform sampling is built and trusted first**, then frame selection is swapped in against it.

Rationale: the captioner is the measuring instrument for the frame-selection claim — "does learned
selection beat uniform sampling" is unmeasurable until frames→captions exists and is trusted. The
uniform-sampling arm is required as the baseline regardless, so none of it is throwaway. Phase 0's
cache stores the **full candidate pool**, not a uniform-K subset, so the scorer later drops in with
no re-embedding.

| Step | Was | Content |
|---|---|---|
| 0 | Phase 0 | Foundation, caching, checkpointing — *code done, data pending* |
| 1 | Phases 2,3,4,7 | Vanilla captioner: connector, LoRA, fusion, greedy decode + correctness gates |
| 2 | Phases 5,6 | Train Stage B (connector) then Stage C (LoRA), uniform sampling throughout |
| 3 | **new** | **Blind baseline control** — same model, visual prefix zeroed |
| 4 | Phase 1 | Frame-relevance scorer + TVSum/SumMe validation |
| 5 | **new** | Cheap proxy check: does selection retain more caption-relevant CLIP signal? |
| 6 | Phase 8 | Headline quality-vs-budget curves, uniform vs motion vs learned |
| 7 | Phases 9,10,11 | Ablations, end-to-end usability, packaging |

**Step 3 is not optional.** MSR-VTT captions are heavily prior-biased ("a man is talking"), so a
prefix-tuned GPT-2 can score respectably while ignoring the visual prefix entirely. If the real
model does not clearly beat the blind one, the vision path is dead — and frame selection then
provably cannot matter, for reasons unrelated to the scorer. This control must pass before step 4.

## Phased Execution Plan

### Phase 0 — Foundation
- Audit the LMA mini-LM (for the optional decoder ablation); pick and load the frozen CLIP vision encoder and the pretrained small LLM decoder.
- Acquire MSR-VTT, TVSum, SumMe, and the ActivityNet Captions subset; build dense candidate-frame extraction + CLIP-embedding caching pipeline for all of them.
- Curate the small real, non-benchmark usability holdout.
- Set up resumable checkpointing as shared infrastructure for every training script.
**Deliverable:** cached-embedding datasets, curated eval sets, checkpointing utility.
**Exit criteria:** embedding cache regenerable from a script; eval/holdout sets verified untouched by training.

### Phase 1 — Frame-Relevance Scorer (headline component)
- Implement and train the scorer on cached embeddings, supervised by CLIP frame-caption similarity.
- Implement the uniform-sampling and motion-heuristic baselines.
- Validate the scorer's predictions against TVSum/SumMe's real human-annotated importance ground truth.
**Deliverable:** trained scorer + correctness report (predicted-vs-CLIP-proxy correlation, and predicted-vs-human-ground-truth correlation on TVSum/SumMe).
**Exit criteria:** both correlations are meaningfully positive, not noise.

### Phase 2 — Connector From Scratch
- Implement cross-attention resampler, mean-pool baseline, temporal transformer.
**Deliverable:** three connector variants, each passing a tiny-batch overfit check.
**Exit criteria:** all three overfit a tiny batch of (selected-frames, caption) pairs.

### Phase 3 — LoRA From Scratch
- Implement low-rank adapters on the frozen LLM's attention/MLP projections.
**Deliverable:** LoRA module + correctness check.
**Exit criteria:** with adapters zero-initialized, LoRA-augmented forward pass exactly matches the base frozen model's output.

### Phase 4 — Cross-Modal Bridge
- Wire scorer → connector → projector → LoRA-augmented LLM into one forward pass.
**Deliverable:** end-to-end pipeline with correct shapes/masking.
**Exit criteria:** overfits a handful of hand-picked (video, caption) pairs.

### Phase 5 — Stage B: Connector Alignment
- Freeze vision encoder and LLM; train connector + projector on MSR-VTT.
**Deliverable:** aligned checkpoint + Stage B metrics.
**Exit criteria:** loss decreases meaningfully; captions topically relevant even if not fluent.

### Phase 6 — Stage C: LoRA Fine-tuning
- Unfreeze LoRA adapters + connector; continue training.
**Deliverable:** fine-tuned checkpoint + Stage C metrics vs. Stage B.
**Exit criteria:** meaningful improvement over Stage B.

### Phase 7 — Decoding
- Hand-written greedy + beam search.
**Deliverable:** decoding module.
**Exit criteria:** beam width 1 exactly matches greedy decoding.

### Phase 8 — Headline Evaluation
- Quality-vs-frame-budget curves: uniform vs. motion heuristic vs. learned scorer, on MSR-VTT and ActivityNet Captions.
- Frame-selection visualizations on real clips.
**Deliverable:** the headline result of the whole project — curves + visualizations.
**Exit criteria:** a clear, honestly-reported comparison exists at every tested budget, on both eval sets.

### Phase 9 — Ablations
- Connector ablation (Phase 2's three variants, same recipe).
- LoRA ablation (frozen / various ranks / full fine-tune).
- Optional decoder ablation (pretrained LLM vs. LMA mini-LM).
- Optional robustness-to-degradation stretch.
**Deliverable:** consolidated ablation tables.
**Exit criteria:** each ablation has a clear, reported outcome — including null results, reported honestly.

### Phase 10 — End-to-End Assembly & Usability
- Wire the full pipeline into a single runnable path; run on the real, non-benchmark holdout (short and long clips).
**Deliverable:** runnable CLI or minimal demo: video in, caption out.
**Exit criteria:** sane, relevant captions on real clips — the hard usability gate.

### Phase 11 — Packaging
- README: the from-scratch ledger, architecture diagram, headline quality-vs-budget curves, all ablations, an explicit limitations section, and precise resume-ready language (e.g. "designed and trained a learned frame-selection module and cross-attention connector, LoRA-adapted a pretrained language decoder" — not "built from scratch," since that's no longer true and shouldn't be claimed).
**Deliverable:** polished repo, saved to `/home/sampath/Desktop/Projects/VidCap/`.
**Exit criteria:** every claim in the README is traceable to a specific file/function; every reused vs. built-from-scratch component is explicitly labeled.

## Compute strategy
- Embedding caching (Stage 0) means Phases 1, 2, 5, 6 train against small cached tensors, not raw video — dramatically cheaper than the earlier from-scratch plan's pretraining stage. Both Kaggle accounts should comfortably handle this workload.
- Use the two accounts for genuine parallelism: e.g., run Phase 9's ablation variants (different LoRA ranks, different connectors) concurrently across accounts once Phase 6's base recipe is fixed.

## Verification
- Every from-scratch component has a correctness gate before the next phase builds on it (scorer correlation check, tiny-batch overfits, LoRA-zero-init-matches-base check, beam-width-1-equals-greedy check).
- Final gate: sane captions on real, non-benchmark clips (Phase 10) and a clear, honestly-reported headline comparison (Phase 8) — usability and a decisive result are both required, neither is optional.
