# VidCap-Adaptive — Context

Single source of truth for this project. Read this before writing or changing any code.

## What this is
A video captioning system: frame(s) in, natural-language caption out. The headline contribution is a **learned frame-relevance selector** — most video captioning pipelines default to naive uniform/fixed-interval frame sampling under a compute budget; this project builds and rigorously proves whether picking frames by learned importance beats that default.

## Why (the story)
Real deployment (accessibility tools describing what a camera sees, content moderation, edge/mobile inference) is frame-budget-constrained — you can't run a vision encoder over every frame. The standard answer is uniform sampling regardless of content. This project asks: given a fixed budget of K frames, can you pick the K that matter, and does it measurably help?

## Hard requirements (non-negotiable)
1. **Usable end-to-end.** Real video in, real caption out — not just training scripts that ran.
2. **Real depth in vision, language, and their intersection.** No shallow wrapping where a from-scratch version is feasible and reasonable.
3. **A decisive, honestly-measured result**, not just working code — the frame-selection claim must be proven or honestly refuted with numbers, not asserted.
4. **Competitive caption quality.** This targets ML-role placement interviews — a working, respectable result matters, not just conceptual depth. Use frozen pretrained backbones for this; do not try to out-train them from scratch.
5. **Precise resume/README language.** Say "designed and trained a learned frame-selector and cross-attention connector, LoRA-adapted a pretrained LLM decoder" — never claim "built from scratch" for parts that are pretrained/frozen.

## What's pretrained/frozen vs. built from scratch

**Frozen, pretrained (this is where competitive quality comes from):**
- Vision encoder: **SigLIP so400m-patch14-384** (1152-d), applied per-frame, never fine-tuned. Its embeddings are precomputed and cached once — every downstream stage trains on cached tensors, not raw pixels. Chosen over CLIP ViT-L/14 for stronger image-text alignment, which the Phase-1 scorer's training labels depend on. **This choice is baked into the cache** — changing it invalidates every cached embedding, so it is fixed before any caching begins.
- LLM decoder: **Qwen2.5-1.5B-Instruct**, adapted only via LoRA, never fully fine-tuned. Swappable at any time (nothing is cached from it). Note: a stronger language prior *raises* the risk of the model ignoring the visual prefix, which is why the blind baseline is a required gate.

Neither backbone can caption on its own — SigLIP scores image-text match but cannot generate; Qwen generates but has never seen an image. The connector/projector that bridges them exists nowhere pretrained and is the thing that must be trained.

**Built from scratch (the actual contribution — must be independently defensible):**
- The frame-relevance scorer (headline component).
- The cross-attention resampler connector, plus mean-pool and temporal-transformer variants (ablation baselines).
- LoRA (low-rank adapters on the frozen LLM's attention/MLP projections) — raw tensor ops, not `peft`.
- The multimodal fusion/attention-masking scheme (video tokens as a visible prefix, text causally masked).
- Greedy and beam-search decoding.
- The staged training recipe and every loss function (scorer's relevance loss, captioning cross-entropy).
- The full evaluation harness (quality-vs-budget curves, ablations, visualizations).

**Optional/stretch:**
- Swap the pretrained LLM decoder for the author's own from-scratch mini-LM (from prior coursework) as an extra ablation.
- Robustness-to-degraded-video training (blur/low-res/compression augmentation), evaluated on a degraded test set.

## How frame importance is defined (read this before touching the scorer)
- **Training signal**: for each candidate frame, compute CLIP similarity between that frame's image embedding and the clip's ground-truth caption's text embedding. High similarity = high importance label. Use a ranking loss across frames within a clip (relative order matters more than absolute score) rather than plain regression.
- **Scorer's actual input**: only the frame's image embedding. Text/captions are used exclusively to generate training labels — never an input to the scorer.
- **At inference**: no caption exists yet. The scorer runs on visual embeddings only, using patterns learned during training (e.g. a clear central subject/action scores higher than a blurry transition or redundant near-duplicate frame). No chicken-and-egg problem, because text never touches the scorer at inference time.
- **Validation against real ground truth** (see datasets below): TVSum/SumMe provide actual human-annotated frame importance, independent of the CLIP-proxy training signal — use these to check that the CLIP-proxy target is actually a good stand-in for real human judgment, not just to check the scorer learned its own proxy.

## Datasets

| Dataset | Role | Why |
|---|---|---|
| **MSR-VTT** (~10k clips, ~20 captions each) | Primary training + standard captioning-quality evaluation | Public, established captioning benchmark with published baselines for sanity-checking (not a target to beat). |
| **TVSum** + **SumMe** | Direct validation of the frame-relevance scorer against real human-annotated importance | These are video-summarization datasets built specifically around non-uniform frame importance — the correct ground truth to prove the scorer (and the CLIP-proxy training signal) actually reflects human judgment, not just a self-referential proxy. |
| **ActivityNet Captions** (subset) | Stress-test evaluation where naive uniform sampling is expected to visibly fail | Long videos with temporally localized/sparse events — public and credible, unlike a personally-curated clip set. This is the dataset that should make the uniform-sampling failure mode obvious and measurable. |
| Small personal video holdout (a handful of real clips) | Final usability sanity check only, never scored/benchmarked | Confirms the whole pipeline produces sane output on genuinely unseen, non-benchmark video. |

## Architecture (summary — see PLAN.md for full detail)
1. Dense candidate-frame extraction per clip (~1fps) → CLIP embeddings computed once, cached.
2. Frame-relevance scorer picks top-K frames from the candidate pool (compared against uniform sampling and a motion-heuristic baseline).
3. Connector (cross-attention resampler, primary; mean-pool and temporal-transformer as ablation baselines) fuses selected frames into video tokens.
4. Projector maps video tokens into the LLM's embedding space.
5. LoRA-adapted frozen LLM decodes the caption via a prefix-attention fusion mask; greedy/beam-search decoding at inference.

Training stages: **A** (scorer, on cached embeddings) → **B** (connector alignment, backbones frozen) → **C** (LoRA fine-tune, connector + LoRA trainable).

## Deliverables (what "done" looks like)
1. Trained frame-relevance scorer + correctness report (predicted vs. TVSum/SumMe ground-truth correlation, and vs. the CLIP-proxy signal).
2. Quality-vs-frame-budget curves (uniform vs. motion-heuristic vs. learned scorer) on MSR-VTT and ActivityNet Captions — the headline result.
3. Connector ablation (3 variants) and LoRA ablation (frozen/ranks/full fine-tune) tables.
4. A runnable CLI/demo: video in, caption out, tested on the personal holdout.
5. A README with the from-scratch ledger, architecture diagram, all metrics/ablations, and an explicit limitations section.

## Compute
Two Kaggle accounts (T4/P100, 12h session cap each). Mandatory resumable checkpointing (model + optimizer state, periodic save, load-and-resume) in every training script. Embedding caching keeps every downstream stage cheap — most training operates on cached tensors, not raw video/pixels.

## Repo layout (as built)
```
vidcap/config.py       paths (VIDCAP_DATA / VIDCAP_OUT), model ids, pool fps/caps
vidcap/video.py        video -> ~1fps candidate frame pool (+ timestamps)
vidcap/clip_cache.py   frozen CLIP load, image/text embedding, atomic per-video .npz cache
vidcap/datasets.py     msrvtt/tvsum/summe/activitynet/holdout loaders + leakage gate
vidcap/lora.py         from-scratch LoRA over GPT-2 Conv1D (no `peft`)
vidcap/model.py        connectors (meanpool/resampler/temporal), projector, VideoCaptioner,
                       uniform_select, blind control flag
vidcap/checkpoint.py   atomic resumable save/load (model + optimizer + step)
scripts/build_cache.py Stage 0 CLI, resumable (skips existing shards)
scripts/smoke_test.py  Phase 0 gate: CLIP + distilgpt2 load and behave
scripts/kaggle_phase0.py  Kaggle driver
test_phase0.py         self-checks, synthetic video, no downloads
```
**Candidate-pool density** (`POOL_FPS` per dataset + `MIN_POOL_FRAMES=32`): the pool must stay
well above the largest frame budget (K=16), or uniform/motion/learned selection return the same
frames and Phase 8 measures nothing. MSR-VTT's ~15s clips need 3fps, not 1fps, for this reason.

Cache format: `out/cache/<dataset>/<video_id>.npz` → `emb (N,512) float32` (raw, **not**
L2-normalized — normalize at use) + `times (N,)` seconds. Caption text embeddings for the
Phase 1 relevance labels: `out/cache/<dataset>/_captions.npz`.

Execution: heavy Stage 0 work runs on Kaggle (local box is CPU-only, 25 GB free). Local is for
authoring + `./run_tests.sh`. Dataset acquisition details: `DATASETS.md`. Step-by-step Kaggle
walkthrough (account setup → cells → keeping results): `KAGGLE.md`.

## Progress

**Phase 0 — code complete, data not yet acquired.**

Done:
- All Phase 0 modules written and passing `test_phase0.py` (frame sampling, embedding cache
  resumability, MSR-VTT loader, leakage gate, checkpoint save/resume) on a synthetic video.
- Frozen backbones verified locally via `scripts/smoke_test.py`: CLIP ViT-B/32 (512-d, correct
  image–text alignment on a controlled red/blue probe) and distilGPT-2 (82M, 6 LoRA-targetable
  `attn.c_attn` projections confirmed present for Phase 3). Weights sit in the local HF cache.
- Execution target decided: Stage 0 runs on Kaggle GPU; this box is CPU-only with ~25 GB free,
  so it is authoring + tests only.

Not done (blocks Phase 1):
- **No dataset downloaded.** MSR-VTT, TVSum, SumMe, ActivityNet, and the personal holdout must
  be attached as Kaggle datasets — see `DATASETS.md`.
- No embedding cache built yet (nothing to build it from).
- TVSum/SumMe loaders are written to the canonical formats but unverified against real files;
  Kaggle mirrors repackage these, so expect a small fix on first run.
- ActivityNet video subset is the known-soft dependency (no clean public mirror); fallback to
  TVSum/SumMe long videos, decided at Phase 8 rather than now.

**Vanilla captioner — architecture built, gates passed, untrained.**

Execution order revised (see PLAN.md "Revised execution order"): the uniform-sampling captioner is
built and trusted *first*, then frame selection is swapped in against it. The captioner is the
measuring instrument for the frame-selection claim, and the uniform arm is a required baseline
regardless, so nothing is throwaway.

Gates passing in `test_model.py` (no data needed):
- LoRA zero-init output is bitwise-identical to the frozen base; nonzero B does change output;
  all non-adapter params frozen. (6 adapters on distilGPT-2 `c_attn`.)
- All three connectors produce correct prefix shapes into the 768-d LLM space.
- `uniform_select` spans the pool and degrades correctly when pool < budget.
- Blind control produces an identical prefix for different frames; sighted does not.
- Tiny-batch overfit: loss 8.03 → 0.00 on 4 pairs, decode reproduces the memorized captions.

**Blind baseline is a required gate, not an optional ablation.** MSR-VTT captions are heavily
prior-biased, so a prefix-tuned GPT-2 can score respectably while ignoring the visual prefix; if
that happens, frame selection provably cannot matter for reasons unrelated to the scorer. The
sighted model must clearly beat the blind one before the scorer work begins.

**Training + evaluation machinery — built, gated, never run on real data.**

Added: `vidcap/metrics.py` (BLEU-4, ROUGE-L, CIDEr-D), `vidcap/data.py` (dataset/collate over
cached shards, pluggable frame selector), `vidcap/decode.py` (hand-written greedy + beam),
`scripts/train.py` (Stage B/C, resumable), `scripts/evaluate.py` (quality-vs-budget curves),
`scripts/validate_metrics.py`, `run_tests.sh`.

**Metrics validated against pycocoevalcap to machine precision** — BLEU-4 |d|=2.8e-11,
ROUGE-L |d|=0, CIDEr-D |d|=4.4e-16. Numbers are therefore comparable to published baselines.
Re-run `scripts/validate_metrics.py` after any edit to `metrics.py`; a subtly non-standard
metric silently invalidates every "matches baseline X" claim. (ROUGE-L initially differed by
0.0086: max precision and max recall must be taken independently across references, not the
best per-reference F-score.)

**Encoder anisotropy — affects the scorer design.** SigLIP embeddings occupy a narrow cone:
a pure-red and a pure-blue frame score **0.924** cosine similarity. Absolute similarity values
therefore carry little signal; only the *ranking* among frames does. This confirms the planned
ranking loss over per-frame regression. Verify the real spread on MSR-VTT frames before
training the scorer. Corollary: never assert absolute cosine thresholds in tests — compare
relative gaps instead.

Not started: the scorer (step 4), and every training run (steps 2, 3, 6, 7).

## First real run: the connector collapsed (2026-09-14)

First end-to-end Stage B on Kaggle, 500 clips evenly sampled (327 train / 24 val / 149 test),
3 epochs, ~60 steps. **The blind control beat the sighted model on every metric**, so the
step-3 gate failed and the scorer work is blocked until it passes.

Measured, not inferred:
- Across-clip cosine of cached SigLIP pooled embeddings: **0.5665** — inputs are well separated.
- Across-clip cosine of the trained connector's prefix: **0.9999** — the connector emits
  essentially the same prefix for every video. Information is present at the input and
  destroyed by the connector.
- Sighted **4.25 train / 3.82 val**; blind **3.44 train / 3.50 val**. Sighted is worse on
  *training* data, so this is not a generalisation gap — the prefix is harmful noise.

Cause: a Flamingo-style learned-query resampler was made the *primary* connector. BLIP-2's
Q-Former needs a contrastive pretraining stage before captioning loss precisely because cold
learned queries collapse to input-independent output. We skipped that stage and reproduced the
documented failure. Two omissions made it worse: frame embeddings were fed unnormalised (and
SigLIP's are anisotropic — see below), and the connector LR was 1e-4 where LLaVA's alignment
stage uses 1e-3.

Decision: the bridge is not this project's contribution, so it should be the standard,
known-to-train design. Primary connector becomes the ClipCap/LLaVA-shaped MLP path
(`meanpool` + expanding projector); the resampler stays as a Phase 9 ablation, which is where
PLAN.md always had it. The null result is worth reporting: "the resampler collapses without
contrastive pretraining at this data scale; the MLP path does not."

## Blind gate PASSED (2026-09-15)

After switching to the standard bridge, 500 clips (327 train / 24 val / 149 test), 3 epochs,
60 steps, meanpool + LayerNorm + lr 1e-3, trained on random frames:

| arm | BLEU-4 | ROUGE-L | CIDEr-D |
|---|---|---|---|
| sighted, uniform K=8 | 0.3025 | 0.5562 | **0.3108** |
| sighted, motion K=8 | 0.3000 | 0.5510 | 0.2939 |
| blind | 0.1377 | 0.4067 | **0.0186** |

CIDEr-D gap **+0.2922** (16x). The vision path is alive; step 3 of the revised execution order
is cleared and the scorer work is unblocked. Val loss: sighted 3.39, blind 3.59.

Caveat on the `B` diagnostic: across-clip prefix cosine is **0.9729**, down from 0.9999 but not
far down. The earlier "below 0.9 means healthy" threshold was a guess and is wrong — in a
high-dimensional anisotropic space a large shared component can coexist with ample usable
variation, which is exactly what the 16x CIDEr gap demonstrates. Treat the blind gate as the
authority and `B` only as a collapse alarm (≈1.0000 = dead).

Also measured: caching runs at **2.63 s/video**, so the remaining 9,500 clips cost ~7 h — its
own session, since Save & Run All caps at 12 h.

Still open: uniform (0.3108) slightly beats motion (0.2939), as expected on 15s single-shot
MSR-VTT where there is no selection headroom. Do not read it as a selection result.

### Connector decision: meanpool

500 clips, 3 epochs, identical settings — meanpool val **3.3928** vs temporal-transformer
**3.9576**. Meanpool it is, for the full run and for the blind control (they must match or the
gate measures nothing).

Caveat on the method: 60 steps rewards fast convergence, not final quality. Temporal was still
descending (4.37 → 4.05 → 3.96) and self-attention usually needs longer to get going, so it may
close the gap at 4,000 steps. Not worth 5 GPU-hours to find out now — revisit as a Phase 9
ablation once the full cache exists. Meanpool's known cost is temporal blindness: it averages
the K frames, so the model cannot express ordering or change over time.

## Parameter budget (measured, not estimated)

| Component | Params | State |
|---|---|---|
| SigLIP so400m-384 | ~0.88B | frozen, never trained |
| Qwen2.5-1.5B-Instruct | 1.54B (28 layers, hidden 1536, 12 heads / 2 kv-heads) | frozen base |
| Cross-attention resampler | **31.89M** | trained from scratch |
| Projector (1152 → 1536) | **12.39M** | trained from scratch |
| LoRA r=8 (q,k,v,o × 28 layers) | **2.18M** (77,824/layer) | trained |
| **Total trainable** | **46.46M** | ~1.9% of the system |

Ablation connectors: mean-pool 0M (but needs a 61.95M projector at expand=8), temporal
transformer 31.95M. LoRA is small because Qwen uses grouped-query attention — k/v project to
256, not 1536, so those adapters are ~⅓ the size of q/o.

**Why these values.** All are conventional defaults chosen before any data existed; none are
tuned, and all should be revisited after the first real run.
- `n_prefix=8` video tokens — ClipCap used 10 for single images; Flamingo-style resamplers use
  32–64 for video. 8 keeps the sequence short (8 prefix + ~20 caption tokens) so training is
  cheap. Raise it first if captions come out generic.
- `lora_r=8` — standard default. Phase 9 ablates ranks explicitly, so this is a starting point.
- `k=8` frames — the midpoint of the 2/4/8/16 budget sweep.
- 2 resampler layers, 4× FF expansion — deliberately small: ~200k training pairs against 46M
  trainable params already risks overfitting the connector.

## Training data adequacy

MSR-VTT: ~10,000 clips × ~20 captions ≈ **200,000 (video, caption) pairs**, official split
6,513 train / 497 val / 2,990 test. Roughly 4,300 pairs per million trainable params.

Enough to train the bridge, but not generously so — for comparison ClipCap trained a similar-sized
mapping network on COCO's ~600k image-caption pairs. Mitigations already in place: a different
caption is sampled per clip per epoch (natural augmentation, which is why ~20 captions/clip
matters more than 10k clips suggests), val-loss checkpointing every epoch, and a deliberately
small connector. Watch for train/val divergence in Stage B; if it appears, shrink the resampler
before reaching for more data.

## Local machine constraints (read before running anything heavy here)
15GB RAM, **zram swap** (compressed, resident in RAM), no GPU, ~20GB free disk. Overcommitting
memory does not page to disk and OOM-kill — it thrashes zram and locks up the desktop.
- Qwen2.5-1.5B in fp32 is **~6.2GB per instance**. Never hold two simultaneously. Never build
  `VideoCaptioner` in a loop (that hung the machine once: three instances, ~12.4GB peak).
- Run every test via `./run_tests.sh`, which applies a 7GB cgroup cap (`MemoryMax`,
  `MemorySwapMax=0`) so overruns crash cleanly instead of hanging.
- Tests are **light by default**: only the LoRA-target gate loads the real Qwen (it must, since
  target module names are architecture-specific). Everything else uses distilgpt2.
  `VIDCAP_HEAVY=1` adds the real-config overfit — run that on Kaggle, not here.

## Reference
Full phased execution plan: `PLAN.md` in this same folder. Project history/pivot reasoning (not needed for day-to-day coding): `~/Desktop/project_requirements_and_constraints.md`.

## PART ONE COMPLETE — full-scale Stage B (2026-09-15)

6,513 train / 497 val, 10 epochs, 4,070 steps, meanpool, bs=16, ~2h20 per arm.

Val loss: 2.9248 → **2.7408 (best, ep6)** → 2.7424 flat to ep9. Blind plateaus at 3.21.
No overfitting; 10 epochs was right, ~7 would have done.

Test (500 clips, greedy):

| K | uniform CIDEr | motion CIDEr |
|---|---|---|
| 2 | 0.4893 | 0.4804 |
| 4 | **0.5397** | 0.5125 |
| 8 | 0.5376 | 0.5393 |
| 16 | 0.5387 | 0.5445 |

Blind @K=16: CIDEr **0.1027**. Gap **+0.4360**. BLEU-4 0.4064 vs 0.2321.

CIDEr ~0.54 / BLEU-4 ~0.41 sits inside the published MSR-VTT baseline band (~0.45-0.60 CIDEr),
with 46M trainable params on a free T4. **The captioner is done and competitive.**

### The finding that governs phase 8

**Quality saturates at K=4.** K=4/8/16 are 0.5397/0.5376/0.5387 — indistinguishable. Motion ~=
uniform everywhere. So on MSR-VTT no selector can beat uniform at K>=4: four evenly-spaced frames
already give the model everything it can use. The only budget with headroom left is K=2
(0.4893, i.e. 0.05 below saturation).

This is the predicted no-headroom result and it means MSR-VTT cannot host the headline
frame-selection experiment. Unresolved: whether the data is genuinely frame-redundant or
whether mean-pool is too coarse to notice which frames it received. The oracle arm
(`evaluate --oracle`) separates those two and must be run before any scorer work.


## Oracle ceiling: headroom EXISTS at low budgets (2026-09-15)

`evaluate --ckpt stageB --oracle --limit 500 --budgets 1,2,4`, CIDEr-D:

| K | uniform | motion | oracle | oracle - uniform |
|---|---|---|---|---|
| 1 | 0.4495 | 0.4572 | **0.5220** | **+0.0725** |
| 2 | 0.4893 | 0.4804 | 0.5273 | +0.0380 |
| 4 | 0.5397 | 0.5125 | 0.5443 | +0.0046 (noise) |

**Oracle K=1 (0.5220) ~= uniform K=4 (0.5397); oracle K=1 BLEU-4 0.4037 ~= uniform K=16 0.4064.**
One well-chosen frame matches 4-16 evenly-spaced ones — PLAN.md's success condition ("matches
uniform quality at a lower budget"), and a 4-16x cut in encoder calls.

Consequences:
- The headline experiment must run at **K=1,2,3** — not the planned 2/4/8/16. At K>=4 MSR-VTT is
  saturated and every selector ties; a sweep centred on 8 would have measured nothing and been
  read as "frame selection does not help".
- Mean-pool is NOT too coarse to see frame choice. If it were, the oracle could not beat uniform
  at K=1. The K>=4 plateau is redundancy in 15s single-shot clips, not blindness in the model.
  The spatial-grid re-cache is therefore not needed.
- The scorer now has a concrete target: recover part of +0.0725 CIDEr at K=1 without the caption.

Caveats to carry into the README, not to discover later:
1. The oracle reads the test caption. It is an upper bound, not an achievable method, and a
   learned scorer will capture only part of the gap.
2. **Circularity**: the oracle ranks frames by SigLIP-to-caption similarity, and the captioner
   consumes SigLIP embeddings — so the oracle is aligned with what this captioner wants by
   construction. That inflates it relative to a human notion of importance. This is exactly what
   the TVSum/SumMe validation exists to answer, and it must be framed that way.
3. 500 clips, greedy. ~0.005 differences are noise; +0.0725 is not.

## HEADLINE RESULT — learned frame selection (2026-09-15)

`evaluate --ckpt stageB --scorer scorer --oracle --limit 500 --budgets 1,2,3,4`, CIDEr-D:

| K | uniform | motion | **learned** | oracle | learned-uniform | % of ceiling |
|---|---|---|---|---|---|---|
| 1 | 0.4495 | 0.4572 | **0.4966** | 0.5220 | **+0.0471** | **65%** |
| 2 | 0.4893 | 0.4804 | **0.5066** | 0.5273 | +0.0173 | 46% |
| 3 | 0.5246 | 0.5136 | 0.5202 | 0.5359 | -0.0044 | — |
| 4 | 0.5397 | 0.5125 | 0.5238 | 0.5443 | -0.0159 | — |

**The claim: learned K=1 (0.4966) > uniform K=2 (0.4893).** One learned frame beats two uniform
ones — ~2x budget reduction at equal quality, matching PLAN.md's success condition. BLEU-4 agrees
and is stronger: learned beats uniform at EVERY budget (K=1 0.3881 vs 0.3514; K=4 0.4007 vs
0.3945), and learned K=2 (0.3942) ~= uniform K=4 (0.3945).

The honest other half: on CIDEr the advantage inverts at K>=3 (-0.0044, -0.0159). Mechanism is
clear — top-K by relevance has **no diversity term**, so the highest-scoring frames are often
near-duplicates of one moment while uniform spreads across time. Report it; do not bury it. The
obvious fix (MMR / redundancy penalty against already-picked frames) is the strongest cheap
follow-up and would likely make the win hold at every budget.

Motion is a genuine null: it never beats uniform except marginally at K=1 (0.4572 vs 0.4495) and
is clearly worse at K>=3. The classical heuristic does not work here.

Still open before this is defensible:
1. **TVSum/SumMe validation** — the circularity objection (scorer trained on SigLIP-caption
   similarity, captioner consumes SigLIP) is the first thing an interviewer will find. Human
   importance labels are the only answer. Needs those datasets cached.
2. Scorer val Spearman not yet recorded here.
3. 500 clips, greedy decode. ~0.005 is noise; +0.0471 is not.

## STATUS — read this first on resuming (2026-09-17)

**Both halves of the project have a measured result. Every hard requirement in this document is
met. What remains is validation and packaging, not "will this work".**

### Where everything lives

| Artifact | Cost to rebuild | Stored as |
|---|---|---|
| Embedding cache, 10k clips | 8.5 GPU-h | Kaggle Dataset |
| `stageB.pt` + `blind.pt` | 4.6 GPU-h | Kaggle Dataset |
| `scorer.pt` + `eval_msrvtt_test.json` | ~30 min | Kaggle Dataset (`results.zip`) |
| All code | — | github.com/Sampath7099/VidCap (public) |

Nothing valuable exists only inside a Kaggle session. One was lost that way already (an
accelerator change wipes `/kaggle/working`) — zip and publish as a Dataset before closing.

### Fresh Kaggle session recipe

Four things, all needed: clone the repo; re-download MSRVTT.zip from the Oxford mirror (~2 min —
the loader enumerates clips by scanning video files even though training reads only cached
shards); symlink the cache Dataset to `/kaggle/working/cache`; copy the checkpoints Dataset to
`/kaggle/working/checkpoints`. `VIDCAP_DATA=/kaggle/working/data` must be set in the kernel or
subprocesses fall back to `/kaggle/input`. After a kernel restart, re-run the clone and path cells
first — `%cd` and `os.environ` do not survive.

### Done

- Stage 0 caching, Stage B connector training, blind control, oracle ceiling, frame scorer,
  four-arm budget curves, end-to-end demo (`scripts/caption.py`).
- Results: see REGENERATED HEADLINE and Scorer correctness below. Use the 2026-09-17 numbers.

### Remaining, in priority order

1. **Run the demo on self-shot clips** (~1 h). `python -m scripts.caption clip.mp4 --k 1
   --select learned,uniform` prints both captions and the chosen timestamps — the qualitative
   figure and the usability check in one command. Clips must never be trained on or scored.
2. **TVSum/SumMe validation** (~3 h, mostly caching). The circularity objection — scorer trained
   on SigLIP-caption similarity, captioner consumes SigLIP — is the sharpest available criticism.
   Human importance labels are the only rebuttal. Loaders exist in `datasets.py` but have never
   seen real files; expect small fixes on first run.
3. **Stage C LoRA** (~1 h GPU). Quality bump, low risk, low insight.
4. **README/packaging** (~3 h). Four-arm curves, from-scratch ledger, and a limitations section
   stating plainly: the oracle is unreachable, CIDEr at K=4 is a marginal loss, run-to-run noise
   is ~0.007, and scorer and captioner share a representation.
5. **Optional**: diversity-aware selection; best-by-val checkpointing (see Scorer correctness);
   connector and LoRA-rank ablations. None are load-bearing for the claim.

Realistically 8-12 h across three or four sessions. No open technical risk remains.

### Next project phase (planned separately)

Video QA / summarisation as an extension, NOT a standalone rebuild — MSRVTT-QA is built on the
same 10k videos already cached, so the expensive asset carries over. The natural contribution is
**query-conditioned frame selection**: which frames matter depends on the question asked, which
is a stronger version of the selector already built. Finish and package part one first.

## REGENERATED HEADLINE (2026-09-17) — use these numbers, not the 2026-09-15 ones

The 2026-09-15 scorer was lost to an accelerator change. Retrained and re-evaluated; these
figures match `scorer.pt` as shipped in `results.zip`. CIDEr-D, 500 test clips, greedy:

| K | uniform | motion | **learned** | oracle | learned-uniform | % of ceiling |
|---|---|---|---|---|---|---|
| 1 | 0.4495 | 0.4572 | **0.4977** | 0.5220 | **+0.0482** | **66%** |
| 2 | 0.4893 | 0.4804 | **0.5107** | 0.5273 | +0.0214 | 56% |
| 3 | 0.5246 | 0.5136 | **0.5270** | 0.5359 | +0.0024 | 21% |
| 4 | 0.5397 | 0.5125 | 0.5302 | 0.5443 | -0.0095 | — |

**Headline: learned K=1 (0.4977) > uniform K=2 (0.4893)** — one learned frame beats two uniform
ones, a 2x budget cut at equal quality, recovering 66% of the oracle ceiling with no caption.

**On BLEU-4 and ROUGE-L, learned beats uniform at EVERY budget** (BLEU-4 0.3886/0.3853/0.4044/
0.4004 vs 0.3514/0.3806/0.3977/0.3945; ROUGE-L 0.5990/0.6046/0.6123/0.6123 vs 0.5871/0.5975/
0.6101/0.6089). Only 1 of 12 metric-budget cells is a loss: CIDEr at K=4.

**Run-to-run noise is now measured, not guessed.** Two independent scorer trainings differ by
~0.007 CIDEr (K=1 0.4966 vs 0.4977; K=3 0.5202 vs 0.5270). So the -0.0095 at K=4 is barely above
the noise floor — report it as marginal, not as a defeat. Any claim resting on a gap under ~0.01
needs more seeds before it is asserted.

The K>=3 reversal seen on 2026-09-15 did NOT reproduce (learned now wins K=3). The
diversity/redundancy hypothesis for it is therefore unconfirmed — it may have been noise. If
diversity-aware selection is implemented, treat it as an improvement to test, not as a fix for a
known defect, and run multiple seeds before claiming either way.

## Scorer correctness (2026-09-17)

`train_scorer --epochs 20`, 6513 train / 497 val, 0.72M params, ~12 min.

**Val Spearman ~0.51** between predicted frame ranking and the CLIP-proxy target on held-out
clips. Meaningfully positive — PLAN.md Phase 1's proxy exit criterion is met. The human
ground-truth half (TVSum/SumMe) is still outstanding and is the stronger claim.

**The scorer converges in ONE epoch and then mildly overfits**: epoch 0 val loss 3.6871 /
rho +0.5271, drifting to 3.7238 / +0.5137 by epoch 19. `--epochs 3` is enough; 20 wastes ~10 min
and ends slightly worse.

**Known defect: checkpoint.save overwrites each epoch, so the SHIPPED scorer is epoch 19, not the
best epoch.** The headline numbers were therefore produced by a slightly-worse-than-best scorer —
the result is understated, not inflated. Fix is to keep a separate best-by-val checkpoint; the
same applies to stageB (best ep6 2.7408 vs shipped ep9 2.7424, negligible there).

## Phase 2 (video Q&A) — CODE ONLY, NEVER TRAINED (2026-09-17)

Commits 9acec09 and 528dbc0 added the full Q&A path — `fetch_qa.py`, the `msrvtt_qa` loader,
`loss_mask` so question tokens condition rather than supervise, prompted greedy decode, exact-match
accuracy with a by-type breakdown, `train.py --task qa`, `evaluate_qa.py`, and composed summaries
in `summarize.py`. It reuses the existing 10k-clip cache, so it cost ~40MB of annotations instead
of a second 8.5 GPU-hour embedding run.

**There is no `qaB` checkpoint and no accuracy number anywhere.** All gates in `test_qa.py` pass,
but gates prove plumbing, not quality. Until `train.py --task qa` runs on Kaggle, treat Phase 2 as
unverified code. Do not describe it as a result.

## Session 2026-09-25 — first real-clip run, README, packaging

**End-to-end verified on CPU, off-Kaggle.** `scripts/caption.py` runs on this 15GB laptop with no
GPU, both models in fp32, ~1-2 min/clip. Needs ~10GB against ~9GB free, so run it under
`systemd-run --user --scope -p MemoryMax=11G -p MemorySwapMax=0` or the desktop locks up.
Checkpoints were unzipped from `results.zip` to `out/checkpoints/`.

**First real out-of-distribution clip** (handheld night video of a lightning strike, 5.7s):

```
myclip.mp4  (35 candidate frames, 5.7s)
  learned  K=2  [0.2s, 2.8s]   'a person is looking at a city'
  uniform  K=2  [0.0s, 5.7s]   'a man is looking at a city'
```

The selector picked 2.8s — the exact frame the lightning fires. Uniform picked two dark frames and
missed the only event in the clip. **The selection claim reproduces on real video outside the
training distribution**; this is now `figures/selection_myclip.png`.

The caption is wrong on both arms (no person, no city). Expected: MSR-VTT is daylight clips of
people doing things. Two things worth carrying forward:
1. The two arms produced *different* captions from different frames — independent corroboration of
   the blind control, on real video.
2. The flash frame is almost entirely blown-out white. It is the most *distinctive* frame but
   carries little describable content — "unusual" and "caption-relevant" are not the same thing,
   and the scorer optimises the latter. Worth a line in the limitations section.

**Still untested: in-distribution real clips.** Cooking, pets, traffic, a person walking. Only
those tell you whether the captioner is good or merely prior-biased. `myclip.mp4` proves the
selector, not the captioner.

**Packaged**: `README.md` (headline table, blind control, architecture, from-scratch ledger,
limitations), `scripts/plot_curves.py` (regenerates the figure from `eval_msrvtt_test.json`, so it
cannot rot), `figures/budget_curves.png`.

**BLOCKER — the repo still cannot be pushed.** `origin/main` is stale at `a5ec1a8`; 9 local commits
are unpushed. `results.zip` (1.3GB) is committed in `9acec09` and GitHub hard-rejects blobs over
100MB. The zip's contents already live in a Kaggle Dataset, so committing it bought nothing. Fix:

```
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --index-filter \
  'git rm --cached --ignore-unmatch results.zip' --prune-empty d533231..HEAD
git push origin main
```

`*.zip` is now in `.gitignore`, so the file stays on disk and stops being tracked. Attach the
checkpoints to a GitHub Release instead if they need a public home.
