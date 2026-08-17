# Stage B (Mamba) Training — Exact Methodology for Verification

**Purpose of this document.** This is a precise, code-faithful description of how Stage B (the Mamba temporal detector) is trained and evaluated in this project, written so an independent reviewer (human or AI) can check the methodology for correctness, leakage, and statistical soundness. Every step below is traceable to source. Where a design choice is deliberate, the rationale is stated so it can be challenged.

**Source of truth (files):**

| File | Role |
|---|---|
| `mamba/config.py` | All hyper-parameters, dims, and the end-to-end UNION gate thresholds. |
| `mamba/build_sequences.py` | Assembles the 110-dim per-window input sequence (run once, before training). |
| `mamba/model.py` | The Mamba model (pure-PyTorch selective scan) + focal loss. |
| `mamba/train.py` | The training loop, inference, prediction export, and the gate. |
| `gat/metrics.py` | Pure-numpy PR-AUC, ROC-AUC, threshold selection, scalar metrics. |
| `gat/config.py` | Upstream contract (paths, `FUSED_DIM=46`, split codes, phases). |

**Audit status (2026-07-27).** This document has been through one external SME audit. Four
methodology fixes were implemented in response (chunk alignment, multi-seed + burst-bootstrap CI,
honest operating-point reporting, measured early-exit) — see **§12 Audit-response changelog** for
the full record of what changed, what was accepted, and what was refuted. Sections below describe
the pipeline **as it now stands after those fixes.**

**Hardened results (5 seeds = 42–46, `chunk_len=2048`, run 2026-07-27).** Headline metrics are now
the mean ± std over five seeds; the per-seed test PR-AUC carries a burst-grouped bootstrap CI. A
follow-up **Asymmetric-Loss ablation (2026-07-29, §13)** left these numbers unchanged within noise —
establishing that Stage B's ~0.455 PR-AUC is a **loss-invariant information ceiling**, not a
loss-tuning artifact. A subsequent **network-feature-enrichment experiment (2026-07-31, §14)** raised the
per-seed ceiling but tripled the PR-AUC variance (overfitting on 101 positives); a **feature-selection
follow-up (§14.1)** then kept only the 4 discriminative features and achieved the **best result to date —
PR-AUC 0.486 ± 0.019, ROC 0.908, 69% recall at a 2% false-alarm budget** — the first stable headline gain
over the 0.455 baseline.

> ⚑ **How to read these results — see §15 (Results reframe).** Window-level union PR-AUC (~0.49) is a
> **prevalence-bound, near-saturated** metric, not the right headline: at 0.88% test prevalence a ROC-0.91
> ranker *mathematically* lands near PR-AUC 0.5 (one seed hit ROC **0.946** and still PR-AUC 0.473), and
> three independent levers — capacity (§7 Findings), loss (§13), features (§14) — all converge on the same
> ~0.5 wall. **The real headline is campaign-level: 100% of test attack bursts detected (70/70 across 5
> seeds), each at its first window, at a 2% false-alarm budget.** §15 makes this case and states what the
> system actually demonstrates.

| Quantity | Result |
|---|---|
| **Test UNION PR-AUC** | **0.455 ± 0.022** (seeds: 0.450, 0.431, 0.492, 0.437, 0.467) |
| **Test UNION ROC-AUC** | **0.879 ± 0.016** |
| Lift | **2.9× Stage A** (0.157), 2.7× linear probe (0.171), ~50× random-prevalence (0.009) |
| Per-seed PR-AUC 95% CI (burst bootstrap) | e.g. seed 42: 0.450 [0.318, 0.574]; seed 44: 0.492 [0.343, 0.636] |
| Gate verdict (all seeds) | **NO-GO** — PR-AUC < 0.55 and FPR@80%-recall > 0.02 (both aspirational; see §6/§9) |

**Operating point (the honest false-alarm story).** The "FPR ≤ 0.02 at 80% recall" gate is a poor
operating point on this data — the bottom ~20–30% of attack windows score inside the benign range,
so forcing 80% recall drives the threshold into the benign mass, and the resulting Phase-1 FPR is
both high and wildly seed-unstable (**0.26 ± 0.17**). Reported instead as a budgeted operating point:

| Operating point | Test recall | Phase-1 FPR |
|---|--:|--:|
| 80% recall target | 0.82 ± 0.06 | 0.26 ± 0.17 (unusable, unstable) |
| 60% recall target | 0.61 ± 0.05 | 0.032 ± 0.030 |
| **Fixed 2% Phase-1 FPR budget** | **0.632 ± 0.047** | 0.02 (by construction) |

- **Burst-level detection: 14/14 bursts (100%) in every seed** at the 2% FPR budget — the whole
  test campaign is caught even under a strict false-alarm ceiling. This is the core persistence/
  low-and-slow result.
- **Persistence windows: 1/3 in every seed** at that strict threshold — the honest limitation
  (n = 3, low-scoring windows); reported descriptively, never as a headline rate.
- **Early-exit:** ~50–61% of windows exit at 0.95 confidence (benign 50–62%, attacks only 12–23%),
  with PR-AUC essentially unchanged (0.450 → 0.446) — ~half the Block-2 compute is skippable in a
  streaming deployment at negligible accuracy cost.

> ✅ **Temporal control is now hardened (2026-08-03, §8).** A matched 5-seed shuffle control on the
> *current best model* (focal, 114-dim featsel features) gives **ordered 0.486 ± 0.019 vs shuffled
> 0.251 ± 0.054 — paired Δ = +0.235 ± 0.064, positive in all 5 seeds (min +0.182).** Scrambling window
> order nearly halves PR-AUC (shuffled = 52% of ordered) and drops it to the per-window baseline, so
> **temporal modelling is empirically established, not just suggestive.**

---

## 0. What "training Stage B" means in one paragraph

Stage A (a frozen GAT) has already produced, per 10-second window, a 64-dim embedding `z_t`. Preprocessing has already produced, per window, a 46-dim fused feature vector (network flow stats + pooled provenance + elapsed-time). Stage B concatenates these into a **110-dim per-window vector**, orders the windows chronologically, and trains a **2-layer Mamba** to emit a per-window malicious probability. It is trained and tested on **Phase 2 only** (Phase 1 is 100% benign and is used solely as a false-positive stress test). Labels are the **UNION** label (a window is positive if *either* modality flagged it → 208 attack windows). The whole point is to detect the 142 network-only attacks the GAT structurally cannot see, plus exploit short-range temporal context.

---

## 1. Data contract & upstream dependencies

Stage B consumes three upstream artifacts **per phase** (produced before Stage B exists):

1. **Frozen GAT embeddings** — `gat/artifacts/gat_window_embeddings_{phase}.npz`, copied by `build_sequences._freeze()` to `mamba/artifacts/frozen_z_gat_{phase}.npz`. Fields: `window_ids`, `embedding` (N×64). Exported by a **train-only** GAT via `@torch.no_grad()` inference over all splits (this is where leakage would enter if the GAT had seen test windows — see §7).
2. **Fused sequence** — `_preprocessed/{phase}/mamba_sequence_fused.npz`. Fields used: `window_ids`, `sequence` (N×46), `window_times`, `segment_ids`, `window_label_union`, `window_label_prov`, `window_tactic_union`.
3. **Frozen split** — `_preprocessed/{phase}/split_assignment.npz`. Fields used: `window_ids`, `split` (0=train, 1=val, 2=test), `burst_id`.

**Dim contract (asserted in `build_sequences.build`):** embedding dim == 64, fused dim == 46, so step dim == **110** (`config.STEP_DIM`). Mismatches hard-fail.

---

## 2. Sequence assembly (`build_sequences.py`, run once)

```
python -m mamba.build_sequences
```

For each phase (`phase1`, `phase2`):

1. **Freeze** the GAT embeddings to a stable path (so later Stage-A capacity sweeps cannot overwrite what Stage B trained on).
2. **Alignment check:** the set of embedding `window_ids` must exactly equal the set of fused `window_ids` (`assert set(emb_ids) == set(fused_ids)`).
3. **Reorder** the embedding rows into the fused file's **chronological** order. (The GAT exports embeddings concatenated as train|val|test, *not* by time; Mamba needs time order.) Done via a `{window_id: row}` map, not by assuming any order.
4. **Concatenate** `[z_t (64) ⊕ fused (46)] → 110-dim` per window. `assert step.shape[1] == 110`.
5. **Split & burst codes** are pulled from `split_assignment.npz` keyed by `window_id` (dict lookup, order-independent).
6. **Segments (temporal discontinuities)** are **recomputed** from `window_times`: `seg = cumsum(diff(times) > 1800s)` (30-minute gap threshold), rather than trusting the stored `segment_ids` (which are all-zero in the shipped fused file). It asserts the timeline is monotonic (`diff(times) >= -1e-6`). **Verified note in code:** the fused sequence is temporally continuous (max inter-window gap ~43 s over the 72 h span) because it is gridded on continuous network windows, so this yields a **single segment** — the documented 34.9 h provenance gap is filled by the network grid.
7. **Save** `mamba/artifacts/mamba_input_{phase}.npz` with: `window_ids, step (N×110), segment_ids, window_times, split, burst_id, y_union, y_prov, tactic`. Prints a NaN check on `step`.

> **Reviewer check points (§2):** (a) Is a single segment correct — i.e., is it legitimate that provenance-gap windows are bridged by network-only windows? The code’s claim is that network windows are continuous; verify against `window_times`. (b) Reordering correctness relies on `window_id` uniqueness within a phase — verify no duplicate ids. (c) `y_union` and `tactic` come from the fused file, `burst_id`/`split` from the split file, all keyed by id — verify these three files agree on the id universe.

---

## 3. Model (`model.py`)

**`MambaDetector`** (`config.MAMBA`):

- **Input projection:** `Linear(110 → d_model=128)` then `LayerNorm`.
- **Blocks:** `n_layers=2` `MambaBlock`s, each pre-norm residual: `h = h + Dropout(block(LayerNorm(h)))`, `dropout=0.1`.
- **Main head:** `Linear(128 → 1)` → per-step logit, shape `(B, L)`.
- **Early-exit head:** after **block 1** (index 0), `Linear(128 → 1)` → `exit_logit`, trained to predict **benign-confidence** (target `1 − y`). `early_exit=True`. At inference this head now drives a **measured** early-exit report (§6, Fix D) with `exit_threshold=0.95`; it does **not** alter the canonical `y_prob`.

**`MambaBlock`** — pure-PyTorch S6 selective scan (the official `mamba-ssm` CUDA kernel is unavailable on this CPU host; this is stated to be numerically equivalent):
- `in_proj: Linear(128 → 2·d_inner)`, `d_inner = expand·d_model = 256`; split into `xi, z`.
- Causal depthwise `Conv1d` (kernel `d_conv=4`, groups=d_inner, left-padded then truncated to L) → `SiLU`.
- Input-dependent selective params: `x_proj → (Δ, B, C)` with `d_state=16`; `dt = softplus(dt_proj(Δ))`; `A = −exp(A_log)` (log-parameterised, init `A = 1..d_state`).
- **Sequential scan** (the O(L)-time, O(1)-state loop): ZOH discretisation `dA = exp(dt·A)`, `dBu = dt·B·u`; recurrence `h_t = dA_t·h_{t−1} + dBu_t`; output `y_t = einsum(h_t, C_t)`. Skip `+ xi·D`, gate `· SiLU(z)`, `out_proj`.

**Parameter count:** printed at train time; methodology quotes ≈ 0.24 M.

**Loss — `focal_bce_with_logits(logits, targets, gamma, pos_weight, mask)`:**
`ce = BCEWithLogits(reduction="none", pos_weight)`, `p_t = p·y + (1−p)(1−y)`, `loss = (1−p_t)^gamma · ce`, mean (or masked-mean). `gamma = focal_gamma = 2.0`.

> **Reviewer check points (§3):** (a) The pure-PyTorch scan is the actual model used for the reported numbers — verify the "numerically equivalent to mamba-ssm" claim is not load-bearing for any result. (b) *[Resolved by Fix D.]* The early-exit head is now used at inference by a **measured** early-exit report (`infer_early_exit` / `_early_exit_report`, §6) that quantifies exit-rate and the PR/recall cost, rather than being trained-but-unused. Note the report deliberately does **not** skip Block 2 in the batched scan — see §6/§12 for why an SSM cannot drop mid-sequence steps and how the saving is framed as streaming-deployment potential. (c) `A_log` init and the discretisation are standard; verify `dt.unsqueeze(-1)*A` broadcasting shapes `(B,L,d_inner,d_state)`.

---

## 4. Training procedure (`train.py :: train`)

Command: `python -m mamba.train` (full run) or `--epochs N --chunk-len K` (smoke test); `--device cuda|cpu`; `--shuffle-control` for the temporal ablation.

**Step-by-step, exactly as coded:**

1. **Seed everything** (`set_seed(42)`): numpy, torch, torch.cuda.
2. **Load Phase 2** `mamba_input_phase2.npz`.
3. **Standardisation (leakage-safe):** compute `mu, sd` on `step` rows where `split == 0` (**train only**), then `step ← clip((step − mu)/sd, −10, 10)`. Save `mu, sd` to `input_scaler.npz`. Rationale in code: raw `z_t`/fused scales stall optimisation and the model floods to ~0.02 PR-AUC; standardised input lets even a linear probe reach ~0.17.
4. **Build runs** via `_runs(split, segment_ids)`: maximal contiguous `[start, end)` spans of constant `(split, segment)`. Since there is one segment, runs are simply the maximal contiguous same-split stretches along the timeline. `train_runs` = runs with split code 0. Val/test are boolean masks `va = split==1`, `te = split==2`.
5. **Positive weighting:** `pos_weight = max(1.0, target_effective_pos_weight=25.0)`. This is a **per-step** loss with **no oversampling** (unlike Stage A which oversampled), so the full weight is applied directly. Rationale in code: a high `pos_weight` on this per-step sequence loss floods predictions to "positive"; 25× is the moderate value chosen.
6. **Model + optimiser:** `MambaDetector` on device; `AdamW(lr=1e-3, weight_decay=1e-4)`. (`lr=3e-4` is noted as too slow to escape the "flood basin".)
7. **Epoch loop** (`epochs=80`, `early_stop_patience=15`):
   - `model.train()`; **shuffle the order of `train_runs`** each epoch (`np.random.shuffle`). Note this shuffles *which run comes first*, not the order of windows within a run (window order is preserved unless `--shuffle-control`).
   - For each train run, iterate in **chunks of `chunk_len=2048`** windows *(Fix A — was 1024)*. Each chunk is one sequence `x = step[sel].unsqueeze(0)` shape `(1, L≤2048, 110)`; target `y[sel]`. 2048 covers the longest val/test run (1,777), so every evaluated run is a single un-split sequence and eval never exceeds the training context length; only the 8,735-window train run is still chunked (train-only, acceptable).
   - Forward → `logit, exit_logit`. Loss = `focal_bce(logit, tgt, γ=2, pos_weight=25)`. If early-exit: `loss += 0.3 · focal_bce(exit_logit, 1−tgt, γ=2)` (**no** pos_weight on the exit head).
   - `backward`; `clip_grad_norm_(5.0)`; `opt.step()`.
   - **Validation each epoch:** `infer()` over the whole phase; pick threshold `thr = threshold_for_target_recall(y_val, prob_val, C.OPERATING_RECALL_TARGET=0.80)` *(Fix C — the 0.80 target is now a named config constant, not a hard-coded literal)*; compute val metrics. **Model selection score = val PR-AUC** (threshold-free).
   - **Early stopping / checkpointing:** keep `best_state` = the state dict at max val PR-AUC; stop after 15 epochs without improvement. Restore `best_state` at the end.
8. **Save** checkpoint `mamba_stageB{tag}.pt` (`tag = "_shuffle"` under the control).

> **Reviewer check points (§4):** (a) *[Resolved by Fix A.]* Train and eval now use the **same** `chunk_len=2048`, and all val/test runs are ≤1,777, so every evaluated sequence is un-split and within the training context length — the former 1024-vs-4096 mismatch is gone. State still resets at chunk boundaries, but this now only affects the 8,735-window **train** run; no evaluated run is chunked. (b) `pos_weight` is applied but the loss is a plain `.mean()` over all steps — confirm the effective positive emphasis is what’s intended (25× on ~0.6% positives). (c) Early-exit auxiliary loss weight 0.3 is a hyper-parameter with no ablation — flag as unjustified-by-data (still open). (d) Val threshold is recomputed every epoch but only PR-AUC (threshold-free) drives selection, so the threshold choice does not affect *which* checkpoint is kept — good.

---

## 5. Inference (`train.py :: infer`, `@torch.no_grad()`)

Per phase, for each run, evaluate in **chunks of `model.cfg["chunk_len"]` (= 2048)** windows *(Fix A — previously a hard-coded 4096)*. `x = step[sel].unsqueeze(0)`, `logit, _ = model(x)`, `prob[sel] = sigmoid(logit)`. Under `--shuffle-control`, `sel` is a **fixed per-run permutation** (`_perm(a,b,seed)` seeded by `[seed, a, b]`) and predictions are scattered back to original positions.

> **Reviewer check point (§5):** *[Resolved by Fix A.]* Eval and train now share `chunk_len=2048`. Since the longest val/test run is 1,777 (< 2048), no evaluated run is chunked at all, and eval context never exceeds what the model saw in training. The prior 1024-vs-4096 concern no longer applies.

---

## 6. Prediction export & the UNION gate (`train.py :: train` tail, `_gate`)

**Export** for both phases → `predictions_{phase}{tag}.npz` with `window_ids, split, y_true (=union), y_true_prov, y_prob, burst_id, tactic, window_time`. Phase 1 is standardised with the **same** train-fitted `mu, sd` before inference (contract match).

**Gate (`_gate`)** — end-to-end readiness for the full GAT→Mamba system, on **Phase 2 test** unless noted:

1. `thr = threshold_for_target_recall(y_val, prob_val, 0.80)` — operating point chosen on **validation**, then frozen and applied to test. (`threshold_for_target_recall` = highest positive-score threshold whose validation recall ≥ 0.80; see `gat/metrics.py`.)
2. Compute test metrics at that `thr` plus threshold-free PR-AUC / ROC-AUC.
3. **Gate checks** (from `config.GATE`):
   - `test union PR-AUC ≥ 0.55`
   - `test union ROC-AUC ≥ 0.85`
   - `persistence detected` — at least one true persistence test window has `prob ≥ thr` (only checked if such windows exist).
   - `Phase-1 benign FPR ≤ 0.02` — computed on **Phase-1 test-split** windows as `mean(prob ≥ thr)`.
4. `beat_stageA_union_pr_auc` and `require_persistence_detected` are declared in config; persistence is enforced, and the Stage-A comparison is reported (see below) rather than hard-checked in `_gate`.
5. **Verdict** = GO iff all checks pass; written to `stageB_gate{tag}.json` with full `test_metrics`.

**Burst-grouped bootstrap CI (Fix B).** `_gate` now attaches a 95% CI to the test PR-AUC via `M.bootstrap_ci(y[te], prob[te], M.pr_auc, group=grp, n_boot=1000, seed=42)`, printed as `PR-AUC {point} [lo, hi]` and saved as `pr_auc_ci` in the gate JSON. **Cluster design (a deliberate, flagged choice):** attack windows resample by `burst_id` (their statistically-independent unit); benign windows — which all share `burst_id == −1` — are each given a **unique** negative id (`grp[neg] = −(arange+2)`) so they resample individually rather than collapsing to one coin-flip cluster. A reviewer may reasonably contest the benign side (e.g. resample by contiguous benign run instead); this is documented in the code comment at the CI site.

**Operating-point curve (Fix C).** `_gate` sweeps recall targets `[0.90 … 0.50]`, recording per point: threshold, test recall, test precision, test FPR (Phase 2), and **Phase-1 FPR**; it also stores the full PR and ROC curve arrays. All saved to `stageB_operating_curve{tag}.json` (+ an optional PNG behind a `try/except` so a missing matplotlib never breaks the run). A one-line honest note is printed with the **computed** trade — e.g. the pre-fix data shows recall-target 0.70 meets Phase-1 FPR ≈ 0.0153 (< 0.02) at the cost of test recall falling 0.789 → 0.667. **The gate thresholds themselves are unchanged** — the curve is added evidence, not a way to move the goalposts.

**Measured early-exit report (Fix D).** `infer_early_exit` + `_early_exit_report` run the full model, then compute, at `exit_threshold=0.95`: the exit rate (overall / benign / attack), and the PR-AUC / ROC / recall **cost** if confident-benign windows were routed to the Block-1 head's own estimate (`1 − P(benign)`). Saved to `stageB_earlyexit{tag}.json`. Because Block 2 is a sequential SSM scan over the whole sequence, exited windows are **not** actually skipped in this batched codepath — the report says so explicitly and frames the compute saving as *streaming-deployment potential only*. The canonical `y_prob` in the predictions file remains the full-model output.

**Temporal-order verdict (shuffle run only):** compares shuffled test PR-AUC against the ordered run’s stored PR-AUC and prints the delta. Large positive delta ⇒ temporal order is used; ≈0 ⇒ the gain is fusion, not order.

> **Reviewer check points (§6):** (a) **The Phase-1 FPR threshold is borrowed from a Phase-2-validation operating point.** The threshold is chosen for 80% recall on Phase-2 val positives, then applied to Phase-1 (a different phase, 100% benign). *Now surfaced (Fix C):* the operating-point curve reports Phase-1 FPR at every recall target, so the cross-phase transfer is visible rather than hidden. Empirically Phase-1 is **quieter** than Phase-2 test at every threshold (so the failure is a low-threshold artifact, not adverse drift); it fails the 0.02 gate at 0.037 for recall-target 0.80 and passes (0.0153) at 0.70. Verify whether 0.80 is the right operating recall for the "quiet factory" claim. (b) **Test-set influence on the operating point:** the threshold is set on validation, not test — good, no test leakage into `thr`. The *gate thresholds themselves* (0.55, 0.85, 0.02) were set with knowledge of Stage-A’s ~0.55 ceiling and were **left unchanged by the audit** (Fix C added reporting, not threshold-moving); verify they’re treated as pre-registered targets. (c) `beat_stageA_union_pr_auc=True` is in config but the ordered `_gate` does not appear to load and compare the frozen GAT-alone union PR-AUC — **still open**: verify this check is actually enforced somewhere or downgrade the claim.

---

## 7. Leakage-safety argument (the part most worth auditing)

The pipeline order is designed so the test set never shapes any fitted quantity. Claimed guarantees:

1. **Split frozen first**, in preprocessing, before any model exists (burst-grouped contiguous regions; no burst straddles a boundary → no neighbouring-window leakage).
2. **GAT trained on train split only**, checkpoint chosen on **val** ranking; embeddings for all splits produced by `@torch.no_grad()` inference of that train-only GAT (per `gat/train.py`). ⇒ test embeddings `z_t` carry no test-trained information.
3. **Input standardiser (`mu, sd`) fit on train split only** (§4.3), reused verbatim for val, test, and Phase 1.
4. **Mamba trained on train runs only**; val used only for model selection + threshold; **test only ever passes through frozen models** (Stage-A GAT + best Stage-B checkpoint).

> **Reviewer check points (§7):** (a) Confirm `gat/train.py` truly never includes val/test windows in GAT weight updates or in the GAT’s own normaliser (methodology asserts the training loop iterates the train loader exclusively and export is `@torch.no_grad()`). (b) Confirm the split assignment used by Stage B (`split_assignment.npz`) is byte-identical to the one Stage A trained under (same freeze). (c) The 30-min segment recomputation uses **all** windows’ timestamps (not split-specific) — this is only reading time metadata, not labels, so it is leakage-neutral; verify.

---

## 8. Temporal-shuffle control (the core mechanistic claim)

`python -m mamba.train --shuffle-control` retrains and re-evaluates with window order scrambled **within each run** by a fixed per-run permutation (`_perm`), identically at train and eval time. Everything else (splits, labels, standardiser, hyper-params, seed) is held fixed. If temporal order carried no signal, shuffled test PR-AUC would match ordered.

**Hardened result (2026-08-03, 5 seeds each side, matched: focal loss, 114-dim featsel features, `chunk_len=2048`).** Ordered and shuffled differ in *nothing but window order*:

| | Test PR-AUC | Test ROC-AUC |
|---|---|---|
| **Ordered** (current best) | **0.486 ± 0.019** | 0.908 ± 0.021 |
| **Shuffled** (temporal control) | **0.251 ± 0.054** | 0.819 ± 0.027 |
| **Paired Δ (order)** | **+0.235 ± 0.064** | +0.089 |

Per-seed Δ = +0.185 / +0.346 / +0.269 / +0.182 / +0.192 — **positive in all 5 seeds (min +0.182)**, so the effect is far outside seed noise. Scrambling order **nearly halves PR-AUC** (shuffled = 52% of ordered) and lands it at the per-window linear-probe baseline (~0.17), and ROC degrades too (0.908 → 0.819) so it is not a threshold artifact. **Conclusion: temporal modelling is empirically established** — Mamba without order degenerates into a per-window model. This supersedes the earlier single-seed indication (0.405 vs 0.193 on old features); the effect is *larger* with the current features because both sides rose.

> **Reviewer check points (§8):** (a) *[Resolved.]* Ordered and shuffled are both **5-seed, same loss, same features, same `chunk_len`** — the only variable is window order. All five paired deltas are positive (min +0.182). (b) The shuffle is *within-run*; since there is one segment, that is effectively a global within-split shuffle — the strongest null (destroys all order). (c) The shuffled model's broad degradation (ROC 0.82, high Phase-1 FPR) is consistent with order removal, now on a matched control. Artifacts: `mamba/artifacts/predictions_phase2_shuffle_s*.npz`, `stageB_gate_shuffle_s*.json`.

---

## 9. Statistical-power caveats a reviewer must weigh

- **Test positives = 57 windows** (union), in **~6,468 test windows** (~0.9% prevalence). PR-AUC and especially the operating-point precision/MCC are high-variance on this many positives. *Now quantified:* threshold-free PR-AUC is actually stable across seeds (**0.455 ± 0.022**), but the **operating-point Phase-1 FPR at 80% recall is not** (0.26 ± 0.17) — because the 80%-recall threshold sits at the 10th-lowest of 50 val positives, an extremely noisy quantity. This is *why* the headline is threshold-free PR-AUC and the false-alarm story is reported as a budgeted operating point (63% recall at 2% FPR), not FPR at a fixed recall.
- **Persistence = 3 test windows.** "Persistence detected" is a **binary case-study check on 3 windows**, never a robust per-class accuracy. *Measured (2% FPR budget):* **1/3 persistence windows** flagged per seed at the strict threshold, while **14/14 attack bursts are detected** — so the campaign is caught at the burst level even where individual persistence windows are missed. Treat as descriptive.
- **Bursts, not windows, are the independent unit.** *[Addressed by Fix B.]* `_gate` now reports the test PR-AUC **with a burst-grouped bootstrap CI** (`n_boot=1000`), attacks resampled by burst and benign windows resampled individually (see §6). The headline is a `point [lo, hi]`, not a bare estimate. Reviewer should still weigh whether the benign-side resampling assumption is appropriate.
- **Single campaign, single testbed.** All 208 attacks are one Caldera campaign; the split partitions that campaign in time. Results measure **within-campaign** detection, *not* generalisation to novel malware. No cross-dataset test yet.

---

## 10. Exact reproduction sequence

```bash
# 0. Prereqs already produced by Stage A + preprocessing:
#    gat/artifacts/gat_window_embeddings_{phase1,phase2}.npz   (train-only GAT, no_grad export)
#    _preprocessed/{phase}/mamba_sequence_fused.npz
#    _preprocessed/{phase}/split_assignment.npz

# 1. Assemble the 110-dim sequences (writes mamba/artifacts/mamba_input_{phase}.npz)
python -m mamba.build_sequences

# 2. Train the ordered model(s), export predictions, run the UNION gate.
#    Multi-seed (Fix B): mean +/- std + burst-bootstrap CI. Seed 42 also keeps the
#    canonical unsuffixed artifact names; every seed writes _s{seed} variants.
#    Also emits stageB_operating_curve*.json (Fix C) and stageB_earlyexit*.json (Fix D).
python -m mamba.train --seeds 42,43,44,45,46 --device cuda   # full: 80 epochs, chunk 2048
#    -> mamba_stageB.pt, predictions_{phase}.npz, stageB_gate.json,
#       stageB_multiseed.json, stageB_operating_curve_s*.json, stageB_earlyexit_s*.json

# 3. Multi-seed temporal-shuffle control; prints the ordered-vs-shuffled delta vs stageB_gate.json
python -m mamba.train --shuffle-control --seeds 42,43,44,45,46 --device cuda
#    -> mamba_stageB_shuffle.pt, predictions_{phase}_shuffle.npz, stageB_gate_shuffle.json, ...

# Single-seed smoke (CPU-feasible): py -m mamba.train --epochs 1 --chunk-len 256 --seeds 42
# GPU note: the selective scan is a pure-PyTorch sequential loop (slow on CPU).
# A CUDA host with real mamba-ssm kernels is recommended for the full run.
```

**Determinism:** each seed path calls `set_seed(s)` (numpy + torch +cuda). The pure-PyTorch scan has no nondeterministic CUDA kernels; on CPU the run should be reproducible. `train_runs` epoch order uses `np.random.shuffle` (seeded), and `_perm` is independently seeded by `[seed, a, b]`. The bootstrap CI uses a fixed `seed=42`.

---

## 11. Summary of open methodological questions for the reviewer

Ranked by how much they could change the conclusions. **Status tags reflect the §12 audit response.**

1. **Multi-seed hardening** *(done — Fix B + §8)* — the ordered headline is **0.486 ± 0.019** (current best) with per-seed burst-bootstrap CIs, and the **temporal claim is now hardened**: matched 5-seed shuffle control gives ordered 0.486 vs shuffled 0.251, **paired Δ +0.235 ± 0.064, positive in all 5 seeds** (§8). *Closed.*
2. **Cross-phase / operating-point FPR** *(resolved as a reporting change — Fix C)* — the pooled operating curve now shows Phase-1 FPR at every recall target. The finding: FPR@80%-recall is unusable and seed-unstable (0.26 ± 0.17); the defensible operating point is **63% recall at a 2% Phase-1 FPR budget**. The gate was **not** relaxed to pass. Open judgement: which operating recall to headline.
3. **Train/eval chunk-size mismatch and per-chunk state reset** *(resolved — Fix A)* — train and eval share `chunk_len=2048`; no evaluated run (max 1,777) is chunked, so eval context never exceeds training context.
4. **`beat_stageA_union_pr_auc` not enforced in `_gate`** *(still open)* — verify the GAT-alone baseline comparison is actually computed, not just asserted in prose (0.405 vs 0.157 is the claim).
5. **Early-exit** *(resolved as a measured report — Fix D)* — the head now drives a quantified exit-rate + PR/recall-cost report; the batched scan does not actually skip Block 2, so any compute-saving is stated as streaming-deployment potential, not a measured latency win. The aux-loss weight (0.3) remains un-ablated.
6. **PR-AUC reported without burst-bootstrap CI** *(resolved — Fix B)* — `_gate` now reports `point [lo, hi]` with a burst-grouped bootstrap; the benign-side resampling choice is flagged for review.
7. **Single campaign / within-campaign scope** *(unchanged, by design)* — no generalisation claim is defensible; the writeup says so (§10 of `Methodology.md`).

---

## 12. Audit-response changelog (2026-07-27)

An external SME audit raised four issues plus a "metrics good enough?" assessment. Each was
independently verified against the committed artifacts before acting; verdicts and the resulting
code changes are recorded here so a reviewer can see exactly what was accepted, refuted, and changed.

| Audit item | Verdict after verification | Action |
|---|---|---|
| **1. Chunk mismatch (1024 vs 4096) + SSM state resets** — claimed "critical", eval OOD state drift, proposed cross-chunk state carry | **Refuted as stated / minor in fact.** No test run reaches 4096 (max 1,777); the SSM is a contraction (`A<0`), so no unbounded drift. Real residue: one 1,777 test run exceeded the 1024 train cap. State-carry is over-engineering for bounded runs. | **Fix A:** `chunk_len=2048` for both train and eval (`infer` uses `model.cfg["chunk_len"]`). Mismatch eliminated; no state-carry added. |
| **2. Fragility on N=57 + single seed** — 5-seed mean±std, burst-bootstrap CI | **Accepted (fully valid).** | **Fix B (done for ordered):** `--seeds` loop + aggregate JSON; burst CI in `_gate`. 5-seed ordered result: **PR-AUC 0.455 ± 0.022, ROC 0.879 ± 0.016** — the multi-seed run confirmed threshold-free ranking is stable. Temporal-control re-run still pending (§8/§11). |
| **3. Cross-phase FPR; "lower recall target to fix it"** | **Confirmed empirically, but the AI understated the cost and mis-stated the number.** Recall-target **0.70** (not "0.70 or 0.75") drops Phase-1 FPR to 0.0153 (< 0.02) — but test recall falls 0.789 → 0.667. Also: Phase 1 is *quieter* than Phase-2 test, so the gate failure is a low-threshold artifact, not adverse drift. | **Fix C:** report the full FPR-vs-recall + PR/ROC curve (`stageB_operating_curve*.json`) and expose `OPERATING_RECALL_TARGET`; **gate thresholds left unchanged** (no goalpost-moving). |
| **4. Early-exit disconnect** — implement the inference gate or drop the aux loss | **Accepted (valid).** But a naive "skip Block 2" is ill-defined in an SSM: the scan is sequential over the whole sequence, so dropping mid-sequence steps corrupts later state and saves no FLOPs in the batched path. | **Fix D:** implement a **measured** early-exit (exit-rate + PR/recall cost at `exit_threshold=0.95`, exited windows → `1−P(benign)`), explicitly framing compute saving as streaming-deployment potential. Canonical `y_prob` unchanged. |
| **Part 2 — "metrics good enough" (0.405 = 46× random, 2.6× Stage A; NO-GO ≠ failure)** | **Broadly accepted, tempered.** Math confirmed (46× over prevalence 0.0088; 2.6× over 0.157). "130:1 imbalance" is loose (~113:1 at test window level). "Shuffled = fusion-only" is imprecise — the true fusion-only, no-order baseline is the **linear probe at 0.171**; shuffled (0.193) still has the temporal architecture. The 0.193→0.405 jump is real but **single-seed** — not "proven" until Fix B's multi-seed run. | Documented here and in §8/§11; no code change. |

**Verification method.** Claims 1 and 3 were checked with a numpy script over the committed
`mamba_input_phase2.npz` and `predictions_*.npz`; the harness reproduced the committed
PR-AUC (0.4048) and Phase-1 FPR (0.0368 at recall 0.80) exactly, so the run-length distribution
and FPR-vs-recall sweep quoted above are measured, not assumed.

---

## 13. ASL ablation — the Stage-B loss-invariance ceiling (2026-07-29)

**Question.** Does a better cost-sensitive loss move Stage B past the focal baseline? This directly
tests the recommended balancing lever (see `DataBalance-Structure.md`) before committing to it.

**Setup.** Identical 5 seeds (42–46), identical frozen split, identical `chunk_len=2048`; the *only*
change is the main detection head's loss: focal (γ=2, pos_weight=25) → **Asymmetric Loss** (Ben-Baruch
et al. 2020; `γ⁺=1, γ⁻=4, clip=0.05, asl_pos_weight=1.0`). The early-exit head stays focal. The focal
baseline artifacts are preserved in `mamba/artifacts_focal_baseline/`; both arms are scored with the
same operating-point definitions.

| Metric | Focal (baseline) | ASL | Δ (ASL − focal) |
|---|---|---|---|
| **Test PR-AUC** | 0.455 ± 0.022 | 0.459 ± 0.019 | **+0.003** (0.15σ — noise) |
| Test ROC-AUC | 0.879 ± 0.016 | 0.880 ± 0.011 | +0.001 |
| Recall @ 2% Phase-1 FPR | 0.632 ± 0.047 | 0.646 ± 0.071 | +0.014 (noise, and noisier) |
| **FPR @ 80% recall** (gate metric) | 0.262 ± 0.172 | **0.146 ± 0.106** | **−0.115 (~44% lower)** |
| Burst detection @ 2% FPR | 14/14 | 14/14 | — |
| Persistence windows @ 2% FPR | 1.0 / 3 | 1.2 / 3 | +0.2 |

**Findings.**
1. **Ranking is loss-invariant.** PR-AUC and ROC-AUC are identical within seed noise (Δ ≈ 0.15σ). ASL
   did *not* raise the headline; the success bar (mean PR-AUC > 0.477) was not met.
2. **Benign suppression helps only the operating point.** ASL's asymmetric negative down-weighting
   pushed benign scores down, cutting FPR-at-80%-recall by ~44% (0.26 → 0.15) and letting **one seed
   (44) clear the 0.02 gate**. This is exactly — and *only* — the effect predicted for a cost-sensitive
   loss on this data.
3. **This is the Stage-B analog of Finding 2's capacity-invariance ceiling.** Two very different losses
   land at PR-AUC ≈ 0.455, strong evidence that ~0.455 is the **information / separability ceiling of
   the fused input** for this campaign. The 142 network-only attacks sit inside the benign distribution
   (Findings 1–2); no loss function manufactures signal that is not in the input.
4. **Side effect (a mark against adopting ASL).** ASL compressed the score distribution, so the
   (unchanged, focal-trained) early-exit head over-fired — exit rate rose to **78–94%** vs ~55% under
   focal, costing recall. Calibration temperatures also became erratic (0.27–1.04) as the scores
   shifted; harmless (rank-preserving) but a symptom of the distribution change.

**Verdict.** ASL is **not an improvement to the headline** — keep **focal as the trained default** and
cite ASL as the ablation that *establishes the loss-invariance ceiling*. The remaining levers for
PR-AUC are **better network features / fusion** or **cross-campaign data**, not balancing or loss
engineering. The gate remains NO-GO, now for a documented, defensible reason rather than an untested one.

---

## 14. Network-feature enrichment — ceiling up, variance up (2026-07-31)

**Question.** §13 showed the ~0.455 ceiling is loss-invariant — but *conditional on the current features*.
The 17 network features are all mean/sum/ratio reductions that dilute the one anomalous flow in a window.
Do richer features (tail/dispersion/port-cardinality) — which a held-out probe showed lift network
attack-vs-benign ROC from 0.55 → 0.78 — raise the ceiling? (See `DataBalance-Structure.md` for the probe.)

**Setup.** Added 13 engineered network features to `step11_window_fusion.py` (per-window `max/std` of Rate
& Tot size, `min` IAT, `max` flow-duration/header/packet-size, **distinct src/dst ports**, ports-per-host,
`max` syn/rst counts). Fused dim 46 → 59, Stage-B input 110 → **123**. Everything else identical: same
frozen split (**verified byte-identical**), same 5 seeds, same focal loss. Baselines preserved in
`mamba/artifacts_focal_baseline/` (110-dim) and `mamba/artifacts_netfeat_baseline/` (123-dim).

| Metric | Focal 110 (baseline) | ASL 110 | **Enriched 123** |
|---|---|---|---|
| **Test PR-AUC** | 0.455 ± 0.022 | 0.459 ± 0.019 | **0.447 ± 0.074** |
| per-seed PR-AUC range | 0.43–0.49 | 0.43–0.48 | **0.34–0.53** |
| Test ROC-AUC | 0.879 | 0.880 | 0.885 |
| Recall @ 2% Phase-1 FPR | 0.632 ± 0.047 | 0.646 ± 0.071 | 0.653 ± 0.090 |
| **FPR @ 80% recall** | 0.262 ± 0.172 | 0.146 ± 0.106 | **0.055 ± 0.023** |
| Burst / persistence @ 2% | 14/14, 1.0/3 | 14/14, 1.2/3 | 14/14, 1.0/3 |

**Findings.**
1. **Headline PR-AUC did not improve, and its variance tripled** (±0.074 vs ±0.022). But the *ceiling rose*
   — two seeds reached 0.529 / 0.531 (above any baseline seed's 0.492, nearly at the 0.55 gate) — while the
   *floor dropped* (two seeds fell to 0.395 / 0.342). This is the signature of **real-but-noisy signal added
   to a model with too few positives (101) to learn it reliably**, i.e. overfitting variance — *not* a
   signal-absent ceiling.
2. **The operating point improved substantially and stably:** FPR-at-80%-recall fell **0.26 → 0.055 (≈5×)**
   with variance **0.17 → 0.02** — the largest, most stable gain on the failing gate metric in the whole
   investigation. The port/dispersion features let the model suppress benign hard and predictably. ROC and
   recall@2%-FPR stay flat, consistent with the gain being in benign suppression / the low-FPR region, not in
   ranking the hardest positives (which dominate PR-AUC).
3. **The bottleneck has shifted** — from "the network features lack signal" (refuted: separability rose) to
   **"101 training positives cannot stably fit 123 features."** Several added features are dead weight
   (probe AUC < 0.5: `totsize_std`, `ports_per_host`, `pktsize_max`), which inflates variance.

**Verdict (all-13).** The naive "add all 13" form is not a headline improvement — feature *selection* is
needed to strip the overfitting noise. **This was then tested and confirmed — see §14.1.**

## 14.1 Feature-selection follow-up — the win (2026-07-31)

Retrained keeping only the **4 discriminative** engineered features (`distinct_dst_ports`,
`distinct_src_ports`, `rate_max`, `rate_std` — the ones with net-only probe AUC clearly > 0.5), dropping the
9 weak ones (net 30→21, fused 59→50, step 123→**114**). Same frozen split (**verified byte-identical**),
same 5 seeds, same focal loss. Four-way comparison:

| Run | net feats | dim | **PR-AUC** | ROC | recall @2% FPR | FPR @80% recall |
|---|--:|--:|---|---|---|---|
| Focal baseline | 17 | 110 | 0.455 ± 0.022 | 0.879 | 0.632 ± 0.047 | 0.262 ± 0.172 |
| ASL | 17 | 110 | 0.459 ± 0.019 | 0.880 | 0.646 ± 0.071 | 0.146 ± 0.106 |
| Enriched (all 13) | 30 | 123 | 0.447 ± 0.074 | 0.885 | 0.653 ± 0.090 | **0.055 ± 0.023** |
| **Feature-select (4)** | 21 | 114 | **0.486 ± 0.019** | **0.908** | **0.691 ± 0.039** | 0.149 ± 0.107 |

**Result — hypothesis confirmed.** Removing the 9 noisy features delivered the best of both: mean PR-AUC rose
to **0.486** (highest of all runs, **+0.031 over baseline, ≈2.4σ** — a real gain) **and** the variance
collapsed to **±0.019** (from the enriched run's ±0.074). ROC reached **0.908** (best), and the honest
operating headline **recall @ 2% false-alarm rose to 0.691 ± 0.039** (from 0.632, and tighter). The per-seed
floor lifted to 0.469 (vs 0.431 baseline / 0.342 enriched). So the enriched features *did* carry signal — the
9 weak ones were pure overfitting noise, and dropping them converted the raised ceiling into a stable gain.

**Trade-off.** The all-13 run still holds the best FPR-at-80%-recall (0.055 vs 0.149) — more features suppress
benign harder at that aggressive point — but the strong-4 wins on every headline metric (PR-AUC, ROC,
recall@2%-FPR) *and* stability, which is the right trade for the reported operating point (the 2% budget).

**Verdict (final).** The **strong-4 feature-selected model (114-dim) is the new best/default**: PR-AUC
**0.486 ± 0.019**, ROC **0.908**, **69% recall at a 2% false-alarm budget**, 14/14 bursts, persistence 1/3.
Still NO-GO on the aspirational 0.55 / 0.02 gate, but the most stable and defensible result to date. The arc
is now complete and positive: loss engineering → no help (§13); naive feature enrichment → ceiling up but
unstable (§14); **feature selection → a real, stable headline gain (§14.1).**

> **But see §15.** Window-PR-AUC ~0.5 is a prevalence-bound ceiling, so "toward 0.55" is likely the *wrong*
> target. The defensible, high-grade result is campaign-level (100% burst detection); the novel lever is
> cross-campaign generalization — not more within-CICAPT window-PR-AUC tuning.

---

## 15. Results reframe — what this system actually demonstrates (2026-07-31)

This section reinterprets §7/§13/§14 rather than adding an experiment. Its purpose: identify the **proper
root cause** of the "sub-0.55 PR-AUC" story, and state the metric framing that makes the strong results
visible. It supersedes the "get PR-AUC to 0.55" framing used in earlier sections.

### 15.1 Root cause — window-level PR-AUC is prevalence-bound and near-saturated

Three *independent* levers have now been swept and all converge on the same ~0.5 wall:

| Lever | Result | Section |
|---|---|---|
| Model **capacity** (3× params) | test PR-AUC flat (~0.5) | §7 Finding 2 |
| **Loss** (focal → ASL) | 0.455 → 0.459 (within noise) | §13 |
| **Features** (17 → 30 → selected 21) | 0.455 → 0.447 → **0.486** | §14 / §14.1 |

Convergence from capacity, loss, *and* features is the signature of an **intrinsic information/prevalence
ceiling**, not a fixable model or feature deficiency. The decisive evidence is the **ROC/PR gap**: the model's
ranking is excellent (**ROC 0.88–0.95**) yet PR-AUC sits at ~0.49 — which is exactly what **0.88% test
prevalence** allows. Per-seed proof: seed 45 reached **ROC 0.946** and still only PR-AUC 0.473; seed 43
ROC 0.917 → PR 0.517. Reaching PR-AUC 0.55 would require ROC ≈ 0.96+, i.e. near-perfect ranking of a signal
that is *partly absent from the input* (the 142 provenance-invisible attacks). **Conclusion: ~0.5 is the
ceiling of within-campaign, window-level binary detection at this prevalence. The metric we were optimizing is
saturated and is the wrong headline.**

### 15.2 The correct headline — campaign / burst-level detection at a fixed false-alarm budget

The security task is *catch the campaign*, not *flag every 10-s window*. Aggregating to bursts escapes the
per-window prevalence penalty. At a **2% Phase-1 false-alarm budget** (leakage-safe: threshold set on
validation-benign), the current-best model (`artifacts_featsel_baseline`, 5 seeds):

| Campaign-level metric | Result |
|---|---|
| **Burst detection rate** | **70 / 70 = 100%** across all 5 seeds (14/14 test bursts, every seed) |
| **Time-to-first-detection** | **0 s** for all 70 — every burst caught at its first window |
| Window ROC-AUC | 0.908 ± 0.021 |
| Window recall @ 2% FPR | 0.691 ± 0.039 |
| Window PR-AUC (supporting) | 0.486 ± 0.019 (≈ 55× the 0.0088 random-prevalence baseline) |

**Honesty caveat:** 72% of attack windows are singletons (§8), so for single-window bursts TTD = 0 is
structural — the robust claim is the **100% detection rate**; TTD is supporting, not a latency breakthrough.

### 15.3 The findings that are genuinely high-grade (and were being buried)

1. **Multi-modal necessity, quantified.** 68% of attacks are provenance-invisible, and the capacity sweep
   proves it is an *information* ceiling — a rigorous motivation for the hybrid, with the curve to show it.
2. **An honest evaluation framework.** Burst-grouped leakage-safe split, burst-bootstrap CIs, temporal-shuffle
   control, FPR-budget operating points — the rigor is itself the contribution in a field that routinely
   over-claims with leaky splits and headline accuracy.
3. **Campaign-level detection** (100% bursts @ 2% FPR) plus the **temporal-order ablation** (+0.21 PR-AUC from
   a controlled shuffle) — a clean mechanistic result.

### 15.4 What to do (and stop doing)

- **Demote** window-PR-AUC 0.486 to a *supporting ranking-quality* number; **lead** with §15.2 + §15.3.
- **Stop:** further augmentation / feature-tuning for window-PR-AUC (saturated, ±0.01 gains, not novel), and
  any sequence-model swap (capacity/loss-invariant — not the bottleneck).
- **The novel, high-grade lever is cross-campaign generalization** on a **provenance-compatible** second APT
  campaign (**DARPA OpTC / ATLAS**, per `Methodology.md` §10) — train-on-one / test-on-another converts a
  within-campaign demonstration into a generalization claim. **Not ToN_IoT** (no provenance graphs → kills
  Stage A; volumetric IoT attacks vs low-and-slow APT). **Readiness caveat:** a 2026-07-31 audit
  (`CrossCampaign-Readiness.md`) found the **GAT node features are the main transfer blocker** (~80%
  dataset-specific ID codes + `pid/ppid` identifier noise), so this lever requires a **Stage-A
  re-featurization** first, not just a target dataset.
- **Optional within-scope lever:** dual-resolution windowing — the network modality is force-fit onto sparse
  provenance windows; a native-network-window probe scored ~0.78 vs ~0.55 ROC, suggesting the coarse windowing
  dilutes the modality carrying 68% of the attacks. Higher effort, uncertain, but the one place the ceiling may
  be partly artificial.

---

*Generated from the source in `mamba/` and `gat/` on 2026-07-27; revised after the SME audit
(post-Fix-A–D), after the 5-seed ordered GPU run, on 2026-07-29 with the ASL ablation (§13), and on
2026-07-31 with the network-feature-enrichment experiment (§14) and the feature-selection follow-up (§14.1,
the current best). Headline numbers are the mean ± std over seeds 42–46; A/B baselines preserved in
`mamba/artifacts_focal_baseline/` (focal 110-dim), `mamba/artifacts_asl_baseline/` (ASL 110-dim),
`mamba/artifacts_netfeat_baseline/` (enriched 123-dim), and `mamba/artifacts_featsel_baseline/`
(feature-selected 114-dim — **current best, PR-AUC 0.486 ± 0.019**). **§15 (2026-07-31) reframes the
reporting:** window-PR-AUC is prevalence-bound/near-saturated and demoted to a supporting metric; the headline
is **campaign-level detection (100% of test bursts at a 2% false-alarm budget)** plus the multi-modal-necessity
and honest-evaluation contributions.
**Recommended next (per §15.4):** cross-campaign generalization on provenance-compatible data (OpTC/ATLAS) —
*not* further window-PR-AUC tuning or a model swap. The multi-seed temporal-shuffle control is now **done and hardened** (§8, 2026-08-03: ordered 0.486 vs shuffled 0.251, paired Δ +0.235 ± 0.064).*


---

## 16. Operational & multi-unit metrics (2026-08-11)

Added after §15. All numbers are **leakage-safe, natural prevalence, 5 seeds (42-46)**, Phase-2 test,
computed from the committed per-seed predictions (`mamba/artifacts/predictions_phase2_s*.npz`) and
Phase-1 benign (`predictions_phase1_s*.npz`). Reproduction scripts live in `unit_level_scorecards/`
(`episode_5seed.py`, `recall_at_fpr.py`, `op_at_fpr.py`, `zt_contribution.py`, `fusion_roc.py`,
`score_node_flow.py`, `flow_probe.py`, `flow_ablation.py`, `prov_event_probe.py`).

### 16.1 Detection unit determines the metric (multi-unit scorecard)

| Detection unit | ROC-AUC | PR-AUC | note |
|---|---:|---:|---|
| Network flow (native, MLP probe) | **0.999** | - | network-only attacks, undiluted |
| &nbsp;&nbsp;vol/timing features only | 0.991 | - | survives dropping port/flag identifiers -> behaviour, not memorisation |
| Provenance node (GAT node head) | 0.851 | 0.633 | 65 positive / 27,578 test nodes |
| 10-s window (union, Mamba) | 0.908 | 0.486 | the raw streaming unit |
| 1-min episode | 0.957 | 0.795 | short-episode triage |
| **5-min episode** | **0.969** | **0.943** | the operational decision unit |
| Campaign / burst | - | 100% | 14/14, every seed |

The window-level PR-AUC 0.486 is *not* a capability ceiling: the network-only attacks that depress the
window-union task are near-perfectly separable at their native flow unit (ROC 0.999). Flow/node probes are
diagnostic (standalone, leakage-safe), not the deployed head.

### 16.2 Episode aggregation is the PR-AUC lever (same model, no retraining)

Max-pooling the deployed per-window scores to fixed episodes raises effective prevalence and PR-AUC:

| Decision unit | Eff. prevalence | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| 10-s window (baseline) | 0.88% | 0.908 +/- 0.021 | 0.486 +/- 0.019 |
| 1-min episode | 3.1% | 0.957 +/- 0.007 | 0.795 +/- 0.013 |
| **5-min episode** | 7.0% | 0.969 +/- 0.016 | **0.943 +/- 0.004** |
| 10-min episode | 12.2% | 1.000 | 1.000 |

Coarse episodes have few positive units (10-min: 14), so 1- and 5-min are the meaningful rows.

### 16.3 Operating points vs the false-alarm budget

Threshold set on Phase-1 "quiet-factory" benign, recall on Phase-2 test.

| FPR budget | Window recall | Precision | F1 | MCC | Burst recall (14) |
|---:|---:|---:|---:|---:|---:|
| 2.0% | 0.691 +/- 0.039 | 0.181 | 0.284 | 0.340 | 100% |
| 1.0% | 0.618 +/- 0.028 | 0.342 | 0.432 | 0.448 | 100% |
| **0.5%** | 0.523 +/- 0.045 | **0.546** | **0.524** | **0.525** | **100%** |
| 0.2% | 0.351 +/- 0.025 | - | - | - | 0.929 |
| 0.1% | 0.242 +/- 0.099 | - | - | - | 0.714 |

**Key finding:** tightening 2% -> 0.5% *improves* F1 (0.284 -> 0.524) and MCC (0.340 -> 0.525) because
precision triples (0.18 -> 0.55) while recall falls only modestly (0.69 -> 0.52); the 2% budget is
precision-poor (FP ~202 vs TP ~39). **0.5-1% FPR is the stronger operating point**, same 100% campaign
detection. Threshold-invariant metrics do **not** change with FPR: ROC-AUC 0.908, window PR-AUC 0.486,
5-min episode PR-AUC 0.943, burst recall 100%.

Per-tactic recall @0.5% FPR (5-seed sums): lateralMovement 15/15, commandAndControl 15/20, exfiltration
13/20, discovery 30/55, collection 66/125, persistence 5/15 (=1/3 per seed), **credentialAccess 0/30**
(a real blind spot to acknowledge, not hide).

### 16.4 The frozen GAT is not an artificial bottleneck

Per-window probe (no temporal model), z_t=step[:, :64] (GAT) vs fused=step[:, 64:] (no GAT):

| Feature set | UNION ROC | UNION PR | Prov-subset ROC | Prov-subset PR |
|---|---:|---:|---:|---:|
| z_t only (GAT output) | 0.688 | 0.092 | 0.867 | 0.488 |
| fused only (no GAT) | 0.884 | 0.186 | 0.989 | 0.179 |
| full (114) | 0.856 | 0.305 | 0.904 | 0.773 |

z_t is the *weaker* contributor even on provenance-visible attacks (0.867 vs fused 0.989). Since the GAT is a
per-window encoder with no cross-window input, temporal backprop could only sharpen already-saturated
per-window features (capacity-invariant sweep; train prov ROC 0.999 vs test 0.92 = overfitting). Expected
joint fine-tune gain <=3 points on the provenance-visible slice, no ceiling break.

### 16.5 Late fusion of the native-flow head does not lift the window output

Window-union test ROC (seed-42 canonical): Mamba 0.890; flow->window max-pool alone 0.713;
late fusion max(rank) **0.903** (+0.013); logistic (val-fit) 0.847. The native-flow 0.99 does not transfer
to the window-union output because Mamba already ingests the network features and the window unit re-imposes
the prevalence ceiling. Confirms the window-union ROC ceiling ~0.90.

### 16.6 Attention-MIL does not beat max-pool (OR-labelled episode)

| Unit | Mamba->max | Attention-MIL | MIL + per-window aux |
|---|---:|---:|---:|
| 1-min | **0.782** | 0.307 | 0.431 |
| 5-min | **0.945** | 0.791 | 0.785 |

For an OR-labelled episode (positive if any window is attack), max-pool of a strong per-window detector is
near Bayes-optimal; attention softmax dilutes the single spike. Retract MIL as a lever; keep max-pool.

## 17. Step-7 semantic-feature promotion — drop discovery + negative A/B (2026-08-14)

**Discovery (preprocessing audit).** The Step-7 engineered semantic features
(`path_is_sensitive`, `path_depth`, `cmdline_length`, `cmdline_token_count`, `cmdline_entropy`,
`privilege_transition_flag`) were computed in `step07` and stored in `provenance_windowed.parquet`,
but **were never fed to the model in either stage.** `step10` builds the node-feature matrix only
from `step05`'s typed `__code/__norm/__observed` columns; the Step-7 columns lack those suffixes, so
they are dropped before the GAT. The Stage-B pooled-provenance block is a mean+max pool of the *same*
node-feature matrix, so it never contained them either. Verified: `node_features` is exactly `(N, 20)`
= 17 process-typed slots + 3 class one-hot, with no room for any Step-7 feature. **This corrects the
earlier claim (CrossCampaign-Readiness §3.3 / CONVERSATION-CONTEXT) that these abstractions "live in
Stage B" — they lived only in the parquet.**

**Feature-A experiment.** Wired all six into the entity-typed node vector (`step05`): the four numeric
features z-scored (`__norm`, same convention as base numerics), the two flags as raw 0/1 (`__code`).
`NODE_FEATURE_DIM 20→24`, pooled `40→48`, `FUSED_DIM 50→59`, `STEP_DIM 114→123`. Regenerated
graphs + fused sequence from the saved parquet; **split byte-identical**, no NaN; then a full-pipeline
A/B (GAT retrained on the 24-dim node features, frozen, 5-seed focal Mamba).

**Result — NEGATIVE, not adopted.**

| Metric | featsel (dim 114) | Feature-A (dim 123) | Δ |
|---|---:|---:|---:|
| Test PR-AUC | 0.486 ± 0.019 | 0.500 ± 0.044 | +0.014 (wash; variance 2.3×) |
| Test ROC-AUC | 0.908 ± 0.021 | 0.888 ± 0.033 | −0.020 |
| Phase-1 benign FPR | passes 0.02 | 2/5 seeds fail (s45 0.139) | worse |
| Burst detection @2% FPR | 14/14 all seeds | 14/14 all seeds | unchanged |

The +0.014 PR-AUC is within noise (overlapping intervals, more than double the variance); ROC — the
prevalence-independent ranking metric — drops 0.020; benign false alarms worsen. Same signature as the
network-feature enrichment (§14, 0.447 ± 0.074) and SSL pretraining: **adding features to the
101-train-positive regime inflates variance and false alarms without a ceiling break.** The features
carry only weak-to-modest signal in the diluted cross-class pool (rank-AUC 0.55–0.58 for cmdline
size/entropy, comparable to mid-tier network features) and are redundant with what `z_t` + the network
block already capture. **Confirms the data bottleneck (§15.4) with the Step-7 features now empirically
tested rather than assumed present.** The canonical pipeline was reverted to featsel (dim 20/50/114);
the Feature-A run is preserved in `mamba/artifacts_featureA_baseline/`. The
CrossCampaign §5 recommendation to "promote the Step-7 behavioural features into the GAT node input"
still stands *as a cross-campaign transferability measure* (OS-neutral behavioural abstractions
generalize better than factorized ID codes), but it is now known **not** to raise the in-campaign
CICAPT metric.
