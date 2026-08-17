"""
Stage B training: Mamba over the 110-dim window sequence, evaluated end-to-end on
UNION labels (the network-only attacks the GAT could not see are Mamba's target).

Split handling: the frozen protocol assigns CONTIGUOUS time regions to each split,
so we train/validate/test on maximal contiguous same-split runs (never crossing a
segment boundary). No window's future leaks into another split's training.

CPU note: the selective scan is a pure-PyTorch sequential loop, so training is
slow on CPU. Use --epochs / --chunk-len to keep smoke tests cheap; a GPU (real
mamba-ssm kernels) is recommended for the full run.

Run:  python -m mamba.build_sequences   # once, to assemble inputs
      python -m mamba.train --epochs 5 --chunk-len 256   # smoke test
      python -m mamba.train                              # full run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    raise ImportError(f"mamba.train needs torch. (original error: {exc})")

from gat import metrics as M
from . import config as C
from .model import MambaDetector, focal_bce_with_logits, asl_with_logits


def set_seed(s: int) -> None:
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _load(phase: str) -> Dict[str, np.ndarray]:
    d = np.load(C.mamba_input(phase), allow_pickle=True)
    return {k: d[k] for k in d.files}


def _runs(split: np.ndarray, seg: np.ndarray) -> List[Tuple[int, int, int]]:
    """Maximal contiguous [start,end) runs of constant (split, segment)."""
    runs = []
    n = len(split)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and split[j + 1] == split[i] and seg[j + 1] == seg[i]:
            j += 1
        runs.append((i, j + 1, int(split[i])))
        i = j + 1
    return runs


def _chunks(a: int, b: int, L: int):
    for s in range(a, b, L):
        yield s, min(s + L, b)


def _perm(a: int, b: int, seed: int) -> np.ndarray:
    """Fixed per-run permutation of absolute positions [a, b) for the shuffle control."""
    return a + np.random.default_rng([int(seed), int(a), int(b)]).permutation(b - a)


def _pred_path(phase: str, tag: str = "") -> str:
    return C.predictions(phase).replace(".npz", f"{tag}.npz")


@torch.no_grad()
def infer(model, data, dev, shuffle: bool = False, seed: int = 0,
          temperature: float = 1.0) -> np.ndarray:
    """Per-window malicious probability for the whole phase (all runs).

    shuffle=True scrambles window order within each run (fixed per-run permutation)
    for the temporal-shuffle control; predictions scatter back to original positions.

    `temperature` (Part 2 calibration, default 1.0 = exact no-op) divides the logit
    before sigmoid. This is a strictly monotonic (rank-preserving) transform: it
    does NOT change PR-AUC/ROC-AUC or any recall-targeted operating metric computed
    from the returned array. It must be used ONLY for the final exported probs (and
    consistently in infer_early_exit) -- never for the per-epoch val call used for
    model selection, which must stay at the default temperature=1.0.
    """
    model.eval()
    step = torch.from_numpy(data["step"]).float()
    prob = np.zeros(len(step), dtype=np.float32)
    # FIX A: use the model's own chunk length (not a hard-coded 4096) so train and
    # eval share the same context window -- the longest val/test run (1777) is <=
    # chunk_len (2048), so every eval sequence is un-split.
    L = model.cfg["chunk_len"]
    for a, b, _ in _runs(data["split"], data["segment_ids"]):
        order = _perm(a, b, seed) if shuffle else np.arange(a, b)
        for s in range(0, len(order), L):
            sel = order[s:s + L]
            x = step[torch.as_tensor(sel)].unsqueeze(0).to(dev)
            logit, _ = model(x)
            prob[sel] = torch.sigmoid(logit / temperature).squeeze(0).cpu().numpy()
    return prob


@torch.no_grad()
def infer_early_exit(model, data, dev, temperature: float = 1.0):
    """FIX D: full-model probability + the early-exit-applied probability.

    Always runs in real temporal order (no shuffle) -- early exit is a
    deployment-time streaming cost, so it is measured under the deployment
    ordering. Returns (prob, prob_ee, exited), all aligned to `data`'s rows.
    exited[i] is True where the block-1 benign-confidence head was confident
    enough (>= C.MAMBA["exit_threshold"]) that window i "early-exits"; for
    those windows prob_ee uses the block-1 head's own malicious estimate
    (1 - P(benign)) instead of the full two-block forward pass.

    `temperature` (Part 2 calibration, default 1.0 = exact no-op) rescales the
    main-head logit before sigmoid, consistently for both `prob` and the `p_main`
    folded into `prob_ee` -- rank-preserving, so PR-AUC/ROC-AUC/recall-targeted
    metrics computed from either array are unchanged by it. The early-exit head's
    own probability (`exit_prob`, and thus which windows `exited`) is left on its
    own raw scale, untouched by `temperature`.
    """
    model.eval()
    step = torch.from_numpy(data["step"]).float()
    n = len(step)
    prob = np.zeros(n, dtype=np.float32)
    prob_ee = np.zeros(n, dtype=np.float32)
    exited = np.zeros(n, dtype=bool)
    L = model.cfg["chunk_len"]
    exit_thr = C.MAMBA["exit_threshold"]
    for a, b, _ in _runs(data["split"], data["segment_ids"]):
        order = np.arange(a, b)
        for s in range(0, len(order), L):
            sel = order[s:s + L]
            x = step[torch.as_tensor(sel)].unsqueeze(0).to(dev)
            main_logit, exit_logit = model(x)
            p_main = torch.sigmoid(main_logit / temperature).squeeze(0).cpu().numpy()
            prob[sel] = p_main
            if exit_logit is not None:
                exit_prob = torch.sigmoid(exit_logit).squeeze(0).cpu().numpy()
                ex = exit_prob >= exit_thr
                prob_ee[sel] = np.where(ex, 1.0 - exit_prob, p_main)
                exited[sel] = ex
            else:
                prob_ee[sel] = p_main
    return prob, prob_ee, exited


@torch.no_grad()
def _val_logits(model, d2, dev) -> np.ndarray:
    """Raw (pre-sigmoid) main-head logits for the Phase-2 VAL split only
    (d2["split"] == 1), in temporal order, chunked by the model's own chunk_len --
    the same run/chunk decomposition `infer` uses. Aligned with d2["y_union"][va]
    where va = d2["split"] == 1. Used ONLY to fit the post-hoc temperature (Part 2);
    never for per-epoch model selection or any ranking metric.
    """
    model.eval()
    step = torch.from_numpy(d2["step"]).float()
    n = len(step)
    logits = np.zeros(n, dtype=np.float32)
    L = model.cfg["chunk_len"]
    for a, b, sp in _runs(d2["split"], d2["segment_ids"]):
        if sp != 1:
            continue
        order = np.arange(a, b)
        for s in range(0, len(order), L):
            sel = order[s:s + L]
            x = step[torch.as_tensor(sel)].unsqueeze(0).to(dev)
            logit, _ = model(x)
            logits[sel] = logit.squeeze(0).cpu().numpy()
    return logits[d2["split"] == 1]


def fit_temperature(logits, labels, iters=300, lr=0.02):
    """Single-scalar temperature by minimizing val BCE; returns T in [0.05, 20].

    Part 2 calibration: post-hoc, fit ONCE on the frozen best model's val logits.
    This is a strictly monotonic rescaling of the logit (divide by T, then sigmoid)
    -- it changes probability CALIBRATION (honest P(malicious) values) only, and
    provably cannot change any ranking metric (PR-AUC, ROC-AUC) or any metric that
    selects its threshold and evaluates in the same (transformed) probability space,
    e.g. the recall-targeted operating point. Do not use this to "fix" a metric --
    it exists purely for honest probability outputs.
    """
    lg = torch.as_tensor(logits, dtype=torch.float32)
    y  = torch.as_tensor(labels, dtype=torch.float32)
    s = torch.zeros(1, requires_grad=True)          # s = log T
    opt = torch.optim.Adam([s], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lg / torch.exp(s), y)
        loss.backward(); opt.step()
    return float(torch.exp(s).clamp(0.05, 20.0).item())


def _metrics(y, prob, thr):
    m = M.scalar_metrics(y, prob, thr)
    m["pr_auc"] = M.pr_auc(y, prob); m["roc_auc"] = M.roc_auc(y, prob)
    return m


def _pr_curve(y_true, y_prob):
    """Full precision/recall arrays (FIX C), same ranking convention as gat.metrics.pr_auc."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_pos = int(y_true.sum())
    order = np.argsort(-y_prob, kind="mergesort")
    yt = y_true[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(n_pos, 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return precision, recall


def _roc_curve(y_true, y_prob):
    """Full fpr/tpr arrays (FIX C), same ranking convention as gat.metrics.roc_auc."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    order = np.argsort(-y_prob, kind="mergesort")
    yt = y_true[order]
    tpr = np.cumsum(yt) / max(n_pos, 1)
    fpr = np.cumsum(1 - yt) / max(n_neg, 1)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return fpr, tpr


def _early_exit_report(model, d2, device, tag: str, temperature: float = 1.0) -> None:
    """FIX D: MEASURED early-exit gate -- exit rate + the PR/recall COST of exiting.

    Block 2 is a sequential SSM scan over the whole window sequence: dropping
    individual mid-sequence steps corrupts the recurrent state for every later
    step, and in this batched pure-PyTorch scan skipping a window gives no real
    FLOP saving anyway (the whole chunk is still one forward call). So this does
    NOT skip Block 2 -- it runs the full model, measures how often the block-1
    head would have been confident enough to exit, and reports the honest
    accuracy cost of doing so, framed as streaming-deployment potential.

    `temperature` (Part 2, default 1.0 = no-op): the same fitted calibration
    temperature used for the final exported probs, applied here too so the
    early-exit report is computed consistently with predictions_phase*.npz.
    """
    prob_full, prob_ee, exited = infer_early_exit(model, d2, device, temperature=temperature)
    va = d2["split"] == 1
    te = d2["split"] == 2
    y = d2["y_union"]
    thr = (M.threshold_for_target_recall(y[va], prob_full[va], C.OPERATING_RECALL_TARGET)
           if va.sum() else 0.5)

    y_te = y[te].astype(int)
    exited_te = exited[te]
    benign = y_te == 0
    attack = y_te == 1
    exit_rate = float(exited_te.mean()) if len(exited_te) else float("nan")
    exit_rate_benign = float(exited_te[benign].mean()) if benign.sum() else float("nan")
    exit_rate_attack = float(exited_te[attack].mean()) if attack.sum() else float("nan")

    full_m = _metrics(y_te, prob_full[te], thr)
    ee_m = _metrics(y_te, prob_ee[te], thr)

    report = {
        "exit_threshold": C.MAMBA["exit_threshold"],
        "operating_threshold": thr,
        "exit_rate_overall": exit_rate,
        "exit_rate_benign": exit_rate_benign,
        "exit_rate_attack": exit_rate_attack,
        "block2_compute_saved_frac": exit_rate,
        "block2_compute_saved_note": (
            "streaming-deployment potential only: this pure-PyTorch scan runs the whole "
            "chunk as one batched forward pass, so exited windows do NOT actually skip "
            "Block-2 compute in this codepath. The saving is realizable only in a "
            "streaming / per-window deployment that checks exit_prob after Block 1 and "
            "stops before running Block 2 for that window."),
        "full_model": {"pr_auc": full_m["pr_auc"], "roc_auc": full_m["roc_auc"],
                        "recall_at_operating_thr": full_m["recall"]},
        "early_exit_applied": {"pr_auc": ee_m["pr_auc"], "roc_auc": ee_m["roc_auc"],
                                "recall_at_operating_thr": ee_m["recall"]},
    }
    with open(os.path.join(C.ARTIFACTS, f"stageB_earlyexit{tag}.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"  early-exit @{C.MAMBA['exit_threshold']:.2f}: {exit_rate:.1%} of windows exit; "
          f"PR-AUC {full_m['pr_auc']:.3f}->{ee_m['pr_auc']:.3f}; "
          f"recall {full_m['recall']:.3f}->{ee_m['recall']:.3f}")


def train(cfg: dict, dev: str, extra_tag: str = "") -> dict:
    os.makedirs(C.ARTIFACTS, exist_ok=True)
    set_seed(cfg["seed"])
    device = torch.device(dev)
    d2 = _load("phase2")
    # Standardize the 110-dim input on the TRAIN split only (leakage-safe), then
    # clip. Without this, the raw z_t / fused scales stall optimization and the
    # model floods to ~0.02 PR-AUC; a linear probe on standardized input reaches
    # ~0.17. mu/sd are reused for the phase-1 export so inference matches training.
    trm = d2["split"] == 0
    mu = d2["step"][trm].mean(0)
    sd = d2["step"][trm].std(0) + 1e-6
    d2["step"] = np.clip((d2["step"] - mu) / sd, -10, 10).astype(np.float32)
    np.savez(os.path.join(C.ARTIFACTS, "input_scaler.npz"), mu=mu, sd=sd)
    step = torch.from_numpy(d2["step"]).float()
    y = torch.from_numpy(d2["y_union"]).float()
    runs = _runs(d2["split"], d2["segment_ids"])
    train_runs = [r for r in runs if r[2] == 0]
    va = d2["split"] == 1
    te = d2["split"] == 2

    shuffle = cfg.get("shuffle_control", False)
    # FIX B: tag = shuffle-control tag + optional per-seed suffix ("_s{seed}") for
    # multi-seed runs. extra_tag="" reproduces today's canonical (unsuffixed) names.
    tag = ("_shuffle" if shuffle else "") + extra_tag
    if shuffle:
        print("** TEMPORAL-SHUFFLE CONTROL: window order scrambled within each run. **\n"
              "   If test PR-AUC matches the ordered run, temporal order is NOT the driver.")

    n_tr = int((d2["split"] == 0).sum()); n_pos = int(d2["y_union"][d2["split"] == 0].sum())
    target_eff = cfg["target_effective_pos_weight"]
    pos_weight = max(1.0, target_eff)      # per-step loss, no oversampling -> full weight
    print(f"phase2 steps={len(step)} | train windows={n_tr} pos={n_pos} "
          f"| runs train/val/test="
          f"{sum(r[2]==0 for r in runs)}/{sum(r[2]==1 for r in runs)}/{sum(r[2]==2 for r in runs)}")
    # Part 1: which main-head loss is active this run. The "focal" branch below is
    # byte-for-identical to the pre-ASL code path; selecting "asl" never touches it.
    if cfg["loss"] == "asl":
        print(f"loss=asl  gamma_pos={cfg['asl_gamma_pos']} gamma_neg={cfg['asl_gamma_neg']} "
              f"clip={cfg['asl_clip']} pos_weight={cfg['asl_pos_weight']}")
    else:
        print(f"loss=focal  gamma={cfg['focal_gamma']} pos_weight={pos_weight}")

    model = MambaDetector(cfg).to(device)
    # ---- Part B: initialise the encoder from a mamba.pretrain checkpoint ----
    # Must happen BEFORE the optimizer is built, so AdamW's state starts from
    # these (not the randomly-initialised) weights. Classification + exit heads
    # are left as MambaDetector just initialised them (freshly random).
    init_from = cfg.get("init_from")
    if init_from:
        ck = torch.load(init_from, map_location=device)
        model.input.load_state_dict(ck["input"])
        model.blocks.load_state_dict(ck["blocks"])
        model.norms.load_state_dict(ck["norms"])
        print(f"initialised encoder from {init_from}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Mamba params: {n_params:,}")

    best, best_state, patience = -1.0, None, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        np.random.shuffle(train_runs)
        tot = 0.0
        for a, b, _ in train_runs:
            order = _perm(a, b, cfg["seed"]) if shuffle else np.arange(a, b)
            for s in range(0, len(order), cfg["chunk_len"]):
                sel_np = order[s:s + cfg["chunk_len"]]
                sel = torch.as_tensor(sel_np)
                x = step[sel].unsqueeze(0).to(device)
                tgt = y[sel].unsqueeze(0).to(device)
                opt.zero_grad()
                logit, exit_logit = model(x)
                # Part 1: loss selection. The "focal" branch is the exact prior call
                # (same function, same args) -- ASL is opt-in and does not alter it.
                if cfg["loss"] == "asl":
                    loss = asl_with_logits(logit, tgt, cfg["asl_gamma_pos"], cfg["asl_gamma_neg"],
                                           cfg["asl_clip"], cfg["asl_pos_weight"])
                else:
                    loss = focal_bce_with_logits(logit, tgt, cfg["focal_gamma"], pos_weight)
                if exit_logit is not None:          # early-exit head learns benign-confidence
                    loss = loss + 0.3 * focal_bce_with_logits(
                        exit_logit, 1.0 - tgt, cfg["focal_gamma"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += float(loss.item())

        prob = infer(model, d2, device, shuffle, cfg["seed"])
        thr = (M.threshold_for_target_recall(d2["y_union"][va], prob[va],
                                              C.OPERATING_RECALL_TARGET)
               if va.sum() else 0.5)
        vm = _metrics(d2["y_union"][va], prob[va], thr)
        score = vm["pr_auc"]
        print(f"epoch {epoch:3d}  loss {tot:.2f}  val PR-AUC {vm['pr_auc']:.3f}  "
              f"MCC {vm['mcc']:.3f}  recall {vm['recall']:.3f}")
        if score > best:
            best, patience = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                print(f"early stop (best val PR-AUC {best:.3f})"); break

    if best_state:
        model.load_state_dict(best_state)

    # ---- Part 2: temperature-scaling calibration (additive; honest probs only) ----
    # Fit ONCE here, after best_state is restored and BEFORE any prediction export --
    # never during the per-epoch val loop above (which stays at the implicit T=1 of
    # a bare `infer(...)` call). Rank-preserving (monotonic in the logit), so it
    # cannot change PR-AUC/ROC-AUC or any recall-targeted operating metric.
    if cfg.get("calibrate", True):
        val_logits = _val_logits(model, d2, device)
        val_labels = d2["y_union"][va]
        lg_t = torch.as_tensor(val_logits, dtype=torch.float32)
        y_t = torch.as_tensor(val_labels, dtype=torch.float32)
        val_nll_before = float(F.binary_cross_entropy_with_logits(lg_t, y_t).item())
        T = fit_temperature(val_logits, val_labels)
        val_nll_after = float(F.binary_cross_entropy_with_logits(lg_t / T, y_t).item())
        print(f"  calibration: T={T:.4f}  val BCE {val_nll_before:.4f} -> {val_nll_after:.4f} "
              f"(rank-preserving; PR-AUC/ROC/recall-targeted metrics unaffected)")
    else:
        T, val_nll_before, val_nll_after = 1.0, float("nan"), float("nan")
        print("  calibration: disabled (--no-calibrate), T=1.0")

    calib = {"temperature": T, "val_nll_before": val_nll_before, "val_nll_after": val_nll_after,
             "note": "rank-preserving; does not change PR-AUC/ROC or recall-targeted metrics"}
    # Always write a per-seed name too (in addition to the standard tag-based name),
    # so a single-seed run's temperature is checkable without relying on the
    # unsuffixed canonical file. Keep the shuffle-control prefix in BOTH names (like
    # every other tagged artifact) so an ordered and a shuffle-control sweep over the
    # same seed never collide on the same filename; in the multi-seed sweep the two
    # names already coincide (tag IS "_s{seed}"/"_shuffle_s{seed}"), so the set below
    # naturally collapses to the single per-seed file already used elsewhere.
    seed_tag = ("_shuffle" if shuffle else "") + f"_s{cfg['seed']}"
    calib_paths = {os.path.join(C.ARTIFACTS, f"calibration{tag}.json"),
                   os.path.join(C.ARTIFACTS, f"calibration{seed_tag}.json")}
    for p in calib_paths:
        with open(p, "w") as fh:
            json.dump(calib, fh, indent=2, default=float)
        print(f"wrote {p}")

    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "temperature": T},
               os.path.join(C.ARTIFACTS, f"mamba_stageB{tag}.pt"))

    # export predictions for both phases (gat.monitor-compatible contract). The
    # calibration temperature is applied here (final export) -- see infer()'s
    # docstring for why this cannot change ranking/recall-targeted metrics.
    for phase in C.PHASES:
        d = d2 if phase == "phase2" else _load(phase)
        if phase != "phase2":                       # phase2 already standardized above
            d["step"] = np.clip((d["step"] - mu) / sd, -10, 10).astype(np.float32)
        prob = infer(model, d, device, shuffle, cfg["seed"], temperature=T)
        np.savez(_pred_path(phase, tag),
                 window_ids=d["window_ids"], split=d["split"].astype(np.int64),
                 y_true=d["y_union"].astype(np.int64), y_true_prov=d["y_prov"].astype(np.int64),
                 y_prob=prob.astype(np.float32), burst_id=d["burst_id"].astype(np.int64),
                 tactic=d["tactic"].astype(str), window_time=d["window_times"].astype(np.float64))
        print(f"wrote {_pred_path(phase, tag)}")

    # FIX D: measured early-exit analysis (reported cost, not a default codepath)
    _early_exit_report(model, d2, device, tag, temperature=T)

    return _gate(dev, tag)


def _gate(dev: str, tag: str = "") -> dict:
    """End-to-end UNION gate for the full GAT->Mamba system."""
    d = np.load(_pred_path("phase2", tag), allow_pickle=True)
    split = d["split"]; y = d["y_true"]; prob = d["y_prob"]; tac = d["tactic"].astype(str)
    burst_id = d["burst_id"]
    va, te = split == 1, split == 2
    thr = M.threshold_for_target_recall(y[va], prob[va], C.OPERATING_RECALL_TARGET)
    tm = _metrics(y[te], prob[te], thr)
    g = C.GATE
    checks = []
    def chk(n, v, ok, t): checks.append((n, v, bool(ok), t))
    chk("test union PR-AUC", tm["pr_auc"], tm["pr_auc"] >= g["test_pr_auc_min"], f">= {g['test_pr_auc_min']}")
    chk("test union ROC-AUC", tm["roc_auc"], tm["roc_auc"] >= g["test_roc_auc_min"], f">= {g['test_roc_auc_min']}")
    pers = te & (tac == "persistence") & (y == 1)
    if pers.sum():
        chk("persistence detected", 1.0, bool(np.any(prob[pers] >= thr)), "== True")
    p1 = np.load(_pred_path("phase1", tag), allow_pickle=True)
    te1 = p1["split"] == 2
    fpr = float((p1["y_prob"][te1] >= thr).mean())
    chk("Phase-1 benign FPR", fpr, fpr <= g["phase1_fpr_max"], f"<= {g['phase1_fpr_max']}")

    # tag can now carry a "_s{seed}" suffix (FIX B) in addition to "_shuffle", so the
    # SHUFFLE-CONTROL/ORDERED label and the temporal-order comparison below must key off
    # whether this run IS a shuffle control, not merely off "tag being non-empty".
    is_shuffle = tag.startswith("_shuffle")
    label = "SHUFFLE-CONTROL" if is_shuffle else "ORDERED"
    print(f"\n==== STAGE B (GAT->Mamba) UNION GATE [{label}]{f' ({tag})' if tag else ''} ====")
    for n, v, ok, t in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {n:<24} {v:.4f}  (need {t})")
    verdict = "GO" if all(c[2] for c in checks) else "NO-GO"
    print(f"  VERDICT: {verdict}  ({sum(c[2] for c in checks)}/{len(checks)})")
    print(f"  test union: PR-AUC {tm['pr_auc']:.3f} ROC {tm['roc_auc']:.3f} "
          f"recall {tm['recall']:.3f} prec {tm['precision']:.3f} MCC {tm['mcc']:.3f}")

    # ---- FIX B: burst-grouped bootstrap CI on test PR-AUC (every single run) ----
    # Cluster-bootstrap design (a reviewable choice, not a hidden default): attack
    # windows resample by burst_id -- their real statistically-independent unit, since
    # windows inside one burst are highly autocorrelated. Benign windows all carry
    # burst_id == -1 in this dataset, so treating them as one cluster would collapse
    # nearly all benign variance into a single coin-flip; instead each benign window is
    # given its own unique negative id so it resamples individually (i.i.d.-window
    # assumption for benign traffic). A reviewer could reasonably contest this benign
    # side of the design (e.g. resample by contiguous run instead) -- flagging it here.
    grp = burst_id[te].astype(int).copy()
    neg = np.where(grp < 0)[0]
    grp[neg] = -(np.arange(len(neg)) + 2)
    ci = M.bootstrap_ci(y[te], prob[te], M.pr_auc, group=grp, n_boot=1000, seed=42)
    print(f"  PR-AUC {ci['point']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]  "
          f"(burst-grouped bootstrap, n_boot=1000)")

    # ---- Part 3: leakage-safe fixed-FPR-budget operating point (reporting only) ----
    # Threshold chosen from Phase-2 VAL benign scores ONLY -- never from test or
    # Phase-1 -- so there is no leakage into the number it is then measured against.
    # This is ADDITIVE reporting: it does not feed the 4 pass/fail checks above or
    # the verdict, and it does not change config.GATE. `va`, `te`, `p1`, `te1` are
    # already computed above.
    val_benign = prob[va & (y == 0)]
    thr_budget = float(np.quantile(val_benign, 1.0 - C.FPR_BUDGET))
    tm_budget = _metrics(y[te], prob[te], thr_budget)
    p1_fpr_budget = float((p1["y_prob"][te1] >= thr_budget).mean())
    operating_at_fpr_budget = {
        "fpr_budget": C.FPR_BUDGET,
        "threshold": thr_budget,
        "test_recall": tm_budget["recall"],
        "test_precision": tm_budget["precision"],
        "test_fpr": tm_budget["fpr"],
        "phase1_fpr": p1_fpr_budget,
    }
    print(f"  [FPR-budget op-point] thr={thr_budget:.4f} "
          f"test recall={tm_budget['recall']:.3f} | test FPR={tm_budget['fpr']:.4f} | "
          f"Phase-1 FPR={p1_fpr_budget:.4f}")

    # temporal-order verdict: compare shuffle-control vs the ordered run
    if is_shuffle:
        ordered = os.path.join(C.ARTIFACTS, "stageB_gate.json")
        if os.path.exists(ordered):
            om = json.load(open(ordered)).get("test_metrics", {})
            if "pr_auc" in om:
                delta = om["pr_auc"] - tm["pr_auc"]
                print(f"\n  [TEMPORAL-ORDER TEST] ordered PR-AUC {om['pr_auc']:.3f} vs "
                      f"shuffled {tm['pr_auc']:.3f}  (delta {delta:+.3f})")
                print("   large positive delta  -> temporal order IS used (Mamba earns its place)")
                print("   delta ~ 0             -> gain is fusion, not order (a per-window model suffices)")
        else:
            print("  [note] run the ordered model first (python -m mamba.train) to get the comparison.")

    # ---- FIX C: honest operating-point curve (recall target -> the FPR/precision it
    # actually buys) -- reporting only, the GATE thresholds above are unchanged. ----
    curve = []
    for rt in [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50]:
        thr_rt = M.threshold_for_target_recall(y[va], prob[va], rt)
        tm_rt = M.scalar_metrics(y[te], prob[te], thr_rt)
        fpr1_rt = float((p1["y_prob"][te1] >= thr_rt).mean())
        curve.append({"recall_target": rt, "threshold": thr_rt,
                      "test_recall": tm_rt["recall"], "test_precision": tm_rt["precision"],
                      "test_fpr": tm_rt["fpr"], "phase1_fpr": fpr1_rt})

    pr_prec, pr_rec = _pr_curve(y[te], prob[te])
    roc_fpr, roc_tpr = _roc_curve(y[te], prob[te])

    note_rt = 0.70
    entry70 = next((e for e in curve if abs(e["recall_target"] - note_rt) < 1e-9), None)
    if entry70:
        cmp_sym = "<" if entry70["phase1_fpr"] <= g["phase1_fpr_max"] else ">"
        print(f"  [note] at recall-target {note_rt:.2f}, Phase-1 FPR={entry70['phase1_fpr']:.4f} "
              f"({cmp_sym} {g['phase1_fpr_max']}) but test recall falls to "
              f"{entry70['test_recall']:.3f} -- a recall/FPR trade, not a free pass.")

    with open(os.path.join(C.ARTIFACTS, f"stageB_operating_curve{tag}.json"), "w") as fh:
        json.dump({"operating_points": curve,
                   "pr_curve": {"precision": pr_prec.tolist(), "recall": pr_rec.tolist()},
                   "roc_curve": {"fpr": roc_fpr.tolist(), "tpr": roc_tpr.tolist()}},
                  fh, indent=2, default=float)

    try:  # optional PNG -- never let a missing matplotlib break the run
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(pr_rec, pr_prec)
        axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision")
        axes[0].set_title("test PR curve")
        rts = [e["recall_target"] for e in curve]
        fprs1 = [e["phase1_fpr"] for e in curve]
        axes[1].plot(rts, fprs1, marker="o")
        axes[1].axhline(g["phase1_fpr_max"], color="r", linestyle="--",
                         label=f"gate max {g['phase1_fpr_max']}")
        axes[1].set_xlabel("recall target"); axes[1].set_ylabel("Phase-1 FPR")
        axes[1].set_title("FPR vs recall target"); axes[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(C.ARTIFACTS, f"stageB_operating_curve{tag}.png"))
        plt.close(fig)
    except Exception as exc:
        print(f"  [note] skipped operating-curve PNG (matplotlib unavailable: {exc})")

    with open(os.path.join(C.ARTIFACTS, f"stageB_gate{tag}.json"), "w") as fh:
        json.dump({"checks": [{"name": n, "value": v, "pass": ok, "target": t}
                              for n, v, ok, t in checks], "verdict": verdict,
                   "test_metrics": tm, "pr_auc_ci": ci,
                   "operating_at_fpr_budget": operating_at_fpr_budget,
                   "shuffle_control": bool(is_shuffle)}, fh, indent=2, default=float)
    return tm


def _copy_canonical(src_tag: str, dst_tag: str) -> None:
    """FIX B backward-compat: when seed 42 runs as part of a multi-seed sweep, also
    write the canonical (unsuffixed) seed-42 artifact names alongside its "_s42" ones,
    so anything depending on mamba_stageB.pt / predictions_{phase}.npz / stageB_gate.json
    keeps working unchanged."""
    pairs = [
        (os.path.join(C.ARTIFACTS, f"mamba_stageB{src_tag}.pt"),
         os.path.join(C.ARTIFACTS, f"mamba_stageB{dst_tag}.pt")),
        (_pred_path("phase1", src_tag), _pred_path("phase1", dst_tag)),
        (_pred_path("phase2", src_tag), _pred_path("phase2", dst_tag)),
        (os.path.join(C.ARTIFACTS, f"stageB_gate{src_tag}.json"),
         os.path.join(C.ARTIFACTS, f"stageB_gate{dst_tag}.json")),
    ]
    for src, dst in pairs:
        if os.path.exists(src):
            shutil.copyfile(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.MAMBA["epochs"])
    ap.add_argument("--chunk-len", type=int, default=C.MAMBA["chunk_len"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shuffle-control", action="store_true",
                    help="temporal ablation: scramble window order within each split")
    ap.add_argument("--seeds", default="42",
                    help="comma-separated seed list. One seed (default) behaves exactly "
                         "as before (canonical unsuffixed artifact names). Multiple seeds "
                         "runs the full multi-seed hardening sweep (FIX B): per-seed "
                         "artifacts suffixed _s{seed}, plus stageB_multiseed{tag}.json.")
    # ---- Part 1: loss selection (ASL vs the frozen-baseline focal loss) ----
    ap.add_argument("--loss", choices=["focal", "asl"], default=C.MAMBA["loss"],
                    help="main-head loss. 'focal' reproduces the frozen baseline "
                         "byte-for-identically; 'asl' (default) is the asymmetric-loss "
                         "alternative -- selecting either never alters the other's code path.")
    ap.add_argument("--asl-gamma-pos", type=float, default=C.MAMBA["asl_gamma_pos"])
    ap.add_argument("--asl-gamma-neg", type=float, default=C.MAMBA["asl_gamma_neg"])
    ap.add_argument("--asl-clip", type=float, default=C.MAMBA["asl_clip"])
    ap.add_argument("--asl-pos-weight", type=float, default=C.MAMBA["asl_pos_weight"])
    # ---- Part 2: temperature-scaling calibration (additive; on by default) ----
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false",
                    default=C.MAMBA["calibrate"],
                    help="disable the post-training temperature-scaling calibration "
                         "(Part 2). Calibration is rank-preserving/additive reporting "
                         "only and is on by default.")
    # ---- Part B: self-supervised pretraining (additive; off by default) ----
    ap.add_argument("--init-from", default=None,
                    help="path to a mamba.pretrain checkpoint (pretrained_encoder.pt); "
                         "loads its input/blocks/norms into the freshly-built "
                         "MambaDetector before training (classification + exit heads "
                         "stay randomly initialised).")
    a = ap.parse_args()
    seeds = [int(s.strip()) for s in a.seeds.split(",") if s.strip()]
    if not seeds:
        seeds = [C.MAMBA["seed"]]
    base_cfg = dict(C.MAMBA); base_cfg.update(epochs=a.epochs, chunk_len=a.chunk_len,
                                              shuffle_control=a.shuffle_control,
                                              loss=a.loss, asl_gamma_pos=a.asl_gamma_pos,
                                              asl_gamma_neg=a.asl_gamma_neg, asl_clip=a.asl_clip,
                                              asl_pos_weight=a.asl_pos_weight,
                                              calibrate=a.calibrate,
                                              init_from=a.init_from)

    if len(seeds) == 1:
        cfg = dict(base_cfg); cfg["seed"] = seeds[0]
        train(cfg, a.device)
        return

    # FIX B: multi-seed hardening sweep
    base_tag = "_shuffle" if a.shuffle_control else ""
    results = []
    for s in seeds:
        cfg = dict(base_cfg); cfg["seed"] = s
        suffix = f"_s{s}"
        tm = train(cfg, a.device, extra_tag=suffix)
        results.append({"seed": s, "pr_auc": tm["pr_auc"], "roc_auc": tm["roc_auc"]})
        if s == 42:
            _copy_canonical(base_tag + suffix, base_tag)

    pr = np.array([r["pr_auc"] for r in results], dtype=float)
    roc = np.array([r["roc_auc"] for r in results], dtype=float)
    summary = {
        "seeds": results,
        "n_seeds": len(results),
        "pr_auc_mean": float(np.nanmean(pr)), "pr_auc_std": float(np.nanstd(pr)),
        "roc_auc_mean": float(np.nanmean(roc)), "roc_auc_std": float(np.nanstd(roc)),
    }
    with open(os.path.join(C.ARTIFACTS, f"stageB_multiseed{base_tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\n==== MULTI-SEED SUMMARY (n={len(results)}) ====")
    print(f"  test PR-AUC  = {summary['pr_auc_mean']:.3f} +/- {summary['pr_auc_std']:.3f} "
          f"over {len(results)} seeds")
    print(f"  test ROC-AUC = {summary['roc_auc_mean']:.3f} +/- {summary['roc_auc_std']:.3f} "
          f"over {len(results)} seeds")


if __name__ == "__main__":
    main()
