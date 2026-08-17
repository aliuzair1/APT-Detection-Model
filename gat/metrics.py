"""
Pure-numpy detection metrics for the imbalanced, window-level GAT baseline.

No sklearn / scipy dependency on purpose: the monitor must run today (before
torch or sklearn are installed) and inside a resource-constrained edge context.
Every function takes plain numpy arrays and returns plain Python floats/dicts so
the output serialises straight to JSON.

Conventions
-----------
* ``y_true`` : int/bool array, 1 = malicious window, 0 = benign.
* ``y_prob`` : float array in [0, 1], P(malicious).
* Threshold-free metrics (PR-AUC, ROC-AUC) are the honest headline for a
  130:1 problem; threshold metrics are always reported *at a stated operating
  point* chosen on validation, never tuned on test.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Confusion-matrix-derived scalars at a fixed threshold
# --------------------------------------------------------------------------- #
def confusion(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def scalar_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    """Precision / recall / specificity / F1 / F2 / MCC / FPR at ``thr``."""
    c = confusion(y_true, y_prob, thr)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)                # == sensitivity / TPR
    specificity = _safe_div(tn, tn + fp)
    fpr = _safe_div(fp, fp + tn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    # F-beta with beta=2 weights recall 4x precision -- the right emphasis for an IDS
    beta2 = 4.0
    f2 = _safe_div((1 + beta2) * precision * recall, beta2 * precision + recall)
    # Matthews correlation coefficient (robust single number under imbalance)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_div(tp * tn - fp * fn, denom) if denom else 0.0
    return {
        "threshold": float(thr),
        "precision": precision, "recall": recall, "specificity": specificity,
        "fpr": fpr, "f1": f1, "f2": f2, "mcc": mcc,
        **{k: float(v) for k, v in c.items()},
    }


# --------------------------------------------------------------------------- #
# Threshold-free ranking metrics
# --------------------------------------------------------------------------- #
def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve), numpy-only.

    AP = sum_n (R_n - R_{n-1}) * P_n  over the ranked score thresholds.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        return float("nan")
    order = np.argsort(-y_prob, kind="mergesort")     # stable, high score first
    yt = y_true[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    # prepend the (recall=0, precision=1) origin
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U rank statistic (handles ties)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_prob, kind="mergesort")
    ranks = np.empty(len(y_prob), dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1)
    # average ranks over tie groups
    sp = y_prob[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    sum_pos = float(ranks[y_true == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def prevalence(y_true: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    return float(y_true.mean()) if len(y_true) else float("nan")


# --------------------------------------------------------------------------- #
# Operating-point selection (choose on validation, apply to test)
# --------------------------------------------------------------------------- #
def threshold_for_target_recall(y_true: np.ndarray, y_prob: np.ndarray,
                                target_recall: float = 0.80) -> float:
    """Highest threshold whose *validation* recall is >= target_recall.

    An IDS operates at a chosen recall and reports the resulting precision/FPR --
    a far more stable operating point than argmax-F2 on a tiny positive set
    (12 val positives here), which overfits to a degenerate high-precision,
    near-zero-recall corner. Never call on test.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    pos = y_prob[y_true == 1]
    if len(pos) == 0:
        return 0.5
    s = np.sort(pos)[::-1]                       # positive scores, high -> low
    k = int(math.ceil(target_recall * len(s)))   # need k of them above threshold
    k = min(max(k, 1), len(s))
    return float(s[k - 1])                        # recall = k/len(s) >= target


def best_threshold_by_f2(y_true: np.ndarray, y_prob: np.ndarray,
                         grid: Sequence[float] | None = None) -> float:
    """Threshold on the *validation* split that maximises F2. Never call on test."""
    y_prob = np.asarray(y_prob, dtype=float)
    if grid is None:
        # candidate thresholds = unique scores (plus a tiny epsilon band)
        grid = np.unique(np.round(y_prob, 6))
        if len(grid) > 512:
            grid = np.quantile(y_prob, np.linspace(0.0, 1.0, 512))
    best_thr, best_f2 = 0.5, -1.0
    for thr in grid:
        f2 = scalar_metrics(y_true, y_prob, float(thr))["f2"]
        if f2 > best_f2:
            best_f2, best_thr = f2, float(thr)
    return best_thr


# --------------------------------------------------------------------------- #
# Burst-level (campaign) metrics -- the persistence / low-and-slow story
# --------------------------------------------------------------------------- #
def burst_detection(y_true: np.ndarray, y_prob: np.ndarray, burst_id: np.ndarray,
                    window_time: np.ndarray, thr: float,
                    tactic: np.ndarray | None = None) -> Dict:
    """A burst is 'detected' if >=1 of its windows is flagged at ``thr``.

    Reports per-burst detection, overall detection rate, time-to-first-detection
    (seconds from the burst's first window to its first flagged window), and a
    per-tactic breakdown when tactics are supplied. benign windows carry
    burst_id == -1 and are ignored here.
    """
    burst_id = np.asarray(burst_id)
    y_prob = np.asarray(y_prob, dtype=float)
    window_time = np.asarray(window_time, dtype=float)
    flagged = y_prob >= thr

    bursts = sorted(int(b) for b in np.unique(burst_id) if b >= 0)
    per_burst = []
    detected = 0
    for b in bursts:
        m = burst_id == b
        times = window_time[m]
        fl = flagged[m]
        t0 = float(times.min())
        is_det = bool(fl.any())
        detected += int(is_det)
        ttd = float(times[fl].min() - t0) if is_det else float("nan")
        entry = {
            "burst_id": b,
            "n_windows": int(m.sum()),
            "detected": is_det,
            "time_to_first_detection_s": ttd,
        }
        if tactic is not None:
            tvals, tcnt = np.unique(np.asarray(tactic)[m], return_counts=True)
            entry["dominant_tactic"] = str(tvals[np.argmax(tcnt)])
        per_burst.append(entry)

    result = {
        "n_bursts": len(bursts),
        "n_detected": detected,
        "detection_rate": _safe_div(detected, len(bursts)),
        "per_burst": per_burst,
    }

    if tactic is not None:
        by_tactic: Dict[str, Dict] = {}
        for e in per_burst:
            t = e["dominant_tactic"]
            d = by_tactic.setdefault(t, {"n_bursts": 0, "n_detected": 0})
            d["n_bursts"] += 1
            d["n_detected"] += int(e["detected"])
        for t, d in by_tactic.items():
            d["detection_rate"] = _safe_div(d["n_detected"], d["n_bursts"])
        result["by_tactic"] = by_tactic
    return result


def per_tactic_recall(y_true: np.ndarray, y_prob: np.ndarray, tactic: np.ndarray,
                      thr: float) -> Dict[str, Dict]:
    """Window-level recall per tactic -- DESCRIPTIVE only (report raw n/N, not %).

    Only counts windows whose union label is positive; benign-labelled windows
    are excluded so recall is computed over genuine attack windows of each tactic.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    tactic = np.asarray(tactic)
    flagged = y_prob >= thr
    out: Dict[str, Dict] = {}
    for t in sorted(set(tactic[(y_true == 1)].tolist())):
        m = (tactic == t) & (y_true == 1)
        n = int(m.sum())
        hit = int(np.sum(flagged[m]))
        out[str(t)] = {"n_windows": n, "detected": hit,
                       "recall": _safe_div(hit, n)}
    return out


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, metric_fn,
                 group: np.ndarray | None = None, n_boot: int = 1000,
                 seed: int = 42, alpha: float = 0.05) -> Dict[str, float]:
    """Percentile bootstrap CI for a threshold-free metric.

    When ``group`` (e.g. burst_id) is given, resamples whole groups -- the
    statistically honest unit here is the burst, not the (autocorrelated) window.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    point = metric_fn(y_true, y_prob)
    stats = []
    if group is None:
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            v = metric_fn(y_true[idx], y_prob[idx])
            if not math.isnan(v):
                stats.append(v)
    else:
        group = np.asarray(group)
        uniq = np.unique(group)
        for _ in range(n_boot):
            chosen = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.where(group == g)[0] for g in chosen])
            v = metric_fn(y_true[idx], y_prob[idx])
            if not math.isnan(v):
                stats.append(v)
    if not stats:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan")}
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return {"point": float(point), "lo": lo, "hi": hi}
