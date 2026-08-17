"""
GAT (Stage A) readiness monitor + Phase-2 GO/NO-GO gate.

Consumes the prediction contract that ``train.py`` writes
(``artifacts/predictions_phase{1,2}.npz``) and produces:

  1. A human-readable dashboard of every statistic worth watching for a
     130:1-imbalanced window-level detector.
  2. A machine-readable ``artifacts/readiness_gate.json``.
  3. A single verdict -- GO / NO-GO for building the Mamba stage (Phase 2) --
     evaluated against the thresholds in ``config.GATE``.

Design goals
------------
* numpy-only: runs before torch/sklearn are installed, and on an edge box.
* Threshold discipline: the operating point is chosen on the *validation*
  split and then applied unchanged to test -- no test-set tuning.
* Honest about scarcity: rare-tactic recall is printed as raw n/N, never as a
  headline percentage.

Usage
-----
    # See the dashboard on synthetic-but-realistic predictions (no training yet):
    python -m gat.monitor --demo

    # Score real predictions after a training run:
    python -m gat.monitor

    # Re-run every 30 s while training writes new predictions (live monitoring):
    python -m gat.monitor --watch 30
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional

import numpy as np

from . import config as C
from . import metrics as M


# --------------------------------------------------------------------------- #
# Small terminal helpers (no external deps)
# --------------------------------------------------------------------------- #
def _c(txt: str, code: str) -> str:
    if os.environ.get("NO_COLOR"):
        return txt
    return f"\033[{code}m{txt}\033[0m"


GREEN, RED, YELLOW, BOLD, DIM = "32", "31", "33", "1", "2"


def _rule(title: str = "", width: int = 74) -> str:
    if not title:
        return "-" * width
    pad = width - len(title) - 2
    return f"-- {_c(title, BOLD)} " + "-" * max(0, pad - 3)


def _fmt(v: float, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v))):
        return "   n/a"
    return f"{v:.{nd}f}"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load(phase: str) -> Optional[Dict[str, np.ndarray]]:
    path = C.predictions_npz(phase)
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


# --------------------------------------------------------------------------- #
# Core evaluation
# --------------------------------------------------------------------------- #
def _scorecard(y: np.ndarray, prob: np.ndarray, thr: float) -> Dict:
    sm = M.scalar_metrics(y, prob, thr)
    sm["pr_auc"] = M.pr_auc(y, prob)
    sm["roc_auc"] = M.roc_auc(y, prob)
    sm["prevalence"] = M.prevalence(y)
    sm["n_windows"] = int(len(y))
    sm["n_positive"] = int(np.sum(y))
    return sm


def evaluate(p2: Dict[str, np.ndarray],
             p1: Optional[Dict[str, np.ndarray]]) -> Dict:
    split = p2["split"].astype(int)
    uni = p2["y_true"].astype(int)                                   # UNION (end-to-end)
    prov = (p2["y_true_prov"].astype(int) if "y_true_prov" in p2     # PROVENANCE (GAT's job)
            else uni)
    prob = p2["y_prob"].astype(float)
    tactic = p2["tactic"].astype(str)
    burst = p2["burst_id"].astype(int)
    wtime = p2["window_time"].astype(float)

    tr, va, te = split == 0, split == 1, split == 2
    # windows the GAT can fairly adjudicate: benign + provenance-visible attacks.
    # (net-only positives are dropped from the Tier-1 scorecard -- invisible in the graph.)
    subset = (uni == 0) | (prov == 1)

    # operating point = recall-targeted on the val PROVENANCE-subset, applied to test
    vm = va & subset
    thr = (M.threshold_for_target_recall(prov[vm], prob[vm], C.OPERATING_RECALL_TARGET)
           if vm.sum() else 0.5)

    report: Dict = {"operating_threshold": thr,
                    "prov": {"splits": {}}, "union": {"splits": {}}}

    for name, mask in [("train", tr), ("val", va), ("test", te)]:
        ps = mask & subset
        report["prov"]["splits"][name] = _scorecard(prov[ps], prob[ps], thr)   # Tier 1
        report["union"]["splits"][name] = _scorecard(uni[mask], prob[mask], thr)  # Tier 2

    # Tier-1 test PR-AUC CI (bootstrap whole bursts over the prov-subset test set)
    ptm = te & subset
    grp = burst.copy()
    grp[prov == 0] = -1
    report["prov_test_pr_auc_ci"] = M.bootstrap_ci(
        prov[ptm], prob[ptm], M.pr_auc, group=grp[ptm], n_boot=500)

    # Tier-2 burst view (union) -- descriptive campaign detection
    report["burst_test"] = M.burst_detection(
        uni[te], prob[te], burst[te], wtime[te], thr, tactic=tactic[te])
    report["per_tactic_recall_test"] = M.per_tactic_recall(
        uni[te], prob[te], tactic[te], thr)

    # persistence is provenance-visible: judge it on the prov label
    pmask = te & (tactic == C.PERSISTENCE_TACTIC) & (prov == 1)
    report["persistence_test"] = (
        {"n_windows": int(pmask.sum()),
         "n_detected": int(np.sum(prob[pmask] >= thr)),
         "detected": bool(np.any(prob[pmask] >= thr))}
        if pmask.sum() else {"n_windows": 0})

    # headroom the Mamba stage must close (net-only positives the GAT can't see)
    report["headroom"] = {
        "union_test_recall": report["union"]["splits"]["test"]["recall"],
        "prov_test_recall": report["prov"]["splits"]["test"]["recall"],
        "net_only_test_positives": int(np.sum(te & (uni == 1) & (prov == 0))),
        "prov_test_positives": int(np.sum(te & (prov == 1))),
    }

    # auxiliary node head (provenance node labels)
    if "node_y_true" in p2 and "node_y_prob" in p2 and len(p2["node_y_true"]):
        nt = p2["node_y_true"].astype(int)
        npb = p2["node_y_prob"].astype(float)
        report["node_head"] = {"n_nodes": int(len(nt)), "n_positive": int(nt.sum()),
                               "roc_auc": M.roc_auc(nt, npb), "pr_auc": M.pr_auc(nt, npb)}

    # Phase-1 benign FPR ("quiet factory")
    if p1 is not None:
        te1 = p1["split"].astype(int) == 2
        pr1 = p1["y_prob"].astype(float)
        n, fp = int(te1.sum()), int(np.sum(pr1[te1] >= thr))
        report["phase1_fpr"] = {"n_benign_windows": n, "false_alarms": fp,
                                "fpr": M._safe_div(fp, n)}
    else:
        report["phase1_fpr"] = None

    return report


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def evaluate_gate(report: Dict) -> Dict:
    """Tier-1 (provenance-subset) checks decide GO. Tier-2 union metrics are
    reported by render() but never gate -- the GAT is expected to lose on the
    network-only positives (that gap is Mamba's job)."""
    g = C.GATE
    te = report["prov"]["splits"]["test"]     # Tier-1 scorecard
    va = report["prov"]["splits"]["val"]
    checks = []

    def chk(name, value, ok, target):
        checks.append({"name": name, "value": value, "target": target, "pass": bool(ok)})

    # RANKING-based Tier-1 gate (see config.GATE). Operating-point MCC/recall are
    # reported by render() but do NOT gate: at PR-AUC ~0.5 they provably conflict,
    # and the ceiling is informational (Stage A is a ranker + embedding producer).
    chk("prov test PR-AUC", te["pr_auc"], te["pr_auc"] >= g["prov_test_pr_auc_min"],
        f">= {g['prov_test_pr_auc_min']}")
    chk("prov test ROC-AUC", te["roc_auc"], te["roc_auc"] >= g["prov_test_roc_auc_min"],
        f">= {g['prov_test_roc_auc_min']}")

    if "node_head" in report:
        na = report["node_head"]["roc_auc"]
        chk("node head ROC-AUC", na, na >= g["node_auc_min"], f">= {g['node_auc_min']}")

    if report.get("phase1_fpr"):
        fpr = report["phase1_fpr"]["fpr"]
        chk("Phase-1 benign FPR", fpr, fpr <= g["phase1_fpr_max"], f"<= {g['phase1_fpr_max']}")

    n_pass = sum(c["pass"] for c in checks)
    verdict = "GO" if n_pass == len(checks) else "NO-GO"
    return {"verdict": verdict, "n_pass": n_pass, "n_total": len(checks), "checks": checks}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _split_table(splits: Dict, highlight: str = "test") -> None:
    hdr = (f"  {'split':<6}{'N':>7}{'pos':>6}{'MCC':>8}{'F2':>8}{'PR-AUC':>9}"
           f"{'ROC':>8}{'recall':>8}{'prec':>8}{'FPR':>8}")
    print(_c(hdr, BOLD))
    for name in ("train", "val", "test"):
        s = splits[name]
        row = (f"  {name:<6}{s['n_windows']:>7}{s['n_positive']:>6}"
               f"{_fmt(s['mcc'],3):>8}{_fmt(s['f2'],3):>8}{_fmt(s['pr_auc'],3):>9}"
               f"{_fmt(s['roc_auc'],3):>8}{_fmt(s['recall'],3):>8}"
               f"{_fmt(s['precision'],3):>8}{_fmt(s['fpr'],4):>8}")
        print(_c(row, BOLD) if name == highlight else row)


def render(report: Dict, gate: Dict) -> None:
    thr = report["operating_threshold"]
    print()
    print(_rule("GAT (Stage A) READINESS DASHBOARD"))
    print(f"  operating threshold (recall-targeted on val prov-subset): {_fmt(thr)}")

    # TIER 1 -- the GAT's fair scorecard (what the gate judges)
    print(_rule("TIER 1  provenance-subset  (net-only positives dropped)  << GATED"))
    _split_table(report["prov"]["splits"])
    ci = report["prov_test_pr_auc_ci"]
    print(f"  {_c('prov test PR-AUC 95% CI (burst-bootstrap)', DIM)}: "
          f"[{_fmt(ci['lo'],3)}, {_fmt(ci['hi'],3)}]  (prevalence "
          f"{_fmt(report['prov']['splits']['test']['prevalence'],4)})")

    # TIER 2 -- end-to-end union scorecard (reported, NOT gated)
    print(_rule("TIER 2  end-to-end UNION labels  (reported, not gated)"))
    _split_table(report["union"]["splits"])
    h = report["headroom"]
    print(f"  {_c('Mamba headroom', YELLOW)}: union test recall {_fmt(h['union_test_recall'],3)} "
          f"vs prov {_fmt(h['prov_test_recall'],3)} -- "
          f"{h['net_only_test_positives']} network-only test positives are invisible to the "
          f"GAT (Mamba's fusion stage must catch these).")

    # phase-1 FPR
    if report.get("phase1_fpr"):
        f = report["phase1_fpr"]
        print(_rule("Phase-1 benign FPR (\"quiet factory\")"))
        print(f"  {f['false_alarms']} false alarms / {f['n_benign_windows']} "
              f"benign test windows  ->  FPR = {_fmt(f['fpr'])}")

    # burst view
    b = report["burst_test"]
    print(_rule("Burst-level detection (campaign view, test split)"))
    print(f"  bursts detected: {b['n_detected']}/{b['n_bursts']}  "
          f"(rate {_fmt(b['detection_rate'],3)})")
    ttds = [e["time_to_first_detection_s"] for e in b["per_burst"]
            if e["detected"] and not np.isnan(e["time_to_first_detection_s"])]
    if ttds:
        print(f"  time-to-first-detection: median {np.median(ttds):.1f}s  "
              f"max {np.max(ttds):.1f}s  (0s = flagged on the burst's first window)")
    if "by_tactic" in b:
        print(f"  {_c('per-tactic burst detection:', DIM)}")
        for t, d in sorted(b["by_tactic"].items()):
            print(f"      {t:<18} {d['n_detected']}/{d['n_bursts']} bursts")

    # per-tactic recall (descriptive)
    print(_rule("Per-tactic WINDOW recall on test (descriptive -- raw n/N)"))
    for t, d in sorted(report["per_tactic_recall_test"].items()):
        star = "  <- persistence claim" if t == C.PERSISTENCE_TACTIC else ""
        print(f"  {t:<18} {d['detected']}/{d['n_windows']} windows "
              f"(recall {_fmt(d['recall'],3)}){_c(star, YELLOW)}")

    # node head
    if "node_head" in report:
        nh = report["node_head"]
        print(_rule("Auxiliary node head (402 labeled attack nodes)"))
        print(f"  ROC-AUC {_fmt(nh['roc_auc'])}  PR-AUC {_fmt(nh['pr_auc'])}  "
              f"(pos {nh['n_positive']}/{nh['n_nodes']})")

    # gate
    print(_rule("PHASE-2 READINESS GATE"))
    for c in gate["checks"]:
        tag = _c(" PASS", GREEN) if c["pass"] else _c(" FAIL", RED)
        print(f"  [{tag} ] {c['name']:<34} {_fmt(c['value'],4)}  (need {c['target']})")
    print()
    print("  " + _c(f" VERDICT: {gate['verdict']} ", f"{BOLD};{'42' if gate['verdict']=='GO' else '41'};30")
          + f"   {gate['n_pass']}/{gate['n_total']} checks passed")
    if gate["verdict"] == "GO":
        print(_c("  -> Stage A is a defensible baseline. Proceed to build Stage B (Mamba).", GREEN))
        print(_c("     Export gat_window_embeddings_phase{1,2}.npz as the Mamba step input.", DIM))
    else:
        fails = [c["name"] for c in gate["checks"] if not c["pass"]]
        print(_c(f"  -> Not ready. Address: {', '.join(fails)}", RED))
    print(_rule())
    print()


# --------------------------------------------------------------------------- #
# Demo data (lets the dashboard run before training exists)
# --------------------------------------------------------------------------- #
def _make_demo() -> None:
    """Synthesize realistic predictions from the true labels + a noisy scorer.

    Reads the *real* split / labels / bursts / tactics from the artifacts so the
    dashboard shows genuine counts, and fabricates only ``y_prob`` -- a decent
    (not perfect) detector, so the gate lands in a believable place.
    """
    os.makedirs(C.ARTIFACTS, exist_ok=True)
    rng = np.random.default_rng(0)
    for phase in C.PHASES:
        seq = np.load(C.fused_npz(phase), allow_pickle=True)
        sp = np.load(C.split_npz(phase), allow_pickle=True)
        uni = seq["window_label_union"].astype(int)
        prov = seq["window_label_prov"].astype(int)
        # Realistic modality ceiling: the GAT scores PROVENANCE positives high,
        # but NETWORK-ONLY positives look benign to it (their graphs are benign).
        prob = np.clip(rng.beta(1.5, 8.0, size=len(uni)), 0, 1)          # benign-like baseline
        prov_pos = prov == 1
        prob[prov_pos] = np.clip(rng.beta(6.0, 2.0, size=int(prov_pos.sum())), 0, 1)
        # a few missed prov positives, for realism
        pidx = np.where(prov_pos)[0]
        if len(pidx):
            miss = rng.choice(pidx, size=max(1, len(pidx) // 6), replace=False)
            prob[miss] = rng.uniform(0.05, 0.25, size=len(miss))
        # net-only positives keep benign-like scores -> Tier-2 union recall < Tier-1
        np.savez(
            C.predictions_npz(phase),
            window_ids=seq["window_ids"],
            split=sp["split"].astype(int),
            y_true=uni,
            y_true_prov=prov,
            y_prob=prob.astype(np.float32),
            burst_id=sp["burst_id"].astype(int),
            tactic=seq["window_tactic_union"].astype(str),
            window_time=seq["window_times"].astype(float),
        )
    print(_c("[demo] wrote synthetic predictions (modality-realistic) to "
             "artifacts/predictions_phase{1,2}.npz", DIM))


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
def run_once(write_json: bool = True) -> Optional[Dict]:
    p2 = _load("phase2")
    p1 = _load("phase1")
    if p2 is None:
        print(_c("No predictions found at " + C.predictions_npz("phase2"), RED))
        print("Run training first, or `python -m gat.monitor --demo` to preview the dashboard.")
        return None
    report = evaluate(p2, p1)
    gate = evaluate_gate(report)
    render(report, gate)
    if write_json:
        os.makedirs(C.ARTIFACTS, exist_ok=True)
        with open(C.gate_report_json(), "w") as fh:
            json.dump({"report": report, "gate": gate,
                       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                      fh, indent=2, default=float)
        print(_c(f"  wrote {C.gate_report_json()}", DIM))
    return {"report": report, "gate": gate}


def main() -> None:
    ap = argparse.ArgumentParser(description="GAT Stage-A readiness monitor + Phase-2 gate")
    ap.add_argument("--demo", action="store_true",
                    help="synthesize realistic predictions (using real labels) and score them")
    ap.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                    help="re-run every N seconds (live monitoring during training)")
    args = ap.parse_args()

    if args.demo:
        _make_demo()

    if args.watch > 0:
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                run_once()
                print(_c(f"  (refreshing every {args.watch}s -- Ctrl-C to stop)", DIM))
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nstopped.")
    else:
        run_once()


if __name__ == "__main__":
    main()
