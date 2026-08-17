"""
Stage-A capacity-invariance sweep -> the ceiling artifact for the paper.

Trains the GAT at several encoder widths with EVERYTHING ELSE FIXED (loss,
weighting, lr, dropout schedule), and records train vs test provenance-subset
PR-AUC. The thesis point: test PR-AUC is flat across a wide capacity range while
train PR-AUC rises -- i.e. the ~0.55 ceiling is an information limit of the
provenance-only single-window input, not an under-capacity artifact.

Outputs (gat/artifacts/):
  stageA_ceiling.json   raw metrics per capacity
  stageA_ceiling.md     markdown table for the paper
  stageA_ceiling.png    train-vs-test PR-AUC plot

Run:  python -m gat.sweep_capacity
NOTE: overwrites gat/artifacts predictions/embeddings, but Stage B uses the
FROZEN copies in mamba/artifacts, so the sweep is safe to run after freezing.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from . import config as C
from . import metrics as M
from .dataset import WindowGraphDataset, fit_normalizer, load_split_index
from .model import GATWindowEncoder, focal_bce_with_logits

# Clean capacity axis: fix depth (2 layers, so `hidden` actually drives width --
# the 1-layer path ignores it) and hold dropout constant, vary only `hidden`.
# MULTI-SEED: PR-AUC on 17 test positives is high-variance, so each capacity is
# run over several seeds and reported as mean +/- std (error bars in the figure).
WIDTHS = [(16, 2), (32, 2), (64, 2)]   # (hidden, n_layers)  ~17k / ~30k / ~55k params
SEEDS = [0, 1, 2]
SWEEP_DROPOUT = 0.3
EPOCHS = 30
PATIENCE = 10


def _prov_subset(y_union, y_prov):
    return (y_union == 0) | (y_prov == 1)


def run_one(hidden: int, n_layers: int, seed: int, data, pw: float) -> dict:
    """Train one (capacity, seed) on PRE-LOADED in-memory graph lists (no disk I/O)."""
    train_list, val_list, test_list = data
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = dict(C.GAT); cfg.update(hidden=hidden, n_layers=n_layers, dropout=SWEEP_DROPOUT)
    dev = torch.device("cpu")
    tl = DataLoader(train_list, batch_size=cfg["batch_size"], shuffle=True)

    model = GATWindowEncoder(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n_params = sum(p.numel() for p in model.parameters())

    def evalset(ds):
        model.eval(); ys, yp, pb = [], [], []
        with torch.no_grad():
            for b in DataLoader(ds, batch_size=cfg["batch_size"]):
                wl, _, _ = model(b.to(dev))
                pb.append(torch.sigmoid(wl).cpu().numpy())
                ys.append(b.y_union.cpu().numpy()); yp.append(b.y.cpu().numpy())
        yu, yv, p = np.concatenate(ys), np.concatenate(yp), np.concatenate(pb)
        sub = _prov_subset(yu, yv)
        return M.pr_auc(yv[sub], p[sub]), M.roc_auc(yv[sub], p[sub])

    best, best_state, wait = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for b in tl:
            b = b.to(dev); opt.zero_grad()
            wl, nl, _ = model(b)
            loss = focal_bce_with_logits(wl, b.y, cfg["focal_gamma"], pw) \
                + cfg["node_loss_weight"] * focal_bce_with_logits(nl, b.node_y, cfg["focal_gamma"], pw)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        vpr, _ = evalset(val_list)
        if vpr > best:
            best, wait = vpr, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    tr_pr, tr_roc = evalset(train_list)
    te_pr, te_roc = evalset(test_list)
    print(f"    seed={seed} hidden={hidden} params={n_params:,}  "
          f"train PR-AUC {tr_pr:.3f}  test PR-AUC {te_pr:.3f}  test ROC {te_roc:.3f}")
    return dict(hidden=hidden, n_layers=n_layers, params=int(n_params), seed=seed,
                train_pr_auc=float(tr_pr), test_pr_auc=float(te_pr),
                train_roc=float(tr_roc), test_roc=float(te_roc))


def _agg(rs, key):
    v = np.array([r[key] for r in rs], dtype=float)
    return float(v.mean()), float(v.std())


def main() -> None:
    os.makedirs(C.ARTIFACTS, exist_ok=True)
    idx = load_split_index("phase2")
    norm = fit_normalizer("phase2", idx)

    # Preload all graphs into memory ONCE (normalized) -- reused across every run,
    # so the 9 trainings never touch disk again. This is the whole speed fix.
    cfg0 = C.GAT
    print("preloading graphs into memory (once)...")
    tr_ds = WindowGraphDataset("phase2", "train", norm, idx, oversample_pos=cfg0["pos_oversample"])
    va_ds = WindowGraphDataset("phase2", "val", norm, idx)
    te_ds = WindowGraphDataset("phase2", "test", norm, idx)
    data = ([tr_ds.get(i) for i in range(len(tr_ds))],
            [va_ds.get(i) for i in range(len(va_ds))],
            [te_ds.get(i) for i in range(len(te_ds))])
    pw = max(1.0, cfg0["target_effective_pos_weight"] / cfg0["pos_oversample"])
    print(f"  cached train/val/test = {len(data[0])}/{len(data[1])}/{len(data[2])} graphs")

    print(f"Multi-seed capacity sweep ({len(WIDTHS)} widths x {len(SEEDS)} seeds):")
    runs = []
    summary = []
    for h, l in WIDTHS:
        print(f"  hidden={h} layers={l}")
        group = [run_one(h, l, s, data, pw) for s in SEEDS]
        runs += group
        te_m, te_s = _agg(group, "test_pr_auc")
        tr_m, tr_s = _agg(group, "train_pr_auc")
        roc_m, roc_s = _agg(group, "test_roc")
        summary.append(dict(hidden=h, n_layers=l, params=group[0]["params"],
                            test_pr_auc_mean=te_m, test_pr_auc_std=te_s,
                            train_pr_auc_mean=tr_m, train_pr_auc_std=tr_s,
                            test_roc_mean=roc_m, test_roc_std=roc_s))

    with open(os.path.join(C.ARTIFACTS, "stageA_ceiling.json"), "w") as fh:
        json.dump({"runs": runs, "summary": summary, "seeds": SEEDS, "epochs": EPOCHS}, fh, indent=2)

    lines = ["# Stage-A capacity-invariance (provenance-subset, mean +/- std over "
             f"{len(SEEDS)} seeds)", "",
             "| hidden | params | train PR-AUC | test PR-AUC | test ROC-AUC |",
             "|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['hidden']} | {s['params']:,} | "
                     f"{s['train_pr_auc_mean']:.3f} ± {s['train_pr_auc_std']:.3f} | "
                     f"{s['test_pr_auc_mean']:.3f} ± {s['test_pr_auc_std']:.3f} | "
                     f"{s['test_roc_mean']:.3f} ± {s['test_roc_std']:.3f} |")
    tem = [s["test_pr_auc_mean"] for s in summary]
    lines += ["", f"**Test PR-AUC stays ~{min(tem):.2f}-{max(tem):.2f} across a "
              f"{summary[-1]['params']//summary[0]['params']}x parameter range** while train "
              f"PR-AUC rises to {max(s['train_pr_auc_mean'] for s in summary):.2f}, and ROC-AUC "
              f"holds ~{min(s['test_roc_mean'] for s in summary):.2f}. Capacity does not raise "
              "generalisation: the ceiling is informational (provenance-only input), not capacity."]
    with open(os.path.join(C.ARTIFACTS, "stageA_ceiling.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = [s["params"] for s in summary]
        plt.figure(figsize=(6.5, 4.2))
        plt.errorbar(x, [s["train_pr_auc_mean"] for s in summary],
                     yerr=[s["train_pr_auc_std"] for s in summary],
                     fmt="o-", capsize=4, label="train PR-AUC")
        plt.errorbar(x, [s["test_pr_auc_mean"] for s in summary],
                     yerr=[s["test_pr_auc_std"] for s in summary],
                     fmt="s-", capsize=4, label="test PR-AUC")
        plt.errorbar(x, [s["test_roc_mean"] for s in summary],
                     yerr=[s["test_roc_std"] for s in summary],
                     fmt="^--", capsize=4, alpha=0.6, label="test ROC-AUC")
        plt.xscale("log"); plt.ylim(0, 1)
        plt.xlabel("encoder parameters (log scale)"); plt.ylabel("prov-subset score")
        plt.title(f"Stage-A capacity invariance (mean±std, {len(SEEDS)} seeds)")
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(C.ARTIFACTS, "stageA_ceiling.png"), dpi=130)
        print("saved stageA_ceiling.{json,md,png}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
