"""
Stage-A GAT training loop.

Trains the two-head GATv2 on Phase-2 window graphs (benign Phase-1 windows are
mixed in as extra negatives), selects the checkpoint by best validation MCC,
then writes the two contract artifacts the rest of the project depends on:

  artifacts/predictions_phase{1,2}.npz      -> consumed by gat.monitor (the gate)
  artifacts/gat_window_embeddings_phase{1,2}.npz -> Stage B (Mamba) step input

Run:
    python -m gat.train                 # full run
    python -m gat.train --epochs 5      # quick smoke test
    python -m gat.monitor               # then score + evaluate the gate
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np

try:
    import torch
    from torch_geometric.loader import DataLoader
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "gat.train needs torch + torch_geometric. See gat/README.md. "
        f"(original error: {exc})"
    )

from . import config as C
from . import metrics as M
from .dataset import (WindowGraphDataset, fit_normalizer, load_prov_labels,
                      load_split_index)
from .model import GATWindowEncoder, focal_bce_with_logits


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _train_pos_count(phase: str, split: str, norm, split_index) -> int:
    """Positive-window count in a split, before oversampling."""
    ds = WindowGraphDataset(phase, split, norm, split_index, oversample_pos=1)
    return int(sum(int(ds.get(i).y.item()) for i in range(len(ds))))


# --------------------------------------------------------------------------- #
# Inference over a whole split -> per-window arrays
# --------------------------------------------------------------------------- #
@torch.no_grad()
def infer(model, loader, device, collect_nodes: bool = False) -> Dict[str, np.ndarray]:
    model.eval()
    wid, prob, yprov, yunion, tac, burst, wtime, emb = [], [], [], [], [], [], [], []
    node_true, node_prob = [], []
    for batch in loader:
        batch = batch.to(device)
        wlogit, nlogit, wemb = model(batch)
        prob.append(torch.sigmoid(wlogit).cpu().numpy())
        yprov.append(batch.y.cpu().numpy())            # training target (prov-visible)
        yunion.append(batch.y_union.cpu().numpy())     # end-to-end eval target
        emb.append(wemb.cpu().numpy())
        burst.append(batch.burst_id.cpu().numpy())
        wtime.append(batch.window_time.cpu().numpy())
        wid.extend(list(batch.window_id))
        tac.extend(list(batch.tactic))
        if collect_nodes:
            node_true.append(batch.node_y.cpu().numpy())
            node_prob.append(torch.sigmoid(nlogit).cpu().numpy())
    out = {
        "window_ids": np.array(wid, dtype=object),
        "y_prob": np.concatenate(prob).astype(np.float32),
        "y_true": np.concatenate(yunion).astype(np.int64),        # UNION = primary scorecard
        "y_true_prov": np.concatenate(yprov).astype(np.int64),    # provenance-subset scorecard
        "tactic": np.array(tac, dtype=object),
        "burst_id": np.concatenate(burst).astype(np.int64),
        "window_time": np.concatenate(wtime).astype(np.float64),
        "embedding": np.concatenate(emb).astype(np.float32),
    }
    if collect_nodes and node_true:
        out["node_y_true"] = np.concatenate(node_true).astype(np.int64)
        out["node_y_prob"] = np.concatenate(node_prob).astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(cfg: dict, device: str) -> None:
    os.makedirs(C.ARTIFACTS, exist_ok=True)
    set_seed(cfg["seed"])
    dev = torch.device(device)

    idx2 = load_split_index("phase2")
    print("Fitting node-feature normaliser on the Phase-2 train split ...")
    norm = fit_normalizer("phase2", idx2)
    norm.to_json(os.path.join(C.ARTIFACTS, "normalizer.json"))

    train_ds = WindowGraphDataset("phase2", "train", norm, idx2,
                                  oversample_pos=cfg["pos_oversample"])
    val_ds = WindowGraphDataset("phase2", "val", norm, idx2)
    # Cache graphs in memory ONCE so epochs never re-read ~16k files from disk
    # (~12x faster training). Graphs are tiny (~4 nodes) so this is cheap RAM.
    print("caching graphs into memory (once) ...")
    train_list = [train_ds.get(i) for i in range(len(train_ds))]
    val_list = [val_ds.get(i) for i in range(len(val_ds))]

    # window-head imbalance (un-oversampled counts, from the provenance labels).
    n_train = sum(1 for m in idx2.values() if m["split"] == C.SPLIT_CODES["train"])
    _prov = load_prov_labels("phase2")
    n_pos = sum(1 for w, m in idx2.items()
                if m["split"] == C.SPLIT_CODES["train"] and _prov.get(w, 0) == 1)
    # target a MODERATE effective positive weight (pos_weight * oversample), not
    # full class balancing -- full balancing floods the false-positive band.
    target_eff = cfg.get("target_effective_pos_weight", 50.0)
    pos_weight = cfg.get("pos_weight", max(1.0, target_eff / cfg["pos_oversample"]))
    # Node-head weight. Effective positive emphasis = pos_weight * pos_oversample,
    # because oversampling already replays positive nodes pos_oversample x. We keep
    # this MODERATE for BOTH heads: fully balancing floods predictions to "positive"
    # everywhere -- the window head did exactly that at ~444x (precision ~0.05),
    # fixed by dropping to ~50x. The node head sees ~65k nodes that are 99.7% benign
    # (~293:1), so targeting *its* raw balance would land at ~250x effective and
    # flood identically. Instead the node head shares the same moderate effective
    # target as the window head (empirically already gives node ROC-AUC ~0.86);
    # override via cfg["node_pos_weight"] if you want to probe higher.
    node_neg = max(norm.node_total - norm.node_pos, 1)
    node_pos_weight = float(cfg.get("node_pos_weight", pos_weight))
    print(f"train windows {n_train}, positives {n_pos}, pos_weight {pos_weight:.1f} "
          f"(eff {pos_weight * cfg['pos_oversample']:.0f}x) | node imbalance "
          f"{node_neg}:{max(norm.node_pos, 1)}, node_pos_weight {node_pos_weight:.1f} "
          f"(eff {node_pos_weight * cfg['pos_oversample']:.0f}x -- deliberately moderate)")

    train_loader = DataLoader(train_list, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_list, batch_size=cfg["batch_size"], shuffle=False)

    model = GATWindowEncoder(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    best_mcc, best_state, patience = -1.0, None, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(dev)
            opt.zero_grad()
            wlogit, nlogit, _ = model(batch)
            loss_w = focal_bce_with_logits(
                wlogit, batch.y, gamma=cfg["focal_gamma"], pos_weight=pos_weight)
            # node head: richer supervision (402 labeled attack nodes), own weight
            loss_n = focal_bce_with_logits(
                nlogit, batch.node_y, gamma=cfg["focal_gamma"], pos_weight=node_pos_weight)
            loss = loss_w + cfg["node_loss_weight"] * loss_n
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * batch.num_graphs
        sched.step()

        # Model selection = Tier-1 (provenance-subset) val MCC: the GAT is judged
        # on what it can actually see. Drop net-only positives from the val set,
        # score against the provenance label, threshold F2-optimal on that subset.
        val = infer(model, val_loader, dev)
        uni, prov = val["y_true"], val["y_true_prov"]
        sub = (uni == 0) | (prov == 1)
        yp, pp = prov[sub], val["y_prob"][sub]
        thr = M.best_threshold_by_f2(yp, pp)
        vm = M.scalar_metrics(yp, pp, thr)
        print(f"epoch {epoch:3d}  loss {total/len(train_ds):.4f}  "
              f"val[prov] MCC {vm['mcc']:.3f}  F2 {vm['f2']:.3f}  "
              f"PR-AUC {M.pr_auc(yp, pp):.3f}  "
              f"| val[union] recall {M.scalar_metrics(uni, val['y_prob'], thr)['recall']:.3f}")

        if vm["mcc"] > best_mcc:
            best_mcc, patience = vm["mcc"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                print(f"early stop at epoch {epoch} (best val MCC {best_mcc:.3f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg},
               os.path.join(C.ARTIFACTS, "gat_stageA.pt"))
    print(f"best val MCC {best_mcc:.3f}; checkpoint saved.")

    # ------------------------------------------------------------------ #
    # Export predictions + embeddings for BOTH phases (all splits)
    # ------------------------------------------------------------------ #
    for phase in C.PHASES:
        idx = idx2 if phase == "phase2" else load_split_index(phase)
        splits, parts = [], []
        for sp_name, code in C.SPLIT_CODES.items():
            ds = WindowGraphDataset(phase, sp_name, norm, idx)
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False)
            res = infer(model, loader, dev, collect_nodes=(sp_name == "test"))
            res["split"] = np.full(len(res["window_ids"]), code, dtype=np.int64)
            parts.append(res)
        merged = _merge(parts)
        np.savez(
            C.predictions_npz(phase),
            window_ids=merged["window_ids"], split=merged["split"],
            y_true=merged["y_true"], y_true_prov=merged["y_true_prov"],
            y_prob=merged["y_prob"],
            burst_id=merged["burst_id"], tactic=merged["tactic"],
            window_time=merged["window_time"],
            node_y_true=merged.get("node_y_true", np.array([], np.int64)),
            node_y_prob=merged.get("node_y_prob", np.array([], np.float32)),
        )
        np.savez(
            C.embeddings_npz(phase),
            window_ids=merged["window_ids"], split=merged["split"],
            embedding=merged["embedding"], y_true=merged["y_true"],
            # self-describing time keys so the export never relies on row order
            window_time=merged["window_time"], burst_id=merged["burst_id"],
        )
        print(f"wrote {C.predictions_npz(phase)} and {C.embeddings_npz(phase)} "
              f"({len(merged['window_ids'])} windows)")


def _merge(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    keys = ["window_ids", "y_prob", "y_true", "y_true_prov", "tactic", "burst_id",
            "window_time", "embedding", "split"]
    for k in keys:
        out[k] = np.concatenate([p[k] for p in parts])
    for k in ["node_y_true", "node_y_prob"]:
        vals = [p[k] for p in parts if k in p]
        if vals:
            out[k] = np.concatenate(vals)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Stage-A GAT baseline")
    ap.add_argument("--epochs", type=int, default=C.GAT["epochs"])
    ap.add_argument("--batch-size", type=int, default=C.GAT["batch_size"])
    ap.add_argument("--lr", type=float, default=C.GAT["lr"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = dict(C.GAT)
    cfg.update(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    train(cfg, args.device)


if __name__ == "__main__":
    main()
