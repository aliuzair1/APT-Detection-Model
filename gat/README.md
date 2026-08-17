# `gat/` — Stage A: GAT relational baseline + Phase-2 readiness gate

This package implements **methodology Phase 1** (the GAT relational baseline) and
the **GO/NO-GO gate** that decides whether the results justify building Stage B
(Mamba). It consumes the frozen `_preprocessed/` artifacts and produces the
window embeddings Mamba will step over.

## Layout

| File | Runs without torch? | Purpose |
|---|:---:|---|
| `config.py`  | ✓ | paths, hyper-parameters, gate thresholds (single source of truth) |
| `metrics.py` | ✓ | pure-numpy MCC / F2 / PR-AUC / burst / per-tactic / bootstrap CI |
| `monitor.py` | ✓ | readiness dashboard + Phase-2 gate; writes `readiness_gate.json` |
| `dataset.py` | ✗ | `window_graphs/*.npz` → PyTorch Geometric `Data` (train-fit normaliser) |
| `model.py`   | ✗ | two-head GATv2 encoder (node head + attention-pooled window head) |
| `train.py`   | ✗ | training loop; writes `predictions_*.npz` + `gat_window_embeddings_*.npz` |

## Quick start

```powershell
# 1. Preview the gate dashboard right now, before any training exists.
#    Uses the REAL labels/splits/bursts and a synthetic scorer.
python -m gat.monitor --demo

# 2. Install the training stack (CPU wheels shown; use a CUDA index if you have a GPU).
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric

# 3. Train the GAT baseline (writes artifacts/predictions_*.npz + embeddings).
python -m gat.train                 # full run (~100 epochs, early-stopped)
python -m gat.train --epochs 5      # fast smoke test

# 4. Score the real predictions and evaluate the Phase-2 gate.
python -m gat.monitor               # one-shot
python -m gat.monitor --watch 30    # live dashboard, refresh every 30s during training
```

## The contract between training and the gate

`train.py` writes, per phase, `artifacts/predictions_{phase}.npz` with keys:
`window_ids, split (0/1/2), y_true` (union), `y_true_prov` (provenance),
`y_prob, burst_id, tactic, window_time` (+ `node_y_true, node_y_prob` for the
test split). `monitor.py` reads only these npz files — it needs no torch, no
sklearn, no pandas — computes every statistic, and prints `VERDICT: GO` / `NO-GO`.

**Two-tier gate.** The GAT is provenance-only, so 142 of 208 Phase-2 positives
(network-only attacks, invisible in the provenance graph) are undetectable at
this stage. The gate therefore judges **Tier 1** (provenance-subset, `y_true_prov`,
net-only positives dropped) and only *reports* **Tier 2** (end-to-end `y_true`
union + burst view). Tier-1 passing → GO. The Tier-2 gap is the headroom the
Mamba fusion stage must close. Gate thresholds live in `config.GATE`; edit them
there, never in the monitor.
