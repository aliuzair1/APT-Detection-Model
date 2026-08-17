"""GAT (Stage A) relational baseline for CICAPT-IIoT malware-persistence detection.

Modules
-------
config   : paths, hyper-parameters, Phase-2 readiness-gate thresholds.
metrics  : pure-numpy detection metrics (MCC, F2, PR/ROC-AUC, burst, per-tactic).
monitor  : readiness dashboard + GO/NO-GO gate (numpy-only; runs without torch).
dataset  : window_graphs -> PyTorch Geometric loader (requires torch + PyG).
model    : two-head GATv2 encoder (node head + attention-pooled window head).
train    : training loop; writes predictions_*.npz and gat_window_embeddings_*.npz.
"""
__all__ = ["config", "metrics", "monitor", "dataset", "model", "train"]
