"""
Stage B (Mamba) configuration.

Consumes the FROZEN Stage-A embeddings + the 46-dim fused features and models the
window SEQUENCE. Where Stage A was provenance-only and capped at test PR-AUC ~0.55
(the capacity-invariance ceiling, see gat/artifacts/stageA_ceiling.*), Stage B adds
the network modality and temporal context -- the signal the per-window GAT could
not see -- and is evaluated end-to-end on the UNION labels.
"""
from __future__ import annotations

import os

from gat import config as GATC   # reuse the verified dataset paths

MAMBA_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(MAMBA_DIR, "artifacts")

PHASES = GATC.PHASES

# Frozen Stage-A embeddings (z_t) -- copied so later Stage-A sweeps never clobber them.
def frozen_emb(phase: str) -> str:
    return os.path.join(ARTIFACTS, f"frozen_z_gat_{phase}.npz")

# The assembled 110-dim per-step sequence input for Mamba.
def mamba_input(phase: str) -> str:
    return os.path.join(ARTIFACTS, f"mamba_input_{phase}.npz")

def predictions(phase: str) -> str:
    return os.path.join(ARTIFACTS, f"predictions_{phase}.npz")

# Part B (self-supervised pretraining): the masked-reconstruction-pretrained
# encoder checkpoint (input+blocks+norms state dicts) -- see pretrain.py.
def pretrained_encoder() -> str:
    return os.path.join(ARTIFACTS, "pretrained_encoder.pt")

# Contract dims (verified in build_sequences.py)
GAT_EMBED_DIM = 64
FUSED_DIM = GATC.FUSED_DIM      # 46
STEP_DIM = GAT_EMBED_DIM + FUSED_DIM   # 110

MAMBA = dict(
    d_model=128,
    d_state=16,
    d_conv=4,
    expand=2,
    n_layers=2,
    dropout=0.1,
    # sequences are hard-split at segment_ids; chunk long segments to this many steps.
    # 2048 (was 1024): covers the longest val/test run (1777) so every eval sequence is
    # un-split and within the training context length; the 8735 train run is still
    # chunked (train-only, acceptable).
    chunk_len=2048,
    # loss / imbalance (union labels; ~0.8% positive). MODERATE weight -- a high
    # pos_weight on this per-step sequence loss floods predictions to "positive".
    focal_gamma=2.0,
    target_effective_pos_weight=25.0,
    # optim
    lr=1e-3,                       # 3e-4 was too slow to escape the flood basin
    weight_decay=1e-4,
    epochs=80,
    early_stop_patience=15,
    seed=42,
    # early-exit head after block 1 (methodology D.2)
    early_exit=True,
    # a window "early-exits" when the block-1 benign-confidence head is this sure
    # (measured, reported analysis only -- see FIX D / stageB_earlyexit{tag}.json)
    exit_threshold=0.95,
    # ---- ASL (Asymmetric Loss) selection, Part 1 ----
    # "focal" reproduces the frozen baseline byte-for-identically; "asl" is the
    # asymmetric-loss alternative (Ben-Baruch et al. 2020) for the extreme-imbalance
    # main head. Selecting "asl" here/via --loss never changes the focal code path.
    loss="asl",              # "focal" | "asl"  (focal reproduces the frozen baseline)
    asl_gamma_pos=1.0,
    asl_gamma_neg=4.0,
    asl_clip=0.05,
    asl_pos_weight=1.0,      # residual positive emphasis on top of ASL asymmetry
    # ---- temperature-scaling calibration, Part 2 ----
    # Fit ONCE after training on frozen (post-best-state) val logits; applied only to
    # the final exported probabilities (and consistently in early-exit). Rank-
    # preserving (monotonic), so it never changes PR-AUC/ROC/recall-targeted metrics
    # or the per-epoch val model-selection (which always runs at T=1).
    calibrate=True,          # fit a temperature on val after training (Part 2)
)

# Operating point used both for the per-epoch val threshold (train) and the gate's
# test threshold (_gate): highest val threshold whose recall >= this target.
OPERATING_RECALL_TARGET = 0.80

# End-to-end (UNION) readiness gate for the full GAT->Mamba system. Unlike Stage A,
# here the operating-point detector metrics DO matter -- Mamba is the detector.
GATE = dict(
    test_pr_auc_min=0.55,          # must beat the Stage-A prov ceiling on the harder union task
    test_roc_auc_min=0.85,
    beat_stageA_union_pr_auc=True, # Mamba must improve union PR-AUC vs frozen GAT alone
    phase1_fpr_max=0.02,
    require_persistence_detected=True,
)

# Leakage-safe fixed-FPR-budget operating point (Part 3, reporting only -- see
# train.py::_gate). threshold = the (1-FPR_BUDGET) quantile of Phase-2 VAL benign
# scores; measured on test/Phase-1. Does NOT feed the GATE checks/verdict above.
FPR_BUDGET = 0.02            # leakage-safe operating point (Part 3): threshold = the
                            # (1-FPR_BUDGET) quantile of Phase-2 VAL benign scores.
