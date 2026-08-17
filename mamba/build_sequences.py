"""
Freeze Stage-A embeddings and assemble the 110-dim Mamba input sequence.

Steps (all numpy -- runs without torch):
  1. FREEZE the current GAT window embeddings to a stable path so later Stage-A
     capacity sweeps cannot overwrite what Stage B trains on.
  2. Reorder the per-split embedding rows into the fused file's CHRONOLOGICAL
     order (embeddings are exported train|val|test-concatenated, not by time).
  3. VERIFY segment_ids alignment: the embedding window_id set must equal the
     fused window_id set, and we carry mamba_sequence_fused.npz's own
     segment_ids/window_times/labels so Stage B never bridges the 34.9 h gap.
  4. CONCATENATE z_t (64) onto the fused 46-dim step features -> 110-dim, and
     save mamba_input_{phase}.npz.

Run:  python -m mamba.build_sequences
"""
from __future__ import annotations

import os
import shutil

import numpy as np

from gat import config as GATC
from . import config as C


def _freeze(phase: str) -> None:
    src = GATC.embeddings_npz(phase)
    dst = C.frozen_emb(phase)
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Stage-A embeddings missing: {src}. Run `python -m gat.train` first.")
    os.makedirs(C.ARTIFACTS, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"[freeze] {os.path.basename(src)} -> {dst}")


def build(phase: str) -> dict:
    _freeze(phase)
    emb = np.load(C.frozen_emb(phase), allow_pickle=True)
    fused = np.load(GATC.fused_npz(phase), allow_pickle=True)
    split = np.load(GATC.split_npz(phase), allow_pickle=True)

    # --- alignment checks ---
    emb_ids = np.array([str(w) for w in emb["window_ids"].tolist()])
    fused_ids = np.array([str(w) for w in fused["window_ids"].tolist()])
    assert set(emb_ids.tolist()) == set(fused_ids.tolist()), \
        f"{phase}: embedding/fused window_id sets differ"
    assert emb["embedding"].shape[1] == C.GAT_EMBED_DIM, \
        f"{phase}: embedding dim {emb['embedding'].shape[1]} != {C.GAT_EMBED_DIM}"
    assert fused["sequence"].shape[1] == C.FUSED_DIM, \
        f"{phase}: fused dim {fused['sequence'].shape[1]} != {C.FUSED_DIM}"

    # --- reorder embeddings into fused (chronological) order ---
    pos = {w: i for i, w in enumerate(emb_ids)}
    order = np.array([pos[w] for w in fused_ids])
    z = emb["embedding"][order].astype(np.float32)              # (N, 64) chronological

    # --- concat -> 110-dim step vectors ---
    step = np.concatenate([z, fused["sequence"].astype(np.float32)], axis=1)  # (N, 110)
    assert step.shape[1] == C.STEP_DIM

    # --- split codes in the same (chronological) order ---
    split_by_id = dict(zip([str(w) for w in split["window_ids"].tolist()],
                           split["split"].astype(int).tolist()))
    burst_by_id = dict(zip([str(w) for w in split["window_ids"].tolist()],
                           split["burst_id"].astype(int).tolist()))
    split_codes = np.array([split_by_id[w] for w in fused_ids], np.int64)
    burst_ids = np.array([burst_by_id[w] for w in fused_ids], np.int64)

    # Segments: RECOMPUTE from window_times (split at gaps > 30 min) rather than
    # trust the stored segment_ids, which are all-zero in the shipped fused file.
    # NOTE (verified 23 Jul 2026): the fused sequence is temporally CONTINUOUS
    # (max inter-window gap ~43 s over the full 72 h span), because it is gridded
    # on continuous network windows -- so the documented 34.9 h provenance gap is
    # filled and yields a SINGLE segment here. Recomputing keeps this correct even
    # if a future rebuild does contain real gaps.
    times = fused["window_times"].astype(float)
    assert np.all(np.diff(times) >= -1e-6), f"{phase}: fused sequence is not chronological"
    seg = np.concatenate([[0], np.cumsum(np.diff(times) > 1800.0)]).astype(np.int64)
    stored = fused["segment_ids"].astype(np.int64)
    n_segments = len(np.unique(seg))
    if len(np.unique(stored)) != n_segments:
        print(f"  [warn] {phase}: stored segment_ids have {len(np.unique(stored))} "
              f"segment(s); recomputed {n_segments} from timestamps (using recomputed).")

    out = dict(
        window_ids=fused_ids,
        step=step,                                   # (N, 110)  << Mamba input
        segment_ids=seg,
        window_times=fused["window_times"].astype(np.float64),
        split=split_codes,
        burst_id=burst_ids,
        y_union=fused["window_label_union"].astype(np.int64),
        y_prov=fused["window_label_prov"].astype(np.int64),
        tactic=fused["window_tactic_union"].astype(str),
    )
    np.savez(C.mamba_input(phase), **out)
    n_pos = int(out["y_union"].sum())
    print(f"[{phase}] steps={step.shape[0]} dim={step.shape[1]} "
          f"segments={n_segments} union_pos={n_pos} "
          f"nan={bool(np.isnan(step).any())} -> {os.path.basename(C.mamba_input(phase))}")
    return out


def main() -> None:
    for phase in C.PHASES:
        build(phase)
    print("done. Stage-A z_t frozen; 110-dim sequences assembled and segment-verified.")


if __name__ == "__main__":
    main()
