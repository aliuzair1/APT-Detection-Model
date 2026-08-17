"""
Part B: self-supervised masked-sequence pretraining of the Stage-B encoder on
the 35,300 benign Phase-1 windows (used in full -- there are no attacks in
Phase-1 to hold out; the point is a benign-dynamics prior). `mamba.train
--init-from` then loads the resulting encoder into a fresh MambaDetector and
fine-tunes it supervised on Phase 2, as a fix for the 101-positive supervised
overfitting.

Objective, per chunk (chunk_len steps of a Phase-1 run, same _runs(split,
segment_ids) contiguous-run decomposition train.py uses): sample a boolean
mask over timesteps (p=0.15), zero the masked rows of the (Phase-1-standardized)
input, encode with the shared MambaDetector.encode(), reconstruct with a small
Linear(d_model, STEP_DIM) head, and take MSE loss over the masked positions
only -- the model has to infer a masked step from its (unmasked) temporal
context, which is what teaches it benign dynamics.

Saves mamba/artifacts/pretrained_encoder.pt = {"input", "blocks", "norms", "cfg"}
-- state dicts for exactly the three submodules `mamba.train --init-from` loads
into a fresh MambaDetector. The classification/exit heads and this file's own
reconstruction head are intentionally NOT saved (fine-tuning starts them fresh).

Run: py -m mamba.pretrain --device cuda              # full
     py -m mamba.pretrain --epochs 1 --chunk-len 256 --device cpu   # CPU smoke
"""
from __future__ import annotations

import argparse
import os

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    raise ImportError(f"mamba.pretrain needs torch. (original error: {exc})")

from . import config as C
from .model import MambaDetector
from .train import _runs, set_seed   # reuse the exact run-decomposition / seeding train.py uses

MASK_PROB = 0.15


def _load_phase1() -> dict:
    d = np.load(C.mamba_input("phase1"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def pretrain(cfg: dict, dev: str) -> str:
    os.makedirs(C.ARTIFACTS, exist_ok=True)
    set_seed(cfg["seed"])
    device = torch.device(dev)

    d1 = _load_phase1()
    n_pos = int(d1["y_union"].sum())
    print(f"phase1 windows={len(d1['y_union'])} non-benign={n_pos} "
          f"(Part B pretrains on the whole of Phase-1 regardless; union_pos is "
          f"expected to be 0 -- reported for the record).")

    # Fit mu, sd on ALL Phase-1 windows (unlike train.py, which fits only on the
    # Phase-2 TRAIN split): Part B has no held-out Phase-1 split to protect --
    # this encoder is fine-tuned on Phase 2 afterwards, where train.py's own
    # leakage-safe TRAIN-only scaler applies.
    mu = d1["step"].mean(0)
    sd = d1["step"].std(0) + 1e-6
    step_std = np.clip((d1["step"] - mu) / sd, -10, 10).astype(np.float32)
    np.savez(os.path.join(C.ARTIFACTS, "pretrain_input_scaler.npz"), mu=mu, sd=sd)

    step = torch.from_numpy(step_std).float()
    runs = _runs(d1["split"], d1["segment_ids"])
    print(f"phase1 steps={len(step)}  runs={len(runs)}  chunk_len={cfg['chunk_len']}  "
          f"mask_prob={MASK_PROB}")

    cfg_pre = dict(cfg)
    model = MambaDetector(cfg_pre).to(device)
    recon = nn.Linear(cfg_pre["d_model"], C.STEP_DIM).to(device)
    n_params = (sum(p.numel() for p in model.parameters())
                + sum(p.numel() for p in recon.parameters()))
    print(f"Mamba(+recon head) params: {n_params:,}")

    opt = torch.optim.Adam(list(model.parameters()) + list(recon.parameters()), lr=1e-3)

    loss_hist = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); recon.train()
        np.random.shuffle(runs)
        tot, n_chunks = 0.0, 0
        for a, b, _ in runs:
            for s in range(a, b, cfg["chunk_len"]):
                e = min(s + cfg["chunk_len"], b)
                x = step[s:e].unsqueeze(0).to(device)                    # (1, L, STEP_DIM)
                L = x.shape[1]
                # deterministic-per-(seed,epoch,chunk) mask -- reproducible, and
                # does not consume/perturb the global numpy RNG used by
                # np.random.shuffle(runs) above (mirrors train.py's _perm pattern).
                mask_np = np.random.default_rng([int(cfg["seed"]), int(epoch), int(s)]).random(L) < MASK_PROB
                mask = torch.from_numpy(mask_np).to(device)
                x_masked = x.clone()
                x_masked[:, mask, :] = 0.0

                opt.zero_grad()
                h, _ = model.encode(x_masked)          # shared encoder (also used by forward)
                pred = recon(h)                        # (1, L, STEP_DIM)
                if mask.any():
                    loss = F.mse_loss(pred[0, mask], x[0, mask])
                else:                                   # degenerate empty-mask chunk (rare)
                    loss = torch.zeros((), device=device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(recon.parameters()), 5.0)
                opt.step()
                tot += float(loss.item()); n_chunks += 1

        avg = tot / max(n_chunks, 1)
        loss_hist.append(avg)
        print(f"epoch {epoch:3d}  chunks {n_chunks}  masked-recon MSE {avg:.4f}")

    tail = ", ".join(f"{v:.4f}" for v in loss_hist[-5:])
    print(f"train loss curve tail (last {min(5, len(loss_hist))} epoch(s)): [{tail}]")

    ck = {"input": model.input.state_dict(), "blocks": model.blocks.state_dict(),
          "norms": model.norms.state_dict(), "cfg": cfg_pre}
    out_path = C.pretrained_encoder()
    torch.save(ck, out_path)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30,
                    help="E_pre: masked-reconstruction pretraining epochs (default 30).")
    ap.add_argument("--chunk-len", type=int, default=C.MAMBA["chunk_len"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=C.MAMBA["seed"])
    a = ap.parse_args()
    cfg = dict(C.MAMBA)
    cfg.update(epochs=a.epochs, chunk_len=a.chunk_len, seed=a.seed)
    pretrain(cfg, a.device)


if __name__ == "__main__":
    main()
