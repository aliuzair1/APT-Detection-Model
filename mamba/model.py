"""
Stage B model: a compact, CPU-runnable Mamba (selective SSM) over window sequences.

The official `mamba-ssm` package needs CUDA/Triton kernels; this environment is
CPU-only, so we implement the S6 selective scan in pure PyTorch (a sequential
scan -- O(L) time, O(1) state memory, the property that justifies Mamba on an
edge gateway). It is slower than the fused CUDA kernel but numerically equivalent
and fine at this sequence length.

Per step it emits a malicious logit; an optional early-exit head after block 1
predicts benign-confidence so ~all benign windows can skip block 2 (methodology D.2).
"""
from __future__ import annotations

from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    raise ImportError(f"mamba.model needs torch. (original error: {exc})")

from . import config as C


class MambaBlock(nn.Module):
    """One selective-SSM block (Mamba), pure-PyTorch sequential scan."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        # selective parameters: Δ, B, C are input-dependent
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        # state matrix A (log-parameterised, per (d_inner, d_state)), and skip D
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)."""
        B, L, _ = x.shape
        xz = self.in_proj(x)                                  # (B, L, 2*d_inner)
        xi, z = xz.chunk(2, dim=-1)

        # short causal conv over time
        xi = xi.transpose(1, 2)                               # (B, d_inner, L)
        xi = self.conv1d(xi)[..., :L]
        xi = xi.transpose(1, 2)                               # (B, L, d_inner)
        xi = F.silu(xi)

        # input-dependent Δ, B, C
        dbl = self.x_proj(xi)                                 # (B, L, 2*d_state+1)
        dt, Bm, Cm = torch.split(dbl, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                     # (B, L, d_inner)
        A = -torch.exp(self.A_log)                            # (d_inner, d_state)

        # sequential selective scan
        y = self._scan(xi, dt, A, Bm, Cm)                     # (B, L, d_inner)
        y = y + xi * self.D                                   # skip connection
        y = y * F.silu(z)                                     # gating
        return self.out_proj(y)

    def _scan(self, u, dt, A, Bm, Cm):
        B, L, d_inner = u.shape
        d_state = self.d_state
        # discretise: dA = exp(dt * A), dB = dt * B  (ZOH, standard Mamba approx)
        dA = torch.exp(dt.unsqueeze(-1) * A)                  # (B, L, d_inner, d_state)
        dBu = dt.unsqueeze(-1) * Bm.unsqueeze(2) * u.unsqueeze(-1)  # (B,L,d_inner,d_state)
        h = torch.zeros(B, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBu[:, t]                      # (B, d_inner, d_state)
            yt = torch.einsum("bds,bs->bd", h, Cm[:, t])     # (B, d_inner)
            ys.append(yt)
        return torch.stack(ys, dim=1)                         # (B, L, d_inner)


class MambaDetector(nn.Module):
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or C.MAMBA
        self.cfg = cfg
        d = cfg["d_model"]
        self.input = nn.Sequential(nn.Linear(C.STEP_DIM, d), nn.LayerNorm(d))
        self.blocks = nn.ModuleList([
            MambaBlock(d, cfg["d_state"], cfg["d_conv"], cfg["expand"])
            for _ in range(cfg["n_layers"])])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(cfg["n_layers"])])
        self.drop = nn.Dropout(cfg["dropout"])
        self.head = nn.Linear(d, 1)
        # early-exit benign-confidence head after block 1 (methodology D.2)
        self.early_exit = cfg.get("early_exit", True)
        self.exit_head = nn.Linear(d, 1) if self.early_exit else None

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """x: (B, L, STEP_DIM) -> post-block hidden state h: (B, L, d_model), and
        the optional block-1 early-exit logit (B, L). Shared by `forward`
        (classification) and `mamba.pretrain` (masked-reconstruction) so both
        consume exactly the same input-projection + Mamba-block stack."""
        h = self.input(x)
        exit_logit = None
        for i, (blk, nrm) in enumerate(zip(self.blocks, self.norms)):
            h = h + self.drop(blk(nrm(h)))                    # pre-norm residual
            if i == 0 and self.exit_head is not None:
                exit_logit = self.exit_head(h).squeeze(-1)    # (B, L)
        return h, exit_logit

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """x: (B, L, STEP_DIM) -> per-step logits (B, L), and optional exit logit."""
        h, exit_logit = self.encode(x)
        return self.head(h).squeeze(-1), exit_logit


def focal_bce_with_logits(logits, targets, gamma=2.0, pos_weight=None, mask=None):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none",
        pos_weight=torch.tensor(pos_weight, device=logits.device) if pos_weight else None)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = (1 - p_t) ** gamma * ce
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1)
    return loss.mean()


def asl_with_logits(logits, targets, gamma_pos=1.0, gamma_neg=4.0, clip=0.05,
                    pos_weight=1.0, eps=1e-8):
    """Asymmetric Loss for the extreme-imbalance binary detection head.
    gamma_neg > gamma_pos suppresses easy negatives harder than positives;
    `clip` is the probability margin m that shifts (and eventually zeros) very-easy
    negatives; `pos_weight` adds residual emphasis on the rare positives on top of
    the asymmetric focusing. Reduction: mean over all elements."""
    p = torch.sigmoid(logits)
    xs_pos = p
    xs_neg = 1.0 - p
    if clip is not None and clip > 0:
        xs_neg = (xs_neg + clip).clamp(max=1.0)            # asymmetric probability shift
    los_pos = targets * torch.log(xs_pos.clamp(min=eps)) * pos_weight
    los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=eps))
    loss = los_pos + los_neg
    # asymmetric focusing
    pt0 = xs_pos * targets
    pt1 = xs_neg * (1.0 - targets)
    pt = pt0 + pt1
    gamma = gamma_pos * targets + gamma_neg * (1.0 - targets)
    loss = loss * ((1.0 - pt) ** gamma)
    return -loss.mean()
