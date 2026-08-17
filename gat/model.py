"""
Two-head GATv2 encoder for per-window provenance graphs.

    node_features(20) (+) node_class_embed(8)
        -> 2x GATv2Conv(hidden, heads, edge_dim=edge_embed) with ELU + dropout
        -> node embeddings
             |-- node head:   Linear -> per-node malicious logit
             |-- window head: attention pooling (learned query) -> window embedding
                              -> Linear -> window malicious logit

The attention-pooled window embedding (``window_embed_dim``, default 64) is the
artifact Stage B (Mamba) consumes: ``forward`` returns it alongside the logits.
"""
from __future__ import annotations

from typing import Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.utils import softmax
    from torch_geometric.nn import global_add_pool
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "gat.model needs torch + torch_geometric. See gat/README.md. "
        f"(original error: {exc})"
    )

from . import config as C


class GATWindowEncoder(nn.Module):
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or C.GAT
        self.cfg = cfg
        hidden = cfg["hidden"]
        heads = cfg["heads"]
        ee = cfg["edge_embed_dim"]
        nce = cfg["node_class_embed_dim"]
        wdim = cfg["window_embed_dim"]

        self.node_class_embed = nn.Embedding(C.NODE_CLASSES, nce)
        self.edge_embed = nn.Embedding(len(C.EDGE_RELATIONS), ee)

        in_dim = C.NODE_FEATURE_DIM + nce
        n_layers = cfg.get("n_layers", 2)
        self.convs = nn.ModuleList()
        if n_layers == 1:
            # single relational layer straight to the embedding dim; concat=False
            # averages the heads so the output width stays wdim.
            self.convs.append(GATv2Conv(in_dim, wdim, heads=heads, edge_dim=ee,
                                        dropout=cfg["dropout"], concat=False))
        else:
            self.convs.append(GATv2Conv(in_dim, hidden, heads=heads, edge_dim=ee,
                                        dropout=cfg["dropout"], concat=True))
            for _ in range(n_layers - 2):
                self.convs.append(GATv2Conv(hidden * heads, hidden, heads=heads,
                                            edge_dim=ee, dropout=cfg["dropout"], concat=True))
            self.convs.append(GATv2Conv(hidden * heads, wdim, heads=1, edge_dim=ee,
                                        dropout=cfg["dropout"], concat=True))
        self.dropout = cfg["dropout"]

        # attention pooling: score each node embedding against a learned query
        self.pool_query = nn.Linear(wdim, 1)

        self.node_head = nn.Sequential(
            nn.Linear(wdim, wdim // 2), nn.ELU(),
            nn.Linear(wdim // 2, 1),
        )
        self.window_head = nn.Sequential(
            nn.Linear(wdim, wdim // 2), nn.ELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(wdim // 2, 1),
        )

    def encode(self, data) -> torch.Tensor:
        x = torch.cat([data.x, self.node_class_embed(data.node_class)], dim=-1)
        x = F.dropout(x, p=self.dropout, training=self.training)   # input dropout (regularizer)
        ea = self.edge_embed(data.edge_type) if data.edge_index.numel() else None
        for i, conv in enumerate(self.convs):
            x = conv(x, data.edge_index, edge_attr=ea)
            if i < len(self.convs) - 1:
                x = F.dropout(F.elu(x), p=self.dropout, training=self.training)
            else:
                x = F.elu(x)
        return x  # (num_nodes, window_embed_dim)

    def pool(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Attention-weighted readout -> one embedding per graph in the batch."""
        scores = self.pool_query(h)                 # (N, 1)
        alpha = softmax(scores, batch)              # per-graph softmax
        return global_add_pool(alpha * h, batch)    # (num_graphs, wdim)

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (window_logit[B], node_logit[N], window_embedding[B, wdim])."""
        h = self.encode(data)
        node_logit = self.node_head(h).squeeze(-1)
        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
            else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        emb = self.pool(h, batch)
        window_logit = self.window_head(emb).squeeze(-1)
        return window_logit, node_logit, emb


# --------------------------------------------------------------------------- #
# Losses tuned for 130:1 imbalance
# --------------------------------------------------------------------------- #
def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor,
                          gamma: float = 2.0, pos_weight: float | None = None
                          ) -> torch.Tensor:
    """Binary focal loss on logits. ``pos_weight`` up-weights the positive class."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none",
        pos_weight=torch.tensor(pos_weight, device=logits.device) if pos_weight else None)
    p_t = p * targets + (1 - p) * (1 - targets)
    return ((1 - p_t) ** gamma * ce).mean()
