"""
window_graphs/*.npz  ->  PyTorch Geometric ``Data`` objects.

Requires torch + torch_geometric (see gat/README.md for install). Everything
here is deterministic and leakage-safe:

* node-feature normalisation statistics are fit on the **train split only**
  (continuous dims z-scored; binary flag dims left untouched);
* edge relation strings are mapped to fixed indices from ``config.EDGE_RELATIONS``;
* the split each window belongs to is read from ``split_assignment.npz`` and is
  index-aligned with the graph files (verified 22 Jul 2026).
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

try:
    import torch
    from torch_geometric.data import Data, Dataset
except Exception as exc:  # pragma: no cover - import guard for numpy-only envs
    raise ImportError(
        "gat.dataset needs torch + torch_geometric. Install with:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "  pip install torch_geometric\n"
        f"(original error: {exc})"
    )

from . import config as C

_REL_TO_IDX = {r: i for i, r in enumerate(C.EDGE_RELATIONS)}


# --------------------------------------------------------------------------- #
# Split index (window_id -> split code), read from split_assignment.npz
# --------------------------------------------------------------------------- #
def load_split_index(phase: str) -> Dict[str, Dict]:
    sp = np.load(C.split_npz(phase), allow_pickle=True)
    wid = [str(w) for w in sp["window_ids"].tolist()]
    return {
        w: {"split": int(s), "burst_id": int(b), "region_id": int(r)}
        for w, s, b, r in zip(wid, sp["split"], sp["burst_id"], sp["region_id"])
    }


def _label_map(phase: str, key: str) -> Dict[str, int]:
    seq = np.load(C.fused_npz(phase), allow_pickle=True)
    wid = [str(w) for w in seq["window_ids"].tolist()]
    return dict(zip(wid, seq[key].astype(int).tolist()))


def load_union_labels(phase: str) -> Dict[str, int]:
    """window_id -> UNION window label (step-11). End-to-end target; used for
    EVALUATION only (the GAT cannot detect the 142 network-only positives)."""
    return _label_map(phase, "window_label_union")


def load_prov_labels(phase: str) -> Dict[str, int]:
    """window_id -> PROVENANCE window label. This is the GAT window head's
    TRAINING target: the subset whose attack signal is actually in the graph.

    (Two-tier design -- see config.GATE. The network-only positives are, by
    construction, invisible in the provenance graph, so training the window head
    on them injects impossible/benign-looking targets; Mamba's fusion stage is
    where those attacks become detectable.)
    """
    return _label_map(phase, "window_label_prov")


def _graph_files(phase: str) -> List[str]:
    return sorted(glob.glob(os.path.join(C.graphs_dir(phase), "*.npz")))


def _wid_of(path: str) -> str:
    # window_graphs/window_<id>.npz  ->  "<id>"
    return os.path.splitext(os.path.basename(path))[0].split("window_", 1)[-1]


# --------------------------------------------------------------------------- #
# Feature normaliser (fit on train, apply everywhere)
# --------------------------------------------------------------------------- #
@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    is_binary: np.ndarray  # True where the dim is a 0/1 flag -> leave as-is
    node_pos: int = 0      # train attack-node count (drives the node-head pos_weight)
    node_total: int = 0

    def apply(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.std
        z = np.clip(z, -10.0, 10.0)                 # bound any out-of-distribution test value
        # Binary passthrough, clipped to the flag range: if a column that was
        # coincidentally 0/1 across the whole train split ever shows a >1 value in
        # val/test, it is clamped rather than fed through raw/unnormalised.
        z[:, self.is_binary] = np.clip(x[:, self.is_binary], 0.0, 1.0)
        return z.astype(np.float32)

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist(),
                       "is_binary": self.is_binary.tolist()}, fh)

    @classmethod
    def from_json(cls, path: str) -> "Normalizer":
        d = json.load(open(path))
        return cls(np.array(d["mean"], np.float32),
                   np.array(d["std"], np.float32),
                   np.array(d["is_binary"], bool))


def fit_normalizer(phase: str, split_index: Dict[str, Dict]) -> Normalizer:
    """Streaming mean/var over train-split node features (dim = NODE_FEATURE_DIM)."""
    dim = C.NODE_FEATURE_DIM
    n = 0
    s = np.zeros(dim, np.float64)
    ss = np.zeros(dim, np.float64)
    node_pos = node_total = 0
    seen_vals = [set() for _ in range(dim)]  # to detect binary flag dims
    for path in _graph_files(phase):
        wid = _wid_of(path)
        meta = split_index.get(wid)
        if meta is None or meta["split"] != C.SPLIT_CODES["train"]:
            continue
        g = np.load(path, allow_pickle=True)
        x = g["node_features"].astype(np.float64)
        if x.size == 0:
            continue
        nlbl = np.asarray(g["node_label"])
        node_total += int(nlbl.size)
        node_pos += int(nlbl.sum())
        n += x.shape[0]
        s += x.sum(0)
        ss += (x * x).sum(0)
        for j in range(dim):
            if len(seen_vals[j]) <= 3:
                seen_vals[j].update(np.unique(x[:, j]).tolist())
    mean = s / max(n, 1)
    var = np.maximum(ss / max(n, 1) - mean ** 2, 1e-8)
    std = np.sqrt(var)
    is_binary = np.array([sv.issubset({0.0, 1.0}) for sv in seen_vals])
    return Normalizer(mean.astype(np.float32), std.astype(np.float32), is_binary,
                      node_pos=node_pos, node_total=node_total)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class WindowGraphDataset(Dataset):
    """One ``Data`` per window graph in a given split.

    ``oversample_pos`` replicates positive-window graphs k times (train only) to
    combat the ~130:1 imbalance without discarding benign windows.
    """

    def __init__(self, phase: str, split: str, normalizer: Normalizer,
                 split_index: Dict[str, Dict] | None = None,
                 oversample_pos: int = 1):
        super().__init__()
        self.phase = phase
        self.normalizer = normalizer
        self.split_index = split_index or load_split_index(phase)
        self.prov = load_prov_labels(phase)     # window head TRAINS on this (graph-visible attacks)
        self.union = load_union_labels(phase)   # carried for EVALUATION only
        code = C.SPLIT_CODES[split]
        files = []
        for path in _graph_files(phase):
            meta = self.split_index.get(_wid_of(path))
            if meta is not None and meta["split"] == code:
                files.append(path)
        # oversample positives by the TRAINING target (provenance-visible positives)
        if oversample_pos > 1:
            extra = []
            for path in files:
                if self.prov.get(_wid_of(path), 0) == 1:
                    extra.extend([path] * (oversample_pos - 1))
            files = files + extra
        self.files = files

    def len(self) -> int:
        return len(self.files)

    def get(self, idx: int) -> "Data":
        path = self.files[idx]
        g = np.load(path, allow_pickle=True)
        x = self.normalizer.apply(np.asarray(g["node_features"], np.float32))
        node_class = np.asarray(g["node_class"], np.int64)
        edge_index = np.asarray(g["edge_index"], np.int64)
        etypes = [str(e) for e in np.asarray(g["edge_type"]).tolist()]
        edge_type = np.array([_REL_TO_IDX.get(e, 0) for e in etypes], np.int64)
        wid = _wid_of(path)
        meta = self.split_index[wid]
        # y (training target) = provenance-visible label; y_union carried for eval.
        y_prov = float(self.prov.get(wid, 0))
        y_union = float(self.union.get(wid, 0))

        data = Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index) if edge_index.size
            else torch.zeros((2, 0), dtype=torch.long),
            node_class=torch.from_numpy(node_class),
            edge_type=torch.from_numpy(edge_type),
            node_y=torch.from_numpy(np.asarray(g["node_label"], np.float32)),
            y=torch.tensor([y_prov], dtype=torch.float32),
        )
        data.y_union = torch.tensor([y_union], dtype=torch.float32)
        # str attrs collate into python lists on the Batch; numeric metadata is
        # stored as 1-elem tensors so it concatenates to shape [batch_size].
        data.window_id = _wid_of(path)
        data.tactic = str(g["window_tactic"])
        data.burst_id = torch.tensor([int(meta["burst_id"])], dtype=torch.long)
        data.window_time = torch.tensor([float(g["window_time"])], dtype=torch.float64)
        return data
