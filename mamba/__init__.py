"""Stage B (Mamba) selective-state-space model over the window sequence.

Builds the 110-dim step vectors ([frozen GAT z_t (64) + fused features (46)]),
respects segment_ids boundaries, and detects the full UNION attack set that the
provenance-only GAT could not see.
"""
__all__ = ["config", "build_sequences", "model", "train"]
