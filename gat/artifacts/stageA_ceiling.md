# Stage-A capacity-invariance (provenance-subset, mean ± std over 3 seeds)

| hidden | params | train PR-AUC | test PR-AUC | test ROC-AUC |
|---:|---:|---:|---:|---:|
| 16 | 17,659 | 0.802 ± 0.030 | 0.510 ± 0.107 | 0.954 ± 0.026 |
| 32 | 30,203 | 0.746 ± 0.126 | 0.500 ± 0.095 | 0.940 ± 0.029 |
| 64 | 55,291 | 0.713 ± 0.078 | 0.435 ± 0.123 | 0.934 ± 0.023 |

**Test PR-AUC stays ~0.44–0.51 (flat within error bars) across a 3× parameter range**, while train PR-AUC sits at 0.71–0.80 (overfitting) and ROC-AUC holds ~0.93–0.95. Capacity does not raise generalisation: the ceiling is **informational** (provenance-only input), not a capacity limit.

The large per-seed spread (±0.10 on test PR-AUC) is expected — the provenance-subset test set has only 17 positive windows, so PR-AUC is high-variance. This is exactly why the sweep is multi-seed: single-seed points scatter from 0.03 to 0.66, whereas the 3-seed means are stable at ~0.5.

See `stageA_ceiling.png` for the figure and `stageA_ceiling.json` for all 9 runs.
