# Data directory

- `raw/`: locally obtained source datasets; preserve source licenses and attribution.
- `processed/`: split manifests and derived data; the expected image folder layout is in the root README.
- `samples/`: tiny hand-curated examples only when redistribution is permitted.
- `synthetic/`: explicitly synthetic training/demo scenarios; never report these as real-world evaluation.

This repository intentionally does not download or redistribute third-party image datasets.

## GP and GA fitness data (`synthetic/`)

| File | Used by | Rows | Label column |
|---|---|---|---|
| `synthetic/gp_priority_labels.csv` | Genetic programming (rule evolution) | 54 | `high_priority` (0 or 1) |
| `synthetic/ga_priority_targets.csv` | Genetic algorithm (membership tuning) | 108 | `priority` (0–100) |

Both files have the inputs `contamination, moisture, recyclability, hazard` (each 0–100) plus a `synthetic` flag. All shipped labels are synthetic: they come from the formulas in `backend/models/teaching_data.py`, not from real waste data.

- `high_priority = 1` if `recyclability − 0.45·contamination − 0.35·hazard ≥ 48` (moisture is ignored)
- `priority = 30 + 0.58·recyclability − 0.38·contamination − 0.32·moisture − 0.42·hazard`, clipped to 0–100

**Regenerate:** `python experiments/make_teaching_data.py`

**Use real data:** replace the rows with expert- or facility-labelled items, keeping the same columns, and set `synthetic` to 0. The UI then reports "labeled examples" instead of "synthetic teaching scenarios". To keep the synthetic files, point the app at other files with the `ECOSORT_GP_DATA` / `ECOSORT_GA_DATA` environment variables. Restart the app after changing data.
