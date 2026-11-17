"""Labelled examples used as fitness data by GP (rule evolution) and GA (membership tuning).

The data lives in CSV files so it can be inspected, shown and replaced:

* GP  -> config.GP_DATA_PATH  columns: contamination, moisture, recyclability, hazard, high_priority (0/1), synthetic (0/1)
* GA  -> config.GA_DATA_PATH  columns: contamination, moisture, recyclability, hazard, priority (0-100), synthetic (0/1)

The shipped files are SYNTHETIC: every label comes from the hand-written formulas below,
not from real waste data. To use real data, replace the CSVs with expert- or facility-
labelled rows (set ``synthetic`` to 0) and keep the same columns.

Regenerate the synthetic files with:  python experiments/make_teaching_data.py
"""
import csv
from pathlib import Path

FEATURES = ("contamination", "moisture", "recyclability", "hazard")


# --- Synthetic label formulas (the "imaginary" labels) ------------------------------
def gp_label(c, m, r, h):
    """High priority (1) when recyclability outweighs contamination and hazard. Moisture is ignored."""
    return int(r - c * .45 - h * .35 >= 48)


def ga_priority(c, m, r, h):
    """Linear priority score: recyclable is good; dirty, wet and hazardous are bad. Clipped to 0-100."""
    return round(max(0.0, min(100.0, 30 + .58 * r - .38 * c - .32 * m - .42 * h)), 2)


def synthetic_gp_rows():
    return [{"contamination": c, "moisture": m, "recyclability": r, "hazard": h,
             "high_priority": gp_label(c, m, r, h), "synthetic": 1}
            for c in (10, 40, 75) for m in (10, 60, 90) for r in (25, 70, 95) for h in (5, 50)]


def synthetic_ga_rows():
    return [{"contamination": c, "moisture": m, "recyclability": r, "hazard": h,
             "priority": ga_priority(c, m, r, h), "synthetic": 1}
            for c in (5, 30, 60, 90) for m in (5, 50, 90) for r in (15, 60, 95) for h in (5, 45, 90)]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# --- Loading with validation --------------------------------------------------------
def _read(path, target, is_binary):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run: python experiments/make_teaching_data.py")
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = set(FEATURES + (target,)) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} is missing column(s): {', '.join(sorted(missing))}")
        for n, raw in enumerate(reader, start=2):
            try:
                x = {k: float(raw[k]) for k in FEATURES}
                y = float(raw[target])
            except (TypeError, ValueError):
                raise ValueError(f"{path.name} line {n}: values must be numbers") from None
            if any(not 0 <= v <= 100 for v in x.values()):
                raise ValueError(f"{path.name} line {n}: inputs must be between 0 and 100")
            if is_binary and y not in (0, 1):
                raise ValueError(f"{path.name} line {n}: {target} must be 0 or 1")
            if not is_binary and not 0 <= y <= 100:
                raise ValueError(f"{path.name} line {n}: {target} must be between 0 and 100")
            rows.append({"x": x, "y": int(y) if is_binary else y,
                         "synthetic": str(raw.get("synthetic", "1")).strip() != "0"})
    if len(rows) < 4:
        raise ValueError(f"{path.name} needs at least 4 rows")
    return rows


def describe(rows, path):
    """Human-readable provenance label shown in the UI next to results."""
    flags = {r["synthetic"] for r in rows}
    kind = "synthetic teaching scenarios" if flags == {True} else ("labeled examples" if flags == {False} else "mixed synthetic + labeled examples")
    return f"{kind} ({len(rows)} rows from {Path(path).name})"


def load_gp_rows(path=None):
    from config import GP_DATA_PATH
    path = path or GP_DATA_PATH
    return _read(path, "high_priority", True), path


def load_ga_data(path=None):
    import numpy as np
    from config import GA_DATA_PATH
    path = path or GA_DATA_PATH
    rows = _read(path, "priority", False)
    X = np.array([[r["x"][k] for k in FEATURES] for r in rows], float)
    y = np.array([r["y"] for r in rows], float)
    return X, y, rows, path
