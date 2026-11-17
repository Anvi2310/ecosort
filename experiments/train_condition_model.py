"""Train the visual contamination/moisture regressor.

Inputs: one or more manifest.csv files with columns file, group, contamination, moisture
(synthetic from make_condition_dataset.py and/or real ones exported from the app via
/api/condition/labels.csv). Splits by ``group`` (source image) so variants of the same
photo never appear in both train and test.

    python experiments/train_condition_model.py data/synthetic/condition/manifest.csv
"""
import argparse, csv, json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
from backend.models.condition_model import extract_features, CHECKPOINT, TARGETS


def load(manifests):
    X, Y, groups, synthetic = [], [], [], []
    for m in manifests:
        root = Path(m).parent
        with open(m) as fh:
            for row in csv.DictReader(fh):
                path = root / row["file"]
                if not path.exists():
                    continue
                X.append(extract_features(path)); Y.append([float(row[t]) for t in TARGETS])
                groups.append(row.get("group") or row["file"]); synthetic.append(row.get("synthetic", "1") == "1")
    return np.array(X), np.array(Y), np.array(groups), np.array(synthetic)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--out", default=str(CHECKPOINT))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    X, Y, groups, synthetic = load(args.manifests)
    if len(X) < 20 or len(set(groups)) < 4:
        raise SystemExit("Need at least 20 images from 4+ source photos.")
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed).split(X, Y, groups))
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=args.seed, n_jobs=-1)
    model.fit(X[tr], Y[tr])
    pred = model.predict(X[te])
    metrics = {t: round(float(mean_absolute_error(Y[te, i], pred[:, i])), 2) for i, t in enumerate(TARGETS)}
    data = "synthetic" if synthetic.all() else ("real" if not synthetic.any() else "mixed")
    meta = {"data": data, "test_mae": metrics, "n_train": int(len(tr)), "n_test": int(len(te)),
            "split": "group-by-source-image 80/20", "seed": args.seed, "manifests": args.manifests}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({"model": model, "meta": meta}, fh)
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    main()
