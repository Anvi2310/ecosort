"""Visual contamination/moisture regressor (optional, dependency-light).

Hand-crafted colour/texture features + a scikit-learn multi-output regressor trained by
``experiments/train_condition_model.py``. Without a checkpoint the app falls back to the
questionnaire and category priors; it never invents a visual estimate.
"""
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from config import MODEL_DIR

CHECKPOINT = MODEL_DIR / "condition_rf.pkl"
TARGETS = ("contamination", "moisture")


def _rgb_to_hsv(a):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(-1), a.min(-1)
    d = mx - mn + 1e-6
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    s = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0)
    return h, s, mx


def extract_features(image, size=160):
    """Fixed-length vector of cues tied to stains (brown/dark blobs) and wetness (gloss, darkening)."""
    img = image if isinstance(image, Image.Image) else Image.open(image)
    a = np.asarray(img.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    h, s, v = _rgb_to_hsv(a)
    gray = a.mean(-1)
    lap = np.abs(4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:])
    brown = (h > 0.02) & (h < 0.12) & (s > 0.35) & (v > 0.15) & (v < 0.7)
    highlights = (v > 0.92) & (s < 0.15)
    dark = v < 0.2
    # Patch statistics: stains create local brightness irregularities in otherwise uniform regions.
    patches = gray[: size // 16 * 16, : size // 16 * 16].reshape(16, size // 16, 16, size // 16).mean((1, 3))
    hist_h = np.histogram(h, bins=12, range=(0, 1), weights=s)[0] / (size * size)
    hist_v = np.histogram(v, bins=8, range=(0, 1))[0] / (size * size)
    return np.concatenate([
        [brown.mean(), highlights.mean(), dark.mean(), s.mean(), s.std(), v.mean(), v.std(),
         lap.mean(), lap.std(), patches.std(), np.abs(np.diff(patches, axis=0)).mean(),
         np.abs(np.diff(patches, axis=1)).mean()],
        hist_h, hist_v,
    ]).astype(np.float32)


class ConditionModel:
    def __init__(self, checkpoint=CHECKPOINT):
        self.model, self.meta, self.status = None, {}, "no_condition_checkpoint"
        path = Path(checkpoint)
        if path.exists():
            try:
                # Local artifact written by our own training script; do not load untrusted pickles.
                with open(path, "rb") as fh:
                    payload = pickle.load(fh)
                self.model, self.meta = payload["model"], payload.get("meta", {})
                self.status = "trained_" + self.meta.get("data", "unknown")
            except Exception as exc:  # corrupt/incompatible checkpoint -> behave as absent
                self.status = f"condition_checkpoint_error: {exc}"

    @property
    def available(self):
        return self.model is not None

    def predict(self, image):
        if not self.available:
            return None
        pred = self.model.predict(extract_features(image)[None, :])[0]
        return {k: round(float(np.clip(v, 0, 100)), 1) for k, v in zip(TARGETS, pred)}
