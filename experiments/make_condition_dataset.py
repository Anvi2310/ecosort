"""Build a labelled contamination/moisture image dataset from clean waste photos.

No public waste dataset labels *how dirty or wet* an item is, so we synthesise the labels:
start from clean photos (e.g. TrashNet / your own), then render controlled amounts of
stains (food/grease/dirt blobs) and wetness (darkening, gloss highlights, droplets).
Because we choose the amount, the label is known exactly.

Labels are SYNTHETIC. Report them as such, split by source image (``group`` column) to
avoid leakage, and validate on a small set of real, hand-rated photos
(see data/CONDITION_LABELING.md) before claiming real-world accuracy.

Usage:
    python experiments/make_condition_dataset.py data/processed/images/train \
        data/synthetic/condition --variants 6 --seed 42
"""
import argparse, csv, hashlib, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

STAIN_COLOURS = [(110, 70, 30), (150, 95, 40), (90, 60, 35), (170, 130, 60), (60, 45, 30), (180, 60, 30)]
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def object_mask(img):
    """Foreground mask (L image): pixels that differ from the median border colour.

    Works for plain-background photos such as TrashNet; falls back to the whole frame
    when the background is busy, so stains are never restricted to a bad mask.
    """
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    dist = np.linalg.norm(a - np.median(border, axis=0), axis=-1)
    fg = (dist > 40).astype(np.uint8) * 255
    mask = Image.fromarray(fg, "L").filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    if not 0.05 < (np.asarray(mask) > 0).mean() < 0.95:
        mask = Image.new("L", img.size, 255)
    return mask


def _points(mask, rng):
    ys, xs = np.nonzero(np.asarray(mask))
    def pick():
        i = rng.randrange(len(xs))
        return float(xs[i]), float(ys[i])
    return pick


def _composite(img, overlay, mask):
    """Paste overlay only where the object is; keep the background untouched."""
    alpha = np.asarray(overlay.getchannel("A"), dtype=np.float32) * (np.asarray(mask, dtype=np.float32) / 255)
    overlay.putalpha(Image.fromarray(alpha.astype(np.uint8), "L"))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def add_stains(img, level, rng, mask=None):
    """level 0-100 -> number/size/opacity of irregular blobs on the object; returns image."""
    if level <= 0:
        return img
    mask = mask or Image.new("L", img.size, 255)
    pick = _points(mask, rng)
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    n = int(1 + level / 6)
    for _ in range(n):
        cx, cy = pick()
        r = rng.uniform(0.02, 0.04 + level / 900) * min(w, h)
        colour = rng.choice(STAIN_COLOURS)
        alpha = int(90 + level * 1.4 * rng.uniform(0.6, 1.0))
        for _ in range(rng.randint(3, 7)):  # clustered circles -> irregular splotch
            dx, dy, rr = rng.gauss(0, r * 0.6), rng.gauss(0, r * 0.6), r * rng.uniform(0.4, 1.0)
            draw.ellipse([cx + dx - rr, cy + dy - rr, cx + dx + rr, cy + dy + rr], fill=(*colour, min(alpha, 235)))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(1, min(w, h) / 120)))
    return _composite(img, overlay, mask)


def add_wetness(img, level, rng, mask=None):
    """level 0-100 -> darkening, saturation boost, specular droplets on the object."""
    if level <= 0:
        return img
    mask = mask or Image.new("L", img.size, 255)
    pick = _points(mask, rng)
    f = level / 100
    wet = ImageEnhance.Color(ImageEnhance.Brightness(img).enhance(1 - 0.35 * f)).enhance(1 + 0.4 * f)
    img = Image.composite(wet, img, mask)
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(int(level * 0.8)):
        x, y = pick()
        r = rng.uniform(0.004, 0.015) * min(w, h)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, int(120 + 100 * f)))
        draw.ellipse([x - r * 1.6, y - r * 0.6, x + r * 1.6, y + r * 1.6], outline=(0, 0, 0, 60))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.8))
    return _composite(img, overlay, mask)


def find_images(root):
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in EXTS)


def build(src, out, variants=6, seed=42, size=384):
    rng = random.Random(seed)
    out = Path(out); (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for src_path in find_images(src):
        base = Image.open(src_path).convert("RGB")
        base.thumbnail((size, size))
        mask = object_mask(base)
        rel = src_path.relative_to(src)
        material = rel.parts[0] if len(rel.parts) > 1 else ""
        for i in range(variants):
            # Stratify so every level band is represented, including clean/dry (label 0).
            c = 0 if i == 0 else round(rng.uniform(0, 100), 1)
            m = 0 if i == 1 else round(rng.uniform(0, 100) if rng.random() < 0.7 else 0, 1)
            img = add_wetness(add_stains(base, c, rng, mask), m, rng, mask)
            name = f"{src_path.stem}_{hashlib.sha1(str(rel).encode()).hexdigest()[:8]}_v{i}.jpg"
            img.save(out / "images" / name, quality=90)
            rows.append({"file": f"images/{name}", "group": str(rel), "material": material,
                         "contamination": c, "moisture": m, "synthetic": 1})
    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "group", "material", "contamination", "moisture", "synthetic"])
        w.writeheader(); w.writerows(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="folder of clean images (sub-folders per material are kept as metadata)")
    ap.add_argument("out", help="output folder for images/ and manifest.csv")
    ap.add_argument("--variants", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = build(args.src, args.out, args.variants, args.seed)
    print(f"wrote {len(rows)} labelled images to {args.out} (synthetic labels)")
