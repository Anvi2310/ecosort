"""Central configuration for EcoSort."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "backend" / "models" / "weights"
INSTANCE_DIR = ROOT / "instance"
DATABASE_PATH = Path(os.environ.get("ECOSORT_DATABASE", INSTANCE_DIR / "ecosort.sqlite3"))
SECRET_KEY = os.environ.get("ECOSORT_SECRET_KEY", "development-only-change-me")
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CATEGORIES = ("Plastic", "Paper", "Metal", "Glass", "Organic", "Electronic waste")
MODEL_VERSION = "ecosort-phase1"
# Fitness data for GP rule evolution and GA membership tuning (synthetic by default; replace with real labels).
GP_DATA_PATH = Path(os.environ.get("ECOSORT_GP_DATA", DATA_DIR / "synthetic" / "gp_priority_labels.csv"))
GA_DATA_PATH = Path(os.environ.get("ECOSORT_GA_DATA", DATA_DIR / "synthetic" / "ga_priority_targets.csv"))
