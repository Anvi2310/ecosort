"""Write the synthetic GP/GA teaching datasets to data/synthetic/.

    python experiments/make_teaching_data.py

Outputs (labels come from the formulas in backend/models/teaching_data.py):
    data/synthetic/gp_priority_labels.csv   54 rows, high_priority 0/1
    data/synthetic/ga_priority_targets.csv  108 rows, priority 0-100
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import GP_DATA_PATH, GA_DATA_PATH
from backend.models.teaching_data import synthetic_gp_rows, synthetic_ga_rows, write_csv

if __name__ == "__main__":
    for path, rows in ((GP_DATA_PATH, synthetic_gp_rows()), (GA_DATA_PATH, synthetic_ga_rows())):
        write_csv(path, rows)
        print(f"wrote {len(rows)} rows -> {path}")
