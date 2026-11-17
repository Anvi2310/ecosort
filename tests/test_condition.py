import csv, io
import numpy as np
import pytest
from PIL import Image
from backend.services.condition_estimator import estimate, catalog
from tests.test_api import client  # noqa: F401  (reuse isolated-DB fixture)


def test_priors_when_nothing_answered():
    r = estimate("Paper", "cardboard")
    assert r["values"]["recyclability"] == 88
    assert r["details"]["contamination"]["source"] == "category prior"


def test_answers_drive_contamination_and_moisture():
    r = estimate("Plastic", "pet_bottle", {"residue": "heavy", "mixed": "fused", "wetness": "wet"})
    assert r["values"]["contamination"] == 100  # 90 + 25 clipped
    assert r["values"]["moisture"] == 85
    assert r["details"]["moisture"]["source"] == "user answers"


def test_hazard_flags_take_worst_case():
    r = estimate("Electronic waste", "small_device", flags=["battery", "sharp"])
    assert r["values"]["hazard"] == 85


def test_visual_used_only_when_user_did_not_answer():
    vis = {"contamination": 70, "moisture": 50}
    assert estimate("Metal", visual=vis)["details"]["contamination"]["source"] == "image model"
    assert estimate("Metal", answers={"residue": "none"}, visual=vis)["values"]["contamination"] == 5


@pytest.mark.parametrize("kwargs", [{"category": "Wood"}, {"category": "Glass", "item": "pet_bottle"},
                                    {"category": "Glass", "answers": {"wetness": "soggy"}},
                                    {"category": "Glass", "flags": ["radioactive"]}])
def test_invalid_inputs_rejected(kwargs):
    with pytest.raises(ValueError):
        estimate(**kwargs)


def test_catalog_covers_all_categories():
    from config import CATEGORIES
    assert set(catalog()["materials"]) == set(CATEGORIES)


def test_estimate_endpoint_and_label_export(client):  # noqa: F811
    assert client.get("/api/condition/catalog").status_code == 200
    r = client.post("/api/condition/estimate", json={"category": "Paper", "answers": {"wetness": "wet"}})
    assert r.status_code == 200 and r.get_json()["values"]["moisture"] == 85
    assert client.post("/api/condition/estimate", json={"category": "Nope"}).status_code == 400
    body = {"category": "Paper", "contamination": 5, "moisture": 85, "recyclability": 88, "hazard": 1,
            "image_ref": "abc.jpg", "condition": {"item": "cardboard", "answers": {"wetness": "wet"}, "user_adjusted": False}}
    assert client.post("/api/analyze", json=body).status_code == 200
    rows = list(csv.DictReader(io.StringIO(client.get("/api/condition/labels.csv").data.decode())))
    assert rows[0]["file"] == "abc.jpg" and float(rows[0]["moisture"]) == 85


def test_synthetic_dataset_and_training(tmp_path):
    from experiments.make_condition_dataset import build
    from experiments.train_condition_model import main
    from backend.models.condition_model import ConditionModel
    src = tmp_path / "clean" / "plastic"; src.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(6):
        colour = rng.integers(120, 255, 3)
        Image.fromarray(np.full((96, 96, 3), colour, dtype=np.uint8)).save(src / f"item{i}.png")
    rows = build(tmp_path / "clean", tmp_path / "synth", variants=5, seed=1, size=96)
    assert len(rows) == 30 and rows[0]["material"] == "plastic"
    ckpt = tmp_path / "condition.pkl"
    meta = main([str(tmp_path / "synth" / "manifest.csv"), "--out", str(ckpt)])
    assert meta["data"] == "synthetic" and set(meta["test_mae"]) == {"contamination", "moisture"}
    model = ConditionModel(ckpt)
    pred = model.predict(tmp_path / "synth" / rows[3]["file"])
    assert model.available and 0 <= pred["contamination"] <= 100
