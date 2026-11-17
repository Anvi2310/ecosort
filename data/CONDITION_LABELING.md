# Condition labelling guide (contamination, moisture, hazard)

No public waste dataset labels *how dirty, wet or dangerous* an item is, and a typical user can't measure these from a photo. EcoSort gets them from three sources, listed from most to least trusted:

| Input | Where it comes from |
|---|---|
| Recyclability | Material knowledge base (category + item type), in `backend/services/condition_estimator.py` |
| Hazard | Item-type baseline plus yes/no safety flags (battery, chemicals, sharp, medical, pressurised). The app never infers hazard from pixels. |
| Contamination | Questionnaire (residue + mixed materials), then the visual model, then the category prior |
| Moisture | Questionnaire (dry / damp / wet), then the visual model, then the category prior |

## Building the dataset

1. **Synthetic, for bootstrapping.** Start from clean photos (TrashNet or your own). `experiments/make_condition_dataset.py` adds stains and wetness in controlled amounts, so each label is known exactly. Mark these labels as synthetic in every result.
2. **Real, human-confirmed.** Every analysis saved from the app records the photo, the questionnaire answers and the final slider values in the `ConditionLabels` table. `GET /api/condition/labels.csv` exports them as a manifest the training script accepts. The images are in `instance/uploads/`.
3. **Expert-rated test set.** Have 2–3 people independently rate about 200 real photos using the rubric below. Keep only items where the raters agree within one band, and use this set only for testing.

## Rubric (rate each image independently)

**Contamination**

| Score | Band | What you see |
|---|---|---|
| 0–10 | clean | No visible residue |
| 20–40 | light | A few drops, crumbs or a small label |
| 50–70 | visible | Clear stains or smears, or residue covering less than about a quarter of the surface |
| 80–100 | heavy | Food left inside, grease covering much of it, or fused to another material |

**Moisture**

| Score | Band | What you see |
|---|---|---|
| 0–10 | dry | |
| 30–50 | damp | Condensation, darkened patches |
| 70–100 | wet | Soaked, dripping, pooled water, soggy paper |

**Hazard:** don't rate it from the photo. Record the item type and the flags.

## Training

```bash
python experiments/make_condition_dataset.py data/processed/images/train data/synthetic/condition --variants 6
python experiments/train_condition_model.py data/synthetic/condition/manifest.csv [instance/uploads/manifest.csv]
```

This writes `backend/models/weights/condition_rf.pkl`. Restart the app to load it. The training script splits by source image and reports test MAE. Before quoting any accuracy, check MAE on the expert-rated real test set, because a model trained only on synthetic data may not transfer to real photos.
