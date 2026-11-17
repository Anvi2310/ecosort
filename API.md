# EcoSort API

Run `python app.py` (default `http://127.0.0.1:5000`). JSON errors use HTTP 400/413/415.

| Method | Endpoint | Input / output |
|---|---|---|
| GET | `/api/health` | service and classifier mode |
| GET | `/api/training/status` | checkpoint and dataset status |
| POST | `/api/classify` | multipart `image`; top-three probabilities only for a fine-tuned checkpoint |
| POST | `/api/analyze` | JSON: category and four 0–100 values; returns scores, activated rules and recommendation; persists analysis |
| GET | `/api/membership` | Gaussian LOW/MEDIUM/HIGH curves and parameters |
| POST | `/api/gp/evolve` | optional population/generations; returns best executable tree and fitness history |
| POST | `/api/ga/optimize` | population/generations and optional inversion/deletion |
| POST | `/api/anfis/train` | at least ten labeled samples with four inputs and a priority target; fixed-seed train/validation/test split, returns held-out metrics |
| GET | `/api/anfis/status` | ANFIS training state and metrics |
| GET | `/api/analytics` | aggregates from locally persisted analyses |
| GET | `/api/history` | recent records |
| GET | `/api/export.csv` | downloadable history CSV |
| GET | `/api/report.pdf` | PDF summary of saved analyses |

Demo GP/GA targets are synthetic teaching scenarios and are labeled in responses. Never interpret them as measured material-recovery performance.

Running GP stores its evolved tree for subsequent decision calls; running GA updates the active membership partition for subsequent calls. A trained ANFIS is blended into priority only after `/api/anfis/train` receives user-labeled examples. In-memory optimizer and ANFIS state resets when the server restarts.
