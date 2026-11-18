# PathSnap: Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University.
**Team:** Swaroop (PES1UG24CS923) · Shashank D (PES1UG24CS433) · Shashank Keshava Murthy (PES1UG24CS434) · Manoj AC (PES1UG25CS826)

Repository: https://github.com/swaroop05v/gps-map-matching

PathSnap matches noisy GPS fixes onto a road network and reconstructs the driven route in four stages:

1. **Candidate generation.** Every road arc within 50 m of a GPS fix becomes a candidate (at most 10 per fix). A KD-tree over densified arcs makes the search fast.
2. **Candidate classifier.** A LightGBM model, chosen from 7 models and tuned with Bayesian optimisation, scores P(candidate is the true arc). It uses distance, heading, context-window, kinematic and topology features.
3. **HMM + Viterbi.** Hidden states are the candidate arcs, emissions are the classifier probabilities and transitions use road-network shortest paths. Viterbi decodes one connected route.
4. **Deployment.** A FastAPI service on Laptop 1 runs the pipeline and scores it with server-side reference data. A Streamlit + Folium UI on Laptop 2 lets the evaluator pick one of the 10 reserved trips.

## Results (student_test, 10 trips)
| Method | Top-1 accuracy | Route continuity | Route F1 |
|---|---|---|---|
| Nearest arc (no ML) | 0.536 | 0.427 | 0.558 |
| Tuned LightGBM, classifier only | 0.910 | 0.821 | 0.807 |
| HMM with distance-only emission | 0.926 | 0.890 | 0.947 |
| **LightGBM + HMM/Viterbi (final)** | **0.947** | **0.904** | **0.978** |

Seven classifiers compared (5-fold GroupKFold CV): `results/model_comparison.csv`. Tuning: `results/tuned_vs_untuned.csv`, `results/optuna_trials.csv`.
Decoder: `results/hmm_student_test_summary.csv`. Full write-up: `report/PathSnap_Report.pdf`.

## Repository layout
```
configs/config.yaml        all thresholds (radius, K, gap rules, decoder defaults)
src/pathsnap/
  io.py                    raw file readers, track cleaning
  geo.py                   local metric projection, point-to-segment geometry
  network.py               road network, KD-tree candidate search, Dijkstra
  labels.py                per-fix reference labels (monotonic DP alignment to .route)
  features.py              preprocessing, gap segmentation, candidate + feature table
  models.py                the 7 candidate classifiers
  metrics.py               Top-1, Macro F1, log loss, route continuity, route P/R/F1
  hmm.py                   HMM states/emissions/transitions, Viterbi, route bridging
  pipeline.py              end-to-end inference used by scripts and the API
  viz.py                   Folium map generation
scripts/
  01_eda.py                dataset exploration -> results/eda, results/figures/eda_*, results/maps/eda_*
  02_build_dataset.py      labels + candidates + features -> data/processed/*.parquet
  03_train_compare.py      7 models, 5-fold GroupKFold CV + student_test evaluation
  03b_plot_models.py       comparison figures + table
  04_bayes_opt.py          Optuna TPE search for LightGBM + tuned vs untuned
  05_hmm_eval.py           decoder tuning, classifier-only vs HMM, final model export, result maps
  06_result_maps.py        zoomed result maps + PNG screenshots of every Folium map
  07_make_report.py        builds report/PathSnap_Report.html + .pdf from results/
api/main.py                FastAPI backend (Laptop 1)
ui/app.py                  Streamlit evaluator UI (Laptop 2)
models/                    final_model.joblib, final_model_meta.json, hmm_config.json
results/                   every CSV, log, figure and map used in the report
report/                    technical report (PDF) and screenshots
```

## 1. Installation
Python 3.10 or newer.
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

## 2. Data
The raw data is not committed. Put it here:
```
data/raw/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}          80 trips
data/raw/student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}   10 trips
data/raw/eval_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}      10 evaluator trips (only on the API laptop)
```
The evaluator partition is read **only** by the API at request time. No training, tuning or analysis script reads it.

## 3. Reproduce all experiments
Run from the repository root:
```bash
python scripts/01_eda.py              # EDA tables, figures, Folium maps
python scripts/02_build_dataset.py    # ~1 min
python scripts/03_train_compare.py logreg gnb lgbm hgb rf knn
python scripts/03_train_compare.py mlp     # separate process (see note below)
python scripts/03b_plot_models.py
N_TRIALS=25 python scripts/04_bayes_opt.py   # Optuna; the reported run used 25 trials (default 30)
python scripts/05_hmm_eval.py         # decoder tuning + final model export + result maps
python scripts/06_result_maps.py      # zoomed maps + screenshots (needs Edge/Chrome)
python scripts/07_make_report.py      # report HTML + PDF
```
Note: PyTorch and LightGBM ship different OpenMP runtimes. When both load in one process on Windows, LightGBM trains about 9x slower, so torch is imported lazily and the MLP runs in its own process.

Random seed 42 is used everywhere. Logs are in `results/logs/`.

## 4. Run the system

### Laptop 1: API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
* The console shows every request: client IP, trip, number of points, accuracy and latency. The same lines go to `results/logs/api_requests.log`.
* Find the laptop's IP with `ipconfig`, e.g. `192.168.1.20`.
* Allow port 8000 through the Windows firewall (admin PowerShell):
  `New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow`
* Test it from Laptop 2: `curl http://192.168.1.20:8000/health`

### Laptop 2: UI
```bash
pip install streamlit streamlit-folium folium requests
streamlit run ui/app.py
```
Set the API URL under **API connection** to `http://<laptop1-ip>:8000`. Use the repository on Laptop 2 for the UI and shared map styling. Data and models are needed on Laptop 1.

Select **Evaluator trips** for the final demonstration or **Student test** for a practice run. Select a trip and run map matching. The map's layer control switches between raw GPS, HMM predictions and classifier-only predictions.

The [redesigned screenshots and evidence index](report/screenshots/gta/README.md) are alongside the original captures. Regenerate the themed offline maps and their screenshots with `python scripts/06_result_maps.py --gta`.

### API endpoints
| Method | Path | Description |
|---|---|---|
| GET | `/health` | model and decoder status |
| GET | `/trips?partition=eval_test` | trip ids on the server (no labels) |
| POST | `/predict` `{"trip_id": "00000015", "partition": "eval_test"}` | full pipeline + server-side scoring |

The response holds the GPS fixes, the predicted route polylines, the classifier-only route, timings and metrics: Top-1 accuracy, route continuity and route precision/recall/F1. **The reference route is never sent to the UI.**

## 5. Leakage controls
* Partitions are split by trip. CV uses `GroupKFold` grouped by trip, so no trip is ever split across folds.
* Scalers are fitted inside each training fold (sklearn `Pipeline`).
* Hyperparameters (Optuna) use only the 80 train trips. The decoder (α, β) uses out-of-fold probabilities on the 80 train trips. student_test is used only for model comparison.
* The prediction pipeline gets the trip with its `.route` removed (`pipeline.predict_trip`). The route is used only afterwards, for server-side scoring.
* No predictions are hardcoded. Every request runs the full pipeline.

## AI assistance disclosure
Parts of the code were written with an AI coding assistant (Claude). All experiments were run by the team on the provided data, and every number in the report comes from the scripts and CSV files in this repository.
