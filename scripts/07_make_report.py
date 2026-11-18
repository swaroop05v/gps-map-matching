"""Build report/PathSnap_Report.html (+ PDF via headless Edge/Chrome) from the result files.

Every table is read from results/*.csv and every figure from results/figures, so the report
always matches the repository. Narrative text lives in report/report_text.md-style
sections inside this script; numbers in the text are pulled from the CSVs.
"""
import base64
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from pathsnap.config import load_config

cfg = load_config()
ROOT = os.path.dirname(cfg["paths"]["results"])
RES = cfg["paths"]["results"]
FIG = os.path.join(RES, "figures")
REP = os.path.join(ROOT, "report")
SHOTS = os.path.join(REP, "screenshots")


def img(name, caption, width=100):
    for base in (FIG, SHOTS):
        p = os.path.join(base, name)
        if os.path.exists(p):
            b64 = base64.b64encode(open(p, "rb").read()).decode()
            ext = name.rsplit(".", 1)[-1]
            return (f'<figure style="width:{width}%"><img src="data:image/{ext};base64,{b64}"/>'
                    f"<figcaption>{caption}</figcaption></figure>")
    return f'<p class="missing">[missing figure: {name}]</p>'


def table(df, floatfmt=4):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    return df.to_html(index=False, escape=False, border=0, classes="tbl")


def csv(name):
    return pd.read_csv(os.path.join(RES, name))


CSS = """
body{font-family:Georgia,'Times New Roman',serif;max-width:1000px;margin:24px auto;color:#111;line-height:1.45;font-size:14px}
h1{font-size:26px;margin-bottom:0} h2{border-bottom:2px solid #333;padding-bottom:3px;margin-top:34px;font-size:20px}
h3{font-size:16px;margin-top:22px} h4{font-size:14px;margin:14px 0 4px}
.tbl{border-collapse:collapse;font-size:11.5px;margin:8px 0;width:100%}
.tbl th,.tbl td{border:1px solid #999;padding:3px 5px;text-align:left;vertical-align:top}
.tbl th{background:#eee;word-break:break-word;font-size:10.5px}
figure{margin:10px auto;text-align:center} figure img{max-width:100%;border:1px solid #ddd}
figcaption{font-size:12px;color:#444;margin-top:3px}
.note{background:#f6f6f6;border-left:4px solid #888;padding:6px 10px;font-size:13px}
.missing{color:#b00}
@media print{h2{page-break-before:auto} figure,table{page-break-inside:avoid}}
"""


def to_pdf(html_path, pdf_path):
    for exe in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                shutil.which("msedge") or "", shutil.which("chrome") or "", shutil.which("google-chrome") or ""]:
        if exe and os.path.exists(exe):
            subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                            f"--print-to-pdf={pdf_path}", "file:///" + html_path.replace("\\", "/")],
                           check=False, timeout=180, capture_output=True)
            if os.path.exists(pdf_path):
                return True
    return False


MODEL_TEXT = {
    "logreg": dict(
        why="A linear model on well-designed geometric features is the natural baseline: if distance, heading agreement and context distance separate correct from wrong candidates almost linearly, a logistic model is enough. It outputs calibrated probabilities, which the HMM needs as emissions.",
        strengths="Strengths: fast (seconds to train, about 2 ms per trip to score), stable across folds, well-calibrated probabilities (low log loss). Limitations: one linear boundary in feature space. It cannot express interactions such as 'heading agreement matters only when the vehicle is moving' or 'distance matters less at intersections'.",
        hyper="C = 1.0 (inverse L2 strength), lbfgs solver, max_iter = 2000, features standardised inside the pipeline (scaler fitted on training folds only).",
        loss="Binary cross-entropy (logistic loss) with L2 regularisation."),
    "gnb": dict(
        why="Gaussian Naive Bayes is a generative probabilistic baseline. We used it to measure how much the strong conditional-independence assumption hurts, because many of our features are correlated by construction (distance, rank, distance-minus-min, context distance).",
        strengths="Strengths: trains in under a second, needs no tuning, and handles the imbalance through class priors. Limitations: the independence assumption double-counts correlated distance features, so the probabilities are badly over-confident. The class-conditional Gaussians also fit skewed features such as distance and overshoot poorly.",
        hyper="var_smoothing = 1e-9 (default), class priors estimated from the data, features standardised.",
        loss="Maximum likelihood of per-class Gaussian feature distributions (no iterative loss)."),
    "knn": dict(
        why="Instance-based learning checks whether candidates with similar feature vectors (similar distance, heading error, context) share the same label, with no parametric assumptions.",
        strengths="Strengths: non-parametric, so it can capture local non-linear patterns. Limitations: it is sensitive to feature scaling and to irrelevant dimensions in a 24-dimensional space. Prediction is slow because every query searches the stored training set, so we fitted it on a 200k-row random subsample. Its probabilities are coarse (weighted votes of 25 neighbours).",
        hyper="k = 25, distance weighting, Euclidean metric on standardised features, 200 000 training rows (random subsample, seed 42).",
        loss="None (lazy learner). Prediction is a distance-weighted neighbour vote."),
    "rf": dict(
        why="A bagged ensemble of decision trees captures non-linear rules and feature interactions (e.g. heading only informative when speed is high) and is robust to feature scaling and outliers such as the long-gap fixes.",
        strengths="Strengths: strong accuracy, no scaling needed, low variance thanks to bagging, and built-in feature importance. Limitations: a large model on disk and in memory, slower inference than boosting, and probabilities that are averages of leaf frequencies, so they are less sharp than boosting's.",
        hyper="150 trees, max_depth = 16, min_samples_leaf = 5, max_samples = 0.3 (bootstrap fraction per tree), sqrt(features) per split.",
        loss="Each tree greedily minimises Gini impurity. The ensemble averages tree probabilities."),
    "hgb": dict(
        why="Gradient boosting corrects the errors of previous trees sequentially, which suits a tabular problem with a few very informative features and many weak interacting ones. The histogram implementation scales to about 1M rows.",
        strengths="Strengths: high accuracy and good probability quality. Limitations: sequential training is slower than LightGBM in our runs, and it has more hyperparameters to tune.",
        hyper="max_iter = 300, learning_rate = 0.1, max_leaf_nodes = 31, no early stopping.",
        loss="Binary cross-entropy (log loss), optimised by gradient boosting."),
    "lgbm": dict(
        why="LightGBM is a leaf-wise gradient-boosted tree ensemble. It is fast on about 1.2M candidate rows, models feature interactions, and directly optimises log loss, so its probabilities work as HMM emissions.",
        strengths="Strengths: the best accuracy/speed trade-off, good calibration, row and column subsampling for regularisation, and fast inference (milliseconds per trip). Limitations: leaf-wise growth can overfit small data without min_child_samples / regularisation, and it is sensitive to hyperparameters (hence the Bayesian optimisation).",
        hyper="n_estimators = 400, learning_rate = 0.05, num_leaves = 63, min_child_samples = 20, subsample = 0.8, colsample_bytree = 0.8, reg_lambda = 0.",
        loss="Binary cross-entropy (log loss) with gradient boosting (second-order Newton steps)."),
    "mlp": dict(
        why="A feed-forward neural network learns non-linear combinations of the features end-to-end and produces probabilities through a sigmoid output. We used it to test whether a learned representation beats hand-designed tree splits on this tabular data.",
        strengths="Strengths: smooth decision function and well-calibrated outputs under BCE. Limitations: needs feature scaling and careful regularisation, trains slower than boosting on CPU, and is not interpretable. On tabular data with skewed features, trees usually do as well or better.",
        hyper="Hidden layers (128, 64), ReLU, dropout 0.1, Adam (lr = 1e-3, weight decay 1e-5), batch size 2048, at most 20 epochs, early stopping with patience 4 on a validation split of 8 held-out training trips.",
        loss="Binary cross-entropy with logits (BCEWithLogitsLoss)."),
}


# Observations per model, written after reading results/model_comparison.csv
OBS = {}


def fmt(v, d=4):
    return "n/a" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.{d}f}"


def build():
    cmp = csv("model_comparison.csv").set_index("model")
    order = [m for m in ["logreg", "gnb", "knn", "rf", "hgb", "lgbm", "mlp"] if m in cmp.index]
    cmp = cmp.loc[order]
    cand_sum = csv("candidate_generation_summary.csv").set_index("partition")
    eda_part = pd.read_csv(os.path.join(RES, "eda", "partition_summary.csv")).set_index("partition")
    miss = pd.read_csv(os.path.join(RES, "eda", "missing_invalid_summary.csv"))
    inv = pd.read_csv(os.path.join(RES, "eda", "trip_inventory.csv"), dtype={"trip_id": str})
    noise = pd.read_csv(os.path.join(RES, "eda", "gps_noise_quantiles_train.csv"), index_col=0)["align_dist_m"]
    unrel = pd.read_csv(os.path.join(RES, "eda", "unreliable_label_pct_per_trip.csv"), dtype={"trip_id": str})
    tvu = csv("tuned_vs_untuned.csv")
    best = json.load(open(os.path.join(RES, "optuna_best_params.json")))
    trials = csv("optuna_trials.csv")
    hmm_tr = pd.read_csv(os.path.join(RES, "hmm_train_grid_summary.csv"), index_col=0)
    hmm_te = pd.read_csv(os.path.join(RES, "hmm_student_test_summary.csv"), index_col=0)
    per_trip = pd.read_csv(os.path.join(RES, "hmm_student_test_per_trip_top1.csv"), dtype={"trip_id": str})
    hcfg = json.load(open(os.path.join(cfg["paths"]["models"], "hmm_config.json")))
    tr, te = eda_part.loc["train"], eda_part.loc["student_test"]
    cs_tr, cs_te = cand_sum.loc["train"], cand_sum.loc["student_test"]
    H = []
    a = H.append

    a(f"""<h1>PathSnap: Distributed Map-Matching &amp; Route Reconstruction</h1>
<p><b>Technical Report</b> · UE24CS352A Machine Learning · PES University<br>
<b>Team:</b> Swaroop (PES1UG24CS923) · Shashank D (PES1UG24CS433) · Shashank Keshava Murthy (PES1UG24CS434) · Manoj AC (PES1UG25CS826)<br>
<b>Repository:</b> {os.environ.get('PATHSNAP_REPO', 'https://github.com/swaroop05v/gps-map-matching')}</p>
<div class="note"><b>Summary.</b> For every GPS fix we generate candidate road arcs within 50 m, score them with a
classifier, and decode a connected route with an HMM/Viterbi decoder over the road graph. We compared seven
classifiers from six algorithm families with 5-fold trip-grouped cross-validation on the 80 training trips.
We selected LightGBM, tuned it with Bayesian optimisation (Optuna TPE) and used its probabilities as HMM emissions.
On the 10 student-testing trips, observation-level Top-1 accuracy goes from
{fmt(hmm_te.loc['nearest_arc','top1_pooled'])} (nearest arc) to {fmt(hmm_te.loc['classifier_only','top1_pooled'])}
(tuned classifier only) to <b>{fmt(hmm_te.loc['hmm_classifier_emission','top1_pooled'])}</b> (classifier + HMM), and route
continuity goes from {fmt(hmm_te.loc['classifier_only','route_continuity_pooled'])} to
{fmt(hmm_te.loc['hmm_classifier_emission','route_continuity_pooled'])}. Every number in this report is produced by the scripts
in the repository and stored in <code>results/</code>.</div>""")

    # ------------------------------------------------------------------ 2.2
    a("<h2>2.2 Dataset Understanding and Exploratory Data Analysis</h2>")
    a(f"""<h3>Dataset organisation</h3>
<p>Each trip folder holds four headerless, whitespace-separated files. <code>.track</code> has
longitude, latitude and timestamp in seconds. <code>.nodes</code> has longitude and latitude, where the row index is the node id.
<code>.arcs</code> has from-node and to-node, where the row index is the arc id. <code>.route</code> lists the arc ids
travelled, in order. <b>Each trip comes with its own road sub-network</b>, so arc ids are only meaningful inside a trip.
A route row does <i>not</i> correspond to a GPS row: trip 00000001 has 1,315 fixes but 525 route arcs.
We therefore had to build per-fix labels ourselves (see Q1 below).
Partitions: <b>train</b> = {int(tr.trips)} trips / {int(tr.gps_points):,} fixes,
<b>student_test</b> = {int(te.trips)} trips / {int(te.gps_points):,} fixes. The 10 evaluator trips are not in our data.
They are read only by the API at demo time.</p>""")
    a(table(eda_part.reset_index()[["partition", "trips", "gps_points", "mean_gps", "min_gps", "max_gps", "mean_nodes",
                                    "mean_arcs", "mean_route_arcs", "trips_with_gaps_gt_60s", "mean_pct_bidirectional", "route_breaks"]], 1))
    a("<p><i>Table 1. Dataset inventory per partition (results/eda/partition_summary.csv). Per-trip detail, including file sizes, "
      "is in results/eda/trip_inventory.csv.</i></p>")

    a("<h3>Missing-value and validity analysis (train + student_test)</h3>")
    a(table(miss, 0))
    a(f"""<p>There are <b>no missing (NaN) values</b> and no out-of-range coordinates in any file. Timestamps are strictly increasing in every trip.
Only 6 consecutive duplicate positions occur in train. The real data-quality problems are elsewhere:
(i) <b>time gaps</b>: {int(tr.trips_with_gaps_gt_60s)} train trips contain gaps over 60 s. The largest is 153,064 s (42 h, trip 00000090),
and in the worst trip only {inv.pct_dt_1s.min():.1f}% of steps are exactly 1 s (median {inv.pct_dt_1s.median():.1f}%).
(ii) <b>off-route GPS</b>: in trips {unrel.iloc[0,0]} and {unrel.iloc[1,0]}, {unrel.iloc[0,1]:.0f}% and {unrel.iloc[1,1]:.0f}% of fixes lie more than 50 m from the
reference route, so the supplied route does not describe those fixes. (iii) <b>Reference-route breaks</b>: {int(inv.route_breaks.sum())} consecutive
route arc pairs ({int((inv.route_breaks>0).sum())} trips) are not connected (to(a) ≠ from(b)), so the reference itself has small errors.</p>""")
    a(img("eda_missing_invalid.png", "Figure 1. Missing/invalid observation counts by type (train and student_test).", 70))
    a(img("eda_sampling.png", "Figure 2. Sampling interval, distance between consecutive fixes and fixes per trip (train)."))
    a(img("eda_gaps.png", "Figure 3. Trips with the largest time gaps (train and student_test)."))

    a("<h3>GPS trajectories and road networks</h3>")
    a(f"""<p>The trips are spread worldwide (Figure 4: Europe, North America, Asia, South America, Africa), so no single projected
coordinate system fits all of them. We use a <b>local equirectangular projection centred on each trip</b>
(x = R·Δlon·cos φ₀, y = R·Δlat). Checked against the haversine distance for every consecutive pair of fixes, the maximum error is
{eda_part.max_proj_err_m.max():.2f} m, far below GPS noise. Road networks range from {int(inv.n_nodes.min())} to {int(inv.n_nodes.max()):,} nodes. Their mean
out-degree is {inv.mean_out_degree.mean():.2f}, and on average {inv.pct_bidirectional_arcs.mean():.1f}% of arcs have a reverse twin (a two-way street is
stored as two arcs with identical geometry). Networks are clipped regions, so they have several weak components (median {inv.weak_components.median():.0f}),
but the largest strongly connected component holds {100*inv.largest_scc_frac.mean():.0f}% of nodes on average. No isolated nodes exist.</p>""")
    a(img("eda_trip_centroids.png", "Figure 4. Trip centroids (train and student_test).", 55))
    a(img("eda_trajectories_network.png", "Figure 5. GPS fixes (blue), reference route (green) and road network (grey) for four train trips."))
    a(img("eda_network.png", "Figure 6. Road-network size, route length vs trajectory length, and connectivity (train and student_test)."))
    a(img("map_eda_train_00000004.png", "Figure 7. Folium map of train trip 00000004: raw GPS fixes, reference route, road network, start/end markers, layer control and legend (results/maps/eda_train_00000004.html)."))

    a("<h3>Q1. How did we preprocess the data?</h3>")
    a(f"""<ol>
<li><b>Cleaning</b> (<code>io.clean_track</code>): drop non-finite or out-of-range rows, sort by time, drop duplicate timestamps. Original row indices are kept.</li>
<li><b>Projection</b> to metres per trip (above). All distances are point-to-segment distances in metres.</li>
<li><b>Gap segmentation</b>: a new segment starts when Δt &gt; {cfg['preprocess']['gap_seconds']:.0f} s or a jump exceeds {cfg['preprocess']['gap_jump_m']:.0f} m. The HMM restarts at segment boundaries.</li>
<li><b>Per-fix reference labels</b> (<code>labels.py</code>): a monotonic dynamic-programming alignment of the fixes onto the ordered route,
D(i,j) = dist(i, route[j]) + min<sub>j′≤j</sub> D(i−1,j′), so fixes can only move forward along the route. This handles loops and
reverse twin arcs, which "nearest route arc" gets wrong. The alignment distance is a GPS-noise estimate: median {noise.loc[0.5]:.1f} m,
95th percentile {noise.loc[0.95]:.0f} m (train). Fixes more than 50 m from their aligned arc ({100-cs_tr.reliable_label_points_pct:.1f}% of train,
{100-cs_te.reliable_label_points_pct:.2f}% of student_test) get an <i>unreliable</i> label. They are excluded from training and scoring,
and kept as input to the decoder.</li>
<li><b>Candidate generation</b> (<code>network.py</code>): arcs are densified every 10 m into a KD-tree. Every arc within
{cfg['candidates']['radius_m']:.0f} m of a fix becomes a candidate (at most {cfg['candidates']['max_k']}). If none exist, the search widens once to 150 m (top 3).
This gives on average {cs_tr.mean_candidates_per_point:.1f} candidates per fix and a <b>candidate miss rate</b> of {cs_tr.candidate_miss_rate_reliable_pct:.2f}% (train) /
{cs_te.candidate_miss_rate_reliable_pct:.2f}% (student_test) on reliable fixes. Only {cs_tr.points_without_candidates_pct:.2f}% of train fixes have no candidate at all.</li>
<li>Learned transformations (StandardScaler for LR/GNB/KNN/MLP) are fitted inside each training fold only.</li></ol>""")
    a(img("eda_gps_noise.png", "Figure 8. Distance from each GPS fix to its aligned reference arc (train), with the 50 m reliability threshold.", 70))

    a("<h3>Q2. Patterns observed and features extracted</h3>")
    a(f"""<p><b>Patterns.</b> (1) The nearest arc is correct for only {cs_tr.nearest_arc_top1_reliable_pct:.1f}% (train) / {cs_te.nearest_arc_top1_reliable_pct:.1f}% (student_test) of fixes.
The main reason is that about 92% of arcs have a reverse twin with identical geometry, so distance alone cannot pick the direction.
(2) Fixes are 1 Hz and about 13 m apart (median), so neighbouring fixes carry strong evidence about which arc the vehicle is on.
(3) Errors cluster at intersections and at the ends of arcs, where a fix projects onto the end node of several arcs.
(4) Heading is undefined when the vehicle is stopped (displacement &lt; 3 m).</p>
<p><b>Features</b> (24 per fix-candidate pair, <code>features.py</code>):</p>""")
    feats = [
        ("Distance", "dist, dist_rank, dist_minus_min, dist_ratio, n_cands", "closeness is the main signal; relative versions make it comparable between dense and sparse areas"),
        ("Direction", "angle_diff, cos_angle (heading over ±2 fixes vs arc bearing), angle_diff_long (±5 fixes), heading_valid", "separates reverse twin arcs and parallel roads; flagged invalid when stopped"),
        ("Projection", "t_proj, overshoot_m, arc_len, log_arc_len", "projections beyond the arc end indicate the vehicle is on the next/previous arc"),
        ("Context", "ctx_mean_dist, ctx_max_dist (fixes i−3..i+3 to the same arc), ctx_nearest_frac, ctx_mean_dist_minus_min", "uses neighbouring fixes to smooth out single noisy fixes"),
        ("Kinematics", "speed, dt_prev, dt_next, turn", "turning and stopping change which features are reliable; gaps reduce context reliability"),
        ("Topology", "out_deg_to, in_deg_from, has_reverse", "intersections and two-way streets are harder"),
    ]
    a(table(pd.DataFrame(feats, columns=["group", "features", "why"])))
    a(img("lgbm_feature_importance.png", "Figure 9. LightGBM (untuned) gain importance (train fit).", 60))
    a(f"<p>{OBS.get('_features', '')}</p>")
    a("""<p><b>Reference-route limitations.</b> We treat the supplied route as reference information, not error-free truth. The
alignment-derived per-fix labels are themselves uncertain near intersections, where a fix is equally close to the end of one arc
and the start of the next. Some routes have breaks or do not cover all fixes. Accuracy figures should be read with this label uncertainty in mind.</p>""")

    # ------------------------------------------------------------------ 3
    a("<h2>3. Model Experimentation and Comparison</h2>")
    a("""<p><b>Task.</b> Binary classification of (fix, candidate arc) pairs (label 1 = the candidate is the fix's reference arc; about 15% positives).
For each fix the highest-scoring candidate is its prediction. <b>Protocol.</b> 5-fold <code>GroupKFold</code> grouped by trip on the 80 train trips
(no trip is ever split across folds), then a refit on all 80 trips and evaluation on the 10 student_test trips. Training uses only reliable fixes (seed 42).
<b>Metrics.</b> <i>Macro F1</i>: pair level, where a pair is predicted positive iff it is its fix's top candidate (no threshold needed); the mean of the
positive and negative class F1. <i>Top-1</i>: observation level, over reliable fixes; a fix whose reference arc is not a candidate counts as wrong.
<i>Log loss</i>: binary cross-entropy of the pair probabilities. <i>Route continuity</i>: share of consecutive predicted-arc changes
a→b with to(a) = from(b). <b>Primary selection metric:</b> CV Top-1 / Macro F1, then log loss, because the selected model's probabilities
feed the HMM, and then inference time, because the model runs in the API.</p>""")
    a("<h3>3.1 Overall model comparison</h3>")
    t = pd.DataFrame({
        "Model": cmp.model_name, "Algorithm family": cmp.family, "Key configuration": cmp.key_config,
        "CV Macro F1 (mean ± std)": [f"{m:.4f} ± {s:.4f}" for m, s in zip(cmp.cv_macro_f1_mean, cmp.cv_macro_f1_std)],
        "CV Top-1": cmp.cv_top1_mean,
        "Student-test Top-1": cmp.test_top1, "Route continuity (student-test, classifier only)": cmp.test_route_continuity,
        "Student-test Macro F1": cmp.test_macro_f1, "Log loss (student-test)": cmp.test_log_loss,
        "Inference time (ms / trip)": cmp.infer_ms_per_trip, "Train time (s)": cmp.train_time_s_full,
    })
    a(table(t))
    a("<p><i>Table 2. results/model_comparison.csv. Route continuity here is for classifier-only predictions (no HMM). Inference time = classifier "
      "scoring of the candidate table, averaged over the 10 student_test trips.</i></p>")
    a(table(pd.DataFrame({"Model": cmp.model_name, "Precision (pos)": cmp.test_precision_pos, "Recall (pos)": cmp.test_recall_pos,
                          "F1 (pos)": cmp.test_f1_pos, "TN": cmp.test_cm_tn, "FP": cmp.test_cm_fp, "FN": cmp.test_cm_fn, "TP": cmp.test_cm_tp})))
    a("<p><i>Table 3. Student-test pair-level precision / recall / F1 of the positive class and confusion-matrix counts.</i></p>")

    a("<h3>3.2 Per-model analysis</h3>")
    for i, m in enumerate(order, 1):
        r = cmp.loc[m]
        tx = MODEL_TEXT[m]
        gap = r.train_top1_20trips - r.cv_top1_mean
        a(f"<h4>Model {i}: {r.model_name} ({r.family})</h4>")
        a(f"""<p><b>A1. Why suitable?</b> {tx['why']}</p><p><b>A2. Strengths and limitations.</b> {tx['strengths']}</p>
<p><b>B5. Hyperparameters (initial values).</b> {tx['hyper']}</p><p><b>B6. Loss / objective.</b> {tx['loss']}</p>
<p><b>C7–C8. Student-testing performance.</b> Top-1 = {r.test_top1:.4f}, Macro F1 = {r.test_macro_f1:.4f}, log loss = {r.test_log_loss:.4f},
positive-class precision/recall = {r.test_precision_pos:.4f}/{r.test_recall_pos:.4f}, classifier-only route continuity = {r.test_route_continuity:.4f},
{r.infer_ms_per_trip:.1f} ms per trip.</p>
<p><b>C9. Over/underfitting.</b> Top-1 on 20 of its own training trips = {r.train_top1_20trips:.4f}, CV (unseen trips) = {r.cv_top1_mean:.4f} ± {r.cv_top1_std:.4f},
student_test = {r.test_top1:.4f} (train − CV gap = {gap*100:+.2f} pp). {OBS.get(m, '')}</p>""")

    a("<h3>3.3 Model visualisations</h3>")
    a(img("models_cv_macro_f1.png", "Figure 10. Cross-validation Macro F1 (mean ± std over 5 trip-grouped folds)."))
    a(img("models_cv_fold_stability.png", "Figure 11. Per-fold Top-1. Stability across folds."))
    a(img("models_test_top1_f1.png", "Figure 12. Student-testing Top-1 accuracy and Macro F1 (classifier only)."))
    a(img("models_log_loss.png", "Figure 13. Log loss (CV and student_test, log scale)."))
    a(img("models_generalisation_gap.png", "Figure 14. Train vs CV vs student-test Top-1, the over/underfitting evidence for C9."))
    a(img("models_time.png", "Figure 15. Training and inference time."))
    a(img("models_confusion_matrices.png", "Figure 16. Confusion matrices on student_test candidate pairs."))
    a(img("mlp_loss_curve.png", "Figure 17. MLP training and validation loss per epoch. The MLP is the only model with an epoch-wise history; no curves are drawn for the other models.", 60))
    a(f"<p><b>Model selection.</b> {OBS.get('_selection', '')}</p>")

    # ------------------------------------------------------------------ 4
    a("<h2>4. Bayesian Hyperparameter Optimisation (LightGBM)</h2>")
    space = best["search_space"]
    sp = pd.DataFrame([(k, v[0], v[1], v[2], best["best_params"].get(k)) for k, v in space.items()],
                      columns=["hyperparameter", "distribution", "low", "high", "best value"])
    ncomp = int((trials.state == "COMPLETE").sum())
    npr = int((trials.state == "PRUNED").sum())
    nstale = int((trials.state == "RUNNING").sum())
    a(f"""<p><b>Q1–Q2. Search space.</b> Eight LightGBM hyperparameters (Table 4) control tree complexity (num_leaves, max_depth, min_child_samples),
learning dynamics (learning_rate, n_estimators) and regularisation (subsample, colsample_bytree, reg_lambda). Scale-type parameters use log-uniform priors.
<b>Q3. Objective.</b> {best['objective']}. Only the 80 training trips are used; student_test is used only after the search, for the comparison below.
<b>Q4. Trials.</b> {ncomp + npr} finished trials of the Tree-structured Parzen Estimator (TPE, seed 42, first 8 random): {ncomp} completed and {npr} pruned early
by a median pruner after a weak first fold.{' ' + str(nstale) + ' stale trial(s) from an interrupted earlier run remain in the study database with state RUNNING and are excluded.' if nstale else ''}
The untuned default configuration is evaluated separately in Table 5. <b>Q5. Best values</b> are shown in Table 4
(best 3-fold CV Macro F1 = {best['best_cv3_macro_f1']:.4f}).</p>""")
    a(table(sp))
    a("<p><i>Table 4. Search space and best configuration (results/optuna_best_params.json; all trials in results/optuna_trials.csv and optuna_study.db).</i></p>")
    a(img("optuna_history.png", "Figure 18. Optimisation history."))
    a(img("optuna_importance.png", "Figure 19. Hyperparameter importance (fANOVA over completed trials).", 65))
    tv = tvu.set_index("model")
    tvt = pd.DataFrame({"model": ["untuned LightGBM", "tuned LightGBM"],
                        "CV Macro F1 (5-fold)": [tv.loc["lgbm_untuned", "cv5_macro_f1"], tv.loc["lgbm_tuned", "cv5_macro_f1"]],
                        "CV Top-1 (5-fold)": [tv.loc["lgbm_untuned", "cv5_top1"], tv.loc["lgbm_tuned", "cv5_top1"]],
                        "CV log loss": [tv.loc["lgbm_untuned", "cv5_log_loss"], tv.loc["lgbm_tuned", "cv5_log_loss"]],
                        "Student-test Top-1": [tv.loc["lgbm_untuned", "test_top1"], tv.loc["lgbm_tuned", "test_top1"]],
                        "Student-test Macro F1": [tv.loc["lgbm_untuned", "test_macro_f1"], tv.loc["lgbm_tuned", "test_macro_f1"]],
                        "Student-test log loss": [tv.loc["lgbm_untuned", "test_log_loss"], tv.loc["lgbm_tuned", "test_log_loss"]],
                        "Inference (ms / trip)": [tv.loc["lgbm_untuned", "infer_ms_per_trip"], tv.loc["lgbm_tuned", "infer_ms_per_trip"]],
                        "Train time (s)": [tv.loc["lgbm_untuned", "train_s"], tv.loc["lgbm_tuned", "train_s"]]})
    a(table(tvt))
    a(f"<p><i>Table 5. Untuned vs tuned on identical partitions (results/tuned_vs_untuned.csv).</i></p><p><b>Q6.</b> {OBS.get('_tuning', '')}</p>")

    # ------------------------------------------------------------------ 5
    a("<h2>5. HMM and Viterbi Route Reconstruction</h2>")
    a(f"""<p><b>Q1. Hidden states.</b> At fix i the states are its candidate arcs (top {hcfg['beam']} by classifier probability), each with the fix's projected
position (offset in metres from the arc start). Candidate ids are the real arc ids of the trip network, so the decoded state maps directly to a road segment.</p>
<p><b>Q2. Emissions.</b> log e<sub>i</sub>(a) = α · log( p<sub>clf</sub>(a|i) / Σ<sub>c</sub> p<sub>clf</sub>(c|i) ), where p<sub>clf</sub> is the
LightGBM probability, renormalised over the fix's candidates, and α weights it against the transitions (α = {hcfg['alpha']}).</p>
<p><b>Q3. Transitions</b> (Newson &amp; Krumm style): log T(a→b) = −| d<sub>route</sub>(a,b) − d<sub>gps</sub>(i−1,i) | / β (β = {hcfg['beta']} m). d<sub>gps</sub> is the
straight-line distance between the two fixes. d<sub>route</sub> is the network distance from the projection on a to the projection on b: on the same
arc it is |offset<sub>b</sub> − offset<sub>a</sub>| (+50 m if it moves backwards more than {hcfg['same_arc_backtrack_tol_m']:.0f} m); otherwise it is
(len<sub>a</sub> − offset<sub>a</sub>) + shortest-path(to(a), from(b)) + offset<sub>b</sub>. Shortest paths are computed with
bounded Dijkstra (limit {hcfg['sp_limit_m']:.0f} m), only for the node pairs needed, all at once per trip. So a transition is likely when the
road distance matches the distance the GPS moved.</p>
<p><b>Viterbi recurrence.</b> δ<sub>i</sub>(b) = log e<sub>i</sub>(b) + max<sub>a</sub> [δ<sub>i−1</sub>(a) + log T(a→b)], with back-pointers, then
backtracking from argmax δ<sub>N</sub>. Everything is in log space and vectorised over (fix, previous state, state).</p>
<p><b>Q4. Missing candidates and gaps.</b> A fix with no candidate within 50 m gets a widened 150 m search. If there is still none, it is skipped by
the decoder and inherits the previous decoded arc. If Δt &gt; 60 s or the jump is &gt; 800 m, the transition becomes uniform (log T = 0), which restarts the
decoder: we do not force a connection we cannot observe.</p>
<p><b>Q5. Disconnected consecutive segments.</b> If no path exists within the Dijkstra limit, the transition gets a floor log-probability of
{hcfg['unreachable_logp']} instead of −∞. Decoding never breaks, but a disconnected jump is chosen only when the emissions demand it overwhelmingly.
After decoding, consecutive matched arcs that do not touch are bridged with the shortest network path (<code>hmm.build_route</code>) to output
a continuous route. Any jump that cannot be bridged is reported as an unbridged break.</p>
<p><b>Decoder tuning</b> (training data only). α ∈ {{0.5, 1, 2}} × β ∈ {{3, 5, 10, 20, 40}} was grid-searched with <i>out-of-fold</i> LightGBM
probabilities on all 80 train trips: each trip is scored by the fold model that did not see it. Selection is by pooled Top-1. Best: {hcfg['selected_from']}.</p>""")
    a(img("hmm_grid_train.png", "Figure 20. Decoder grid search: pooled OOF Top-1 on the 80 train trips.", 70))
    a("<p><b>Q6. Did HMM decoding improve the route?</b></p>")
    order_h = ["nearest_arc", "classifier_only", "hmm_distance_emission", "hmm_classifier_emission"]
    names_h = ["Nearest arc (no ML)", "Classifier only (tuned LightGBM)", "HMM, distance-only emission", "Classifier + HMM/Viterbi (final)"]
    ht = hmm_te.loc[order_h].reset_index(drop=True)
    ht.insert(0, "method", names_h)
    a(table(ht[["method", "top1_pooled", "top1_mean_trip", "route_continuity_pooled", "route_precision", "route_recall", "route_f1",
                "trips_top1_ge_95pct", "unbridged_breaks"]]))
    a("<p><i>Table 6. Student_test (10 trips). Top-1 pooled over fixes and averaged per trip; route precision/recall/F1 compare the set of arcs "
      "in the reconstructed (bridged) route with the reference route; trips_top1_ge_95pct is a sequence-level measure.</i></p>")
    htr = hmm_tr.loc[order_h[:3] + [hcfg['selected_from']]].reset_index(drop=True)
    htr.insert(0, "method", names_h[:3] + [f"Classifier + HMM ({hcfg['selected_from']})"])
    a(table(htr[["method", "top1_pooled", "route_continuity_pooled", "route_f1", "trips_top1_ge_95pct"]]))
    a("<p><i>Table 7. Same comparison on the 80 train trips with out-of-fold probabilities.</i></p>")
    a(img("hmm_vs_classifier_test.png", "Figure 21. Classifier-only vs classifier + HMM on student_test."))
    a(img("hmm_delta_per_trip.png", "Figure 22. Per-trip change in Top-1 from HMM decoding (student_test).", 75))
    a(table(per_trip[["trip_id", "nearest_arc", "classifier_only", "hmm_distance_emission", "hmm_classifier_emission", "delta_hmm_minus_clf"]]))
    a(f"<p><i>Table 8. Per-trip Top-1 on student_test.</i></p><p>{OBS.get('_hmm', '')}</p>")

    # ------------------------------------------------------------------ 6
    a("<h2>6. Geographical Visualisation Using Folium</h2>")
    a("""<p>All maps come from <code>src/pathsnap/viz.py</code> (scripts 01, 05 and 06) and are committed as interactive HTML in
<code>results/maps/</code>. Layers: road network (grey), reference route (green, offline maps only), classifier-only route (orange),
HMM/Viterbi route (red), raw GPS fixes (blue circle markers), start/end markers, a legend and a layer control. The maps are fitted to the trip
extent, or zoomed to an area where the methods disagree.</p>""")
    for f, cap in OBS.get("_maps", []):
        a(img(f, cap))

    # ------------------------------------------------------------------ 7
    a("<h2>7. API, UI and Deployment</h2>")
    a("""<p><b>Architecture.</b> Laptop 1 runs the FastAPI service (<code>uvicorn api.main:app --host 0.0.0.0 --port 8000</code>). It holds the
final model (<code>models/final_model.joblib</code>), the decoder configuration (<code>models/hmm_config.json</code>), the road networks and the
evaluator reference routes. Laptop 2 runs the Streamlit UI (<code>streamlit run ui/app.py</code>), which only makes HTTP calls to the API.
Endpoints: <code>GET /health</code>, <code>GET /trips?partition=eval_test</code> (trip ids only) and <code>POST /predict {trip_id, partition}</code>.
For each request the API loads the trip, <b>removes its route before prediction</b> (<code>pipeline.predict_trip</code>), runs candidate generation →
features → LightGBM → Viterbi → route bridging, and only then scores against its own reference. The response holds the GPS fixes, the predicted
route geometry and the metrics. The reference route never leaves the server. Every request is logged to the console and to
<code>results/logs/api_requests.log</code> (client IP, trip, number of fixes, accuracy, latency).</p>""")
    for f, cap in OBS.get("_shots", []):
        a(img(f, cap))
    logp = os.path.join(RES, "logs", "api_requests.log")
    if os.path.exists(logp):
        lines = [l for l in open(logp, encoding="utf-8", errors="ignore").read().splitlines() if "PREDICT" in l][-10:]
        a("<p><b>API log excerpt</b> (results/logs/api_requests.log, written by the API process on the receiving machine):</p>"
          "<pre style='font-size:9.5px;white-space:pre-wrap'>" + "\n".join(lines) + "</pre>")

    a("<h2>8. Reproducibility, Integrity and AI-use Disclosure</h2>")
    a("""<p>All numbers come from <code>results/*.csv</code>, produced by <code>scripts/01–06</code> (seed 42, logs in <code>results/logs/</code>).
The evaluator partition is never read by any training, tuning or analysis script. Hyperparameters and decoder parameters were chosen on
train data only, and student_test was used only for comparison. No predictions are hardcoded. <b>AI assistance:</b> the team used an AI coding
assistant (Claude) to help write code and draft this report. All experiments were run by the team on the provided data, and the team has reviewed
and can explain every component.</p>""")
    return "\n".join(H)


def main():
    os.makedirs(REP, exist_ok=True)
    # team-written interpretation of the results (kept as data so the text sits next to the numbers it refers to)
    for fn in ("observations_models.json", "observations.json"):
        obs_path = os.path.join(REP, fn)
        if os.path.exists(obs_path):
            OBS.update(json.load(open(obs_path, encoding="utf-8")))
    body = build()
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>PathSnap Technical Report</title><style>{CSS}</style></head><body>{body}</body></html>"
    hp = os.path.join(REP, "PathSnap_Report.html")
    open(hp, "w", encoding="utf-8").write(html)
    ok = to_pdf(hp, os.path.join(REP, "PathSnap_Report.pdf"))
    print("html", hp, "pdf", ok)


if __name__ == "__main__":
    main()
