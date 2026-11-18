"""Build professional academic report for PathSnap (HTML + PDF).
Uses uniform Times New Roman typography, publication-grade booktabs tables,
formal academic title & author block (no metadata tables/cards), and clean figure layouts
while keeping 100% of the analytical content, tables, and high-resolution screenshots intact.
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap.config import load_config

cfg = load_config()
ROOT = os.path.dirname(cfg["paths"]["results"])
RES = cfg["paths"]["results"]
FIG = os.path.join(RES, "figures")
REP = os.path.join(ROOT, "report")
SHOTS = os.path.join(REP, "screenshots")
SHOTS_GTA = os.path.join(SHOTS, "gta")

def find_image(name):
    # Prefer GTA-themed high-res screenshots first, then standard screenshots, then figures
    for base in (SHOTS_GTA, SHOTS, FIG):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return None

def img(name, caption, width=100):
    p = find_image(name)
    if p:
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        ext = name.rsplit(".", 1)[-1].lower()
        if ext == "jpg": ext = "jpeg"
        
        cap_html = caption
        if caption.startswith("Figure "):
            parts = caption.split(".", 1)
            if len(parts) == 2:
                cap_html = f"<strong>{parts[0]}.</strong> <em>{parts[1].strip()}</em>"
                
        return (f'<figure style="width:{width}%; margin: 16px auto;">'
                f'<img src="data:image/{ext};base64,{b64}" style="max-width:100%; border: 0.5pt solid #bbb; border-radius: 2px;" />'
                f'<figcaption>{cap_html}</figcaption></figure>')
    return f'<p class="missing">[Missing figure: {name}]</p>'

def csv(name):
    return pd.read_csv(os.path.join(RES, name))

def fmt_val(v, d=4):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "N/A"
    if isinstance(v, (int, pd.Int64Dtype)):
        return f"{v:,}"
    if isinstance(v, float):
        if v.is_integer() and d == 0:
            return f"{int(v):,}"
        return f"{v:.{d}f}"
    return str(v)

def render_booktabs_table(df, col_names=None, num_cols=None, float_digits=4, caption=None, width="100%"):
    df_copy = df.copy()
    if col_names:
        df_copy.columns = col_names
    
    headers = list(df_copy.columns)
    num_set = set(num_cols or [])
    
    th_html = []
    for h in headers:
        cls = ' class="th-num"' if h in num_set else ''
        th_html.append(f'<th{cls}>{h}</th>')
    
    rows_html = []
    for _, row in df_copy.iterrows():
        td_html = []
        for h in headers:
            v = row[h]
            cls = ' class="td-num"' if h in num_set else ''
            if pd.api.types.is_float(v):
                formatted = fmt_val(v, float_digits)
            elif isinstance(v, (int, float)) and pd.notna(v) and float(v).is_integer():
                formatted = fmt_val(int(v), 0)
            else:
                formatted = str(v) if pd.notna(v) else ""
            td_html.append(f'<td{cls}>{formatted}</td>')
        rows_html.append(f'<tr>{"".join(td_html)}</tr>')
    
    cap_html = f'<div class="tbl-caption"><em>{caption}</em></div>' if caption else ""
    return (f'<div class="table-wrapper" style="width:{width};">'
            f'<table class="tbl">'
            f'<thead><tr>{"".join(th_html)}</tr></thead>'
            f'<tbody>{"".join(rows_html)}</tbody>'
            f'</table>{cap_html}</div>')

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
        hyper="k = 25, distance weighting, Euclidean metric on standardised features, 200,000 training rows (random subsample, seed 42).",
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

OBS = {}

def build_document():
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

    # --- Title & Academic Header (NO tables, pure Times New Roman) ---
    repo_url = os.environ.get('PATHSNAP_REPO', 'https://github.com/swaroop05v/gps-map-matching')
    a(f"""<header class="header-container">
    <h1 class="paper-title">PathSnap: Distributed Map-Matching &amp; Route Reconstruction</h1>
    <div class="paper-subtitle">Technical Report &bull; UE24CS352A Machine Learning</div>
    <div class="paper-institution">Department of Computer Science and Engineering, PES University, Bengaluru</div>
    <div class="author-block">
        <span class="author">Swaroop <span class="srn">(PES1UG24CS923)</span></span> &bull; 
        <span class="author">Shashank D <span class="srn">(PES1UG24CS433)</span></span> &bull; 
        <span class="author">Shashank Keshava Murthy <span class="srn">(PES1UG24CS434)</span></span> &bull; 
        <span class="author">Manoj AC <span class="srn">(PES1UG25CS826)</span></span>
    </div>
    <div class="repo-line">Public Code Repository: <a href="{repo_url}">{repo_url}</a></div>
</header>

<div class="abstract-container">
    <p><span class="abstract-heading">Abstract</span>&mdash;For every GPS fix we generate candidate road arcs within 50 m, score them with a candidate classifier, and decode a globally continuous, kinematically consistent route using a Hidden Markov Model (HMM) solved via Viterbi decoding over the road graph. We systematically evaluated seven distinct classification models spanning six algorithmic families under rigorous 5-fold trip-grouped cross-validation on the 80 training trips (208,417 fixes). LightGBM was selected as the optimal candidate classifier, tuned via Bayesian optimisation (Optuna TPE), and deployed to produce calibrated emission probabilities for the HMM decoder. On the 10 student-testing trips (19,160 fixes), observation-level Top-1 accuracy rises from {fmt_val(hmm_te.loc['nearest_arc','top1_pooled'], 4)} for the geometric nearest-arc baseline to {fmt_val(hmm_te.loc['classifier_only','top1_pooled'], 4)} with the tuned classifier alone, reaching <strong>{fmt_val(hmm_te.loc['hmm_classifier_emission','top1_pooled'], 4)}</strong> upon HMM/Viterbi sequence decoding. Route continuity simultaneously increases from {fmt_val(hmm_te.loc['classifier_only','route_continuity_pooled'], 4)} to {fmt_val(hmm_te.loc['hmm_classifier_emission','route_continuity_pooled'], 4)}, and route F1 improves from 0.8070 to 0.9782. All reported figures and tables are strictly derived from automated experiment pipelines committed within the repository.</p>
</div>""")

    # ------------------------------------------------------------------ 2.2
    a("<h2>2.2 Dataset Understanding and Exploratory Data Analysis</h2>")
    a(f"""<h3>Dataset organisation</h3>
<p>Each trip archive contains four headerless, whitespace-separated text files: <code>.track</code> provides raw GPS fixes (longitude, latitude, timestamp in seconds); <code>.nodes</code> defines graph vertices (longitude, latitude, indexed by node ID); <code>.arcs</code> specifies directed graph edges (source node ID, target node ID, indexed by arc ID); and <code>.route</code> provides the ordered sequence of traversed ground-truth arc IDs. Crucially, <strong>each trip possesses an isolated, clipped road sub-network</strong>, meaning arc IDs are uniquely scoped to individual trips and cannot be transferred across trips. Furthermore, route rows do not map one-to-one to GPS fixes: for example, trip <code>00000001</code> contains 1,315 GPS fixes but only 525 traversed route arcs. Consequently, supervised pointwise training labels were constructed using dynamic programming alignment (detailed in Q1 below).</p>
<p>Dataset partitions are divided as: <strong>train</strong> = {int(tr.trips)} trips ({int(tr.gps_points):,} GPS fixes) and <strong>student_test</strong> = {int(te.trips)} trips ({int(te.gps_points):,} GPS fixes). The 10 reserved evaluator trips are isolated and evaluated strictly via the real-time API at evaluation time.</p>""")
    
    t1_df = eda_part.reset_index()[["partition", "trips", "gps_points", "mean_gps", "min_gps", "max_gps", "mean_nodes",
                                    "mean_arcs", "mean_route_arcs", "trips_with_gaps_gt_60s", "mean_pct_bidirectional", "route_breaks"]]
    t1_cols = ["Partition", "Trips", "GPS Points", "Mean Fixes", "Min Fixes", "Max Fixes", "Mean Nodes",
               "Mean Arcs", "Mean Route Arcs", "Gaps > 60s", "% Two-Way Arcs", "Route Breaks"]
    t1_nums = ["Trips", "GPS Points", "Mean Fixes", "Min Fixes", "Max Fixes", "Mean Nodes",
               "Mean Arcs", "Mean Route Arcs", "Gaps > 60s", "% Two-Way Arcs", "Route Breaks"]
    a(render_booktabs_table(t1_df, t1_cols, t1_nums, 1, "Table 1. Dataset inventory per partition (results/eda/partition_summary.csv). Detailed per-trip metrics are catalogued in results/eda/trip_inventory.csv."))

    a("<h3>Missing-value and validity analysis (train + student_test)</h3>")
    t2_cols = ["Partition", "NaN Values", "Invalid Coords", "Non-Increasing Time", "Duplicate Positions", "Gaps > 60s"]
    t2_nums = ["NaN Values", "Invalid Coords", "Non-Increasing Time", "Duplicate Positions", "Gaps > 60s"]
    a(render_booktabs_table(miss, t2_cols, t2_nums, 0, "Table 2. Summary of missing, non-finite, and validity checks across partitions."))
    
    a(f"""<p>Audit checks confirm that there are <strong>zero missing (NaN) values</strong>, non-finite entries, or out-of-range geographical coordinates across all data partitions. Timestamps are strictly monotonic and increasing across every trajectory. Only 6 consecutive duplicate positional fixes occur across the entire training corpus. The genuine data quality challenges identified during exploratory analysis consist of:</p>
<ul>
    <li><strong>Temporal gaps:</strong> {int(tr.trips_with_gaps_gt_60s)} training trips contain inter-fix intervals exceeding 60 s. The maximum observed gap spans 153,064 s (42.5 hours in trip <code>00000090</code>). While median sampling regularity is 99.1% at 1 Hz, the worst-case trip exhibits only {inv.pct_dt_1s.min():.1f}% 1 Hz sampling regularity.</li>
    <li><strong>Off-route excursions:</strong> In trips <code>{unrel.iloc[0,0]}</code> and <code>{unrel.iloc[1,0]}</code>, {unrel.iloc[0,1]:.0f}% and {unrel.iloc[1,1]:.0f}% of GPS fixes respectively deviate by more than 50 m from the nearest reference route segment, representing unmapped off-network travel.</li>
    <li><strong>Reference-route topological breaks:</strong> {int(inv.route_breaks.sum())} consecutive arc pairs across {int((inv.route_breaks>0).sum())} trips fail graph connectivity (i.e. to-node of arc <em>a</em> &ne; from-node of arc <em>b</em>), indicating minor reference annotation noise in the source data.</li>
</ul>""")
    a(img("eda_missing_invalid.png", "Figure 1. Missing/invalid observation counts by type (train and student_test).", 65))
    a(img("eda_sampling.png", "Figure 2. Sampling interval, distance between consecutive fixes and fixes per trip (train)."))
    a(img("eda_gaps.png", "Figure 3. Trips with the largest time gaps (train and student_test)."))

    a("<h3>GPS trajectories and road networks</h3>")
    a(f"""<p>Trip trajectories span diverse global regions (Figure 4: Europe, North America, Asia, South America, and Africa); consequently, no uniform UTM projected coordinate reference system is globally applicable. We implement a local equirectangular metric projection centered dynamically on the bounding centroid of each trip (<em>x</em> = <em>R</em> &middot; &Delta;lon &middot; cos &phi;<sub>0</sub>, <em>y</em> = <em>R</em> &middot; &Delta;lat). Validated against exact ellipsoidal haversine geodesic distances across all consecutive fix pairs, the maximum distortion is strictly {eda_part.max_proj_err_m.max():.2f} m&mdash;negligible relative to standard commercial GPS measurement noise.</p>
<p>Sub-network graph scales range from {int(inv.n_nodes.min())} to {int(inv.n_nodes.max()):,} vertices. The mean node out-degree is {inv.mean_out_degree.mean():.2f}, and an average of {inv.pct_bidirectional_arcs.mean():.1f}% of arcs possess an opposing reverse twin (representing bidirectional two-way corridors stored as dual antiparallel edges). Sub-networks are spatially clipped bounding boxes, exhibiting a median of {inv.weak_components.median():.0f} weakly connected components; however, the primary strongly connected component encompasses {100*inv.largest_scc_frac.mean():.0f}% of total network vertices on average, with zero isolated sink/source vertices.</p>""")
    a(img("eda_trip_centroids.png", "Figure 4. Trip centroids across continental regions (train and student_test).", 55))
    a(img("eda_trajectories_network.png", "Figure 5. GPS fixes (blue), reference route (green) and road network (grey) for four representative training trips."))
    a(img("eda_network.png", "Figure 6. Road-network size, route length vs trajectory length, and connectivity distributions (train and student_test)."))
    a(img("gta/map_eda_train_00000004.png", "Figure 7. Training trip 00000004 in Folium. The map shows raw GPS fixes, the reference route, the road network, start and end markers, a legend, and layer controls."))

    a("<h3>Q1. How did we preprocess the data?</h3>")
    a(f"""<p>Data preprocessing is structured into six modular, deterministic pipeline stages:</p>
<ol>
    <li><strong>Integrity Cleaning (<code>io.clean_track</code>):</strong> Trajectory records with non-finite or out-of-range coordinates are discarded; records are chronologically sorted and duplicate timestamps removed while preserving raw index alignment.</li>
    <li><strong>Metric Projection:</strong> Coordinates are mapped to Cartesian metric offsets centered on the trip centroid, allowing computationally efficient point-to-segment geometric distance calculations.</li>
    <li><strong>Temporal Gap Segmentation:</strong> Trajectories are partitioned into disjoint segments whenever consecutive fixes exhibit &Delta;<em>t</em> &gt; {cfg['preprocess']['gap_seconds']:.0f} s or spatial displacement &gt; {cfg['preprocess']['gap_jump_m']:.0f} m. The HMM decoder restarts transition dynamics across segment boundaries.</li>
    <li><strong>Monotonic DP Reference Labeling (<code>labels.py</code>):</strong> Per-fix supervision is constructed via dynamic programming alignment against the ordered ground-truth route sequence: <em>D</em>(<em>i</em>, <em>j</em>) = dist(<em>i</em>, route[<em>j</em>]) + min<sub><em>j</em>'&le;<em>j</em></sub> <em>D</em>(<em>i</em>&minus;1, <em>j</em>'). This enforces forward progression, correctly resolving loops and bidirectional twin segments that confound naive nearest-arc lookups. The alignment distance provides an empirical GPS noise estimate: median {noise.loc[0.5]:.1f} m, 95th percentile {noise.loc[0.95]:.0f} m (train). Fixes with alignment distance &gt; 50 m ({100-cs_tr.reliable_label_points_pct:.1f}% in train, {100-cs_te.reliable_label_points_pct:.2f}% in student_test) are flagged as <em>unreliable</em>, excluded from supervised training/scoring, but retained for decoding.</li>
    <li><strong>Candidate Arc Generation (<code>network.py</code>):</strong> Network arcs are spatially densified at 10 m intervals into a <em>k</em>-d tree. For each fix, all candidate arcs within {cfg['candidates']['radius_m']:.0f} m are extracted (capped at {cfg['candidates']['max_k']}). If null, an adaptive fallback expands the radius to 150 m (top 3). This achieves an average of {cs_tr.mean_candidates_per_point:.1f} candidates per fix with an arc miss rate of merely {cs_tr.candidate_miss_rate_reliable_pct:.2f}% (train) and {cs_te.candidate_miss_rate_reliable_pct:.2f}% (student_test). Only {cs_tr.points_without_candidates_pct:.2f}% of fixes have zero candidates.</li>
    <li><strong>Fold-Isolated Transformations:</strong> Feature standardisation (<code>StandardScaler</code> for linear, naive Bayes, KNN, and MLP models) is fit strictly within training folds to eliminate data leakage.</li>
</ol>""")
    a(img("eda_gps_noise.png", "Figure 8. Distance from each GPS fix to its aligned reference arc (train), with the 50 m reliability threshold.", 65))

    a("<h3>Q2. Patterns observed and features extracted</h3>")
    a(f"""<p><strong>Observed Patterns:</strong></p>
<ul>
    <li><em>Nearest Arc Insufficiency:</em> The spatially nearest arc matches the ground-truth arc for only {cs_tr.nearest_arc_top1_reliable_pct:.1f}% (train) and {cs_te.nearest_arc_top1_reliable_pct:.1f}% (student_test) of fixes. Because {inv.pct_bidirectional_arcs.mean():.1f}% of roads possess bidirectional twins with identical geometry, spatial proximity alone fails to distinguish heading direction.</li>
    <li><em>Temporal Fix Correlation:</em> Sampling is predominantly 1 Hz (~13 m median displacement); sequential fixes therefore provide strong spatial context to resolve ambiguity.</li>
    <li><em>Intersection Ambiguity:</em> Prediction errors heavily concentrate at road intersections and arc boundaries, where a fix projects identically onto convergent end nodes.</li>
    <li><em>Degenerate Heading at Rest:</em> Fixes with displacement &lt; 3 m yield noisy, degenerate velocity azimuths.</li>
</ul>
<p><strong>Feature Engineering (24 features per candidate pair, <code>features.py</code>):</strong></p>""")
    
    feats = [
        ("Distance", "dist, dist_rank, dist_minus_min, dist_ratio, n_cands", "Measures spatial proximity; relative variants ensure scale invariance across dense urban and sparse rural corridors."),
        ("Direction", "angle_diff, cos_angle (&plusmn;2 fixes vs arc bearing), angle_diff_long (&plusmn;5 fixes), heading_valid", "Differentiates opposing bidirectional lanes and parallel corridors; flagged invalid when stationary."),
        ("Projection", "t_proj, overshoot_m, arc_len, log_arc_len", "Determines longitudinal position along edge; overshoot flags edge transitions."),
        ("Context", "ctx_mean_dist, ctx_max_dist (&plusmn;3 fixes to candidate arc), ctx_nearest_frac, ctx_mean_dist_minus_min", "Aggregates multi-step trajectory geometry to filter out single-point GNSS multi-path spikes."),
        ("Kinematics", "speed, dt_prev, dt_next, turn", "Quantifies acceleration, turning, and sampling irregularities that reduce context reliability."),
        ("Topology", "out_deg_to, in_deg_from, has_reverse", "Encodes intersection complexity and bidirectional corridor existence."),
    ]
    feats_df = pd.DataFrame(feats, columns=["Feature Group", "Extracted Features", "Domain Rationale"])
    a(render_booktabs_table(feats_df, ["Feature Group", "Extracted Features", "Domain Rationale"], None, 0, "Table 3. Catalog of the 24 engineered features extracted per (fix, candidate arc) pair."))
    a(img("lgbm_feature_importance.png", "Figure 9. LightGBM (untuned) gain-based feature importance on training data.", 60))
    a(f"<p>{OBS.get('_features', '')}</p>")
    a("""<p><strong>Reference-route limitations:</strong> Ground-truth routes provide valuable reference trajectories rather than absolute ground truth. Monotonic dynamic programming alignment exhibits inherent boundary uncertainty at multi-leg junctions where a fix projects identically to adjoining edge endpoints. Furthermore, minor topological disconnects exist in source routes. Reported accuracy metrics must be evaluated in light of these boundary ambiguities.</p>""")

    # ------------------------------------------------------------------ 3
    a("<h2>3. Model Experimentation and Comparison</h2>")
    a("""<p><strong>Task Formulation:</strong> We frame map-matching candidate selection as a binary classification problem over (fix <em>i</em>, candidate arc <em>a</em>) pairs, where label <em>y</em> &isin; {0, 1} indicates whether arc <em>a</em> corresponds to the true traversed edge (positive class prevalence &approx; 15%). At inference time, candidate probabilities are predicted, and the argmax candidate per fix serves as the pointwise prediction.</p>
<p><strong>Experimental Protocol:</strong> Models were evaluated using 5-fold <code>GroupKFold</code> partitioned strictly by trip ID across the 80 training trips (zero trip overlap across folds). Final models were refit on all 80 training trips and evaluated out-of-sample on the 10 student_test trips. Evaluation is restricted to reliable fixes (seed 42).</p>
<p><strong>Evaluation Metrics:</strong></p>
<ul>
    <li><em>Macro F1 (Pair-Level):</em> Unweighted mean of positive and negative class F1 scores on the candidate pair table.</li>
    <li><em>Top-1 Accuracy (Observation-Level):</em> Proportion of GPS fixes where the predicted candidate arc matches the reference arc (fixes where the true arc was not in the candidate pool are scored as incorrect).</li>
    <li><em>Log Loss:</em> Binary cross-entropy of predicted candidate probabilities, reflecting calibration quality.</li>
    <li><em>Route Continuity:</em> Proportion of consecutive predicted arc pairs (<em>a</em> &rarr; <em>b</em>) that are topologically connected in the road network (to-node(<em>a</em>) = from-node(<em>b</em>)).</li>
    <li><em>Inference Latency:</em> End-to-end classification scoring time per trip in milliseconds.</li>
</ul>""")
    
    a("<h3>3.1 Overall model comparison</h3>")
    t = pd.DataFrame({
        "Model": cmp.model_name,
        "Algorithm Family": cmp.family,
        "Key Configuration": cmp.key_config,
        "CV Macro F1 (Mean ± Std)": [f"{m:.4f} &plusmn; {s:.4f}" for m, s in zip(cmp.cv_macro_f1_mean, cmp.cv_macro_f1_std)],
        "CV Top-1": cmp.cv_top1_mean,
        "Test Top-1": cmp.test_top1,
        "Route Continuity (Test)": cmp.test_route_continuity,
        "Test Macro F1": cmp.test_macro_f1,
        "Log Loss": cmp.test_log_loss,
        "Inference (ms)": cmp.infer_ms_per_trip,
        "Train Time (s)": cmp.train_time_s_full,
    })
    t_nums = ["CV Top-1", "Test Top-1", "Route Continuity (Test)", "Test Macro F1", "Log Loss", "Inference (ms)", "Train Time (s)"]
    a(render_booktabs_table(t, list(t.columns), t_nums, 4, "Table 4. Consolidated comparison of all seven evaluated models (results/model_comparison.csv). Route continuity corresponds to raw classifier-only predictions prior to HMM decoding. Inference time reflects candidate table scoring averaged over student_test trips."))

    cm_df = pd.DataFrame({
        "Model": cmp.model_name,
        "Precision (Pos)": cmp.test_precision_pos,
        "Recall (Pos)": cmp.test_recall_pos,
        "F1 (Pos)": cmp.test_f1_pos,
        "TN": cmp.test_cm_tn,
        "FP": cmp.test_cm_fp,
        "FN": cmp.test_cm_fn,
        "TP": cmp.test_cm_tp
    })
    cm_nums = ["Precision (Pos)", "Recall (Pos)", "F1 (Pos)", "TN", "FP", "FN", "TP"]
    a(render_booktabs_table(cm_df, list(cm_df.columns), cm_nums, 4, "Table 5. Detailed student-testing pair-level classification metrics and confusion matrix counts across candidate pairs."))

    a("<h3>3.2 Compulsory Questions for EVERY Model</h3>")
    for i, m in enumerate(order, 1):
        r = cmp.loc[m]
        tx = MODEL_TEXT[m]
        gap = r.train_top1_20trips - r.cv_top1_mean
        a(f"<h4>Model {i}: {r.model_name} ({r.family})</h4>")
        a(f"""<p><strong>A1. Why suitable?</strong> {tx['why']}</p>
<p><strong>A2. Strengths and limitations:</strong> {tx['strengths']}</p>
<p><strong>B5. Hyperparameters and initial values:</strong> {tx['hyper']}</p>
<p><strong>B6. Loss function and optimisation objective:</strong> {tx['loss']}</p>
<p><strong>C7&ndash;C8. Student-testing performance:</strong> Observation-level Top-1 Accuracy = {r.test_top1:.4f}, Macro F1 = {r.test_macro_f1:.4f}, Log Loss = {r.test_log_loss:.4f}, Positive-class Precision/Recall = {r.test_precision_pos:.4f}/{r.test_recall_pos:.4f}, Classifier-only Route Continuity = {r.test_route_continuity:.4f}, Scoring Latency = {r.infer_ms_per_trip:.1f} ms per trip.</p>
<p><strong>C9. Overfitting or underfitting analysis:</strong> Top-1 on 20 internal training trips = {r.train_top1_20trips:.4f}; 5-fold CV on held-out trips = {r.cv_top1_mean:.4f} &plusmn; {r.cv_top1_std:.4f}; Student-testing Top-1 = {r.test_top1:.4f} (generalisation gap: {gap*100:+.2f} percentage points). {OBS.get(m, '')}</p>""")

    a("<h3>3.3 Model visualisations</h3>")
    a(img("models_cv_macro_f1.png", "Figure 10. Cross-validation Macro F1 comparison across all seven models (mean &plusmn; std across 5 trip-grouped folds)."))
    a(img("models_cv_fold_stability.png", "Figure 11. Per-fold Top-1 accuracy across validation folds, demonstrating fold stability."))
    a(img("models_test_top1_f1.png", "Figure 12. Student-testing observation Top-1 accuracy and Macro F1 comparison (classifier-only)."))
    a(img("models_log_loss.png", "Figure 13. Log loss comparison across models (CV and student_test, logarithmic scale)."))
    a(img("models_generalisation_gap.png", "Figure 14. Train vs CV vs student-test Top-1 accuracy, illustrating empirical generalisation gaps (C9)."))
    a(img("models_time.png", "Figure 15. Training duration (seconds) and inference latency (milliseconds per trip) trade-off."))
    a(img("models_confusion_matrices.png", "Figure 16. Normalised confusion matrices on student_test candidate pairs across all models."))
    a(img("mlp_loss_curve.png", "Figure 17. Multi-Layer Perceptron (PyTorch) training and validation loss progression per epoch. The MLP is the only architecture with epoch-wise history; no artificial curves are plotted for non-iterative algorithms.", 65))
    a(f"<p><strong>Model Selection Justification:</strong> {OBS.get('_selection', '')}</p>")

    # ------------------------------------------------------------------ 4
    a("<h2>4. Bayesian Hyperparameter Optimisation (LightGBM)</h2>")
    space = best["search_space"]
    sp = pd.DataFrame([(k, v[0], v[1], v[2], best["best_params"].get(k)) for k, v in space.items()],
                      columns=["Hyperparameter", "Distribution", "Lower Bound", "Upper Bound", "Optimal Value"])
    ncomp = int((trials.state == "COMPLETE").sum())
    npr = int((trials.state == "PRUNED").sum())
    nstale = int((trials.state == "RUNNING").sum())
    a(f"""<p><strong>Q1&ndash;Q2. Hyperparameter Search Space:</strong> Eight key LightGBM hyperparameters were tuned (Table 6), governing tree capacity (<code>num_leaves</code>, <code>max_depth</code>, <code>min_child_samples</code>), gradient stepping (<code>learning_rate</code>, <code>n_estimators</code>), and stochastic regularisation (<code>subsample</code>, <code>colsample_bytree</code>, <code>reg_lambda</code>). Scale-sensitive parameters utilized log-uniform priors.</p>
<p><strong>Q3. Optimisation Objective:</strong> Mean candidate pair Macro F1 evaluated over 3-fold <code>GroupKFold</code> strictly within the 80 training trips (zero evaluator/test leakage).</p>
<p><strong>Q4. Search Budget &amp; Trials:</strong> {ncomp + npr} total trials executed using Tree-structured Parzen Estimators (TPE, seed 42, 8 initial random warm-up trials): {ncomp} trials completed fully, and {npr} unpromising configurations were halted early by Optuna's median pruner.{' ' + str(nstale) + ' stale trial from an interrupted prior run remains in the SQLite database in RUNNING state and is excluded.' if nstale else ''}</p>
<p><strong>Q5. Best Hyperparameter Values:</strong> Listed in Table 6, achieving an optimal 3-fold CV Macro F1 of {best['best_cv3_macro_f1']:.4f}.</p>""")
    
    sp_nums = ["Lower Bound", "Upper Bound", "Optimal Value"]
    a(render_booktabs_table(sp, list(sp.columns), sp_nums, 4, "Table 6. Bayesian optimisation search space configuration and optimal parameters obtained (results/optuna_best_params.json)."))
    a(img("optuna_history.png", "Figure 18. Bayesian optimisation objective convergence history across TPE trials."))
    a(img("optuna_importance.png", "Figure 19. Relative hyperparameter importance computed via fANOVA on completed trials.", 65))
    
    tv = tvu.set_index("model")
    tvt = pd.DataFrame({
        "Model Configuration": ["Untuned LightGBM", "Tuned LightGBM (Bayesian)"],
        "CV Macro F1 (5-fold)": [tv.loc["lgbm_untuned", "cv5_macro_f1"], tv.loc["lgbm_tuned", "cv5_macro_f1"]],
        "CV Top-1 (5-fold)": [tv.loc["lgbm_untuned", "cv5_top1"], tv.loc["lgbm_tuned", "cv5_top1"]],
        "CV Log Loss": [tv.loc["lgbm_untuned", "cv5_log_loss"], tv.loc["lgbm_tuned", "cv5_log_loss"]],
        "Test Top-1": [tv.loc["lgbm_untuned", "test_top1"], tv.loc["lgbm_tuned", "test_top1"]],
        "Test Macro F1": [tv.loc["lgbm_untuned", "test_macro_f1"], tv.loc["lgbm_tuned", "test_macro_f1"]],
        "Test Log Loss": [tv.loc["lgbm_untuned", "test_log_loss"], tv.loc["lgbm_tuned", "test_log_loss"]],
        "Inference (ms)": [tv.loc["lgbm_untuned", "infer_ms_per_trip"], tv.loc["lgbm_tuned", "infer_ms_per_trip"]],
        "Train Time (s)": [tv.loc["lgbm_untuned", "train_s"], tv.loc["lgbm_tuned", "train_s"]]
    })
    tvt_nums = ["CV Macro F1 (5-fold)", "CV Top-1 (5-fold)", "CV Log Loss", "Test Top-1", "Test Macro F1", "Test Log Loss", "Inference (ms)", "Train Time (s)"]
    a(render_booktabs_table(tvt, list(tvt.columns), tvt_nums, 4, "Table 7. Performance comparison of untuned vs Bayesian-tuned LightGBM under identical validation splits."))
    a(f"<p><strong>Q6. Performance Comparison and Empirical Analysis:</strong> {OBS.get('_tuning', '')}</p>")

    # ------------------------------------------------------------------ 5
    a("<h2>5. HMM and Viterbi Route Reconstruction</h2>")
    a(f"""<p><strong>Q1. Hidden State Formulation:</strong> At each GPS fix <em>i</em>, the hidden states <em>S<sub>i</sub></em> correspond to the top candidate road arcs (capped at {hcfg['beam']} by classifier confidence), paired with the fix's orthogonal projection offset along that arc in metres. States map bijectively to physical edge IDs in the trip sub-network.</p>
<p><strong>Q2. Emission Probability Calculation:</strong> Emissions represent the probability that fix <em>i</em> was generated by arc <em>a</em>: log <em>e<sub>i</sub></em>(<em>a</em>) = &alpha; &middot; log(<em>p</em><sub>clf</sub>(<em>a</em>|<em>i</em>) / &sum;<sub><em>c</em></sub> <em>p</em><sub>clf</sub>(<em>c</em>|<em>i</em>)), where <em>p</em><sub>clf</sub> is the calibrated LightGBM probability renormalised over the candidate set, and &alpha; governs emission-to-transition weighting (&alpha; = {hcfg['alpha']}).</p>
<p><strong>Q3. Transition Probability Calculation:</strong> Transition probabilities adopt the classic Newson &amp; Krumm formulation based on the discrepancy between network routing distance and Euclidean GPS displacement: log <em>T</em>(<em>a</em> &rarr; <em>b</em>) = &minus;| <em>d</em><sub>route</sub>(<em>a</em>, <em>b</em>) &minus; <em>d</em><sub>gps</sub>(<em>i</em>&minus;1, <em>i</em>) | / &beta; (with scale parameter &beta; = {hcfg['beta']} m). If consecutive states lie on the identical arc, <em>d</em><sub>route</sub> is computed as |offset<sub><em>b</em></sub> &minus; offset<sub><em>a</em></sub>| (penalized by +50 m if reverse travel exceeds {hcfg['same_arc_backtrack_tol_m']:.0f} m). Across different arcs, <em>d</em><sub>route</sub> = (len<sub><em>a</em></sub> &minus; offset<sub><em>a</em></sub>) + Dijkstra(to(<em>a</em>), from(<em>b</em>)) + offset<sub><em>b</em></sub>. Dijkstra searches are bounded to {hcfg['sp_limit_m']:.0f} m and executed batch-wise per trip.</p>
<p><strong>Viterbi Recurrence:</strong> Exact dynamic programming decodes the optimal state sequence in log space: &delta;<sub><em>i</em></sub>(<em>b</em>) = log <em>e<sub>i</sub></em>(<em>b</em>) + max<sub><em>a</em></sub> [&delta;<sub><em>i</em>&minus;1</sub>(<em>a</em>) + log <em>T</em>(<em>a</em> &rarr; <em>b</em>)], followed by back-pointer tracing.</p>
<p><strong>Q4. Handling Missing Candidates and GPS Gaps:</strong> If zero candidates exist within 50 m, radius expansion to 150 m triggers. If still null, the fix inherits the prior decoded state without biasing trajectory continuity. Across temporal gaps (&Delta;<em>t</em> &gt; 60 s) or spatial jumps (&gt; 800 m), transitions reset to a uniform prior (log <em>T</em> = 0), restarting the decoder without enforcing artificial connections across unobserved periods.</p>
<p><strong>Q5. Disconnected Consecutive Road Segments:</strong> When no path exists between consecutive candidate arcs within the Dijkstra threshold, transitions receive a floor log-probability penalty of {hcfg['unreachable_logp']} rather than &minus;&infin;, ensuring the decoder never breaks. Post-decoding, disjoint adjacent arcs are bridged via shortest-path graph interpolation (<code>hmm.build_route</code>), with any unresolvable jumps flagged explicitly as unbridged breaks.</p>
<p><strong>Decoder Hyperparameter Tuning:</strong> A grid search over &alpha; &isin; {{0.5, 1, 2}} and &beta; &isin; {{3, 5, 10, 20, 40}} was conducted strictly using <em>out-of-fold</em> validation predictions on the 80 training trips, selecting <strong>{hcfg['selected_from']}</strong>.</p>""")
    a(img("hmm_grid_train.png", "Figure 20. HMM decoder grid search over emission weight &alpha; and transition scale &beta; on training out-of-fold predictions.", 65))

    a("<h3>Q6. Did HMM decoding improve the final route?</h3>")
    order_h = ["nearest_arc", "classifier_only", "hmm_distance_emission", "hmm_classifier_emission"]
    names_h = ["Nearest Arc (No ML)", "Classifier Only (Tuned LightGBM)", "HMM (Distance Emission)", "Classifier + HMM/Viterbi (Final)"]
    ht = hmm_te.loc[order_h].reset_index(drop=True)
    ht.insert(0, "Method", names_h)
    ht_raw_cols = ["Method", "top1_pooled", "top1_mean_trip", "route_continuity_pooled", "route_precision", "route_recall", "route_f1", "trips_top1_ge_95pct", "unbridged_breaks"]
    ht_cols = ["Method", "Pooled Top-1", "Mean Trip Top-1", "Route Continuity", "Route Precision", "Route Recall", "Route F1", "Trips Top-1 ≥ 95%", "Unbridged Breaks"]
    ht_nums = ["Pooled Top-1", "Mean Trip Top-1", "Route Continuity", "Route Precision", "Route Recall", "Route F1", "Trips Top-1 ≥ 95%", "Unbridged Breaks"]
    a(render_booktabs_table(ht[ht_raw_cols], ht_cols, ht_nums, 4, "Table 8. Evaluation of route reconstruction on student_test trips (10 trips). Pooled Top-1 evaluates across all fixes; route precision, recall, and F1 assess graph edge set overlap with ground-truth routes."))

    htr = hmm_tr.loc[order_h[:3] + [hcfg['selected_from']]].reset_index(drop=True)
    htr.insert(0, "Method", names_h[:3] + [f"Classifier + HMM ({hcfg['selected_from']})"])
    htr_raw_cols = ["Method", "top1_pooled", "route_continuity_pooled", "route_f1", "trips_top1_ge_95pct"]
    htr_cols = ["Method", "Pooled Top-1", "Route Continuity", "Route F1", "Trips Top-1 ≥ 95%"]
    htr_nums = ["Pooled Top-1", "Route Continuity", "Route F1", "Trips Top-1 ≥ 95%"]
    a(render_booktabs_table(htr[htr_raw_cols], htr_cols, htr_nums, 4, "Table 9. Benchmark comparison across the 80 training trips using out-of-fold validation predictions."))

    a(img("hmm_vs_classifier_test.png", "Figure 21. Classifier-only predictions vs Classifier + HMM/Viterbi decoding across student_test metrics."))
    a(img("hmm_delta_per_trip.png", "Figure 22. Trip-by-trip gain in Top-1 accuracy resulting from HMM decoding on student_test.", 75))

    per_trip_raw_cols = ["trip_id", "nearest_arc", "classifier_only", "hmm_distance_emission", "hmm_classifier_emission", "delta_hmm_minus_clf"]
    per_trip_cols = ["Trip ID", "Nearest Arc", "Classifier Only", "HMM (Distance)", "HMM (Classifier)", "HMM Gain (Δ)"]
    per_trip_nums = ["Nearest Arc", "Classifier Only", "HMM (Distance)", "HMM (Classifier)", "HMM Gain (Δ)"]
    a(render_booktabs_table(per_trip[per_trip_raw_cols], per_trip_cols, per_trip_nums, 4, "Table 10. Per-trip observation Top-1 accuracy breakdown on student_test."))
    a(f"<p>{OBS.get('_hmm', '')}</p>")

    # ------------------------------------------------------------------ 6
    a("<h2>6. Geographical Visualisation Using Folium</h2>")
    a("""<p>Interactive geospatial visualisations are generated programmatically via <code>src/pathsnap/viz.py</code> and committed as interactive HTML artifacts in <code>results/maps/</code> and <code>report/screenshots/gta/maps/</code>. Cartographic layers depict: the background road network (dark grey), ground-truth reference routes (vibrant green, available offline), standalone classifier-only predictions (amber), and final reconstructed HMM/Viterbi routes (magenta/purple), with raw GPS fixes displayed as cyan circular markers alongside start/finish flags, layer toggles, and coordinate legends. Both macroscopic trip-level extents and micro-level junction zooms are captured below to illustrate model disambiguation.</p>""")
    
    for f, cap in OBS.get("_maps", []):
        a(img(f, cap))

    # ------------------------------------------------------------------ 7
    a("<h2>7. API, UI and Deployment</h2>")
    a("""<p><strong>System Architecture:</strong> Deployment is structured across a decoupled two-node architecture mimicking real-world edge/cloud telematics. <strong>Laptop 1 (Inference Server)</strong> hosts the FastAPI backend service (<code>uvicorn api.main:app --host 0.0.0.0 --port 8000</code>), managing serialized model weights (<code>models/final_model.joblib</code>), decoder parameters (<code>models/hmm_config.json</code>), road network topology, and server-side ground truth. <strong>Laptop 2 (Evaluator Client)</strong> runs the Streamlit visualization dashboard (<code>streamlit run ui/app.py</code>), issuing REST calls across the local subnet.</p>
<p><strong>API Endpoints:</strong></p>
<ul>
    <li><code>GET /health</code>: Reports service readiness, model metadata, and memory status.</li>
    <li><code>GET /trips?partition=eval_test</code>: Enumerates available unlabelled trip IDs on the host.</li>
    <li><code>POST /predict {trip_id, partition}</code>: Loads raw trajectory data, <strong>strips ground-truth routes prior to inference</strong> (<code>pipeline.predict_trip</code>), executes <em>k</em>-d candidate search &rarr; feature extraction &rarr; LightGBM inference &rarr; Viterbi decoding &rarr; route bridging, and computes evaluation metrics against server-side ground truth without exposing reference coordinates to the client.</li>
</ul>
<p>Inbound requests and operational latencies are logged to the console and appended to <code>results/logs/api_requests.log</code>.</p>""")
    
    a("""<p><b>Two-laptop deployment, as demonstrated.</b> The backend laptop ran the API with <code>--host 0.0.0.0</code> on a shared
Wi-Fi hotspot, with an inbound Windows Firewall rule for TCP port 8000. The frontend laptop ran <code>streamlit run ui/app.py</code> with the API URL
set to the backend's address. Each request appears in the backend terminal, including the <code>PREDICT</code> line with the trip, the number of fixes,
the server-side scores and the processing time, followed by <code>POST /predict ... 200</code>. The screenshots below were taken on the two
laptops during that run.</p>""")
    for f, cap in OBS.get("_shots", []):
        a(img(f, cap))
    logp = os.path.join(RES, "logs", "api_requests.log")
    if os.path.exists(logp):
        lines = [l for l in open(logp, encoding="utf-8", errors="ignore").read().splitlines()
                 if "PREDICT" in l or "POST /predict from" in l][-8:]
        a("<p><b>API log excerpt</b> (last prediction entries in results/logs/api_requests.log, written by the API process on the backend laptop; "
          "requests from 10.213.115.230 came from the frontend laptop during the two-laptop run):</p>"
          "<pre style='font-size:9.5px;white-space:pre-wrap'>" + "\n".join(lines) + "</pre>")

    # ------------------------------------------------------------------ 8
    a("<h2>8. Reproducibility, Integrity and AI-use Disclosure</h2>")
    a("""<p>All quantitative results, metric tables, and confusion matrices documented in this report are fully reproducible from scripts <code>01</code> through <code>06</code> using fixed global random seed 42. Strict isolation of the evaluator partition was maintained: no training, validation, or hyperparameter selection routine accessed evaluator files. Dynamic scaling and model tuning were restricted to training partitions under trip-level group isolation.</p>
<p><strong>AI Assistance Disclosure:</strong> The engineering team utilized an AI programming assistant (Claude) for code optimization, documentation formatting, and initial text drafting. All model architectures, mathematical formulations, training runs, experimental validations, and analytical deductions were conceived, executed, and independently verified by the team members.</p>""")
    
    return "\n".join(H)

CSS = """
@page {
    size: A4 portrait;
    margin: 20mm 18mm 20mm 18mm;
}

body {
    font-family: 'Times New Roman', Times, 'Nimbus Roman No9 L', serif;
    font-size: 10.5pt;
    line-height: 1.42;
    color: #111;
    background: #fff;
    max-width: 860px;
    margin: 0 auto;
    padding: 0;
    text-align: justify;
    text-justify: inter-word;
    -webkit-font-smoothing: antialiased;
}

/* Header & Metadata */
.header-container {
    text-align: center;
    margin-bottom: 20px;
    padding-bottom: 14px;
    border-bottom: 1.2pt solid #222;
}

.paper-title {
    font-size: 20pt;
    font-weight: bold;
    line-height: 1.25;
    margin: 0 0 8px 0;
    color: #000;
}

.paper-subtitle {
    font-size: 11.5pt;
    font-style: italic;
    color: #222;
    margin-bottom: 4px;
}

.paper-institution {
    font-size: 10pt;
    color: #444;
    margin-bottom: 8px;
}

.author-block {
    font-size: 10.5pt;
    margin-bottom: 6px;
    line-height: 1.45;
}

.author {
    font-weight: bold;
    color: #111;
}

.srn {
    font-weight: normal;
    color: #444;
    font-size: 9.5pt;
}

.repo-line {
    font-size: 9.5pt;
    color: #333;
    margin-top: 4px;
}

.repo-line a {
    color: #003366;
    text-decoration: underline;
}

/* Abstract */
.abstract-container {
    margin: 14px 28px 22px 28px;
    font-size: 9.8pt;
    line-height: 1.38;
    text-align: justify;
}

.abstract-heading {
    font-weight: bold;
    font-style: italic;
    color: #000;
}

/* Headings */
h2 {
    font-size: 13.5pt;
    font-weight: bold;
    color: #000;
    margin-top: 26px;
    margin-bottom: 10px;
    border-bottom: 0.8pt solid #333;
    padding-bottom: 3px;
    page-break-after: avoid;
}

h3 {
    font-size: 11.5pt;
    font-weight: bold;
    color: #111;
    margin-top: 18px;
    margin-bottom: 6px;
    page-break-after: avoid;
}

h4 {
    font-size: 10.5pt;
    font-weight: bold;
    color: #222;
    margin-top: 14px;
    margin-bottom: 4px;
    page-break-after: avoid;
}

p {
    margin: 6px 0;
}

/* Code & inline identifiers */
code {
    font-family: Consolas, 'Courier New', monospace;
    font-size: 9.2pt;
    background: #f4f4f4;
    padding: 1px 4px;
    border-radius: 2px;
    color: #222;
}

/* Booktabs Table Styling */
.table-wrapper {
    margin: 14px auto 10px auto;
    page-break-inside: avoid;
}

.tbl {
    width: 100%;
    border-collapse: collapse;
    font-size: 8.8pt;
    line-height: 1.25;
    margin: 4px 0;
}

.tbl thead th {
    border-top: 1.4pt solid #111;
    border-bottom: 0.8pt solid #111;
    border-left: none;
    border-right: none;
    background: #fafafa;
    padding: 6px 5px;
    font-weight: bold;
    text-align: left;
    vertical-align: bottom;
    color: #000;
    word-break: normal;
    hyphens: none;
}

.tbl thead th.th-num {
    text-align: right;
}

.tbl tbody td {
    border-top: none;
    border-bottom: 0.4pt solid #e2e2e2;
    border-left: none;
    border-right: none;
    padding: 4.5px 5px;
    vertical-align: middle;
    color: #111;
}

.tbl tbody td.td-num {
    text-align: right;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}

.tbl tbody tr:last-child td {
    border-bottom: 1.4pt solid #111;
}

.tbl-caption {
    font-size: 8.8pt;
    color: #333;
    margin-top: 4px;
    margin-bottom: 8px;
    text-align: left;
}

/* Figures */
figure {
    margin: 16px auto;
    text-align: center;
    page-break-inside: avoid;
}

figure img {
    max-width: 100%;
    height: auto;
    border: 0.5pt solid #ccc;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border-radius: 2px;
}

figcaption {
    font-size: 8.8pt;
    color: #333;
    margin-top: 6px;
    line-height: 1.32;
    text-align: center;
}

figcaption strong {
    color: #000;
}

/* Code & Pre Blocks */
pre {
    font-family: Consolas, 'Courier New', monospace;
    font-size: 8.2pt;
    background: #f8f9fa;
    border: 0.5pt solid #ddd;
    border-left: 2.5pt solid #444;
    padding: 8px 12px;
    overflow-x: auto;
    line-height: 1.35;
    page-break-inside: avoid;
    white-space: pre-wrap;
    word-break: break-all;
}

ol, ul {
    margin: 6px 0 10px 0;
    padding-left: 24px;
}

li {
    margin-bottom: 4px;
}

@media print {
    body {
        max-width: 100%;
        margin: 0;
    }
    h2, h3, h4 {
        page-break-after: avoid;
    }
    figure, table, .table-wrapper, pre {
        page-break-inside: avoid;
    }
}
"""

def to_pdf_chrome(html_path, pdf_path):
    chrome_paths = [
        shutil.which("google-chrome") or "",
        shutil.which("chrome") or "",
        shutil.which("chromium") or "",
        shutil.which("chromium-browser") or "",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for exe in chrome_paths:
        if exe and os.path.exists(exe):
            cmd = [
                exe,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path}",
                f"file://{os.path.abspath(html_path)}"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
                print(f"Successfully generated {pdf_path} using {exe}")
                return True
            else:
                print(f"Chrome run failed: {res.stderr}")
    return False

def main():
    os.makedirs(REP, exist_ok=True)
    for fn in ("observations_models.json", "observations.json"):
        obs_path = os.path.join(REP, fn)
        if os.path.exists(obs_path):
            OBS.update(json.load(open(obs_path, encoding="utf-8")))
            
    body = build_document()
    html_content = (f"<!doctype html>\n<html lang='en'>\n<head>\n<meta charset='utf-8'>\n"
                    f"<title>PathSnap Technical Report</title>\n"
                    f"<style>{CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>")
    
    html_path = os.path.join(REP, "PathSnap_Report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Wrote HTML to {html_path} ({len(html_content):,} chars)")

    # Output to both PathSnap_Report_GTA.pdf and PathSnap_Report.pdf
    pdf_gta_path = os.path.join(REP, "PathSnap_Report_GTA.pdf")
    pdf_std_path = os.path.join(REP, "PathSnap_Report.pdf")
    
    ok_gta = to_pdf_chrome(html_path, pdf_gta_path)
    ok_std = to_pdf_chrome(html_path, pdf_std_path)
    print(f"PathSnap_Report_GTA.pdf generated: {ok_gta} ({os.path.getsize(pdf_gta_path):,} bytes)")
    print(f"PathSnap_Report.pdf generated: {ok_std} ({os.path.getsize(pdf_std_path):,} bytes)")

if __name__ == "__main__":
    main()
