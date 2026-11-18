"""HMM / Viterbi route reconstruction: decoder tuning and comparison.

1. Out-of-fold classifier probabilities for every training trip (5-fold GroupKFold by trip,
   the fold model never saw the trip it scores).
2. Grid search of the decoder parameters (alpha, beta) on the 80 training trips (OOF probs).
3. Student-test comparison of:
     nearest arc | classifier only | HMM with distance-only emission | HMM with classifier emission
   on Top-1 accuracy, route continuity, route arc-set precision/recall/F1 and per-trip results.
4. Saves models/hmm_config.json, results/hmm_*.csv, figures and Folium maps.
"""
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from pathsnap.config import load_config
from pathsnap.features import FEATURES, PreparedTrip
from pathsnap.hmm import build_route, build_states, viterbi
from pathsnap.io import list_trips, load_trip
from pathsnap.metrics import continuity, route_set_prf
from pathsnap.models import make_model
from pathsnap.viz import trip_map

cfg = load_config()
P = cfg["paths"]
RES, FIG, MAPS = P["results"], os.path.join(P["results"], "figures"), os.path.join(P["results"], "maps")
ALPHAS = [0.5, 1.0, 2.0]
BETAS = [3.0, 5.0, 10.0, 20.0, 40.0]
LABEL_MAX_DIST = 50.0


def methods_for(prep, cand, prob, hcfgs, extra=True):
    """Return dict method_name -> per-fix predicted arcs."""
    n = len(prep.xy)
    out = {}
    if extra:
        nearest = np.full(n, -1)
        r0 = cand[cand.dist_rank == 0]
        nearest[r0.pt_idx.to_numpy()] = r0.arc_id.to_numpy()
        out["nearest_arc"] = nearest
        clf = np.full(n, -1)
        order = np.lexsort((-prob, cand.pt_idx.to_numpy()))
        ps = cand.pt_idx.to_numpy()[order]
        first = np.r_[True, ps[1:] != ps[:-1]]
        clf[ps[first]] = cand.arc_id.to_numpy()[order][first]
        out["classifier_only"] = clf
        h = copy.deepcopy(cfg["hmm"])
        out["hmm_distance_emission"] = viterbi(prep, build_states(cand, prob, n, h["beam"], 1.0, emission="distance"), h)
    for name, h in hcfgs.items():
        out[name] = viterbi(prep, build_states(cand, prob, n, h["beam"], h["alpha"]), h)
    return out


def score(prep, pred, route):
    ok = prep.align_dist <= LABEL_MAX_DIST
    correct = pred == prep.ref_arc
    cont, cok, ctot = continuity(pred, prep.net.arcs)
    full_route, info = build_route(prep, pred)
    pr, rc, f1 = route_set_prf(full_route, route)
    return {"n_ok": int(ok.sum()), "n_correct": int(correct[ok].sum()), "top1": float(correct[ok].mean()) if ok.any() else np.nan,
            "continuity": cont, "cont_ok": cok, "cont_tot": ctot, "route_precision": pr, "route_recall": rc, "route_f1": f1,
            "unbridged_breaks": info["unbridged_breaks"]}


def run_trip(part, tid, model, hcfgs, extra):
    tr = load_trip(P[part], tid)
    prep = PreparedTrip(tr, cfg, with_labels=True)
    cand = prep.candidate_table()
    prob = model.predict_proba(cand[FEATURES].to_numpy(np.float32))[:, 1]
    t0 = time.perf_counter()
    preds = methods_for(prep, cand, prob, hcfgs, extra)
    el = time.perf_counter() - t0
    rows = []
    for name, pred in preds.items():
        r = score(prep, pred, tr.route)
        r.update({"partition": part, "trip_id": tid, "method": name})
        rows.append(r)
    return rows, el


def pooled(df):
    g = df.groupby("method")
    return pd.DataFrame({
        "top1_pooled": g.n_correct.sum() / g.n_ok.sum(),
        "top1_mean_trip": g.top1.mean(),
        "route_continuity_pooled": g.cont_ok.sum() / g.cont_tot.sum(),
        "route_continuity_mean_trip": g.continuity.mean(),
        "route_precision": g.route_precision.mean(), "route_recall": g.route_recall.mean(), "route_f1": g.route_f1.mean(),
        "trips_top1_ge_95pct": g.top1.apply(lambda s: (s >= 0.95).mean()),
        "unbridged_breaks": g.unbridged_breaks.sum(),
    })


def main():
    best_path = os.path.join(RES, "optuna_best_params.json")
    params = json.load(open(best_path))["best_params"] if os.path.exists(best_path) else {}
    print("classifier params", params)

    # ---------------- 1. OOF fold models
    cand = pd.read_parquet(os.path.join(P["processed"], "candidates_train.parquet"))
    candl = cand[cand.label_ok == 1].reset_index(drop=True)
    X, y, groups = candl[FEATURES].to_numpy(np.float32), candl.label.to_numpy(), candl.trip_id.to_numpy()
    trip_fold, fold_models = {}, []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        m = make_model("lgbm", params)
        m.fit(X[tr], y[tr])
        fold_models.append(m)
        for t in np.unique(groups[va]):
            trip_fold[t] = k
        print("fold model", k)
    train_trips = list_trips(P["train"])
    for t in train_trips:  # trips with no reliable labels at all -> any fold
        trip_fold.setdefault(t, 0)

    # ---------------- 2. decoder grid search on train (OOF)
    grid = {}
    for a in ALPHAS:
        for b in BETAS:
            h = copy.deepcopy(cfg["hmm"]); h["alpha"], h["beta"] = a, b
            grid[f"hmm_a{a}_b{b}"] = h
    t0 = time.perf_counter()
    res = Parallel(n_jobs=10, verbose=5)(delayed(run_trip)("train", t, fold_models[trip_fold[t]], grid, True) for t in train_trips)
    grid_s = time.perf_counter() - t0
    tr_rows = pd.DataFrame([r for rr, _ in res for r in rr])
    tr_rows.to_csv(os.path.join(RES, "hmm_train_grid_per_trip.csv"), index=False)
    tr_sum = pooled(tr_rows)
    tr_sum.to_csv(os.path.join(RES, "hmm_train_grid_summary.csv"))
    gnames = [n for n in tr_sum.index if n.startswith("hmm_a")]
    best_name = tr_sum.loc[gnames, "top1_pooled"].idxmax()
    best_h = grid[best_name]
    print(tr_sum.round(4).to_string())
    print("best decoder", best_name, "grid seconds", grid_s)

    hm = np.array([[tr_sum.loc[f"hmm_a{a}_b{b}", "top1_pooled"] for b in BETAS] for a in ALPHAS])
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(hm, cmap="viridis")
    ax.set_xticks(range(len(BETAS)), [str(b) for b in BETAS]); ax.set_yticks(range(len(ALPHAS)), [str(a) for a in ALPHAS])
    for i in range(len(ALPHAS)):
        for j in range(len(BETAS)):
            ax.text(j, i, f"{hm[i, j]:.4f}", ha="center", va="center", color="white", fontsize=8)
    ax.set(xlabel="beta (transition scale, m)", ylabel="alpha (emission weight)", title="HMM decoder grid: OOF Top-1 on train trips")
    plt.colorbar(im); plt.tight_layout(); plt.savefig(os.path.join(FIG, "hmm_grid_train.png"), dpi=130); plt.close()

    final_h = {k: v for k, v in best_h.items()}
    final_h.update({"selected_from": best_name, "selection": "max pooled OOF Top-1 on 80 train trips",
                    "classifier_params": params})
    with open(os.path.join(P["models"], "hmm_config.json"), "w") as f:
        json.dump(final_h, f, indent=2)

    # ---------------- 3. student_test comparison with the final (all-train) model
    final_model = make_model("lgbm", params)
    final_model.fit(X, y)
    joblib.dump(final_model, os.path.join(P["models"], "final_model.joblib"))
    with open(os.path.join(P["models"], "final_model_meta.json"), "w") as f:
        json.dump({"algorithm": "LightGBM (binary candidate classifier)", "params": params, "features": FEATURES,
                   "trained_on": "80 train trips, fixes with alignment distance <= 50 m",
                   "candidate_config": cfg["candidates"], "feature_config": cfg["features"],
                   "preprocess_config": cfg["preprocess"]}, f, indent=2)
    te_trips = list_trips(P["student_test"])
    res = Parallel(n_jobs=10)(delayed(run_trip)("student_test", t, final_model, {"hmm_classifier_emission": best_h}, True) for t in te_trips)
    te_rows = pd.DataFrame([r for rr, _ in res for r in rr])
    te_rows.to_csv(os.path.join(RES, "hmm_student_test_per_trip.csv"), index=False)
    te_sum = pooled(te_rows)
    te_sum["decode_s_total"] = np.nan
    te_sum.to_csv(os.path.join(RES, "hmm_student_test_summary.csv"))
    print(te_sum.round(4).to_string())

    order = ["nearest_arc", "classifier_only", "hmm_distance_emission", "hmm_classifier_emission"]
    s = te_sum.loc[order]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.2
    for i, (col, lab) in enumerate([("top1_pooled", "Top-1 accuracy"), ("route_continuity_pooled", "Route continuity"),
                                     ("route_precision", "Route precision"), ("route_f1", "Route arc-set F1")]):
        ax.bar(np.arange(len(order)) + (i - 1.5) * w, s[col], w, label=lab)
    ax.set_xticks(range(len(order)), ["Nearest arc", "Classifier only", "HMM (distance emission)", "HMM (classifier emission)"])
    ax.set_ylim(0.4, 1.0); ax.set_title("Route reconstruction on student_test (10 trips)"); ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "hmm_vs_classifier_test.png"), dpi=130); plt.close()

    piv = te_rows.pivot(index="trip_id", columns="method", values="top1")
    piv["delta_hmm_minus_clf"] = piv["hmm_classifier_emission"] - piv["classifier_only"]
    piv.to_csv(os.path.join(RES, "hmm_student_test_per_trip_top1.csv"))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(piv.index.str.lstrip("0").str.rjust(1, "0"), piv.delta_hmm_minus_clf * 100,
           color=np.where(piv.delta_hmm_minus_clf >= 0, "#2a9d8f", "#e76f51"))
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel="student_test trip", ylabel="Top-1 change (percentage points)", title="HMM minus classifier-only, per trip")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "hmm_delta_per_trip.png"), dpi=130); plt.close()

    # ---------------- 4. Folium maps: best, worst and median trip
    srt = piv.sort_values("delta_hmm_minus_clf")
    picks = {"smallest_gain": srt.index[0], "largest_gain": srt.index[-1], "median_gain": srt.index[len(srt) // 2]}
    for tag, tid in picks.items():
        tr = load_trip(P["student_test"], tid)
        prep = PreparedTrip(tr, cfg, with_labels=True)
        cand_t = prep.candidate_table()
        prob = final_model.predict_proba(cand_t[FEATURES].to_numpy(np.float32))[:, 1]
        preds = methods_for(prep, cand_t, prob, {"hmm": best_h}, True)
        clf_route, _ = build_route(prep, preds["classifier_only"])
        hmm_route, _ = build_route(prep, preds["hmm"])
        m = trip_map(tr.nodes, tr.arcs, tr.track, ref_route=tr.route,
                     layers=[("Classifier-only route", clf_route, "#f4a261"), ("HMM / Viterbi route", hmm_route, "#d62728")],
                     title=f"student_test trip {tid} ({tag})")
        m.save(os.path.join(MAPS, f"result_{tag}_{tid}.html"))
    print("done")


if __name__ == "__main__":
    main()
