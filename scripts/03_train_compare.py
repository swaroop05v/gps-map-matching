"""Train and compare the candidate classifiers.

* 5-fold GroupKFold (grouped by trip) on the 80 training trips -> CV metrics + OOF probabilities
* refit on all 80 training trips -> evaluation on the 10 student_test trips
* results/model_comparison.csv, results/cv_folds.csv, figures, saved models

Usage: python scripts/03_train_compare.py [model names...]
"""
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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

from pathsnap.config import load_config
from pathsnap.features import FEATURES
from pathsnap.io import list_trips, load_trip
from pathsnap.metrics import observation_top1, pair_metrics, route_continuity_by_trip, top_choice
from pathsnap.models import MODEL_INFO, SEED, make_model

cfg = load_config()
P = cfg["paths"]
RES, FIG = P["results"], os.path.join(P["results"], "figures")
EXP = os.path.join(P["models"], "experiments")
OOF = os.path.join(P["processed"], "oof")
for d in (FIG, EXP, OOF):
    os.makedirs(d, exist_ok=True)

KNN_MAX_ROWS = 200_000


def load(part):
    cand = pd.read_parquet(os.path.join(P["processed"], f"candidates_{part}.parquet"))
    pts = pd.read_parquet(os.path.join(P["processed"], f"points_{part}.parquet"))
    return cand, pts


def arcs_by_trip():
    out = {}
    for part in ["train", "student_test"]:
        for t in list_trips(P[part]):
            out[t] = np.loadtxt(os.path.join(P[part], t, f"{t}.arcs"), dtype=np.int64, ndmin=2)
    return out


def subsample(X, y, n, seed=SEED):
    if len(X) <= n:
        return X, y
    idx = np.random.RandomState(seed).choice(len(X), n, replace=False)
    return X[idx], y[idx]


def fit_model(name, X, y, groups=None, params=None):
    model = make_model(name, params)
    if name == "knn":
        X, y = subsample(X, y, KNN_MAX_ROWS)
    t0 = time.perf_counter()
    if name == "mlp":
        # internal validation split by trip for early stopping / loss curves
        ug = np.unique(groups)
        rs = np.random.RandomState(SEED)
        val_g = set(rs.choice(ug, max(1, len(ug) // 10), replace=False))
        vm = np.array([g in val_g for g in groups])
        model.fit(X[~vm], y[~vm], X[vm], y[vm])
    else:
        model.fit(X, y)
    return model, time.perf_counter() - t0


def evaluate(name, model, cand, pts, arcs):
    X = cand[FEATURES].to_numpy(np.float32)
    t0 = time.perf_counter()
    prob = model.predict_proba(X)[:, 1]
    infer = time.perf_counter() - t0
    m, yhat = pair_metrics(cand, prob)
    pred = top_choice(cand, prob)
    top1, _ = observation_top1(pts, pred)
    cont, cont_pooled = route_continuity_by_trip(pred, arcs)
    m.update({"top1": top1, "route_continuity": cont, "route_continuity_pooled": cont_pooled,
              "infer_s": infer, "infer_ms_per_1k_rows": 1000 * infer / max(len(X), 1) * 1000 / 1000,
              "infer_ms_per_trip": 1000 * infer / cand["trip_id"].nunique()})
    return m, prob, yhat


def main(names):
    cand_tr, pts_tr = load("train")
    cand_te, pts_te = load("student_test")
    # train only on fixes whose reference label is reliable
    cand_tr = cand_tr[cand_tr["label_ok"] == 1].reset_index(drop=True)
    arcs = arcs_by_trip()
    X = cand_tr[FEATURES].to_numpy(np.float32)
    y = cand_tr["label"].to_numpy()
    groups = cand_tr["trip_id"].to_numpy()
    print(f"train rows {len(X)}, positives {y.mean():.3f}, test rows {len(cand_te)}")

    gkf = GroupKFold(n_splits=5)
    fold_rows, summary = [], []
    cmp_path = os.path.join(RES, "model_comparison.csv")
    prev = pd.read_csv(cmp_path) if os.path.exists(cmp_path) else pd.DataFrame()

    for name in names:
        print(f"===== {name}")
        oof = np.zeros(len(X))
        fold_metrics = []
        for k, (tr, va) in enumerate(gkf.split(X, y, groups)):
            model, ttrain = fit_model(name, X[tr], y[tr], groups[tr])
            cva = cand_tr.iloc[va]
            m, p, _ = evaluate(name, model, cva, pts_tr[pts_tr.trip_id.isin(cva.trip_id.unique())], arcs)
            oof[va] = p
            m.update({"model": name, "fold": k, "train_s": ttrain})
            fold_metrics.append(m)
            print(f"  fold {k}: macroF1 {m['macro_f1']:.4f} top1 {m['top1']:.4f} logloss {m['log_loss']:.4f} ({ttrain:.1f}s)")
        fold_rows += fold_metrics
        fdf = pd.DataFrame(fold_metrics)
        pd.DataFrame({"trip_id": cand_tr.trip_id, "pt_idx": cand_tr.pt_idx, "arc_id": cand_tr.arc_id, "prob": oof}) \
            .to_parquet(os.path.join(OOF, f"oof_{name}.parquet"))

        # training-set fit quality (to diagnose over/underfitting)
        model, ttrain = fit_model(name, X, y, groups)
        ug = np.unique(groups)
        tr_trips = set(np.random.RandomState(1).choice(ug, 20, replace=False))
        sel = cand_tr.trip_id.isin(tr_trips).to_numpy()
        mtr, _, _ = evaluate(name, model, cand_tr[sel], pts_tr[pts_tr.trip_id.isin(tr_trips)], arcs)
        mte, prob_te, yhat_te = evaluate(name, model, cand_te, pts_te, arcs)
        cm = confusion_matrix(cand_te["label"], yhat_te)
        joblib.dump(model, os.path.join(EXP, f"{name}.joblib"))
        pd.DataFrame({"trip_id": cand_te.trip_id, "pt_idx": cand_te.pt_idx, "arc_id": cand_te.arc_id, "prob": prob_te}) \
            .to_parquet(os.path.join(OOF, f"test_{name}.parquet"))
        if name == "mlp":
            with open(os.path.join(RES, "mlp_history.json"), "w") as f:
                json.dump(model.history, f)
        full, fam, conf = MODEL_INFO[name]
        row = {
            "model": name, "model_name": full, "family": fam, "key_config": conf,
            "cv_macro_f1_mean": fdf.macro_f1.mean(), "cv_macro_f1_std": fdf.macro_f1.std(),
            "cv_top1_mean": fdf.top1.mean(), "cv_top1_std": fdf.top1.std(), "cv_log_loss_mean": fdf.log_loss.mean(),
            "train_macro_f1_20trips": mtr["macro_f1"], "train_top1_20trips": mtr["top1"], "train_log_loss_20trips": mtr["log_loss"],
            "test_macro_f1": mte["macro_f1"], "test_top1": mte["top1"], "test_log_loss": mte["log_loss"],
            "test_precision_pos": mte["precision_pos"], "test_recall_pos": mte["recall_pos"], "test_f1_pos": mte["f1_pos"],
            "test_route_continuity": mte["route_continuity"],
            "train_time_s_full": ttrain, "infer_ms_per_trip": mte["infer_ms_per_trip"],
            "test_cm_tn": cm[0, 0], "test_cm_fp": cm[0, 1], "test_cm_fn": cm[1, 0], "test_cm_tp": cm[1, 1],
        }
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        summary.append(row)
        # write incrementally
        cur = pd.DataFrame(summary)
        allrows = pd.concat([prev[~prev.model.isin(cur.model)] if len(prev) else prev, cur], ignore_index=True)
        allrows.to_csv(cmp_path, index=False)
        fpath = os.path.join(RES, "cv_folds.csv")
        fprev = pd.read_csv(fpath) if os.path.exists(fpath) else pd.DataFrame()
        fcur = pd.DataFrame(fold_rows)
        pd.concat([fprev[~fprev.model.isin(fcur.model)] if len(fprev) else fprev, fcur], ignore_index=True).to_csv(fpath, index=False)


if __name__ == "__main__":
    names = sys.argv[1:] or ["logreg", "gnb", "knn", "rf", "hgb", "lgbm", "mlp"]
    main(names)
