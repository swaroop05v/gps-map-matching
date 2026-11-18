"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected model (LightGBM).

* objective: mean pair-level Macro F1 over a 3-fold GroupKFold (grouped by trip) on the 80
  training trips; the student_test and evaluator partitions are never used in the search
* every trial is recorded in results/optuna_trials.csv and results/optuna_study.db
* the best configuration is refit on all training trips and compared with the untuned model
  on the same partitions (CV Macro F1, student-test Top-1 / Macro F1, log loss, inference time)
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
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.config import load_config
from pathsnap.features import FEATURES
from pathsnap.metrics import observation_top1, pair_metrics, route_continuity_by_trip, top_choice
from pathsnap.models import SEED, make_model

sys.path.insert(0, os.path.dirname(__file__))
cfg = load_config()
P = cfg["paths"]
RES, FIG = P["results"], os.path.join(P["results"], "figures")
N_TRIALS = int(os.environ.get("N_TRIALS", 30))
MODEL = "lgbm"

SPACE = {
    "learning_rate": ("log-uniform", 0.01, 0.3),
    "n_estimators": ("int", 100, 800),
    "num_leaves": ("int", 15, 255),
    "max_depth": ("int", 3, 14),
    "min_child_samples": ("int", 10, 300),
    "subsample": ("uniform", 0.5, 1.0),
    "colsample_bytree": ("uniform", 0.4, 1.0),
    "reg_lambda": ("log-uniform", 1e-4, 10.0),
}


def suggest(trial):
    p = {}
    for k, (kind, lo, hi) in SPACE.items():
        if kind == "int":
            p[k] = trial.suggest_int(k, lo, hi)
        elif kind == "uniform":
            p[k] = trial.suggest_float(k, lo, hi)
        else:
            p[k] = trial.suggest_float(k, lo, hi, log=True)
    return p


def cv_score(params, cand, X, y, groups, pts, n_splits, trial=None):
    scores, losses, top1s = [], [], []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        m = make_model(MODEL, params)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        cva = cand.iloc[va]
        mm, _ = pair_metrics(cva, p)
        t1, _ = observation_top1(pts[pts.trip_id.isin(cva.trip_id.unique())], top_choice(cva, p))
        scores.append(mm["macro_f1"]); losses.append(mm["log_loss"]); top1s.append(t1)
        if trial is not None:
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(losses)), float(np.mean(top1s))


def main():
    cand = pd.read_parquet(os.path.join(P["processed"], "candidates_train.parquet"))
    cand = cand[cand.label_ok == 1].reset_index(drop=True)
    pts = pd.read_parquet(os.path.join(P["processed"], "points_train.parquet"))
    X = cand[FEATURES].to_numpy(np.float32)
    y = cand["label"].to_numpy()
    groups = cand["trip_id"].to_numpy()

    storage = f"sqlite:///{os.path.join(RES, 'optuna_study.db')}"
    study = optuna.create_study(study_name="pathsnap_lgbm", direction="maximize", storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))
    # trial 0 = the untuned default configuration, so the search starts from it
    if len(study.trials) == 0:
        study.enqueue_trial({"learning_rate": 0.05, "n_estimators": 400, "num_leaves": 63, "max_depth": 14,
                             "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1e-4})

    def objective(trial):
        params = suggest(trial)
        t0 = time.perf_counter()
        f1, f1s, ll, t1 = cv_score(params, cand, X, y, groups, pts, 3, trial)
        trial.set_user_attr("cv_f1_std", f1s)
        trial.set_user_attr("cv_log_loss", ll)
        trial.set_user_attr("cv_top1", t1)
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return f1

    t0 = time.perf_counter()
    remaining = N_TRIALS - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    search_s = time.perf_counter() - t0

    df = study.trials_dataframe()
    df.to_csv(os.path.join(RES, "optuna_trials.csv"), index=False)
    best = study.best_params
    with open(os.path.join(RES, "optuna_best_params.json"), "w") as f:
        json.dump({"best_params": best, "best_cv3_macro_f1": study.best_value, "n_trials": len(study.trials),
                   "search_space": SPACE, "objective": "mean pair-level Macro F1, 3-fold GroupKFold by trip (train only)",
                   "sampler": "TPE (seed 42, 8 random start-up trials)", "pruner": "MedianPruner",
                   "search_seconds_this_run": search_s}, f, indent=2)
    print("best", best, study.best_value)

    # optimisation history + parameter importance plots
    comp = df[df.state == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(comp.number, comp.value, s=18, label="trial CV Macro F1")
    ax.plot(comp.number, comp.value.cummax(), color="red", label="best so far")
    ax.set(xlabel="trial", ylabel="3-fold CV Macro F1", title="Optuna TPE optimisation history (LightGBM, train trips)")
    ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG, "optuna_history.png"), dpi=130); plt.close()
    try:
        imp = optuna.importance.get_param_importances(study)
        pd.Series(imp).to_csv(os.path.join(RES, "optuna_param_importance.csv"))
        fig, ax = plt.subplots(figsize=(7, 4))
        pd.Series(imp).sort_values().plot.barh(ax=ax, color="#3b6ea5")
        ax.set(title="Hyperparameter importance (fANOVA)", xlabel="importance")
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "optuna_importance.png"), dpi=130); plt.close()
    except Exception as e:  # noqa
        print("importance failed", e)

    # ---- untuned vs tuned on identical partitions (5-fold CV + student_test)
    cand_te = pd.read_parquet(os.path.join(P["processed"], "candidates_student_test.parquet"))
    pts_te = pd.read_parquet(os.path.join(P["processed"], "points_student_test.parquet"))
    import importlib
    tc = importlib.import_module("03_train_compare")
    arcs = tc.arcs_by_trip()
    rows = []
    for tag, params in [("untuned", {}), ("tuned", best)]:
        f1, f1s, ll, t1 = cv_score(params, cand, X, y, groups, pts, 5)
        m = make_model(MODEL, params)
        t0 = time.perf_counter(); m.fit(X, y); ttrain = time.perf_counter() - t0
        Xte = cand_te[FEATURES].to_numpy(np.float32)
        t0 = time.perf_counter(); p = m.predict_proba(Xte)[:, 1]; tinf = time.perf_counter() - t0
        mm, _ = pair_metrics(cand_te, p)
        pred = top_choice(cand_te, p)
        top1, _ = observation_top1(pts_te, pred)
        cont, _ = route_continuity_by_trip(pred, arcs)
        rows.append({"model": f"lgbm_{tag}", "params": json.dumps(params), "cv5_macro_f1": f1, "cv5_macro_f1_std": f1s,
                     "cv5_top1": t1, "cv5_log_loss": ll, "test_top1": top1, "test_macro_f1": mm["macro_f1"],
                     "test_log_loss": mm["log_loss"], "test_route_continuity": cont, "train_s": ttrain,
                     "infer_ms_per_trip": 1000 * tinf / cand_te.trip_id.nunique()})
        print(rows[-1])
        if tag == "tuned":
            os.makedirs(P["models"], exist_ok=True)
            joblib.dump(m, os.path.join(P["models"], "experiments", "lgbm_tuned.joblib"))
            pd.DataFrame({"trip_id": cand_te.trip_id, "pt_idx": cand_te.pt_idx, "arc_id": cand_te.arc_id, "prob": p}) \
                .to_parquet(os.path.join(P["processed"], "oof", "test_lgbm_tuned.parquet"))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "tuned_vs_untuned.csv"), index=False)


if __name__ == "__main__":
    main()
