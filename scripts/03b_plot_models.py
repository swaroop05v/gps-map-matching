"""Model comparison figures, all generated from results/model_comparison.csv, results/cv_folds.csv
and results/mlp_history.json (actual experiment outputs)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.config import load_config
from pathsnap.features import FEATURES

cfg = load_config()
RES = cfg["paths"]["results"]
FIG = os.path.join(RES, "figures")
ORDER = ["logreg", "gnb", "knn", "rf", "hgb", "lgbm", "mlp"]


def main():
    cmp = pd.read_csv(os.path.join(RES, "model_comparison.csv"))
    cmp = cmp.set_index("model").loc[[m for m in ORDER if m in set(cmp.model)]].reset_index()
    folds = pd.read_csv(os.path.join(RES, "cv_folds.csv"))
    labels = cmp.model_name.str.replace(" (", "\n(", regex=False)
    x = np.arange(len(cmp))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x, cmp.cv_macro_f1_mean, yerr=cmp.cv_macro_f1_std, capsize=4, color="#3b6ea5")
    for i, v in enumerate(cmp.cv_macro_f1_mean):
        ax.text(i, v + 0.004, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels, fontsize=8); ax.set_ylim(0.85, 1.0)
    ax.set(ylabel="Macro F1 (mean ± std over 5 folds)", title="5-fold GroupKFold CV Macro F1 (80 train trips)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "models_cv_macro_f1.png"), dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - 0.2, cmp.test_top1, 0.4, label="Top-1 accuracy (observation level)", color="#2a9d8f")
    ax.bar(x + 0.2, cmp.test_macro_f1, 0.4, label="Macro F1 (pair level)", color="#e9c46a")
    for i in range(len(cmp)):
        ax.text(i - 0.2, cmp.test_top1[i] + 0.004, f"{cmp.test_top1[i]:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.2, cmp.test_macro_f1[i] + 0.004, f"{cmp.test_macro_f1[i]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels, fontsize=8); ax.set_ylim(0.8, 1.0); ax.legend(loc="lower right")
    ax.set(title="Student-test (10 trips): Top-1 accuracy and Macro F1, classifier only")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "models_test_top1_f1.png"), dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - 0.2, cmp.cv_log_loss_mean, 0.4, label="CV", color="#8172b2")
    ax.bar(x + 0.2, cmp.test_log_loss, 0.4, label="student_test", color="#e07b39")
    ax.set_yscale("log"); ax.set_xticks(x, labels, fontsize=8); ax.legend()
    ax.set(ylabel="binary log loss (log scale)", title="Log loss of candidate probabilities (lower is better)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "models_log_loss.png"), dpi=130); plt.close()

    # stability across folds
    fig, ax = plt.subplots(figsize=(11, 4.5))
    data = [folds[folds.model == m].top1.values for m in cmp.model]
    ax.boxplot(data)
    ax.set_xticks(np.arange(1, len(cmp) + 1), labels, fontsize=8)
    ax.set(ylabel="Top-1 accuracy per fold", title="Stability across the 5 CV folds (observation-level Top-1)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "models_cv_fold_stability.png"), dpi=130); plt.close()

    # over/underfitting: train vs CV vs test top-1
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - 0.27, cmp.train_top1_20trips, 0.27, label="train (fit data, 20 trips)", color="#bbbbbb")
    ax.bar(x, cmp.cv_top1_mean, 0.27, label="CV (held-out trips)", color="#3b6ea5")
    ax.bar(x + 0.27, cmp.test_top1, 0.27, label="student_test", color="#2a9d8f")
    ax.set_xticks(x, labels, fontsize=8); ax.set_ylim(0.8, 1.0); ax.legend(loc="lower right")
    ax.set(ylabel="Top-1 accuracy", title="Train vs cross-validation vs student-test Top-1 (generalisation gap)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "models_generalisation_gap.png"), dpi=130); plt.close()

    # efficiency
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].bar(x, cmp.train_time_s_full, color="#6c757d"); axs[0].set_yscale("log")
    axs[0].set_xticks(x, cmp.model, fontsize=8); axs[0].set(title="Training time on 80 trips (s, log)")
    axs[1].bar(x, cmp.infer_ms_per_trip, color="#6c757d"); axs[1].set_yscale("log")
    axs[1].set_xticks(x, cmp.model, fontsize=8); axs[1].set(title="Inference time per trip (ms, log)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "models_time.png"), dpi=130); plt.close()

    # confusion matrices (student_test, pair level, positive = top-ranked candidate)
    n = len(cmp)
    fig, axs = plt.subplots(1, n, figsize=(3 * n, 3.2))
    for ax, (_, r) in zip(np.atleast_1d(axs), cmp.iterrows()):
        cm = np.array([[r.test_cm_tn, r.test_cm_fp], [r.test_cm_fn, r.test_cm_tp]])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(r.model, fontsize=10)
    plt.suptitle("Confusion matrices on student_test candidate pairs")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "models_confusion_matrices.png"), dpi=130); plt.close()

    hp = os.path.join(RES, "mlp_history.json")
    if os.path.exists(hp):
        h = json.load(open(hp))
        fig, ax = plt.subplots(figsize=(7, 4))
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], "o-", label="training loss")
        ax.plot(ep[:len(h["val_loss"])], h["val_loss"], "o-", label="validation loss (8 held-out train trips)")
        ax.set(xlabel="epoch", ylabel="binary cross-entropy", title="MLP training history (final fit)")
        ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG, "mlp_loss_curve.png"), dpi=130); plt.close()

    mp = os.path.join(cfg["paths"]["models"], "experiments", "lgbm.joblib")
    if os.path.exists(mp):
        m = joblib.load(mp)
        imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES).sort_values()
        imp.to_csv(os.path.join(RES, "lgbm_feature_importance_gain.csv"))
        fig, ax = plt.subplots(figsize=(7, 7))
        (imp / imp.sum()).plot.barh(ax=ax, color="#3b6ea5")
        ax.set(xlabel="share of total gain", title="LightGBM feature importance (gain)")
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "lgbm_feature_importance.png"), dpi=130); plt.close()

    cols = ["model_name", "family", "key_config", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_mean", "test_top1",
            "test_route_continuity", "test_macro_f1", "test_log_loss", "test_precision_pos", "test_recall_pos",
            "train_time_s_full", "infer_ms_per_trip"]
    cmp[cols].round(4).to_csv(os.path.join(RES, "model_comparison_table.csv"), index=False)
    print(cmp[cols].round(4).to_string())


if __name__ == "__main__":
    main()
