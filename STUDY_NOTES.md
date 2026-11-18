# Study notes: Shashank Keshava Murthy (models, metrics, Bayesian optimisation)

*For studying only. Do not commit (add `STUDY_NOTES.md` to `.gitignore`).*

## Your files in plain English

| File | What it does |
|---|---|
| `src/pathsnap/models.py` | Builds the 7 classifiers: Logistic Regression, Gaussian NB, KNN, Random Forest, HistGradientBoosting, LightGBM, and a PyTorch MLP (`TorchMLP`, which keeps train/val loss per epoch). Torch is imported lazily (see the decisions below). |
| `src/pathsnap/metrics.py` | All metric definitions: pair-level Macro F1, log loss, observation-level Top-1, route continuity, route precision/recall/F1. |
| `scripts/03_train_compare.py` | For each model: 5-fold `GroupKFold` by trip on the 80 train trips, then a refit on all 80 and evaluation on student_test. Saves `model_comparison.csv`, `cv_folds.csv`, models in `models/experiments/` and probabilities. |
| `scripts/03b_plot_models.py` | Comparison figures: CV F1 with error bars, test Top-1/F1, log loss, fold stability, train vs CV vs test gap, timings, confusion matrices, MLP loss curve, LightGBM feature importance. |
| `scripts/04_bayes_opt.py` | Optuna TPE search for LightGBM: 8 hyperparameters, objective = mean 3-fold CV Macro F1 (train only), median pruner. Then untuned vs tuned on identical partitions → `tuned_vs_untuned.csv`. |
| `models/final_model.joblib` | The tuned LightGBM the API uses. `final_model_meta.json` records its params and features. |

## How the task is set up as ML
Each row is one (GPS fix, candidate arc) pair: label 1 if the candidate is the fix's reference arc, else 0 (about 15% positives). For each fix, the candidate with the highest probability is the prediction.

## Metrics (know these definitions)
* **Top-1 (observation level):** the share of reliable fixes whose top candidate equals the reference arc. A fix whose true arc isn't a candidate counts as wrong.
* **Macro F1 (pair level):** a pair is predicted positive if it is its fix's top candidate (no threshold to tune). We average the F1 of class 1 and class 0.
* **Log loss:** binary cross-entropy of the probabilities. It matters because the HMM uses these probabilities.
* **Candidate miss rate:** 0.66% train / 0.12% student_test.

## Results (student_test; CV is 5-fold mean)
| Model | CV Macro F1 | CV Top-1 | Test Top-1 | Test log loss |
|---|---|---|---|---|
| Logistic Regression | 0.9378 | 0.892 | 0.883 | 0.103 |
| Gaussian NB | 0.9234 | 0.868 | 0.848 | 0.648 |
| KNN (k = 25) | 0.9350 | 0.887 | 0.882 | 0.129 |
| Random Forest | **0.9443** | **0.903** | 0.901 | 0.073 |
| HistGradientBoosting | 0.9395 | 0.895 | 0.906 | 0.085 |
| **LightGBM** | 0.9435 | 0.901 | **0.914** | **0.072** |
| MLP | 0.9423 | 0.899 | 0.899 | 0.079 |

**Why LightGBM:** it ties RF on CV (the difference is smaller than one fold std of about 0.011), but it is better on student_test Top-1, F1 and log loss. It is also 30x smaller (2.8 MB vs 87 MB), fast at inference, and has regularisation knobs worth tuning.

**Over/underfitting trick:** Logistic Regression can't memorise, yet it has a 3.7 pp train-vs-CV gap. That gap is just "the 20 sampled train trips are easier", so we compare every model against it. LightGBM and KNN are about 2.6 pp above it (mild overfitting), RF 1.4 pp, MLP and HGB about 1 to 1.5 pp. GNB is low everywhere, which is underfitting from model bias.

**Tuning result (be honest about it):** the best params (lr 0.028, 467 trees, 119 leaves, max_depth 6, min_child_samples 188, subsample and colsample about 0.57) are *more regularised*. CV Macro F1 0.9435 → 0.9445, CV log loss 0.082 → 0.077. On student_test, Top-1 went 0.914 → 0.910, which is within trip-to-trip noise on 10 trips. We kept the tuned model because it was chosen on train data only and has better probabilities. 25 trials (14 complete, 11 pruned) took about 12.5 minutes. One stale trial from an interrupted run sits in the DB as RUNNING and is excluded.

## Other decisions
* **GroupKFold by trip:** fixes from one trip are highly correlated (1 Hz), so a random split would leak and inflate scores.
* **Scaler inside a `Pipeline`:** it is fitted on training folds only, so there is no leakage.
* **KNN on a 200k-row subsample:** prediction is about 1.35 s per trip even so, much slower than the others.
* **Lazy torch import:** loading PyTorch and LightGBM in one process on Windows made LightGBM about 9x slower (they ship different OpenMP runtimes), so torch loads only inside the MLP and the MLP runs in its own process.

## 5 likely viva questions
1. **Why GroupKFold and not KFold?** Fixes from the same trip are near-duplicates, so a trip split across folds would let the model "see" the test trip. Grouping by trip simulates unseen trips like the evaluator set.
2. **What objective did Bayesian optimisation maximise, and on what data?** Mean pair-level Macro F1 over a 3-fold GroupKFold on the 80 train trips. Student_test was used only after the search, to compare tuned vs untuned.
3. **How does TPE differ from grid search?** It models good vs bad regions of the search space from past trials and samples where improvement is likely. It needs far fewer trials than a grid, and the median pruner stops weak trials after one fold.
4. **Why is Naive Bayes's log loss so high (0.65)?** It assumes features are independent, but our distance features are strongly correlated, so it counts the same evidence several times and becomes over-confident.
5. **Did tuning help?** On the objective, yes: CV F1 +0.1 pp, better log loss and lower fold std. On student_test it was −0.4 pp Top-1, which is within the noise of 10 trips. The search surface was flat (all completed trials within 0.9437–0.9451).
