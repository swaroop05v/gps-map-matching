"""Evaluation metrics.

Definitions used everywhere in the project:
* pair-level Macro F1: every (fix, candidate) pair has a binary label. A pair is predicted
  positive iff it is the highest-scoring candidate of its fix (no threshold to tune).
  Macro F1 = mean of the F1 of the positive and the negative class.
* Log loss: binary cross-entropy of the pair probabilities.
* Observation-level Top-1 accuracy: fraction of fixes (with a reliable reference label) whose
  top-scoring candidate equals the reference arc. Fixes with no candidates, or whose reference
  arc is missing from the candidate set, count as wrong.
* Candidate miss rate: fraction of reliable fixes whose reference arc is not a candidate.
* Route continuity: after collapsing repeated arcs, fraction of consecutive predicted arc
  changes (a -> b) where b starts where a ends (to(a) == from(b)).
* Route arc-set precision / recall / F1: unique predicted arcs vs unique reference route arcs.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def top_choice(cand, prob):
    """Per fix, the arc with the highest probability. Returns DataFrame trip_id, pt_idx, pred_arc, pred_prob."""
    d = cand[["trip_id", "pt_idx", "arc_id"]].copy()
    d["p"] = prob
    d = d.sort_values(["trip_id", "pt_idx", "p"], ascending=[True, True, False], kind="stable")
    first = d.drop_duplicates(["trip_id", "pt_idx"], keep="first")
    return first.rename(columns={"arc_id": "pred_arc", "p": "pred_prob"})[["trip_id", "pt_idx", "pred_arc", "pred_prob"]]


def is_top_mask(cand, prob):
    top = top_choice(cand, prob)
    key = cand["trip_id"].astype(str) + "_" + cand["pt_idx"].astype(str) + "_" + cand["arc_id"].astype(str)
    tkey = top["trip_id"].astype(str) + "_" + top["pt_idx"].astype(str) + "_" + top["pred_arc"].astype(str)
    return key.isin(set(tkey)).to_numpy().astype(int)


def pair_metrics(cand, prob):
    y = cand["label"].to_numpy()
    yhat = is_top_mask(cand, prob)
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return {
        "macro_f1": f1_score(y, yhat, average="macro"),
        "precision_pos": precision_score(y, yhat, zero_division=0),
        "recall_pos": recall_score(y, yhat, zero_division=0),
        "f1_pos": f1_score(y, yhat, zero_division=0),
        "log_loss": log_loss(y, p, labels=[0, 1]),
    }, yhat


def observation_top1(points, pred):
    """points: per-fix table with ref_arc, label_ok. pred: trip_id, pt_idx, pred_arc."""
    pts = points[points["label_ok"] == 1][["trip_id", "pt_idx", "ref_arc"]]
    m = pts.merge(pred[["trip_id", "pt_idx", "pred_arc"]], on=["trip_id", "pt_idx"], how="left")
    correct = (m["pred_arc"] == m["ref_arc"]).to_numpy()
    return float(correct.mean()), m.assign(correct=correct)


def continuity(arc_seq, arcs):
    seq = [a for a in arc_seq if a is not None and a >= 0]
    if len(seq) < 2:
        return 1.0, 0, 0
    s = np.array(seq)
    s = s[np.r_[True, s[1:] != s[:-1]]]
    if len(s) < 2:
        return 1.0, 0, 0
    ok = arcs[s[:-1], 1] == arcs[s[1:], 0]
    return float(ok.mean()), int(ok.sum()), int(len(ok))


def route_set_prf(pred_arcs, ref_route):
    P = set(int(a) for a in pred_arcs if a is not None and a >= 0)
    R = set(int(a) for a in ref_route)
    if not P or not R:
        return 0.0, 0.0, 0.0
    tp = len(P & R)
    pr, rc = tp / len(P), tp / len(R)
    f1 = 0.0 if tp == 0 else 2 * pr * rc / (pr + rc)
    return pr, rc, f1


def route_continuity_by_trip(pred, arcs_by_trip):
    """Mean over trips of the continuity of the per-fix predicted arc sequence."""
    vals, oks, tots = [], 0, 0
    for tid, g in pred.sort_values(["trip_id", "pt_idx"]).groupby("trip_id"):
        c, ok, tot = continuity(g["pred_arc"].to_numpy(), arcs_by_trip[tid])
        vals.append(c)
        oks += ok
        tots += tot
    return float(np.mean(vals)), (oks / tots if tots else 1.0)
