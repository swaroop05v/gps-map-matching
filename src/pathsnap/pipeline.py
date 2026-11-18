"""End-to-end inference for one trip: raw files -> features -> classifier -> Viterbi -> route.

Used by the evaluation scripts and by the API, so both run exactly the same code.
Reference labels are NOT used here; scoring happens separately in score_trip().
"""
import time

import numpy as np

from .features import FEATURES, PreparedTrip
from .hmm import build_route, build_states, viterbi
from .io import Trip
from .metrics import continuity, route_set_prf


def predict_trip(trip: Trip, model, cfg, hcfg, use_hmm=True):
    timings = {}
    t0 = time.perf_counter()
    # labels are deliberately stripped before prediction
    blind = Trip(trip.trip_id, trip.track, trip.nodes, trip.arcs, None)
    prep = PreparedTrip(blind, cfg, with_labels=False)
    cand = prep.candidate_table()
    timings["features_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    prob = model.predict_proba(cand[FEATURES].to_numpy(np.float32))[:, 1] if len(cand) else np.zeros(0)
    timings["classifier_s"] = time.perf_counter() - t1

    n = len(prep.xy)
    # classifier-only: argmax per fix
    clf_pred = np.full(n, -1, dtype=np.int64)
    if len(cand):
        order = np.lexsort((-prob, cand["pt_idx"].to_numpy()))
        pts_sorted = cand["pt_idx"].to_numpy()[order]
        first = np.r_[True, pts_sorted[1:] != pts_sorted[:-1]]
        clf_pred[pts_sorted[first]] = cand["arc_id"].to_numpy()[order][first]

    t2 = time.perf_counter()
    if use_hmm:
        states = build_states(cand, prob, n, hcfg["beam"], hcfg["alpha"])
        hmm_pred = viterbi(prep, states, hcfg)
    else:
        hmm_pred = clf_pred
    route, rinfo = build_route(prep, hmm_pred)
    timings["hmm_s"] = time.perf_counter() - t2
    timings["total_s"] = time.perf_counter() - t0
    return {"prep": prep, "cand": cand, "prob": prob, "clf_pred": clf_pred, "hmm_pred": hmm_pred,
            "route": route, "route_info": rinfo, "timings": timings}


def score_trip(prep_pred, ref_prep_or_labels, ref_route, arcs, label_ok=None):
    """Compare a per-fix prediction with reference per-fix arcs (server-side only)."""
    pred = np.asarray(prep_pred)
    ref = np.asarray(ref_prep_or_labels)
    mask = np.ones(len(ref), bool) if label_ok is None else np.asarray(label_ok, bool)
    acc = float((pred[mask] == ref[mask]).mean()) if mask.any() else float("nan")
    cont, _, _ = continuity(pred, arcs)
    pr, rc, f1 = route_set_prf(pred, ref_route)
    return {"top1_accuracy": acc, "route_continuity": cont, "route_precision": pr, "route_recall": rc, "route_f1": f1,
            "n_scored_points": int(mask.sum())}
