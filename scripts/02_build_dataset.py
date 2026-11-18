"""Build the candidate/feature tables for the train and student_test partitions.

Output:
  data/processed/candidates_<partition>.parquet  one row per (GPS fix, candidate arc)
  data/processed/points_<partition>.parquet      one row per GPS fix (labels, miss info)
  results/candidate_generation_summary.csv
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap.config import load_config
from pathsnap.features import PreparedTrip
from pathsnap.io import list_trips, load_trip

LABEL_MAX_DIST = 50.0  # fixes further than this from their aligned route arc get an unreliable label


def process(part_dir, tid, cfg):
    t0 = time.time()
    trip = load_trip(part_dir, tid)
    pt = PreparedTrip(trip, cfg, with_labels=True)
    cand = pt.candidate_table()
    pts = pt.point_table()
    pts["label_ok"] = (pts["align_dist"] <= LABEL_MAX_DIST).astype(np.int8)
    cand = cand.merge(pts[["pt_idx", "label_ok"]], on="pt_idx", how="left")
    return cand, pts, time.time() - t0


def main():
    cfg = load_config()
    os.makedirs(cfg["paths"]["processed"], exist_ok=True)
    rows = []
    for part in ["train", "student_test"]:
        d = cfg["paths"][part]
        tids = list_trips(d)
        res = Parallel(n_jobs=8, verbose=5)(delayed(process)(d, t, cfg) for t in tids)
        cand = pd.concat([r[0] for r in res], ignore_index=True)
        pts = pd.concat([r[1] for r in res], ignore_index=True)
        cand.to_parquet(os.path.join(cfg["paths"]["processed"], f"candidates_{part}.parquet"))
        pts.to_parquet(os.path.join(cfg["paths"]["processed"], f"points_{part}.parquet"))
        ok = pts["label_ok"] == 1
        rows.append({
            "partition": part,
            "trips": len(tids),
            "gps_points": len(pts),
            "candidate_rows": len(cand),
            "mean_candidates_per_point": cand.groupby(["trip_id", "pt_idx"]).size().mean(),
            "points_without_candidates_pct": 100 * (pts["n_cands"] == 0).mean(),
            "reliable_label_points_pct": 100 * ok.mean(),
            "candidate_miss_rate_all_pct": 100 * (1 - pts["ref_in_cands"].mean()),
            "candidate_miss_rate_reliable_pct": 100 * (1 - pts.loc[ok, "ref_in_cands"].mean()),
            "nearest_arc_top1_reliable_pct": 100 * cand[(cand.dist_rank == 0) & (cand.label_ok == 1)]["label"].mean(),
            "build_seconds_total": sum(r[2] for r in res),
        })
        print(rows[-1])
    os.makedirs(cfg["paths"]["results"], exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(cfg["paths"]["results"], "candidate_generation_summary.csv"), index=False)


if __name__ == "__main__":
    main()
