"""Exploratory data analysis on the train and student_test partitions.

Writes tables to results/eda/ and figures to results/figures/eda_*.png and
Folium maps to results/maps/eda_*.html. The evaluator partition is never read here.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.config import load_config
from pathsnap.geo import LocalProjection, haversine
from pathsnap.io import list_trips, load_trip
from pathsnap.viz import trip_map

cfg = load_config()
RES = cfg["paths"]["results"]
FIG = os.path.join(RES, "figures")
EDA = os.path.join(RES, "eda")
MAPS = os.path.join(RES, "maps")
for d in (FIG, EDA, MAPS):
    os.makedirs(d, exist_ok=True)


def trip_stats(part, tid):
    d = cfg["paths"][part]
    tr = load_trip(d, tid)
    track, nodes, arcs, route = tr.track, tr.nodes, tr.arcs, tr.route
    sizes = {ext: os.path.getsize(os.path.join(d, tid, f"{tid}.{ext}")) for ext in ["track", "nodes", "arcs", "route"]}
    dt = np.diff(track[:, 2])
    lon, lat = track[:, 0], track[:, 1]
    step = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
    # projection check: local equirectangular vs haversine
    proj = LocalProjection(lon.mean(), lat.mean())
    xy = proj.to_xy(lon, lat)
    step_xy = np.sqrt((np.diff(xy, axis=0) ** 2).sum(1))
    proj_err = np.abs(step_xy - step).max() if len(step) else 0
    n = len(nodes)
    G = csr_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
    n_weak, _ = connected_components(G, directed=True, connection="weak")
    n_strong, lab_s = connected_components(G, directed=True, connection="strong")
    deg = np.bincount(arcs[:, 0], minlength=n) + np.bincount(arcs[:, 1], minlength=n)
    key = set(map(tuple, arcs.tolist()))
    bidir = np.mean([(b, a) in key for a, b in arcs.tolist()]) if len(arcs) else 0
    arc_len = haversine(nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1], nodes[arcs[:, 1], 0], nodes[arcs[:, 1], 1])
    route_breaks = int(np.sum(arcs[route[:-1], 1] != arcs[route[1:], 0])) if route is not None and len(route) > 1 else 0
    route_len_m = float(arc_len[route].sum()) if route is not None else np.nan
    return {
        "partition": part, "trip_id": tid,
        "n_gps": len(track), "n_nodes": n, "n_arcs": len(arcs), "n_route_arcs": 0 if route is None else len(route),
        "size_track_kb": sizes["track"] / 1024, "size_nodes_kb": sizes["nodes"] / 1024,
        "size_arcs_kb": sizes["arcs"] / 1024, "size_route_kb": sizes["route"] / 1024,
        "nan_values": int(np.isnan(track).sum()),
        "invalid_coords": int(((np.abs(lon) > 180) | (np.abs(lat) > 90)).sum()),
        "non_increasing_t": int((dt <= 0).sum()),
        "duplicate_positions": int((step == 0).sum()),
        "duration_s": float(track[-1, 2] - track[0, 2]),
        "pct_dt_1s": 100 * float(np.mean(dt == 1)) if len(dt) else 100,
        "max_dt_s": float(dt.max()) if len(dt) else 0,
        "n_gaps_gt_60s": int((dt > 60).sum()),
        "median_step_m": float(np.median(step)) if len(step) else 0,
        "max_step_m": float(step.max()) if len(step) else 0,
        "track_length_km": float(step.sum() / 1000),
        "route_length_km": route_len_m / 1000,
        "lon_min": lon.min(), "lon_max": lon.max(), "lat_min": lat.min(), "lat_max": lat.max(),
        "extent_km_ew": float(haversine(lon.min(), lat.mean(), lon.max(), lat.mean()) / 1000),
        "extent_km_ns": float(haversine(lon.mean(), lat.min(), lon.mean(), lat.max()) / 1000),
        "weak_components": n_weak, "strong_components": n_strong,
        "largest_scc_frac": float(np.bincount(lab_s).max() / n),
        "isolated_nodes": int((deg == 0).sum()),
        "mean_out_degree": float(len(arcs) / n),
        "pct_bidirectional_arcs": 100 * float(bidir),
        "median_arc_len_m": float(np.median(arc_len)),
        "route_breaks": route_breaks,
        "max_proj_err_m": float(proj_err),
    }, dt, step


def main():
    jobs = [(p, t) for p in ["train", "student_test"] for t in list_trips(cfg["paths"][p])]
    out = Parallel(n_jobs=8)(delayed(trip_stats)(p, t) for p, t in jobs)
    df = pd.DataFrame([o[0] for o in out])
    df.to_csv(os.path.join(EDA, "trip_inventory.csv"), index=False)

    # partition summary
    summ = df.groupby("partition").agg(
        trips=("trip_id", "count"), gps_points=("n_gps", "sum"), mean_gps=("n_gps", "mean"),
        min_gps=("n_gps", "min"), max_gps=("n_gps", "max"), mean_nodes=("n_nodes", "mean"),
        mean_arcs=("n_arcs", "mean"), mean_route_arcs=("n_route_arcs", "mean"),
        trips_with_gaps_gt_60s=("n_gaps_gt_60s", lambda s: int((s > 0).sum())),
        total_nan=("nan_values", "sum"), total_invalid=("invalid_coords", "sum"),
        total_non_increasing_t=("non_increasing_t", "sum"), total_dup_positions=("duplicate_positions", "sum"),
        mean_pct_bidirectional=("pct_bidirectional_arcs", "mean"), route_breaks=("route_breaks", "sum"),
        max_proj_err_m=("max_proj_err_m", "max"),
    )
    summ.to_csv(os.path.join(EDA, "partition_summary.csv"))
    print(summ.T.to_string())

    # missing / invalid value analysis
    miss = df.groupby("partition")[["nan_values", "invalid_coords", "non_increasing_t", "duplicate_positions", "n_gaps_gt_60s"]].sum()
    miss.to_csv(os.path.join(EDA, "missing_invalid_summary.csv"))
    fig, ax = plt.subplots(figsize=(8, 4))
    miss.T.plot.bar(ax=ax)
    ax.set_title("Missing / invalid GPS observations (train + student_test)")
    ax.set_ylabel("count")
    plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_missing_invalid.png"), dpi=130); plt.close()

    # sampling interval + step length
    dts = np.concatenate([o[1] for o, (p, _) in zip(out, jobs) if p == "train"])
    steps = np.concatenate([o[2] for o, (p, _) in zip(out, jobs) if p == "train"])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].hist(np.clip(dts, 0, 20), bins=40, color="#3b6ea5")
    axs[0].set_yscale("log"); axs[0].set_title("Sampling interval dt (train, clipped at 20 s)"); axs[0].set_xlabel("seconds")
    axs[1].hist(np.clip(steps, 0, 60), bins=60, color="#e07b39")
    axs[1].set_title("Distance between consecutive fixes (train)"); axs[1].set_xlabel("metres (clipped at 60)")
    axs[2].hist(df[df.partition == "train"]["n_gps"], bins=30, color="#55a868")
    axs[2].set_title("GPS observations per trip (train)"); axs[2].set_xlabel("observations")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_sampling.png"), dpi=130); plt.close()

    # gaps per trip
    fig, ax = plt.subplots(figsize=(12, 4))
    d2 = df.sort_values("max_dt_s", ascending=False).head(25)
    ax.bar(d2["trip_id"].str.lstrip("0"), d2["max_dt_s"], color=np.where(d2.partition == "train", "#3b6ea5", "#e07b39"))
    ax.set_yscale("log"); ax.set_ylabel("largest gap (s, log)"); ax.set_title("Trips with the largest GPS time gaps (blue=train, orange=student_test)")
    plt.xticks(rotation=90); plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_gaps.png"), dpi=130); plt.close()

    # network size and connectivity
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for p, c in [("train", "#3b6ea5"), ("student_test", "#e07b39")]:
        s = df[df.partition == p]
        axs[0].scatter(s.n_nodes, s.n_arcs, label=p, color=c, s=15)
        axs[1].scatter(s.n_gps, s.n_route_arcs, label=p, color=c, s=15)
        axs[2].scatter(s.n_arcs, s.strong_components, label=p, color=c, s=15)
    axs[0].set(xscale="log", yscale="log", xlabel="nodes", ylabel="arcs", title="Road-network size per trip")
    axs[1].set(xlabel="GPS observations", ylabel="route arcs", title="Trajectory vs reference route length")
    axs[2].set(xscale="log", yscale="log", xlabel="arcs", ylabel="strongly connected components", title="Network connectivity")
    for a in axs:
        a.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_network.png"), dpi=130); plt.close()

    # GPS coordinate distribution (all trip centroids)
    fig, ax = plt.subplots(figsize=(6, 6))
    for p, c in [("train", "#3b6ea5"), ("student_test", "#e07b39")]:
        s = df[df.partition == p]
        ax.scatter((s.lon_min + s.lon_max) / 2, (s.lat_min + s.lat_max) / 2, label=p, color=c, s=20)
    ax.set(xlabel="longitude", ylabel="latitude", title="Trip centroids (all partitions given to students)")
    ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_trip_centroids.png"), dpi=130); plt.close()

    # trajectory + road-network plots for 4 trips
    sample = ["00000001", "00000003", "00000004", "00000008"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 12))
    for ax, tid in zip(axs.ravel(), sample):
        tr = load_trip(cfg["paths"]["train"], tid)
        proj = LocalProjection(tr.track[:, 0].mean(), tr.track[:, 1].mean())
        nxy = proj.to_xy(tr.nodes[:, 0], tr.nodes[:, 1])
        txy = proj.to_xy(tr.track[:, 0], tr.track[:, 1])
        pad = 300
        lo, hi = txy.min(0) - pad, txy.max(0) + pad
        A, B = nxy[tr.arcs[:, 0]], nxy[tr.arcs[:, 1]]
        m = ((A >= lo) & (A <= hi)).all(1)
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(np.stack([A[m], B[m]], 1), colors="#bbbbbb", linewidths=0.6, label="road network"))
        R = tr.route
        ax.add_collection(LineCollection(np.stack([A[R], B[R]], 1), colors="#1a9e3a", linewidths=3, label="reference route"))
        ax.scatter(txy[:, 0], txy[:, 1], s=2, c="#1f4fd8", label="GPS fixes", zorder=3)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_title(f"train trip {tid}: {len(tr.track)} fixes, {len(R)} route arcs"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_trajectories_network.png"), dpi=130); plt.close()

    # GPS noise: distance from fix to its aligned reference arc (needs scripts/02 output)
    pp = os.path.join(cfg["paths"]["processed"], "points_train.parquet")
    if os.path.exists(pp):
        pts = pd.read_parquet(pp)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(np.clip(pts.align_dist, 0, 100), bins=100, color="#8172b2")
        ax.axvline(50, color="red", ls="--", label="50 m reliability threshold")
        ax.set(xlabel="distance from GPS fix to its aligned reference arc (m, clipped 100)", ylabel="fixes",
               title="GPS noise relative to reference route (train)")
        ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG, "eda_gps_noise.png"), dpi=130); plt.close()
        q = pts.align_dist.quantile([0.5, 0.9, 0.95, 0.99]).rename("align_dist_m")
        q.to_csv(os.path.join(EDA, "gps_noise_quantiles_train.csv"))
        bad = pts.groupby("trip_id").apply(lambda g: 100 * (g.align_dist > 50).mean()).sort_values(ascending=False)
        bad.rename("pct_fixes_gt_50m_from_route").to_csv(os.path.join(EDA, "unreliable_label_pct_per_trip.csv"))

    # Folium maps
    for tid in ["00000001", "00000004"]:
        tr = load_trip(cfg["paths"]["train"], tid)
        m = trip_map(tr.nodes, tr.arcs, tr.track, ref_route=tr.route, title=f"EDA train trip {tid}")
        m.save(os.path.join(MAPS, f"eda_train_{tid}.html"))
    tr = load_trip(cfg["paths"]["train"], "00000002")
    m = trip_map(tr.nodes, tr.arcs, tr.track, ref_route=tr.route, title="EDA train trip 00000002 (GPS off-route)")
    m.save(os.path.join(MAPS, "eda_train_00000002_offroute.html"))
    print("EDA done")


if __name__ == "__main__":
    main()
