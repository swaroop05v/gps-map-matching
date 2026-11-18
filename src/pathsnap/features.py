"""Trip preprocessing, candidate generation and feature engineering.

One row = one (GPS observation, candidate road arc) pair. The classifier learns
P(candidate is the reference arc | features).
"""
import numpy as np
import pandas as pd

from .geo import LocalProjection, angle_diff, bearing_deg, point_to_segment
from .io import clean_track
from .labels import align_track_to_route
from .network import RoadNetwork

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands",
    "t_proj", "overshoot_m", "arc_len", "log_arc_len",
    "heading_valid", "angle_diff", "cos_angle", "angle_diff_long",
    "speed", "dt_prev", "dt_next", "turn",
    "ctx_mean_dist", "ctx_max_dist", "ctx_nearest_frac", "ctx_mean_dist_minus_min",
    "out_deg_to", "in_deg_from", "has_reverse",
]


class PreparedTrip:
    """Everything derived from the raw files that the model and decoder need."""

    def __init__(self, trip, cfg, with_labels=True):
        self.trip_id = trip.trip_id
        track, self.orig_idx = clean_track(trip.track)
        self.track = track
        lon0, lat0 = (track[:, 0].mean(), track[:, 1].mean()) if len(track) else (trip.nodes[:, 0].mean(), trip.nodes[:, 1].mean())
        self.proj = LocalProjection(lon0, lat0)
        self.xy = self.proj.to_xy(track[:, 0], track[:, 1])
        self.t = track[:, 2]
        c = cfg["candidates"]
        self.net = RoadNetwork(trip.nodes, trip.arcs, self.proj, c["densify_step_m"])
        self.cfg = cfg
        self._kinematics()
        self.segment_id = self._segments()
        self.ref_arc = None
        self.align_dist = None
        if with_labels and trip.route is not None and len(trip.route):
            _, self.ref_arc, self.align_dist = align_track_to_route(self.xy, self.net, trip.route)
        self.route = trip.route if with_labels else None
        self.cands = self.net.candidates(self.xy, c["radius_m"], c["max_k"], c["fallback_radius_m"], c["fallback_k"])

    # -------------------------------------------------------------- kinematics
    def _heading(self, h):
        n = len(self.xy)
        i0 = np.clip(np.arange(n) - h, 0, n - 1)
        i1 = np.clip(np.arange(n) + h, 0, n - 1)
        d = self.xy[i1] - self.xy[i0]
        disp = np.sqrt((d * d).sum(1))
        return bearing_deg(d[:, 0], d[:, 1]), disp

    def _kinematics(self):
        f = self.cfg["features"]
        n = len(self.xy)
        self.heading, disp = self._heading(f["heading_half_window"])
        self.heading_valid = disp >= f["min_heading_disp_m"]
        self.heading_long, disp_l = self._heading(f["heading_long_half_window"])
        self.heading_long_valid = disp_l >= f["min_heading_disp_m"]
        step = np.zeros(n)
        dt = np.ones(n)
        if n > 1:
            dd = self.xy[1:] - self.xy[:-1]
            step[1:] = np.sqrt((dd * dd).sum(1))
            dt[1:] = np.diff(self.t)
        self.dt_prev = dt.copy()
        self.dt_prev[0] = 0
        self.dt_next = np.r_[dt[1:], 0] if n > 1 else np.zeros(n)
        self.step = step
        i0 = np.clip(np.arange(n) - 1, 0, n - 1)
        i1 = np.clip(np.arange(n) + 1, 0, n - 1)
        dist01 = np.sqrt(((self.xy[i1] - self.xy[i0]) ** 2).sum(1))
        self.speed = dist01 / np.maximum(self.t[i1] - self.t[i0], 1.0)
        # turning: heading of the next window vs the previous window
        h = self.cfg["features"]["heading_half_window"]
        prev_h = np.r_[np.full(min(h, n), np.nan), self.heading[:-h]] if n > h else np.full(n, np.nan)
        nxt_h = np.r_[self.heading[h:], np.full(min(h, n), np.nan)] if n > h else np.full(n, np.nan)
        turn = angle_diff(prev_h, nxt_h)
        self.turn = np.nan_to_num(turn, nan=0.0)

    def _segments(self):
        p = self.cfg["preprocess"]
        n = len(self.xy)
        brk = np.zeros(n, dtype=bool)
        if n > 1:
            brk[1:] = (np.diff(self.t) > p["gap_seconds"]) | (self.step[1:] > p["gap_jump_m"])
        return np.cumsum(brk)

    # -------------------------------------------------------------- feature table
    def candidate_table(self):
        net = self.net
        n = len(self.xy)
        counts = np.array([len(c[0]) for c in self.cands])
        if counts.sum() == 0:
            return pd.DataFrame(columns=["trip_id", "pt_idx", "arc_id", "label"] + FEATURES)
        pt = np.repeat(np.arange(n), counts)
        arc = np.concatenate([c[0] for c in self.cands]).astype(np.int64)
        dist = np.concatenate([c[1] for c in self.cands])
        tproj = np.concatenate([c[2] for c in self.cands])
        rank = np.concatenate([np.arange(k) for k in counts])
        mind = np.repeat(np.array([c[1][0] if len(c[1]) else np.nan for c in self.cands]), counts)
        meand = np.repeat(np.array([c[1].mean() if len(c[1]) else np.nan for c in self.cands]), counts)

        _, _, t_raw = point_to_segment(self.xy[pt], net.A[arc], net.B[arc])
        L = net.length[arc]
        overshoot = np.maximum(0, -t_raw) * L + np.maximum(0, t_raw - 1) * L

        hv = self.heading_valid[pt]
        ad = angle_diff(self.heading[pt], net.bearing[arc])
        ad = np.where(hv, ad, 90.0)
        cosang = np.where(hv, np.cos(np.deg2rad(ad)), 0.0)
        adl = np.where(self.heading_long_valid[pt], angle_diff(self.heading_long[pt], net.bearing[arc]), 90.0)

        # context: distances of neighbouring fixes to the same arc
        w = self.cfg["features"]["context_half_window"]
        ctx = []
        for o in range(-w, w + 1):
            if o == 0:
                continue
            j = np.clip(pt + o, 0, n - 1)
            d, _, _ = point_to_segment(self.xy[j], net.A[arc], net.B[arc])
            ctx.append(d)
        ctx = np.vstack(ctx)
        nearest = np.array([c[0][0] if len(c[0]) else -1 for c in self.cands])
        frac = np.zeros(len(pt))
        for o in range(-w, w + 1):
            j = np.clip(pt + o, 0, n - 1)
            frac += nearest[j] == arc
        frac /= (2 * w + 1)
        ctx_mean = ctx.mean(0)
        # min over candidates of each point's ctx_mean
        ctx_min = pd.Series(ctx_mean).groupby(pt).transform("min").to_numpy()

        df = pd.DataFrame({
            "trip_id": self.trip_id,
            "pt_idx": pt,
            "arc_id": arc,
            "dist": dist,
            "dist_rank": rank,
            "dist_minus_min": dist - mind,
            "dist_ratio": dist / (meand + 1.0),
            "n_cands": counts[pt],
            "t_proj": tproj,
            "overshoot_m": overshoot,
            "arc_len": L,
            "log_arc_len": np.log1p(L),
            "heading_valid": hv.astype(np.int8),
            "angle_diff": ad,
            "cos_angle": cosang,
            "angle_diff_long": adl,
            "speed": self.speed[pt],
            "dt_prev": self.dt_prev[pt],
            "dt_next": self.dt_next[pt],
            "turn": self.turn[pt],
            "ctx_mean_dist": ctx_mean,
            "ctx_max_dist": ctx.max(0),
            "ctx_nearest_frac": frac,
            "ctx_mean_dist_minus_min": ctx_mean - ctx_min,
            "out_deg_to": net.out_deg[net.arcs[arc, 1]],
            "in_deg_from": net.in_deg[net.arcs[arc, 0]],
            "has_reverse": net.has_reverse[arc].astype(np.int8),
        })
        if self.ref_arc is not None:
            df["label"] = (arc == self.ref_arc[pt]).astype(np.int8)
        else:
            df["label"] = -1
        return df

    def point_table(self):
        """Per-observation info used for evaluation (candidate miss rate, label quality)."""
        n = len(self.xy)
        d = {
            "trip_id": self.trip_id,
            "pt_idx": np.arange(n),
            "orig_idx": self.orig_idx,
            "t": self.t,
            "segment_id": self.segment_id,
            "n_cands": [len(c[0]) for c in self.cands],
        }
        if self.ref_arc is not None:
            d["ref_arc"] = self.ref_arc
            d["align_dist"] = self.align_dist
            d["ref_in_cands"] = [int(r in c[0]) for r, c in zip(self.ref_arc, self.cands)]
        return pd.DataFrame(d)
