"""Reading the raw trip files (.track, .nodes, .arcs, .route)."""
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N, 3) lon, lat, t
    nodes: np.ndarray            # (M, 2) lon, lat   (row index = node_id)
    arcs: np.ndarray             # (A, 2) from_node, to_node (row index = arc_id)
    route: Optional[np.ndarray]  # (R,) arc ids in travel order, None if not available


def _load(path, dtype=float, ncols=None):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, ncols), dtype=dtype) if ncols else np.zeros(0, dtype=dtype)
    a = np.loadtxt(path, dtype=dtype, ndmin=2 if ncols else 1)
    return a


def list_trips(partition_dir):
    if not os.path.isdir(partition_dir):
        return []
    return sorted(d for d in os.listdir(partition_dir) if os.path.isdir(os.path.join(partition_dir, d)))


def load_trip(partition_dir, trip_id, with_route=True):
    d = os.path.join(partition_dir, trip_id)
    track = _load(os.path.join(d, f"{trip_id}.track"), float, 3)
    nodes = _load(os.path.join(d, f"{trip_id}.nodes"), float, 2)
    arcs = _load(os.path.join(d, f"{trip_id}.arcs"), np.int64, 2)
    route = None
    rp = os.path.join(d, f"{trip_id}.route")
    if with_route and os.path.exists(rp):
        route = _load(rp, np.int64).reshape(-1)
    return Trip(trip_id, track, nodes, arcs, route)


def clean_track(track):
    """Basic cleaning: drop non-finite rows, sort by time, drop duplicate timestamps.

    Returns the cleaned track and the index of each kept row in the original file,
    so predictions can always be reported against the original observation index.
    """
    idx = np.arange(len(track))
    ok = np.isfinite(track).all(axis=1)
    ok &= (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    track, idx = track[ok], idx[ok]
    order = np.argsort(track[:, 2], kind="stable")
    track, idx = track[order], idx[order]
    if len(track) > 1:
        keep = np.r_[True, np.diff(track[:, 2]) > 0]
        track, idx = track[keep], idx[keep]
    return track, idx
