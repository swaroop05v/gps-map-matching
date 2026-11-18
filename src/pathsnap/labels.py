"""Per-observation reference labels.

The .route file only lists the arcs the vehicle travelled, in order. It does not say
which GPS fix belongs to which arc. We derive a label for every fix with a monotonic
alignment (dynamic programming):

    cost(i, j) = distance from GPS fix i to route arc j
    D(i, j)    = cost(i, j) + min_{j' <= j} D(i-1, j')

i.e. fixes are assigned to route positions that never go backwards in the route, and the
total fix-to-arc distance is minimised. Unlike "nearest route arc", this handles routes
that revisit the same place and reverse-direction twin arcs correctly.
"""
import numpy as np

from .geo import point_to_segment


def align_track_to_route(track_xy, net, route):
    """Return (route_pos, ref_arc, align_dist) for every GPS fix."""
    N, L = len(track_xy), len(route)
    if N == 0 or L == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    A, B = net.A[route], net.B[route]
    back = np.empty((N, L), dtype=np.int16 if L < 32000 else np.int32)
    idx = np.arange(L)
    D_prev = None
    for i in range(N):
        d, _, _ = point_to_segment(track_xy[i][None, :], A, B)
        if D_prev is None:
            D = d
            back[i] = idx
        else:
            cm = np.minimum.accumulate(D_prev)
            arg = np.maximum.accumulate(np.where(D_prev <= cm, idx, 0))
            D = d + cm
            back[i] = arg
        D_prev = D
    pos = np.empty(N, dtype=np.int64)
    j = int(np.argmin(D_prev))
    for i in range(N - 1, -1, -1):
        pos[i] = j
        if i > 0:
            j = int(back[i][j])
    ref_arc = route[pos]
    dist, _, _ = point_to_segment(track_xy, net.A[ref_arc], net.B[ref_arc])
    return pos, ref_arc, dist
