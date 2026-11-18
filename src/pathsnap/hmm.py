"""HMM + Viterbi route reconstruction.

Hidden state at GPS fix i: one of the candidate road arcs of that fix (top-B by classifier
probability), together with the projected position of the fix on that arc.

Emission:   log e_i(a)    = alpha * log( p_clf(a | i) / sum_c p_clf(c | i) )
Transition: log T(a -> b) = -| d_route(a, b) - d_gps(i-1, i) | / beta
    d_route = network distance from the projection on arc a to the projection on arc b:
      * same arc:     |offset_b - offset_a|  (+50 m penalty if it moves backwards > tol)
      * different:    (len_a - offset_a) + shortest_path(to(a), from(b)) + offset_b
      * no path within the Dijkstra limit -> log T = unreachable_logp (a floor, never -inf)
    Consecutive fixes in different segments (time gap / big jump) get a flat transition,
    so the decoder restarts there.

Viterbi:  delta_i(b) = log e_i(b) + max_a [ delta_{i-1}(a) + log T(a -> b) ]
"""
import numpy as np

NEG = -1e18


def _pad(cands_per_point, B):
    n = len(cands_per_point)
    arcs = np.full((n, B), -1, dtype=np.int64)
    emit = np.full((n, B), NEG)
    off = np.zeros((n, B))
    for i, (a, e, o) in enumerate(cands_per_point):
        k = min(len(a), B)
        arcs[i, :k], emit[i, :k], off[i, :k] = a[:k], e[:k], o[:k]
    return arcs, emit, off


def build_states(cand_df, prob, n_points, beam, alpha, emission="classifier", sigma=8.0):
    """Per fix: candidate arcs sorted by score (top-beam), emission log-probs, offsets along arc."""
    d = cand_df[["pt_idx", "arc_id", "t_proj", "arc_len", "dist"]].copy()
    d["p"] = prob
    if emission == "classifier":
        s = d.groupby("pt_idx")["p"].transform("sum")
        d["e"] = alpha * np.log(d["p"] / np.maximum(s, 1e-12) + 1e-12)
    else:  # distance-only Gaussian emission (classic Newson & Krumm baseline)
        d["e"] = -0.5 * (d["dist"] / sigma) ** 2
    d = d.sort_values(["pt_idx", "e"], ascending=[True, False], kind="stable")
    d["rk"] = d.groupby("pt_idx").cumcount()
    d = d[d["rk"] < beam]
    out = [(np.zeros(0, int), np.zeros(0), np.zeros(0)) for _ in range(n_points)]
    for pt, g in d.groupby("pt_idx", sort=False):
        out[pt] = (g["arc_id"].to_numpy(), g["e"].to_numpy(), (g["t_proj"] * g["arc_len"]).to_numpy())
    return out


def viterbi(prep, states, hcfg):
    """Decode the most likely arc for every fix. Returns array of arc ids (-1 if none)."""
    net = prep.net
    n = len(prep.xy)
    B = int(hcfg["beam"])
    valid = np.array([len(s[0]) > 0 for s in states])
    vidx = np.where(valid)[0]
    pred = np.full(n, -1, dtype=np.int64)
    if len(vidx) == 0:
        return pred
    arcs, emit, off = _pad([states[i] for i in vidx], B)
    S = len(vidx)
    if S == 1:
        pred[vidx[0]] = arcs[0, np.argmax(emit[0])]
        return _ffill(pred)

    # pairwise arrays for each step s: prev = s, cur = s + 1
    a = arcs[:-1, :, None]          # (S-1, B, 1)
    b = arcs[1:, None, :]           # (S-1, 1, B)
    oa = off[:-1, :, None]
    ob = off[1:, None, :]
    xy = prep.xy[vidx]
    dgps = np.sqrt(((xy[1:] - xy[:-1]) ** 2).sum(1))[:, None, None]
    seg = prep.segment_id[vidx]
    cross = (seg[1:] != seg[:-1])[:, None, None]

    A = np.broadcast_to(a, (S - 1, B, B))
    Bm = np.broadcast_to(b, (S - 1, B, B))
    ok = (A >= 0) & (Bm >= 0)
    Asafe, Bsafe = np.where(ok, A, 0), np.where(ok, Bm, 0)
    to_a = net.arcs[Asafe, 1]
    from_b = net.arcs[Bsafe, 0]
    same = Asafe == Bsafe
    adjacent = to_a == from_b

    # shortest paths only for pairs that need them
    need = ok & ~same & ~adjacent & ~cross
    sp = np.zeros(A.shape)
    if need.any():
        src, tgt = to_a[need], from_b[need]
        keys = src * net.n_nodes + tgt
        uk, inv = np.unique(keys, return_inverse=True)
        us, ut = uk // net.n_nodes, uk % net.n_nodes
        vals = np.full(len(uk), np.inf)
        srcs = np.unique(us)
        tg_all = np.unique(ut)
        D = net.node_distances(srcs, tg_all, hcfg["sp_limit_m"])
        vals = D[np.searchsorted(srcs, us), np.searchsorted(tg_all, ut)]
        sp[need] = vals[inv]
    len_a = net.length[Asafe]
    diff = ob - oa
    d_same = np.abs(diff) + np.where(diff < -hcfg["same_arc_backtrack_tol_m"], 50.0, 0.0)
    d_diff = (len_a - oa) + sp + ob
    droute = np.where(same, d_same, d_diff)
    logT = -np.abs(droute - dgps) / hcfg["beta"]
    logT = np.where(np.isfinite(logT), logT, hcfg["unreachable_logp"])
    logT = np.maximum(logT, hcfg["unreachable_logp"])
    logT = np.where(np.broadcast_to(cross, logT.shape), 0.0, logT)
    logT = np.where(ok, logT, NEG)

    delta = emit[0].copy()
    back = np.zeros((S, B), dtype=np.int64)
    for s in range(S - 1):
        sc = delta[:, None] + logT[s]
        back[s + 1] = np.argmax(sc, axis=0)
        delta = sc[back[s + 1], np.arange(B)] + emit[s + 1]
    j = int(np.argmax(delta))
    path = np.empty(S, dtype=np.int64)
    for s in range(S - 1, -1, -1):
        path[s] = j
        j = back[s, j]
    pred[vidx] = arcs[np.arange(S), path]
    return _ffill(pred)


def _ffill(pred):
    out = pred.copy()
    last = -1
    for i in range(len(out)):
        if out[i] >= 0:
            last = out[i]
        else:
            out[i] = last
    return out


def build_route(prep, pred, limit=3000.0):
    """Collapse the per-fix arcs into a route and bridge non-adjacent arcs with shortest paths."""
    net = prep.net
    seq, segs = [], []
    for a, sg in zip(pred, prep.segment_id):
        if a < 0:
            continue
        if not seq or seq[-1] != a:
            seq.append(int(a))
            segs.append(sg)
    route, bridged, broken = [], 0, 0
    for k, a in enumerate(seq):
        if route:
            prev = route[-1]
            if net.arcs[prev, 1] != net.arcs[a, 0]:
                path = None
                if segs[k] == segs[k - 1]:
                    path = net.shortest_arc_path(int(net.arcs[prev, 1]), int(net.arcs[a, 0]), limit)
                if path:
                    route.extend(path)
                    bridged += 1
                else:
                    broken += 1
        route.append(a)
    return route, {"bridged_gaps": bridged, "unbridged_breaks": broken}
