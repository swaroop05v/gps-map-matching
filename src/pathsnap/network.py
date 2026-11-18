"""Road network for one trip: arc geometry, spatial index and shortest paths."""
import heapq

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .geo import bearing_deg, point_to_segment


class RoadNetwork:
    def __init__(self, nodes_lonlat, arcs, proj, densify_step=10.0):
        self.proj = proj
        self.nodes_lonlat = nodes_lonlat
        self.nodes_xy = proj.to_xy(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.arcs = arcs.astype(np.int64)
        self.n_nodes = len(nodes_lonlat)
        self.n_arcs = len(arcs)
        self.A = self.nodes_xy[self.arcs[:, 0]]
        self.B = self.nodes_xy[self.arcs[:, 1]]
        d = self.B - self.A
        self.length = np.sqrt((d * d).sum(1))
        self.bearing = bearing_deg(d[:, 0], d[:, 1])
        self.out_deg = np.bincount(self.arcs[:, 0], minlength=self.n_nodes)
        self.in_deg = np.bincount(self.arcs[:, 1], minlength=self.n_nodes)
        key = self.arcs[:, 0] * self.n_nodes + self.arcs[:, 1]
        rkey = self.arcs[:, 1] * self.n_nodes + self.arcs[:, 0]
        self.has_reverse = np.isin(rkey, key)
        self._build_index(densify_step)
        self._graph = None
        self._adj = None

    # ------------------------------------------------------------------ spatial index
    def _build_index(self, step):
        nseg = np.maximum(1, np.ceil(self.length / step)).astype(np.int64)
        counts = nseg + 1
        arc_of = np.repeat(np.arange(self.n_arcs), counts)
        start = np.repeat(np.cumsum(counts) - counts, counts)
        k = np.arange(len(arc_of)) - start
        t = k / nseg[arc_of]
        pts = self.A[arc_of] + t[:, None] * (self.B[arc_of] - self.A[arc_of])
        self._sample_arc = arc_of
        self._step = step
        self._tree = cKDTree(pts) if len(pts) else None

    def candidates(self, P, radius, max_k, fallback_radius=None, fallback_k=3):
        """For each point in P (N,2) return a list of (arc_ids, dists, t) within radius (top max_k)."""
        out = []
        if self._tree is None:
            return [(np.zeros(0, int), np.zeros(0), np.zeros(0))] * len(P)
        lists = self._tree.query_ball_point(P, r=radius + self._step / 2)
        for i, idx in enumerate(lists):
            res = self._exact(P[i], idx, radius, max_k)
            if len(res[0]) == 0 and fallback_radius:
                idx2 = self._tree.query_ball_point(P[i], r=fallback_radius + self._step / 2)
                res = self._exact(P[i], idx2, fallback_radius, fallback_k)
            out.append(res)
        return out

    def _exact(self, p, sample_idx, radius, max_k):
        if len(sample_idx) == 0:
            return np.zeros(0, int), np.zeros(0), np.zeros(0)
        arcs = np.unique(self._sample_arc[np.asarray(sample_idx)])
        d, t, _ = point_to_segment(p[None, :], self.A[arcs], self.B[arcs])
        m = d <= radius
        arcs, d, t = arcs[m], d[m], t[m]
        order = np.argsort(d, kind="stable")[:max_k]
        return arcs[order], d[order], t[order]

    # ------------------------------------------------------------------ graph
    @property
    def graph(self):
        if self._graph is None:
            w = np.maximum(self.length, 0.01)
            # keep the shortest of parallel arcs between the same node pair
            order = np.lexsort((w, self.arcs[:, 1], self.arcs[:, 0]))
            a = self.arcs[order]
            first = np.r_[True, (a[1:] != a[:-1]).any(1)]
            a, ww = a[first], w[order][first]
            self._graph = csr_matrix((ww, (a[:, 0], a[:, 1])), shape=(self.n_nodes, self.n_nodes))
        return self._graph

    def node_distances(self, sources, targets, limit):
        """Shortest-path distance (m) matrix from source nodes to target nodes, inf if > limit."""
        sources = np.asarray(sources, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.int64)
        out = np.full((len(sources), len(targets)), np.inf)
        chunk = max(1, int(2e7 // max(self.n_nodes, 1)))
        for s in range(0, len(sources), chunk):
            D = dijkstra(self.graph, directed=True, indices=sources[s:s + chunk], limit=limit)
            out[s:s + chunk] = D[:, targets]
        return out

    @property
    def adjacency(self):
        if self._adj is None:
            adj = [[] for _ in range(self.n_nodes)]
            for aid, (u, v) in enumerate(self.arcs):
                adj[u].append((v, max(self.length[aid], 0.01), aid))
            self._adj = adj
        return self._adj

    def shortest_arc_path(self, src_node, dst_node, limit=5000.0):
        """Arc ids of the shortest path src_node -> dst_node (empty list if none / same node)."""
        if src_node == dst_node:
            return []
        adj = self.adjacency
        dist = {src_node: 0.0}
        prev = {}
        pq = [(0.0, src_node)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst_node:
                break
            if d > dist.get(u, np.inf) or d > limit:
                continue
            for v, w, aid in adj[u]:
                nd = d + w
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    prev[v] = (u, aid)
                    heapq.heappush(pq, (nd, v))
        if dst_node not in prev:
            return None
        path, v = [], dst_node
        while v != src_node:
            u, aid = prev[v]
            path.append(aid)
            v = u
        return path[::-1]

    def arcs_lonlat(self, arc_ids):
        arc_ids = np.asarray(arc_ids, dtype=np.int64)
        a = self.nodes_lonlat[self.arcs[arc_ids, 0]]
        b = self.nodes_lonlat[self.arcs[arc_ids, 1]]
        return a, b
