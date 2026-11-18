"""Folium map generation (used by EDA, result maps and the UI)."""
import folium
import numpy as np


def _arc_lines(nodes_lonlat, arcs, arc_ids):
    arc_ids = np.asarray(arc_ids, dtype=np.int64)
    a = nodes_lonlat[arcs[arc_ids, 0]]
    b = nodes_lonlat[arcs[arc_ids, 1]]
    return [[(float(p[1]), float(p[0])), (float(q[1]), float(q[0]))] for p, q in zip(a, b)]


def _merge_route(nodes_lonlat, arcs, arc_ids):
    """Chain consecutive connected arcs into polylines (lat, lon) so routes draw as lines."""
    lines, cur = [], []
    prev_to = None
    for aid in arc_ids:
        u, v = arcs[aid]
        pu, pv = nodes_lonlat[u], nodes_lonlat[v]
        if prev_to is not None and u == prev_to and cur:
            cur.append((float(pv[1]), float(pv[0])))
        else:
            if len(cur) > 1:
                lines.append(cur)
            cur = [(float(pu[1]), float(pu[0])), (float(pv[1]), float(pv[0]))]
        prev_to = v
    if len(cur) > 1:
        lines.append(cur)
    return lines


def add_legend(m, items, title="Legend"):
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};margin-right:6px;vertical-align:middle"></span>{n}</div>'
        for n, c in items
    )
    html = (
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
        'padding:8px 10px;border:1px solid #999;border-radius:6px;font-size:13px;line-height:1.6">'
        f"<b>{title}</b>{rows}</div>"
    )
    m.get_root().html.add_child(folium.Element(html))


def trip_map(nodes_lonlat, arcs, track, ref_route=None, layers=None, network_margin_deg=0.003,
             max_network_arcs=15000, point_step=1, title=None, bounds=None):
    """Build an interactive map.

    track: (N, >=2) lon, lat
    ref_route: arc ids of the reference route (only for offline / report maps)
    layers: list of (name, arc_ids, colour) for predicted routes
    """
    lat_c, lon_c = float(np.mean(track[:, 1])), float(np.mean(track[:, 0]))
    m = folium.Map(location=[lat_c, lon_c], zoom_start=15, tiles="OpenStreetMap", control_scale=True)
    legend = []

    # road network near the trajectory
    lo0, lo1 = track[:, 0].min() - network_margin_deg, track[:, 0].max() + network_margin_deg
    la0, la1 = track[:, 1].min() - network_margin_deg, track[:, 1].max() + network_margin_deg
    mid = (nodes_lonlat[arcs[:, 0]] + nodes_lonlat[arcs[:, 1]]) / 2
    inside = np.where((mid[:, 0] >= lo0) & (mid[:, 0] <= lo1) & (mid[:, 1] >= la0) & (mid[:, 1] <= la1))[0]
    if len(inside) > max_network_arcs:
        inside = inside[:: int(np.ceil(len(inside) / max_network_arcs))]
    if len(inside):
        fg = folium.FeatureGroup(name="Road network (arcs)", show=True)
        folium.PolyLine(_arc_lines(nodes_lonlat, arcs, inside), color="#888888", weight=1.5, opacity=0.6).add_to(fg)
        fg.add_to(m)
        legend.append(("Road network", "#888888"))

    # raw GPS drawn below the route layers so the routes stay visible
    fg = folium.FeatureGroup(name="Raw GPS observations", show=True)
    for lon, lat in track[::point_step, :2]:
        folium.CircleMarker((float(lat), float(lon)), radius=2.5, color="#1f4fd8", fill=True, fill_opacity=0.7, weight=1).add_to(fg)
    fg.add_to(m)
    legend.append(("Raw GPS", "#1f4fd8"))

    if ref_route is not None and len(ref_route):
        fg = folium.FeatureGroup(name="Reference route (ground truth)", show=True)
        for line in _merge_route(nodes_lonlat, arcs, ref_route):
            folium.PolyLine(line, color="#1a9e3a", weight=7, opacity=0.6).add_to(fg)
        fg.add_to(m)
        legend.append(("Reference route", "#1a9e3a"))

    for name, arc_ids, colour in layers or []:
        if arc_ids is None or len(arc_ids) == 0:
            continue
        fg = folium.FeatureGroup(name=name, show=True)
        for line in _merge_route(nodes_lonlat, arcs, arc_ids):
            folium.PolyLine(line, color=colour, weight=4, opacity=0.9).add_to(fg)
        fg.add_to(m)
        legend.append((name, colour))

    folium.Marker((float(track[0, 1]), float(track[0, 0])), tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((float(track[-1, 1]), float(track[-1, 0])), tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    if bounds is None:
        bounds = [[float(track[:, 1].min()), float(track[:, 0].min())], [float(track[:, 1].max()), float(track[:, 0].max())]]
    m.fit_bounds(bounds)
    folium.LayerControl(collapsed=False).add_to(m)
    add_legend(m, legend, title or "PathSnap")
    return m
