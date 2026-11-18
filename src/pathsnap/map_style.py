"""Shared Folium presentation for live routes and report maps."""
from html import escape

import folium

GPS = "#54d8ff"
PREDICTED = "#ef9aff"
CLASSIFIER = "#ffbe69"
REFERENCE = "#a6f780"
NETWORK = "#83969f"

MAP_CSS = """
<style>
.leaflet-container {background:#14212a!important; font-family:Arial,sans-serif}
.leaflet-tile-pane {filter:invert(1) hue-rotate(175deg) saturate(.45) brightness(.86) contrast(.88)}
.leaflet-control-layers,.leaflet-bar a {background:#17262f!important;color:#edf7fa!important}
.leaflet-control-layers {border:1px solid #566a74!important;border-radius:4px!important;padding:8px}
.leaflet-control-scale-line {background:#17262f;color:#edf7fa;border-color:#a8bdc6}
.leaflet-control-attribution {background:#17262f!important;color:#e5f3f7!important}
.leaflet-control-attribution a {color:#a6f780!important}
.route-key {position:fixed;bottom:32px;left:12px;z-index:9999;background:#121e26;color:#edf7fa;
  border:1px solid #526874;border-left:3px solid #a6f780;padding:10px 13px;font:13px/1.55 Arial,sans-serif;
  pointer-events:none;max-width:75%}
.route-key strong {display:block;letter-spacing:.04em;margin-bottom:4px}
.route-swatch {display:inline-block;width:13px;height:4px;margin-right:7px;vertical-align:middle}
.route-pin {display:grid;place-items:center;width:24px;height:24px;box-sizing:border-box;
  color:#14212a;font:bold 11px Arial,sans-serif;border:2px solid #14212a;border-radius:50%;
  box-shadow:0 0 0 1px #b5c9d1}
</style>
"""


def street_map(location=(0, 0), zoom=2):
    fmap = folium.Map(location=location, zoom_start=zoom, control_scale=True, prefer_canvas=True)
    fmap.get_root().header.add_child(folium.Element(MAP_CSS))
    return fmap


def route_endpoints(fmap, first, last):
    group = folium.FeatureGroup(name="Start and finish")
    for point, label, initial, color in (
        (first, "Start", "S", REFERENCE), (last, "Finish", "F", CLASSIFIER)
    ):
        icon = folium.DivIcon(
            html=f'<span class="route-pin" style="background:{color}">{initial}</span>',
            icon_size=(24, 24), icon_anchor=(12, 12), class_name="pathsnap-pin",
        )
        folium.Marker(point, tooltip=label, icon=icon).add_to(group)
    group.add_to(fmap)


def map_key(fmap, items, title=None):
    rows = "".join(
        f'<div><span class="route-swatch" style="background:{color}"></span>{escape(name)}</div>'
        for name, color in items
    )
    heading = f"<strong>{escape(title)}</strong>" if title else ""
    fmap.get_root().html.add_child(folium.Element(f'<div class="route-key">{heading}{rows}</div>'))


def live_route_map(result):
    points = result["gps_points"]
    fmap = street_map(points[0], 15)
    gps = folium.FeatureGroup(name="Raw GPS")
    for point in points:
        folium.CircleMarker(point, radius=2, color=GPS, weight=1, fill=True, fill_opacity=.72).add_to(gps)
    gps.add_to(fmap)

    for title, key, color, visible, dash in (
        ("Predicted route · HMM", "predicted_route", PREDICTED, True, None),
        ("Classifier-only route", "classifier_only_route", CLASSIFIER, False, "7 6"),
    ):
        layer = folium.FeatureGroup(name=title, show=visible)
        for line in result[key]:
            folium.PolyLine(line, color="#1a1022", weight=8, opacity=.75, dash_array=dash).add_to(layer)
            folium.PolyLine(line, color=color, weight=4, opacity=.98, dash_array=dash).add_to(layer)
        layer.add_to(fmap)

    route_endpoints(fmap, points[0], points[-1])
    fmap.fit_bounds([
        [min(p[0] for p in points), min(p[1] for p in points)],
        [max(p[0] for p in points), max(p[1] for p in points)],
    ], padding=(24, 24), max_zoom=17)
    folium.LayerControl(collapsed=False).add_to(fmap)
    map_key(fmap, [("Raw GPS", GPS), ("HMM route", PREDICTED), ("Classifier-only route", CLASSIFIER)])
    return fmap
