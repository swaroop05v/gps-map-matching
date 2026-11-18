"""PathSnap evaluator UI (runs on Laptop 2).

It only talks to the API over HTTP. It never reads the dataset or any reference route.

Run:  streamlit run ui/app.py
      (set the API address in the sidebar, e.g. http://192.168.x.x:8000)
"""
import os
import time

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", page_icon="🗺️", layout="wide")

if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.history = []

st.sidebar.title("PathSnap")
api = st.sidebar.text_input("API URL (Laptop 1)", os.environ.get("PATHSNAP_API_URL", "http://localhost:8000")).rstrip("/")
partition = st.sidebar.selectbox("Trip set", ["eval_test", "student_test"], index=0,
                                 help="eval_test = the ten reserved evaluation trips")

try:
    h = requests.get(f"{api}/health", timeout=5).json()
    st.sidebar.success(f"API online · {h['model']} · α={h['decoder']['alpha']} β={h['decoder']['beta']}")
except Exception as e:  # noqa
    st.sidebar.error(f"API not reachable: {e}")
    h = None

st.title("PathSnap: map-matching and route reconstruction")
st.caption("Pick a trip. The API runs candidate generation, the LightGBM classifier and the HMM/Viterbi decoder, "
           "and scores the route against its own reference data.")

trips = []
if h:
    try:
        trips = requests.get(f"{api}/trips", params={"partition": partition}, timeout=10).json()["trips"]
    except Exception as e:  # noqa
        st.error(f"Could not load trips: {e}")

if trips:
    cols = st.columns(min(len(trips), 10))
    if "trip" not in st.session_state or st.session_state.trip not in trips:
        st.session_state.trip = trips[0]
    for i, t in enumerate(trips):
        if cols[i % len(cols)].button(t.lstrip("0") or "0", key=f"b{t}", use_container_width=True,
                                      type="primary" if t == st.session_state.trip else "secondary"):
            st.session_state.trip = t
            st.rerun()
    st.write(f"Selected trip: **{st.session_state.trip}** ({len(trips)} trips available in `{partition}`)")
    if st.button("▶ Run map matching", type="primary"):
        with st.spinner("Sending the trip ID to the API..."):
            t0 = time.time()
            r = requests.post(f"{api}/predict", json={"trip_id": st.session_state.trip, "partition": partition}, timeout=300)
            rt = time.time() - t0
        if r.ok:
            st.session_state.result = r.json()
            st.session_state.result["_roundtrip_s"] = rt
            m = st.session_state.result["metrics"].get("hmm", {})
            st.session_state.history.append({"trip": st.session_state.trip, "partition": partition,
                                             "top1": m.get("top1_accuracy"), "continuity": m.get("route_continuity"),
                                             "route_f1": m.get("route_f1"), "seconds": round(rt, 2)})
        else:
            st.error(f"API error {r.status_code}: {r.text}")
elif h:
    st.warning(f"No trips found on the server for `{partition}`.")

res = st.session_state.result
if res:
    st.subheader(f"Result for trip {res['trip_id']}")
    mh = res["metrics"].get("hmm", {})
    mc = res["metrics"].get("classifier_only", {})
    pct = lambda v: "n/a" if v is None else f"{100 * v:.2f}%"  # noqa
    c = st.columns(3)
    c[0].metric("Top-1 accuracy (HMM)", pct(mh.get("top1_accuracy")),
                None if not mc else f"{100 * (mh['top1_accuracy'] - mc['top1_accuracy']):+.2f} pp vs classifier")
    c[1].metric("Route continuity", pct(mh.get("route_continuity")))
    c[2].metric("Route F1 (arc set)", pct(mh.get("route_f1")))
    c = st.columns(3)
    c[0].metric("GPS points", res["n_gps_points"])
    c[1].metric("Predicted route arcs", len(res["predicted_arc_ids"]))
    c[2].metric("API time", f"{res['timings_s']['request_total_s']:.2f} s")

    show_clf = st.checkbox("Also show classifier-only route (before HMM)", value=False)
    pts = res["gps_points"]
    lat = sum(p[0] for p in pts) / len(pts)
    lon = sum(p[1] for p in pts) / len(pts)
    fmap = folium.Map(location=[lat, lon], zoom_start=15, control_scale=True)
    g = folium.FeatureGroup("Raw GPS observations")
    for p in pts:
        folium.CircleMarker(p, radius=2, color="#1f4fd8", fill=True, fill_opacity=0.8, weight=1).add_to(g)
    g.add_to(fmap)
    if show_clf:
        g = folium.FeatureGroup("Classifier-only route")
        for line in res["classifier_only_route"]:
            folium.PolyLine(line, color="#f4a261", weight=4, opacity=0.9).add_to(g)
        g.add_to(fmap)
    g = folium.FeatureGroup("Predicted route (HMM/Viterbi)")
    for line in res["predicted_route"]:
        folium.PolyLine(line, color="#d62728", weight=5, opacity=0.85).add_to(g)
    g.add_to(fmap)
    folium.Marker(pts[0], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker(pts[-1], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
    fmap.fit_bounds([[min(p[0] for p in pts), min(p[1] for p in pts)], [max(p[0] for p in pts), max(p[1] for p in pts)]])
    folium.LayerControl(collapsed=False).add_to(fmap)
    st.markdown("🔵 raw GPS &nbsp;&nbsp; 🔴 predicted route (HMM)" + (" &nbsp;&nbsp; 🟠 classifier-only" if show_clf else ""))
    st_folium(fmap, height=560, use_container_width=True, returned_objects=[])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Classifier-only vs HMM**")
        if mc:
            st.table({"metric": ["Top-1 accuracy", "Route continuity", "Route precision", "Route recall", "Route F1"],
                      "classifier only": [pct(mc[k]) for k in ["top1_accuracy", "route_continuity", "route_precision", "route_recall", "route_f1"]],
                      "classifier + HMM": [pct(mh[k]) for k in ["top1_accuracy", "route_continuity", "route_precision", "route_recall", "route_f1"]]})
        st.caption(f"Scored points: {res['metrics'].get('scored_points')} · candidate miss rate: "
                   f"{pct(res['metrics'].get('candidate_miss_rate'))} · segments: {res['n_segments']}")
    with col2:
        st.markdown("**Timings (server)**")
        st.json(res["timings_s"])
    with st.expander("Raw API response (without geometry)"):
        st.json({k: v for k, v in res.items() if k not in ("gps_points", "predicted_route", "classifier_only_route", "predicted_arc_ids")})

if st.session_state.history:
    st.subheader("Runs this session")
    st.dataframe(st.session_state.history, use_container_width=True)
