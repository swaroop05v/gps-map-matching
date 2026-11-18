"""Streamlit evaluator console. Run from the repository with streamlit run ui/app.py."""
import os
from pathlib import Path
import sys
import time

import requests
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathsnap.map_style import live_route_map

st.set_page_config(page_title="PathSnap", layout="wide")
st.markdown((Path(__file__).parent / "style.css").read_text(encoding="utf-8"), unsafe_allow_html=True)

PARTITIONS = {"Evaluator trips": "eval_test", "Student test": "student_test"}
METRICS = (
    ("Top-1 accuracy", "top1_accuracy"),
    ("Route continuity", "route_continuity"),
    ("Route F1", "route_f1"),
    ("Route precision", "route_precision"),
    ("Route recall", "route_recall"),
)


def get_json(url, method="GET", **kwargs):
    response = requests.request(method, url, **kwargs)
    response.raise_for_status()
    return response.json()


def pct(value):
    return "—" if value is None else f"{value:.2%}"


if "history" not in st.session_state:
    st.session_state.history = []
if "route_result" not in st.session_state:
    st.session_state.route_result = None

st.markdown('<header class="dispatch-header"><h1>Path<span>Snap</span></h1>'
            '<p>Route reconstruction</p></header>', unsafe_allow_html=True)

controls, main = st.columns([1, 3.8], gap="medium")
with controls:
    with st.container(border=True):
        st.subheader("Trip selector")
        partition = PARTITIONS[st.radio("Dataset", list(PARTITIONS), horizontal=False)]
        with st.expander("API connection"):
            api = st.text_input("API URL", os.getenv("PATHSNAP_API_URL", "http://localhost:8000")).strip().rstrip("/")
        trips = []
        try:
            get_json(f"{api}/health", timeout=5)
            trips = get_json(f"{api}/trips", params={"partition": partition}, timeout=10)["trips"]
            st.caption(f"API connected · {len(trips)} trips")
        except (requests.RequestException, ValueError, KeyError):
            st.error("API unavailable. Check the API connection and try again.")
            st.button("Retry connection")
        if trips:
            trip = st.selectbox("Trip ID", trips, key=f"trip_{api}_{partition}")
            run = st.button("Run map matching", type="primary", use_container_width=True)
        else:
            trip, run = None, False
            st.caption("Select a dataset with available trips.")

if run:
    with main, st.spinner(f"Reconstructing trip {trip}…"):
        start = time.perf_counter()
        try:
            result = get_json(f"{api}/predict", "POST", json={"trip_id": trip, "partition": partition}, timeout=300)
            score = result["metrics"].get("hmm", {})
            st.session_state.route_result = (api, result)
            st.session_state.history.append({
                "Trip": trip, "Dataset": partition, "Top-1": pct(score.get("top1_accuracy")),
                "Continuity": pct(score.get("route_continuity")), "Route F1": pct(score.get("route_f1")),
                "Round trip (s)": round(time.perf_counter() - start, 2),
            })
        except (requests.RequestException, ValueError, KeyError) as exc:
            st.error(f"Map matching failed for trip {trip}. {exc}")

saved = st.session_state.route_result
res = saved[1] if saved and (saved[0], saved[1]["partition"], saved[1]["trip_id"]) == (api, partition, trip) else None
with main:
    if res is None:
        with st.container(border=True):
            st.subheader("Route map")
            st.info("Select a trip and run map matching to display the route and metrics.")
    else:
        metrics = res["metrics"]
        hmm = metrics.get("hmm", {})
        clf = metrics.get("classifier_only", {})
        st.subheader(f"Trip {res['trip_id']}")
        columns = st.columns(3, gap="small")
        for col, (label, key) in zip(columns, METRICS[:3]):
            with col:
                current, baseline = hmm.get(key), clf.get(key)
                delta = f"{100 * (current - baseline):+.2f} pp" if current is not None and baseline is not None else None
                st.metric(label, pct(current), delta=delta)
        if res["gps_points"]:
            st_folium(live_route_map(res), height=540, use_container_width=True, returned_objects=[],
                     key=f"map_{api}_{partition}_{trip}")
        else:
            st.info("This trip has no GPS observations to display.")

        with st.expander("Model comparison and trip details", expanded=True):
            st.caption(f"{res['n_gps_points']:,} GPS points · {len(res['predicted_arc_ids']):,} predicted arcs · "
                       f"{res['n_segments']} segments · {res['timings_s']['request_total_s']:.2f} s API time")
            if hmm or clf:
                st.table([{"Metric": name, "Classifier only": pct(clf.get(key)), "HMM / Viterbi": pct(hmm.get(key))}
                          for name, key in METRICS])
                st.caption(f"Scored points: {metrics.get('scored_points', '—')} · "
                           f"Candidate miss rate: {pct(metrics.get('candidate_miss_rate'))}")
            else:
                st.caption("Reference scoring is unavailable for this trip.")
        with st.expander("API timings and decoder settings"):
            st.json({"timings_s": res["timings_s"], "decoder": res["decoder"], "route_info": res["route_info"]})
    if st.session_state.history:
        with st.expander("Run history"):
            st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)
