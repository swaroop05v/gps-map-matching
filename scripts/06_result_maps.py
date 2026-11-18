"""Result maps (Folium) for the report + PNG screenshots via headless Edge/Chrome.

* zoomed maps on the stretch of a student_test trip where classifier-only and HMM disagree most
* screenshots of every map in results/maps -> report/screenshots/map_*.png
Uses the final exported model (models/final_model.joblib, models/hmm_config.json).
"""
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np
import pandas as pd

from pathsnap.config import load_config
from pathsnap.features import FEATURES, PreparedTrip
from pathsnap.hmm import build_route, build_states, viterbi
from pathsnap.io import load_trip
from pathsnap.viz import trip_map

cfg = load_config()
P = cfg["paths"]
MAPS = os.path.join(P["results"], "maps")
SHOTS = os.path.join(os.path.dirname(P["results"]), "report", "screenshots")
os.makedirs(SHOTS, exist_ok=True)


def zoom_map(tid, model, h, tag, window=60):
    tr = load_trip(P["student_test"], tid)
    prep = PreparedTrip(tr, cfg, with_labels=True)
    cand = prep.candidate_table()
    prob = model.predict_proba(cand[FEATURES].to_numpy(np.float32))[:, 1]
    n = len(prep.xy)
    clf = np.full(n, -1)
    order = np.lexsort((-prob, cand.pt_idx.to_numpy()))
    ps = cand.pt_idx.to_numpy()[order]
    first = np.r_[True, ps[1:] != ps[:-1]]
    clf[ps[first]] = cand.arc_id.to_numpy()[order][first]
    hmm = viterbi(prep, build_states(cand, prob, n, h["beam"], h["alpha"]), h)
    diff = (clf != hmm).astype(float)
    score = np.convolve(diff, np.ones(window), mode="same")
    c = int(np.argmax(score))
    lo, hi = max(0, c - window), min(n, c + window)
    clf_route, _ = build_route(prep, clf)
    hmm_route, _ = build_route(prep, hmm)
    layers = [("Classifier-only route", clf_route, "#f4a261"), ("HMM / Viterbi route", hmm_route, "#d62728")]
    # full-extent map
    trip_map(tr.nodes, tr.arcs, tr.track, ref_route=tr.route, layers=layers,
             title=f"student_test {tid} ({tag.replace('_', ' ')})").save(os.path.join(MAPS, f"result_{tag}_{tid}.html"))
    # zoom on the stretch where classifier-only and HMM disagree most
    seg = tr.track[prep.orig_idx[lo:hi]]
    pad = 0.0008
    bounds = [[float(seg[:, 1].min() - pad), float(seg[:, 0].min() - pad)], [float(seg[:, 1].max() + pad), float(seg[:, 0].max() + pad)]]
    m = trip_map(tr.nodes, tr.arcs, tr.track, ref_route=tr.route, layers=layers, bounds=bounds,
                 title=f"student_test {tid}: zoom on fixes {lo}-{hi}")
    name = f"result_zoom_{tag}_{tid}.html"
    m.save(os.path.join(MAPS, name))
    ok = prep.align_dist[lo:hi] <= 50
    acc_c = float((clf[lo:hi] == prep.ref_arc[lo:hi])[ok].mean())
    acc_h = float((hmm[lo:hi] == prep.ref_arc[lo:hi])[ok].mean())
    return name, lo, hi, acc_c, acc_h


def screenshot(html_name, png_name, w=1200, hgt=800):
    exe = next((e for e in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                            shutil.which("msedge"), shutil.which("google-chrome")] if e and os.path.exists(e)), None)
    if not exe:
        print("no browser for screenshots")
        return
    url = "file:///" + os.path.join(MAPS, html_name).replace("\\", "/")
    subprocess.run([exe, "--headless", "--disable-gpu", "--hide-scrollbars", f"--window-size={w},{hgt}",
                    "--virtual-time-budget=10000", f"--screenshot={os.path.join(SHOTS, png_name)}", url],
                   capture_output=True, timeout=120)


def main():
    model = joblib.load(os.path.join(P["models"], "final_model.joblib"))
    h = json.load(open(os.path.join(P["models"], "hmm_config.json")))
    piv = pd.read_csv(os.path.join(P["results"], "hmm_student_test_per_trip_top1.csv"), dtype={"trip_id": str}).sort_values("delta_hmm_minus_clf")
    rows = []
    for tag, tid in [("largest_gain", piv.trip_id.iloc[-1]), ("smallest_gain", piv.trip_id.iloc[0]),
                     ("median_gain", piv.trip_id.iloc[len(piv) // 2])]:
        name, lo, hi, ac, ah = zoom_map(tid, model, h, tag)
        rows.append({"map": name, "trip_id": tid, "fix_from": lo, "fix_to": hi, "top1_classifier_window": ac, "top1_hmm_window": ah})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(os.path.join(P["results"], "zoom_maps_summary.csv"), index=False)
    for f in sorted(os.listdir(MAPS)):
        if f.endswith(".html"):
            screenshot(f, "map_" + f.replace(".html", ".png"))
            print("shot", f)


if __name__ == "__main__":
    main()
