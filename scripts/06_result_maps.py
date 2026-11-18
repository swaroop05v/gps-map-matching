"""Result maps (Folium) for the report + PNG screenshots via headless Edge/Chrome.

* zoomed maps on the stretch of a student_test trip where classifier-only and HMM disagree most
* screenshots of every map in results/maps -> report/screenshots/map_*.png
Uses the final exported model (models/final_model.joblib, models/hmm_config.json).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

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


def zoom_map(tid, model, h, tag, window=60, maps=MAPS, themed=False):
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
             title=f"student_test {tid} ({tag.replace('_', ' ')})",
             themed=themed).save(os.path.join(maps, f"result_{tag}_{tid}.html"))
    # zoom on the stretch where classifier-only and HMM disagree most
    seg = tr.track[prep.orig_idx[lo:hi]]
    pad = 0.0008
    bounds = [[float(seg[:, 1].min() - pad), float(seg[:, 0].min() - pad)], [float(seg[:, 1].max() + pad), float(seg[:, 0].max() + pad)]]
    m = trip_map(tr.nodes, tr.arcs, tr.track, ref_route=tr.route, layers=layers, bounds=bounds,
                 title=f"student_test {tid}: zoom on fixes {lo}-{hi}", themed=themed)
    name = f"result_zoom_{tag}_{tid}.html"
    m.save(os.path.join(maps, name))
    ok = prep.align_dist[lo:hi] <= 50
    acc_c = float((clf[lo:hi] == prep.ref_arc[lo:hi])[ok].mean())
    acc_h = float((hmm[lo:hi] == prep.ref_arc[lo:hi])[ok].mean())
    return name, lo, hi, acc_c, acc_h


def screenshot(html_path, png_path, w=1440, hgt=920):
    exe = next((e for e in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                            shutil.which("msedge"), shutil.which("google-chrome"),
                            shutil.which("chromium")] if e and os.path.exists(e)), None)
    if not exe:
        raise RuntimeError("Edge or Chromium is required to capture maps")
    result = subprocess.run([exe, "--headless", "--disable-gpu", "--hide-scrollbars",
                             f"--window-size={w},{hgt}", "--virtual-time-budget=12000",
                             f"--screenshot={Path(png_path).resolve()}", Path(html_path).resolve().as_uri()],
                            capture_output=True, text=True, timeout=120)
    if result.returncode or not Path(png_path).is_file():
        raise RuntimeError(result.stderr.strip() or f"Browser failed to capture {html_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gta", action="store_true", help="Use the dark theme and preserve the original captures")
    parser.add_argument("--maps-only", action="store_true", help="Generate interactive HTML without opening a browser")
    args = parser.parse_args()
    shots = Path(SHOTS) / "gta" if args.gta else Path(SHOTS)
    maps = shots / "maps" if args.gta else Path(MAPS)
    shots.mkdir(parents=True, exist_ok=True)
    maps.mkdir(parents=True, exist_ok=True)
    if args.gta:
        for tid, suffix in (("00000001", ""), ("00000002", "_offroute"), ("00000004", "")):
            trip = load_trip(P["train"], tid)
            title = f"Training trip {tid}" + (" · Off-route GPS" if suffix else "")
            trip_map(trip.nodes, trip.arcs, trip.track, ref_route=trip.route,
                     title=title, themed=True).save(str(maps / f"eda_train_{tid}{suffix}.html"))
    model = joblib.load(os.path.join(P["models"], "final_model.joblib"))
    h = json.load(open(os.path.join(P["models"], "hmm_config.json")))
    piv = pd.read_csv(os.path.join(P["results"], "hmm_student_test_per_trip_top1.csv"), dtype={"trip_id": str}).sort_values("delta_hmm_minus_clf")
    rows = []
    for tag, tid in [("largest_gain", piv.trip_id.iloc[-1]), ("smallest_gain", piv.trip_id.iloc[0]),
                     ("median_gain", piv.trip_id.iloc[len(piv) // 2])]:
        name, lo, hi, ac, ah = zoom_map(tid, model, h, tag, maps=maps, themed=args.gta)
        rows.append({"map": name, "trip_id": tid, "fix_from": lo, "fix_to": hi, "top1_classifier_window": ac, "top1_hmm_window": ah})
        print(rows[-1])
    if not args.gta:
        pd.DataFrame(rows).to_csv(os.path.join(P["results"], "zoom_maps_summary.csv"), index=False)
    if not args.maps_only:
        for path in sorted(maps.glob("*.html")):
            output = shots / f"map_{path.stem}.png"
            screenshot(path, output)
            print("captured", output)


if __name__ == "__main__":
    main()
