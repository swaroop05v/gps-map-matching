"""PathSnap API (runs on Laptop 1).

Endpoints
  GET  /health                 -> model + decoder loaded?
  GET  /trips?partition=...    -> list of trip ids available on the server (no labels)
  POST /predict                -> {"trip_id": "...", "partition": "eval_test"}
                                  runs the full pipeline (features -> LightGBM -> HMM/Viterbi),
                                  then scores the prediction against the server-side reference route.

The reference route never leaves the server: the response only contains the GPS fixes,
the predicted route geometry and the accuracy numbers.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import json
import logging
import os
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.config import load_config
from pathsnap.features import PreparedTrip
from pathsnap.hmm import build_route
from pathsnap.io import list_trips, load_trip
from pathsnap.metrics import continuity, route_set_prf
from pathsnap.pipeline import predict_trip

LABEL_MAX_DIST = 50.0

cfg = load_config()
PARTITIONS = {
    "eval_test": os.environ.get("PATHSNAP_EVAL_DIR", cfg["paths"]["eval_test"]),
    "student_test": cfg["paths"]["student_test"],
}

os.makedirs(os.path.join(cfg["paths"]["results"], "logs"), exist_ok=True)
log = logging.getLogger("pathsnap.api")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(cfg["paths"]["results"], "logs", "api_requests.log"))):
    h.setFormatter(_fmt)
    log.addHandler(h)

MODEL = joblib.load(os.path.join(cfg["paths"]["models"], "final_model.joblib"))
with open(os.path.join(cfg["paths"]["models"], "hmm_config.json")) as f:
    HCFG = json.load(f)
log.info("model loaded: %s | decoder alpha=%s beta=%s beam=%s", type(MODEL).__name__, HCFG["alpha"], HCFG["beta"], HCFG["beam"])

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str
    partition: str = "eval_test"


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path,
             request.client.host if request.client else "?", response.status_code, 1000 * (time.perf_counter() - t0))
    return response


def _dir(partition):
    if partition not in PARTITIONS:
        raise HTTPException(400, f"unknown partition {partition}")
    return PARTITIONS[partition]


@app.get("/health")
def health():
    return {"status": "ok", "model": type(MODEL).__name__, "decoder": {k: HCFG[k] for k in ("alpha", "beta", "beam")},
            "partitions": {k: len(list_trips(v)) for k, v in PARTITIONS.items()}}


@app.get("/trips")
def trips(partition: str = "eval_test"):
    ids = list_trips(_dir(partition))
    return {"partition": partition, "trips": ids, "count": len(ids)}


def _lines(net, route):
    """Route arcs -> list of polylines [[lat, lon], ...] chaining connected arcs."""
    lines, cur, prev_to = [], [], None
    for a in route:
        u, v = net.arcs[a]
        pu, pv = net.nodes_lonlat[u], net.nodes_lonlat[v]
        if prev_to is not None and u == prev_to and cur:
            cur.append([float(pv[1]), float(pv[0])])
        else:
            if len(cur) > 1:
                lines.append(cur)
            cur = [[float(pu[1]), float(pu[0])], [float(pv[1]), float(pv[0])]]
        prev_to = v
    if len(cur) > 1:
        lines.append(cur)
    return lines


@app.post("/predict")
def predict(req: PredictRequest):
    d = _dir(req.partition)
    if req.trip_id not in list_trips(d):
        raise HTTPException(404, f"trip {req.trip_id} not found in {req.partition}")
    t0 = time.perf_counter()
    trip = load_trip(d, req.trip_id, with_route=True)
    # ---- prediction (the pipeline receives the trip without its route)
    out = predict_trip(trip, MODEL, cfg, HCFG, use_hmm=True)
    prep = out["prep"]
    clf_route, _ = build_route(prep, out["clf_pred"])
    pred_s = time.perf_counter() - t0

    # ---- server-side scoring against the reference route (never sent to the client)
    metrics = {}
    if trip.route is not None and len(trip.route):
        ref = PreparedTrip(trip, cfg, with_labels=True)
        ok = ref.align_dist <= LABEL_MAX_DIST
        for tag, pred, rt in [("hmm", out["hmm_pred"], out["route"]), ("classifier_only", out["clf_pred"], clf_route)]:
            pr, rc, f1 = route_set_prf(rt, trip.route)
            metrics[tag] = {
                "top1_accuracy": float((pred[ok] == ref.ref_arc[ok]).mean()) if ok.any() else None,
                "route_continuity": continuity(pred, prep.net.arcs)[0],
                "route_precision": pr, "route_recall": rc, "route_f1": f1,
            }
        metrics["scored_points"] = int(ok.sum())
        metrics["candidate_miss_rate"] = float(np.mean([r not in c[0] for r, c, o in zip(ref.ref_arc, prep.cands, ok) if o])) if ok.any() else None

    track = prep.track
    step = max(1, len(track) // 3000)
    resp = {
        "trip_id": req.trip_id,
        "partition": req.partition,
        "n_gps_points": int(len(track)),
        "n_segments": int(prep.segment_id.max() + 1) if len(track) else 0,
        "gps_points": [[float(la), float(lo)] for lo, la in track[::step, :2]],
        "predicted_route": _lines(prep.net, out["route"]),
        "classifier_only_route": _lines(prep.net, clf_route),
        "predicted_arc_ids": [int(a) for a in out["route"]],
        "route_info": out["route_info"],
        "metrics": metrics,
        "timings_s": {**{k: round(v, 3) for k, v in out["timings"].items()}, "predict_total_s": round(pred_s, 3),
                      "request_total_s": round(time.perf_counter() - t0, 3)},
        "decoder": {k: HCFG[k] for k in ("alpha", "beta", "beam")},
        "served_at": datetime.now().isoformat(timespec="seconds"),
    }
    acc = metrics.get("hmm", {}).get("top1_accuracy")
    log.info("PREDICT trip=%s partition=%s points=%d route_arcs=%d top1_hmm=%s top1_clf=%s continuity=%s time=%.2fs",
             req.trip_id, req.partition, len(track), len(out["route"]),
             f"{acc:.4f}" if acc is not None else "n/a",
             f"{metrics['classifier_only']['top1_accuracy']:.4f}" if metrics else "n/a",
             f"{metrics['hmm']['route_continuity']:.4f}" if metrics else "n/a", time.perf_counter() - t0)
    return resp
