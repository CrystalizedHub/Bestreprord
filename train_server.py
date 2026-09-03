#!/usr/bin/env python3
"""
Shared dodge-training aggregation server.

This is the backend that lets every copy of the autofarm script contribute its
dodge-learned escapes so the model improves for ALL users ("goes to everyone,
benefits everyone").

The Roblox client (see Train.endpoint in autofarm.lua) does:
    GET  /train   ->  {"names": {...}}   (the current shared model)
    POST /train   body {"names": {..}}  ->  merges the client's counts into the
                                            store, saves, returns the merged
                                            {"names": {...}}

Counts are summed, so the more users dodge, the more data accumulates and the
better the bias becomes for every client that pulls it.

HOW TO HOST (pick one):
  1) Local (quick test):     python3 train_server.py  ->  http://127.0.0.1:8080
  2) Free cloud (production): deploy this to Render / Railway / Hugging Face
     Spaces / Fly.io, set the port from the env var PORT, then set
     Train.endpoint in autofarm.lua to that public HTTPS URL, e.g.
         Train.endpoint = "https://your-app-/train"
     Rebuild/ship the updated script. Everyone running it then syncs through it.

Dependencies:  pip install flask
"""

import os
import json
import threading

from flask import Flask, request, jsonify

app = Flask(__name__)

PORT = int(os.environ.get("PORT", "8080"))
DATA_FILE = os.environ.get("TRAIN_DATA", "train_data.json")

_lock = threading.Lock()
_model = {"names": {}}


def load_state():
    global _model
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                _model = json.load(f)
            _model.setdefault("names", {})
    except Exception:
        _model = {"names": {}}


def save_state():
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(_model, f)
    except Exception:
        pass


def merge_counts(dst, src):
    """Sum src counts into dst in place (g and b buckets per attack name)."""
    for name, r in (src or {}).items():
        if not isinstance(r, dict):
            continue
        lr = dst.setdefault(name, {"g": {}, "b": {}})
        for k in ("g", "b"):
            srcb = r.get(k)
            if not isinstance(srcb, dict):
                continue
            dstb = lr.setdefault(k, {})
            for bk, ct in srcb.items():
                dstb[int(bk)] = int(dstb.get(int(bk), 0)) + int(ct or 0)


@app.route("/train", methods=["GET"])
def get_train():
    with _lock:
        return jsonify({"names": _model["names"]})


@app.route("/train", methods=["POST"])
def post_train():
    body = request.get_json(silent=True) or {}
    incoming = body.get("names") or {}
    with _lock:
        merge_counts(_model["names"], incoming)
        save_state()
        return jsonify({"names": _model["names"]})


@app.route("/", methods=["GET"])
def ping():
    return jsonify({"ok": True, "mode": "dodge-train-shared"})


if __name__ == "__main__":
    load_state()
    print(f"dodge-train server on port {PORT}  (data -> {DATA_FILE})")
    app.run(host="0.0.0.0", port=PORT)
