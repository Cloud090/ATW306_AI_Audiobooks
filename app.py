import os, json, threading
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

LOCK = threading.Lock()
STATE_PATH = "/tmp/current_link.json"

def load_state():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(d):
    tmp_path = STATE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(d, f)
    os.replace(tmp_path, STATE_PATH)

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "EA-Endpoint",
        "status": "ok",
        "set_endpoint": "/set (POST, header X-Auth-Token required)",
        "get_endpoint": "/current (GET)"
    })

@app.route("/current", methods=["GET"])
def current():
    with LOCK:
        state = load_state()
    url = state.get("url")
    if not url:
        return jsonify({"error": "no url set"}), 404
    return jsonify({"url": url})

@app.route("/set", methods=["POST"])
def set_url():
    auth_token = request.headers.get("X-Auth-Token", "")
    expected = os.environ.get("AUTH_TOKEN", "")
    if not expected or auth_token != expected:
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    if not url.startswith("http"):
        return jsonify({"error": "invalid url"}), 400

    with LOCK:
        state = {"url": url}
        save_state(state)
    return jsonify({"ok": True, "url": url})
