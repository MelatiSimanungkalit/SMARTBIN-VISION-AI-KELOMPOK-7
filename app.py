from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import json
import base64
import io
from PIL import Image

app = Flask(__name__)

# ── Load Model ─────────────────────────────────────────────
MODEL   = None
CLASSES = None

CLASS_INFO = {
    "Anorganik": {
        "icon": "🔵",
        "color": "#3b82f6",
        "bin": "Biru",
        "tips": "Pisahkan dari sampah organik. Termasuk plastik, kaca, logam, dan kertas.",
        "contoh": ["plastik", "botol kaca", "kaleng", "kertas", "kardus"]
    },
    "B3": {
        "icon": "☢️",
        "color": "#ef4444",
        "bin": "Merah (Khusus B3)",
        "tips": "BERBAHAYA! Jangan campur dengan sampah lain. Antar ke fasilitas pengolahan B3 terdekat.",
        "contoh": ["baterai", "lampu neon", "cat bekas", "obat kadaluarsa", "elektronik rusak"]
    },
    "Organik": {
        "icon": "🌿",
        "color": "#22c55e",
        "bin": "Hijau",
        "tips": "Bisa dijadikan kompos. Pisahkan dari sampah anorganik untuk mempercepat pengomposan.",
        "contoh": ["sisa makanan", "daun", "kulit buah", "sayur", "ranting"]
    },
}

KEYWORD_MAP = {
    # Organik
    "sisa makanan": "Organik", "daun": "Organik", "kulit buah": "Organik",
    "nasi": "Organik", "sayur": "Organik", "buah": "Organik", "apel": "Organik",
    "pisang": "Organik", "ranting": "Organik", "rumput": "Organik",
    "ampas kopi": "Organik", "cangkang telur": "Organik",
    # Anorganik
    "plastik": "Anorganik", "botol plastik": "Anorganik", "kantong": "Anorganik",
    "sedotan": "Anorganik", "styrofoam": "Anorganik", "kresek": "Anorganik",
    "botol kaca": "Anorganik", "kaca": "Anorganik", "toples": "Anorganik",
    "kaleng": "Anorganik", "besi": "Anorganik", "aluminium": "Anorganik",
    "logam": "Anorganik", "paku": "Anorganik",
    "koran": "Anorganik", "kardus": "Anorganik", "kertas": "Anorganik",
    "buku": "Anorganik", "karton": "Anorganik",
    "baju": "Anorganik", "kaos": "Anorganik", "kain": "Anorganik",
    # B3
    "baterai": "B3", "baterai aa": "B3", "kabel": "B3",
    "handphone": "B3", "hp": "B3", "laptop": "B3",
    "charger": "B3", "elektronik": "B3", "remote": "B3",
    "lampu neon": "B3", "cat": "B3", "tiner": "B3",
    "obat": "B3", "pestisida": "B3", "oli bekas": "B3",
}

def load_model():
    global MODEL, CLASSES
    try:
        MODEL = tf.keras.models.load_model("smartbin_best.h5")
        with open("class_names.json", "r") as f:
            data = json.load(f)
            CLASSES = data["classes"]
        print("✅ Model loaded:", CLASSES)
    except Exception as e:
        print(f"⚠️  Model tidak ditemukan: {e}")

def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Routes ─────────────────────────────────────────────────
@app.route("/")
def index():
    model_ready = MODEL is not None
    return render_template("index.html", model_ready=model_ready, class_info=CLASS_INFO)

@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model belum dimuat. Pastikan smartbin_best.h5 ada di folder ini."}), 503

    data = request.get_json()
    img_data = data.get("image", "")

    if "," in img_data:
        img_data = img_data.split(",")[1]
    img_bytes = base64.b64decode(img_data)
    img       = Image.open(io.BytesIO(img_bytes))

    arr   = preprocess(img)
    preds = MODEL.predict(arr, verbose=0)[0]
    order = np.argsort(preds)[::-1]

    results = []
    for i in order:
        cls  = CLASSES[i]
        prob = float(preds[i])
        info = CLASS_INFO.get(cls, {})
        results.append({
            "label"  : cls,
            "display": cls,
            "prob"   : prob,
            "pct"    : round(prob * 100, 1),
            "color"  : info.get("color", "#64748b"),
            "icon"   : info.get("icon", ""),
            "bin"    : info.get("bin", ""),
            "tips"   : info.get("tips", ""),
            "contoh" : info.get("contoh", []),
        })

    return jsonify({"results": results})

@app.route("/search", methods=["POST"])
def search():
    query = request.get_json().get("query", "").lower().strip()
    for kw in sorted(KEYWORD_MAP, key=len, reverse=True):
        if kw in query:
            cls  = KEYWORD_MAP[kw]
            info = CLASS_INFO[cls]
            return jsonify({
                "found"  : True,
                "label"  : cls,
                "display": cls,
                "color"  : info["color"],
                "icon"   : info["icon"],
                "bin"    : info["bin"],
                "tips"   : info["tips"],
                "contoh" : info["contoh"],
            })
    return jsonify({"found": False})

if __name__ == "__main__":
    load_model()
    app.run(debug=True, port=5000)
