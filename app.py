# =============================================================================
# SECTION 1 — IMPORTS
# =============================================================================
from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import google.generativeai as genai


# =============================================================================
# SECTION 2 — CONFIGURATION & PATHS
# =============================================================================
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# API Keys
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")


# =============================================================================
# SECTION 2B — AUTO DOWNLOAD KERAS MODELS FROM GOOGLE DRIVE
# (Runs on Render startup if models are missing)
# To add more models: add to KERAS_MODELS dict below
# =============================================================================
KERAS_MODELS = {
    "disease_model_final.keras": "1uhbA0wWV0Ts-pzvndRGGrCdwCYeDyeK-",
    "resnet50_finetuned.keras":  "1KGpP8zlS1NbY1_efBAj4ovs-HyavG7G_",
}

def download_models_if_missing() -> None:
    """Download .keras models from Google Drive if not present locally."""
    try:
        import gdown
    except ImportError:
        print("⚠️  gdown not installed — skipping model download")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)

    for filename, file_id in KERAS_MODELS.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f"  ✅ Already exists: {filename}")
            continue
        print(f"  ⬇️  Downloading: {filename} ...")
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, dest, quiet=False)
            print(f"  ✅ Downloaded: {filename}")
        except Exception as e:
            print(f"  ❌ Failed to download {filename}: {e}")

# Run download before models are loaded
download_models_if_missing()


# =============================================================================
# SECTION 3 — FLASK APP SETUP
# =============================================================================
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "app", "templates"),
    static_folder=os.path.join(BASE_DIR, "app", "static"),
    static_url_path="/static",
)
app.secret_key = os.environ.get("RAHAT_SECRET_KEY", "dev-change-me-in-production")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =============================================================================
# SECTION 4 — MODEL VARIABLES (filled by load_models)
# =============================================================================
crop_model: Any = None
crop_scaler: Any = None
crop_label_encoder: Any = None

fertilizer_model: Any = None
fertilizer_scaler: Any = None
fertilizer_le_soil: Any = None
fertilizer_le_crop: Any = None
fertilizer_le_fertilizer: Any = None

disease_model: Any = None
disease_class_labels: Any = None
disease_class_info: Any = None

load_errors: Dict[str, str] = {}


# =============================================================================
# SECTION 5 — MODEL LOADING
# =============================================================================
def _safe_load(path: str, loader, label: str) -> Any:
    """Load a model file safely, recording any errors."""
    try:
        obj = loader(path)
        print(f"  ✅ Loaded: {label}")
        return obj
    except Exception as e:
        load_errors[label] = str(e)
        print(f"  ❌ Failed: {label} — {e}")
        return None


def load_models() -> None:
    """Load all ML artifacts once at startup."""
    global crop_model, crop_scaler, crop_label_encoder
    global fertilizer_model, fertilizer_scaler
    global fertilizer_le_soil, fertilizer_le_crop, fertilizer_le_fertilizer
    global disease_model, disease_class_labels, disease_class_info

    load_errors.clear()
    print("\n📦 Loading models...")

    # --- Crop models ---
    crop_model = _safe_load(
        os.path.join(MODELS_DIR, "crop_model.pkl"),
        joblib.load, "crop_model"
    )
    crop_scaler = _safe_load(
        os.path.join(MODELS_DIR, "crop_scaler.pkl"),
        joblib.load, "crop_scaler"
    )
    crop_label_encoder = _safe_load(
        os.path.join(MODELS_DIR, "crop_label_encoder.pkl"),
        joblib.load, "crop_label_encoder"
    )

    # --- Fertilizer models ---
    fertilizer_model = _safe_load(
        os.path.join(MODELS_DIR, "fertilizer_model.pkl"),
        joblib.load, "fertilizer_model"
    )
    fertilizer_scaler = _safe_load(
        os.path.join(MODELS_DIR, "fertilizer_scaler.pkl"),
        joblib.load, "fertilizer_scaler"
    )
    fertilizer_le_soil = _safe_load(
        os.path.join(MODELS_DIR, "fertilizer_le_soil.pkl"),
        joblib.load, "fertilizer_le_soil"
    )
    fertilizer_le_crop = _safe_load(
        os.path.join(MODELS_DIR, "fertilizer_le_crop.pkl"),
        joblib.load, "fertilizer_le_crop"
    )
    fertilizer_le_fertilizer = _safe_load(
        os.path.join(MODELS_DIR, "fertilizer_le_fertilizer.pkl"),
        joblib.load, "fertilizer_le_fertilizer"
    )

    # --- Disease model ---
    def _load_keras(p: str):
        import tensorflow as tf
        return tf.keras.models.load_model(p)

    disease_model = _safe_load(
        os.path.join(MODELS_DIR, "disease_model_final.keras"),
        _load_keras, "disease_model_final"
    )
    disease_class_labels = _safe_load(
        os.path.join(MODELS_DIR, "disease_class_labels.pkl"),
        joblib.load, "disease_class_labels"
    )
    disease_class_info = _safe_load(
        os.path.join(MODELS_DIR, "disease_class_info.pkl"),
        joblib.load, "disease_class_info"
    )

    if load_errors:
        print(f"\n⚠️  {len(load_errors)} model(s) failed to load: {list(load_errors.keys())}")
    else:
        print("\n✅ All models loaded successfully!\n")


# =============================================================================
# SECTION 6 — HELPER FUNCTIONS
# =============================================================================
def _allowed_file(filename: str) -> bool:
    return "." in filename and \
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _decode_crop_name(raw_pred: Any) -> str:
    if isinstance(raw_pred, (str, np.str_)):
        return str(raw_pred).title()
    idx = int(np.asarray(raw_pred).ravel()[0])
    if crop_label_encoder is None:
        return str(idx)
    try:
        return str(crop_label_encoder.inverse_transform([idx])[0]).title()
    except Exception:
        if hasattr(crop_label_encoder, "classes_") and \
                0 <= idx < len(crop_label_encoder.classes_):
            return str(crop_label_encoder.classes_[idx]).title()
        return str(idx)


def _decode_fertilizer_name(raw_pred: Any) -> str:
    if isinstance(raw_pred, (str, np.str_)):
        return str(raw_pred)
    idx = int(np.asarray(raw_pred).ravel()[0])
    if fertilizer_le_fertilizer is None:
        return str(idx)
    try:
        return str(fertilizer_le_fertilizer.inverse_transform([idx])[0])
    except Exception:
        if hasattr(fertilizer_le_fertilizer, "classes_") and \
                0 <= idx < len(fertilizer_le_fertilizer.classes_):
            return str(fertilizer_le_fertilizer.classes_[idx])
        return str(idx)


def _encode_le(le: Any, value: str) -> int:
    v = value.strip()
    try:
        return int(le.transform([v])[0])
    except Exception:
        raise ValueError(f"Unknown category: {value!r}")


def _get_confidence_warning(confidence: float) -> Optional[str]:
    if confidence < 70:
        return "⚠️ Low confidence — please take a clearer, closer photo of the affected leaf."
    elif confidence < 85:
        return "📸 Moderate confidence — for better results use a close-up photo of a single leaf."
    return None


# =============================================================================
# SECTION 6B — WEATHER API HELPER
# =============================================================================
import requests as http_requests

def get_weather(lat: float, lon: float) -> Dict:
    try:
        weather_url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}"
            f"&appid={OPENWEATHER_API_KEY}"
            f"&units=metric"
        )
        response = http_requests.get(weather_url, timeout=5)
        data = response.json()

        if response.status_code != 200:
            return {"error": data.get("message", "Weather API error")}

        temperature = round(data["main"]["temp"], 1)
        humidity    = round(data["main"]["humidity"], 1)
        city        = data.get("name", "Your Location")
        country     = data["sys"].get("country", "")
        condition   = data["weather"][0]["description"].title()
        icon_code   = data["weather"][0]["icon"]
        icon_url    = f"https://openweathermap.org/img/wn/{icon_code}@2x.png"

        rainfall = 0.0
        if "rain" in data:
            rainfall = data["rain"].get("1h", 0.0)

        return {
            "temperature": temperature,
            "humidity":    humidity,
            "rainfall":    round(rainfall, 1),
            "city":        city,
            "country":     country,
            "condition":   condition,
            "icon_url":    icon_url,
            "error":       None
        }

    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# SECTION 6C — WEATHER API ROUTE
# =============================================================================
@app.route("/api/weather")
def weather_api():
    from flask import jsonify
    try:
        lat = float(request.args.get("lat", 0))
        lon = float(request.args.get("lon", 0))
        if lat == 0 and lon == 0:
            return jsonify({"error": "Invalid coordinates"})
        weather = get_weather(lat, lon)
        return jsonify(weather)
    except Exception as e:
        return jsonify({"error": str(e)})


# =============================================================================
# SECTION 6D — GROQ AI FERTILIZER ADVICE
# =============================================================================
from groq import Groq

GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
groq_client   = Groq(api_key=GROQ_API_KEY)

def get_gemini_fertilizer_advice(
    soil_type: str,
    crop_type: str,
    nitrogen: float,
    potassium: float,
    phosphorous: float,
    temperature: float,
    humidity: float,
    moisture: float,
) -> str:
    try:
        prompt = f"""
You are an expert Indian agricultural scientist advising a farmer.
Be concise, practical and helpful.

Farmer's soil and crop data:
- Crop: {crop_type}
- Soil Type: {soil_type}
- Nitrogen (N): {nitrogen} kg/ha
- Phosphorous (P): {phosphorous} kg/ha
- Potassium (K): {potassium} kg/ha
- Temperature: {temperature}°C
- Humidity: {humidity}%
- Soil Moisture: {moisture}%

Give advice in this exact format:

1. RECOMMENDED FERTILIZER
- Best fertilizer name for this crop and soil
- Why it suits this specific soil and crop
- Dosage per acre
- When to apply (sowing/vegetative/flowering stage)

2. BIO FERTILIZER ALTERNATIVES
- 2 specific bio fertilizers suitable for {crop_type}
- Benefits of each bio fertilizer
- How and when to apply

3. ORGANIC OPTIONS
- 1-2 organic alternatives (FYM, compost, green manure)
- Quantity per acre

4. IMPORTANT TIPS
- 2-3 specific tips for {crop_type} on {soil_type} soil
- Any nutrient deficiency warnings based on NPK values

Keep advice simple and practical for Indian farmers.
"""
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Indian agricultural scientist. Give practical, specific farming advice."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"AI advice unavailable. Error: {str(e)}"


# =============================================================================
# SECTION 7 — CROP RECOMMENDATION ROUTE
# =============================================================================
@app.route("/crop", methods=["GET", "POST"])
def crop():
    if request.method == "POST":
        try:
            N           = float(request.form["N"])
            P           = float(request.form["P"])
            K           = float(request.form["K"])
            temperature = float(request.form["temperature"])
            humidity    = float(request.form["humidity"])
            ph          = float(request.form["ph"])
            rainfall    = float(request.form["rainfall"])

            features   = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            scaled     = crop_scaler.transform(features)
            prediction = crop_model.predict(scaled)
            crop_name  = _decode_crop_name(prediction)

            return render_template(
                "crop_result.html",
                crop_name=crop_name,
                N=N, P=P, K=K,
                temperature=temperature,
                humidity=humidity,
                ph=ph,
                rainfall=rainfall,
            )
        except Exception as e:
            return render_template("crop.html", error=str(e))

    return render_template("crop.html", error=None)


# =============================================================================
# SECTION 8 — FERTILIZER RECOMMENDATION ROUTE
# =============================================================================
@app.route("/fertilizer", methods=["GET", "POST"])
def fertilizer():
    if request.method == "POST":
        try:
            temperature  = float(request.form["temperature"])
            humidity     = float(request.form["humidity"])
            moisture     = float(request.form["moisture"])
            soil_type    = request.form["soil_type"]
            crop_type    = request.form["crop_type"]
            nitrogen     = float(request.form["nitrogen"])
            potassium    = float(request.form["potassium"])
            phosphorous  = float(request.form["phosphorous"])

            ai_advice = get_gemini_fertilizer_advice(
                soil_type=soil_type,
                crop_type=crop_type,
                nitrogen=nitrogen,
                potassium=potassium,
                phosphorous=phosphorous,
                temperature=temperature,
                humidity=humidity,
                moisture=moisture,
            )

            return render_template(
                "fertilizer_result.html",
                ai_advice=ai_advice,
                temperature=temperature,
                humidity=humidity,
                moisture=moisture,
                soil_type=soil_type,
                crop_type=crop_type,
                nitrogen=nitrogen,
                potassium=potassium,
                phosphorous=phosphorous,
            )
        except Exception as e:
            return render_template("fertilizer.html", error=str(e))

    return render_template("fertilizer.html", error=None)


# =============================================================================
# SECTION 9 — PLANT DISEASE DETECTION ROUTE
# =============================================================================
@app.route("/disease", methods=["GET", "POST"])
def disease():
    if request.method == "GET":
        return render_template("disease.html", error=None)

    try:
        if "leaf_image" not in request.files:
            return render_template("disease.html", error="No image uploaded")

        file = request.files["leaf_image"]
        if not file or not file.filename:
            return render_template("disease.html", error="No image selected")

        if not _allowed_file(file.filename):
            return render_template("disease.html", error="Allowed formats: PNG, JPG, JPEG")

        ext      = file.filename.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file.save(filepath)

        from PIL import Image as PILImage
        from tensorflow.keras.applications.resnet50 import preprocess_input

        img   = PILImage.open(filepath).convert("RGB")
        img   = img.resize((224, 224), PILImage.Resampling.LANCZOS)
        arr   = np.asarray(img, dtype=np.float32)
        batch = np.expand_dims(arr, axis=0)
        batch = preprocess_input(batch)

        predictions   = disease_model.predict(batch, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence    = float(predictions[0][predicted_idx] * 100)

        predicted_class = str(disease_class_labels[predicted_idx])
        remedy = (
            disease_class_info.get(predicted_class, "Consult local agricultural expert")
            if isinstance(disease_class_info, dict)
            else "Consult local agricultural expert"
        )

        is_healthy         = "healthy" in predicted_class.lower()
        display_name       = predicted_class.replace("_", " ").strip()
        confidence_warning = _get_confidence_warning(confidence)

        return render_template(
            "disease_result.html",
            image_path=filename,
            predicted_class=predicted_class,
            display_name=display_name,
            confidence=round(confidence, 1),
            remedy=remedy,
            is_healthy=is_healthy,
            confidence_warning=confidence_warning,
        )

    except Exception as e:
        return render_template("disease.html", error=str(e))


# =============================================================================
# SECTION 10 — HOME ROUTE
# =============================================================================
@app.route("/")
def index():
    return render_template("index.html", load_errors=load_errors)


# =============================================================================
# SECTION 11 — RUN APP
# =============================================================================
load_models()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
