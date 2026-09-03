from flask import Blueprint, jsonify, render_template, request

from app.ml.predict_service import prediction_service

main = Blueprint("main", __name__)


@main.route("/")
def home():
    return render_template("index.html")


@main.route("/recognition")
def recognition():
    return render_template("recognition.html")


@main.route("/predict", methods=["POST"])
def predict():
    """
    Accepts hand landmarks detected by the browser-side MediaPipe
    Hands solution and returns an ML gesture prediction.

    Expected JSON body:
        { "landmarks": [ { "x": .., "y": .., "z": .. }, ... 21 items ] }

    Response JSON (see predict_service.GesturePredictionService.predict
    for the full shape):
        { "status": "ok", "prediction": "...", "confidence": 0.0-1.0, ... }
        { "status": "model_unavailable", "message": "..." }
        { "status": "invalid_input", "message": "..." }
    """
    body = request.get_json(silent=True) or {}
    landmarks = body.get("landmarks")

    result = prediction_service.predict(landmarks)

    status_code = 200
    if result["status"] == "invalid_input":
        status_code = 400

    return jsonify(result), status_code