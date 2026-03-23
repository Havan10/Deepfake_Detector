import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

# CONFIGURATION
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_PATH = "ml/deepfake_model.h5"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model safely
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    print(f"CRITICAL ERROR: {MODEL_PATH} not found!")

def predict_video(video_path, frames_to_check=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // frames_to_check, 1)
    predictions = []
    
    count = 0
    while len(predictions) < frames_to_check:
        ret, frame = cap.read()
        if not ret: break
        if count % step == 0:
            frame_resized = cv2.resize(frame, (224, 224)) / 255.0
            frame_expanded = np.expand_dims(frame_resized, axis=0)
            pred = model.predict(frame_expanded, verbose=0)[0][0]
            predictions.append(pred)
        count += 1
    cap.release()
    return np.mean(predictions) if predictions else 0.5

@app.route("/")
def index():
    return render_template("index.html")

# This is the route the JavaScript calls
@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files["file"]
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    score = predict_video(path)
    label = "FAKE" if score > 0.5 else "REAL"
    conf = score if label == "FAKE" else (1 - score)

    return jsonify({
        "result": label,
        "confidence": f"{conf * 100:.2f}%"
    })

if __name__ == "__main__":
    # Using Port 8080 to avoid browser cache issues
    app.run(debug=True, port=8080)