from flask import Blueprint, request, render_template, session, redirect, url_for, jsonify
from .utils.classification import load_classifier
from .utils.localization import localize_kidney, map_coordinates_to_regions
from .utils.report import generate_medical_report
from .utils.chatbot import chatbot_response
from .utils.risk_model import predict_kidney_risk

import torch
from torchvision import transforms
from PIL import Image
import os
import glob
import time
import re
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datetime import datetime

bp = Blueprint('routes', __name__)
classifier = load_classifier()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@bp.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        session.clear()
        image_file = request.files.get("image")

        if image_file and image_file.filename != "":
            try:
                # Save uploaded image
                upload_dir = "static/uploaded"
                os.makedirs(upload_dir, exist_ok=True)
                original_filename = os.path.splitext(image_file.filename)[0]
                safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '', original_filename) + ".png"
                image_path = os.path.join(upload_dir, safe_filename)
                image_file.save(image_path)

                # Classify
                image = Image.open(image_path).convert("RGB")
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                input_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    output = classifier(input_tensor)
                    pred_idx = torch.argmax(output).item()
                    labels = ["cyst", "normal", "stone", "tumor"]
                    predicted_label = labels[pred_idx]

                # Localization if abnormal
                boxes = []
                localized_image_url = None
                if predicted_label != "normal":
                    output_folder = os.path.join("static", "localized")
                    os.makedirs(output_folder, exist_ok=True)

                    # Cleanup old
                    for f in glob.glob(os.path.join(output_folder, "*_localized.png")):
                        try:
                            os.remove(f)
                        except Exception:
                            continue

                    # YOLOv8 localization with region labels
                    boxes, localized_image = localize_kidney(image_path)

                    timestamp = int(time.time())
                    localized_filename = f"{safe_filename}_{timestamp}_localized.png"
                    localized_path = os.path.join(output_folder, localized_filename)

                    # Save the image with annotations (rectangle, region name)
                    localized_image.save(localized_path)

                    localized_image_url = os.path.join("localized", localized_filename).replace("\\", "/")

                # Report
                report = generate_medical_report(predicted_label, len(boxes))

                # Store in session
                session["label"] = predicted_label
                session["report"] = report
                session["boxes"] = boxes
                session["localized_image_url"] = localized_image_url

                # Log for debugging
                logging.info(f"Processed image - Label: {predicted_label}, Boxes: {boxes}")

                return redirect(url_for("routes.results"))

            except Exception as e:
                logging.error(f"Error processing image: {str(e)}")
                return render_template("index.html", error="Processing failed.", label=None)

        return render_template("index.html", error="No image selected.", label=None)

    return render_template("index.html", label=session.get("label"))

@bp.route("/redirect")
def redirect_after_alert():
    return render_template("redirect.html")

@bp.route("/results")
def results():
    label = session.get("label")
    report = session.get("report")
    boxes = session.get("boxes", [])
    localized_image_url = session.get("localized_image_url")

    if label is None:
        return redirect(url_for("routes.index"))

    # Convert boxes to region names
    regions = map_coordinates_to_regions(boxes)

    # Log for debugging
    logging.info(f"Results - Label: {label}, Regions: {regions}, Image URL: {localized_image_url}")

    return render_template("results.html",
                          label=label,
                          report=report,
                          regions=regions,
                          localized_image_url=localized_image_url,
                          current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@bp.route("/pdf_preview")
def pdf_preview():
    label = session.get("label")
    report = session.get("report")
    boxes = session.get("boxes", [])
    localized_image_url = session.get("localized_image_url")

    if label is None:
        return redirect(url_for("routes.index"))

    # Convert boxes to region names
    regions = map_coordinates_to_regions(boxes)

    # Log for debugging
    logging.info(f"PDF Preview - Label: {label}, Regions: {regions}, Image URL: {localized_image_url}")

    return render_template("report_pdf.html",
                          label=label,
                          report=report,
                          regions=regions,
                          localized_image_url=localized_image_url,
                          current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@bp.route("/chat", methods=["POST"])
def chat():
    user_input = request.form.get("message") or request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    response = chatbot_response(user_input)
    return jsonify({"response": response})

@bp.route("/risk-quiz", methods=["GET", "POST"])
def risk_quiz():
    prediction = None
    explanation = None
    error = None
    if request.method == "POST":
        try:
            prediction, explanation = predict_kidney_risk(request.form)
        except ValueError as e:
            error = str(e)
    return render_template("risk_quiz.html", prediction=prediction, explanation=explanation, error=error)

@bp.route("/api/localized-images")
def get_localized_images():
    folder = os.path.join("static", "localized")
    image_files = [
        f for f in sorted(
            os.listdir(folder),
            key=lambda x: os.path.getmtime(os.path.join(folder, x)),
            reverse=True
        ) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    return jsonify(image_files)