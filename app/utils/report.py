from datetime import datetime

def generate_medical_report(predicted_label, num_boxes):
    today = datetime.today().strftime('%Y-%m-%d')
    label_map = {
        "cyst": "- Usually observation unless painful or large\n- Aspiration or surgery if needed",
        "stone": "- Hydration and pain control\n- Shock wave lithotripsy or ureteroscopy",
        "tumor": "- Biopsy and staging\n- Surgery, ablation or targeted therapy",
        "normal": "✅ No abnormalities detected."
    }

    treatment = label_map.get(predicted_label, "Consult a specialist.")
    return f"""
🩺 Nephrology Diagnostic Report – {today}
-----------------------------------------
🔹 Classification Result: {predicted_label.upper()}
🔹 Abnormal Regions Detected: {num_boxes}

📄 Recommended Actions:
{treatment}

📍 Note: Kindly follow up with a certified nephrologist.
"""