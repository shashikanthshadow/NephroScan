


# NephroScan: An AI-Powered System for Kidney Disease Detection and Risk Prediction

NephroScan is an AI-powered system designed to assist in the diagnosis and risk prediction of kidney conditions. [cite\_start]It integrates multiple machine learning models within a user-friendly Flask web application to automate the analysis of CT scans and urine test data.



-----

## 🧐 Features

  * **Image Classification**: Classifies kidney CT scans as **Cyst**, **Normal**, **Stone**, or **Tumor** with a **95%** accuracy using a fine-tuned ResNet18 model.
  * **Abnormality Localization**: Pinpoints the exact location of cysts, stones, and tumors on CT scans using the YOLOv8 Nano model, achieving an **89.8% mAP**.
  * **Kidney Stone Risk Prediction**: Predicts the risk of kidney stones with a **96%** accuracy on a large test set using a Random Forest classifier and urine analysis dat.
  * **Interactive Chatbot**: An integrated chatbot ("NephroBot") provides simplified, context-aware answers to user questions about kidney health and diagnoses.
  * **Automated Report Generation**: Generates a downloadable PDF report containing the diagnostic results, an annotated image, and risk predictions.
  * **Web Application**: A Flask-based web interface allows for easy uploading of CT scans and input of risk quiz data.

![NephroScan](assets/demo.gif)




-----

## 📂 Project Structure

```
├── NephroScan/
│   ├── static/             # CSS, JS, and image assets
│   ├── templates/          # HTML templates for the web app
│   ├── models/             # Pre-trained model weights (e.g., ResNet18, YOLOv8)
│   ├── classification.py   # ResNet18 model implementation
│   ├── localization.py     # YOLOv8 model implementation
│   ├── chatbot.py          # RAG-based chatbot logic
│   ├── report.py           # PDF report generation script
│   ├── risk_model.py       # Random Forest model implementation
│   └── run.py              # Main Flask application entry point
├── requirements.txt        # Python dependencies
└── README.md
```



-----

## 🛠️ Installation

### Prerequisites

  * Python 3.8+
  * A GPU is recommended for faster YOLOv8 inference[cite: 584].

### Steps

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/shashikanthshadow/NephroScan.git
    cd NephroScan
    ```
2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```
      * The `requirements.txt` file includes `Flask`, `PyTorch`, `Ultralytics YOLOv8`, `scikit-learn`, `FAISS`, and `all-MiniLM-L6-v2`.


-----

## 🚀 Usage

1.  **Run the Flask application**:
    ```bash
    python run.py
    ```
2. Open your web browser and navigate to `http://127.0.0.1:5000` to access the application.
3. Upload a CT scan, fill out the risk quiz, or interact with the chatbot to use the system's features.


-----


