# 🧠 NephroScan – AI-Powered Kidney Disease Detection Platform

**NephroScan** is a comprehensive Flask-based AI application for kidney disease screening, diagnosis, and health risk assessment using deep learning and large language models. It supports CT scan classification, abnormality localization, report generation, chatbot queries using RAG (Retrieval-Augmented Generation), and kidney stone risk prediction.

---

## 🚀 Features

- 🔍 **CT Scan Classification** – Predicts `normal`, `cyst`, `stone`, or `tumor` using ResNet18.
- 🧭 **Disease Localization** – Highlights affected kidney regions with YOLOv8.
- 📝 **Auto Medical Reports** – Generates AI-based diagnosis and summaries.
- 💬 **Chatbot (RAG)** – Answers nephrology questions with MiniLM + FAISS.
- 📊 **Risk Assessment Quiz** – Predicts kidney stone risk using Random Forest.
- 🗂️ **Session State** – Preserves user inputs and results across views.
- 🖼️ **Recent Localized API** – Returns list of recent localized images.

---

## 📁 Project Structure

```plaintext
NephroScan/
├── app/
│   ├── __init__.py
│   ├── main.py                # Flask entrypoint
│   ├── routes.py              # Routes: Upload, Chat, Report, Quiz
│   ├── utils/
│   │   ├── classification.py  # ResNet18 inference
│   │   ├── localization.py    # YOLOv8 localization
│   │   ├── chatbot.py         # Chatbot with MiniLM & FAISS
│   │   ├── report.py          # Report generator
│   │   └── risk_model.py      # Kidney risk prediction logic
├── templates/
│   ├── index.html
│   ├── results.html
│   ├── report_pdf.html
│   └── risk_quiz.html
├── static/
│   ├── uploaded/              # User-uploaded CT scans
│   └── localized/             # Localized/annotated output
├── models/
│   ├── ResNet18_Optimized_AntiOverfit.pth
│   ├── yolov8_localizer.pt
│   ├── kidney_stone_rf_model.joblib
│   ├── kidney_stone_scaler.joblib
│   └── kindey stone urine analysis.csv
├── rag/
│   ├── faiss_index.faiss      # Vector store
│   └── documents/             # Source documents for RAG
├── requirements.txt           # Python dependencies
├── kidney_env.yml             # Conda environment file
└── run.py                     # Shortcut to run app
⚙️ Installation
1. Clone the Repository

git clone https://github.com/yourusername/NephroScan.git
cd NephroScan
2. Set Up Environment
Using conda:
conda env create -f kidney_env.yml
conda activate kidney_env

Or manually with venv:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Download / Place Required Models
Ensure the following files are in the models/ directory:

ResNet18_Optimized_AntiOverfit.pth

yolov8_localizer.pt

kidney_stone_rf_model.joblib

kidney_stone_scaler.joblib

Also, rag/faiss_index.faiss must exist (generated from document embeddings).

▶️ Running the App
python run.py
Visit http://localhost:5000 to access the app.

🌐 Main Routes
Route	Function
/	Upload CT image and view classification
/results	Shows localized image and report
/pdf_preview	PDF-like printable report
/chat (POST)	Chatbot Q&A with RAG
/risk-quiz	Health quiz for risk prediction
/api/localized-images	Lists recent localized image files

🤖 AI Models Used
Task	Model
Classification	ResNet18 (Custom-trained)
Localization	YOLOv8 (Kidney regions)
Chatbot (RAG)	MiniLM + FAISS vector search
Risk Prediction	Random Forest (trained on CSV dataset)

🧪 Sample Use Cases
Upload a CT image → get disease prediction + bounding box.

Ask: “What causes kidney stones?” in chatbot.

Try the risk quiz to get insights into your health.

📄 License
This project is open-sourced under the MIT License.

👨‍💻 Author
K Shashikanth Rao
📧 shashi19rao@gmail.com










