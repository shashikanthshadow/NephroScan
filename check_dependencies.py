import importlib

required_packages = [
    "flask", "torch", "torchvision", "PIL", "faiss", "sentence_transformers",
    "transformers", "cv2", "ultralytics", "requests"
]

for package in required_packages:
    try:
        importlib.import_module(package)
        print(f"[✅] {package} is installed.")
    except ImportError:
        print(f"[❌] {package} is NOT installed.")
