from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from textblob import TextBlob
from fuzzywuzzy import fuzz
import re

# Load the context (text file) content
def load_context():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONTEXT_PATH = os.path.join(BASE_DIR, "../../rag/index_documents.txt")
    with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# Initialize the embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load the context documents
context_docs = load_context()

# Initialize FAISS index
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.join(BASE_DIR, "../../rag/")
INDEX_PATH = os.path.join(RAG_DIR, "faiss_index.faiss")
index = faiss.read_index(INDEX_PATH)

# Function for spelling correction using TextBlob
def correct_spelling(user_input):
    blob = TextBlob(user_input)
    corrected = blob.correct()
    return str(corrected)

# Function for fuzzy matching
def fuzzy_match(query, options, threshold=80):
    best_match = None
    highest_score = 0
    for option in options:
        score = fuzz.ratio(query.lower(), option.lower())
        if score > highest_score and score >= threshold:
            highest_score = score
            best_match = option
    return best_match

# Retrieve context based on user query
def retrieve_context(query, top_k=3):
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    results = [context_docs[i] for i in I[0]]
    return results

# Detect if input is location-related
def get_user_location(user_input):
    location_keywords = ["city", "town", "place", "area", "near me", "location"]
    matched_location = fuzzy_match(user_input, location_keywords)
    return bool(matched_location)

# Extract location from phrases like 'near manipal', 'in bangalore'
def extract_possible_location(user_input):
    match = re.search(r"(near|in|around|at)\s+([a-zA-Z\s]+)", user_input.lower())
    if match:
        return match.group(2).strip().title()
    return user_input.title()

# Main chatbot logic (map functionality removed)
def chatbot_response(user_input):
    corrected_input = correct_spelling(user_input)
    response = {"text": ""}

    # Step 1: Ask for location if general location intent is detected
    if get_user_location(corrected_input):
        response["text"] = "🧠 I see you're looking for nephrologists near you. Please specify your location (e.g., 'Bangalore', 'New York')."
        return response

    # Step 2: Detect disease-related keywords
    medical_keywords = ["cyst", "kidney", "disease", "tumor", "stone", "cancer"]
    matched_term = fuzzy_match(corrected_input, medical_keywords)
    if matched_term:
        disease_info = {
            "cyst": "- What is a Kidney Cyst?\n- A kidney cyst is a fluid-filled sac that forms within the kidney. Most are benign but may cause symptoms if infected or large.\n- How is a Kidney Cyst Treated?\n- Most kidney cysts require no treatment unless symptomatic. Large cysts may require aspiration or surgery.",
            "stone": "- What is a Kidney Stone?\n- A kidney stone is a solid mineral deposit formed in the kidneys. They may cause severe pain when moving through the urinary tract.\n- How are Kidney Stones Treated?\n- Treatment includes hydration, pain control, and possibly procedures like lithotripsy or surgery.",
            "tumor": "- What is a Renal Tumor?\n- A kidney tumor may be benign or malignant. RCC is the most common cancer.\n- How are Renal Tumors Treated?\n- Treatment includes surgery, ablation, or immunotherapy based on staging.",
            "cancer": "- What is Kidney Cancer?\n- It includes various malignancies in the kidney.\n- Treatment usually involves surgery and systemic therapy.",
            "kidney": "- What is the Function of the Kidney?\n- Kidneys filter blood and manage electrolytes.\n- Common conditions: infections, stones, cysts, tumors."
        }
        response["text"] = f"🧠 Based on nephrology knowledge:\n\n{disease_info[matched_term]}\n\nNeed more help? Ask me another question!"
        return response

    # Step 3: Fallback to RAG context retrieval
    context = retrieve_context(corrected_input)
    answer = "\n".join([f"- {c}" for c in context])
    response["text"] = f"🧠 Based on nephrology knowledge:\n{answer}\n\nNeed more help? Ask me another question!"
    return response
