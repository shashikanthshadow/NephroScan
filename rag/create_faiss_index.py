import os
from sentence_transformers import SentenceTransformer
import faiss

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Get base directory
base_dir = os.path.dirname(__file__)
# Use index_documents.txt
context_path = os.path.join(base_dir, "index_documents.txt")

# Load knowledge base
with open(context_path, "r", encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

# Embed text
embeddings = model.encode(lines)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index
faiss.write_index(index, os.path.join(base_dir, "faiss_index.faiss"))

# Save matching docs
with open(os.path.join(base_dir, "index_documents.txt"), "w", encoding='utf-8') as f:
    for line in lines:
        f.write(line + "\n")

print("✅ FAISS index created and saved to rag/")