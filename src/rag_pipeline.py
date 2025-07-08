import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import huggingface_hub

# ---------------- PATH SETUP ----------------
# Get project root based on this file's location
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vector_store_path = os.path.join(project_root, "vector_store")

# Full paths to FAISS index and metadata
index_path = os.path.join(vector_store_path, "faiss_index.index")
metadata_path = os.path.join(vector_store_path, "metadata.pkl")

# ---------------- LOAD COMPONENTS ----------------
# Load FAISS index
index = faiss.read_index(index_path)

# Load metadata
with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Optional: extend HuggingFace timeout (useful for slow internet)
huggingface_hub.constants.HF_HUB_READ_TIMEOUT = 60

# Load language model
generator = pipeline("text2text-generation", model="google/flan-t5-large", device=-1)

# ---------------- RETRIEVER FUNCTION ----------------
def retrieve_relevant_chunks(question, k=5):
    question_embedding = embedding_model.encode([question])
    _, indices = index.search(np.array(question_embedding), k)
    
    top_chunks = [metadata[i] for i in indices[0]]
    return top_chunks

# ---------------- PROMPT TEMPLATE ----------------
def build_prompt(context_chunks, question):
    context_text = "\n\n".join([chunk["text"] for chunk in context_chunks[:2]])  # limit context
    prompt = f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context_text}

Question:
{question}

Answer:
"""
    return prompt.strip()

# ---------------- GENERATOR FUNCTION ----------------
def generate_answer(question, k=5):
    chunks = retrieve_relevant_chunks(question, k=k)
    prompt = build_prompt(chunks, question)
    result = generator(prompt, max_new_tokens=256)[0]["generated_text"]
    
    return {
        "question": question,
        "answer": result.strip(),
        "retrieved_sources": chunks[:2]
    }
