import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import huggingface_hub

# Load vector store & metadata
index = faiss.read_index("../vector_store/faiss_index.index")
with open("../vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
huggingface_hub.constants.HF_HUB_READ_TIMEOUT = 60
# Load language model
generator = pipeline("text2text-generation", model="google/flan-t5-large", device=-1)

# --------- RETRIEVER FUNCTION ---------
def retrieve_relevant_chunks(question, k=5):
    question_embedding = embedding_model.encode([question])
    _, indices = index.search(np.array(question_embedding), k)
    
    top_chunks = [metadata[i] for i in indices[0]]
    return top_chunks

# --------- PROMPT TEMPLATE ---------
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

# --------- GENERATOR FUNCTION ---------
def generate_answer(question, k=5):
    chunks = retrieve_relevant_chunks(question, k=k)
    prompt = build_prompt(chunks, question)
    result = generator(prompt, max_new_tokens=256)[0]["generated_text"]
    
    return {
        "question": question,
        "answer": result.strip(),
        "retrieved_sources": chunks[:2]
    }

