import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_pipeline import generate_answer


st.set_page_config(page_title="CrediTrust Complaint Insight Chatbot", layout="wide")

st.title("💬 CrediTrust Complaint Insight Chatbot")
st.markdown("Ask a question to explore real customer complaints across financial products.")

# Input box
user_question = st.text_input("Ask a question (e.g., Why are users unhappy with BNPL?)", "")

# Ask button
if st.button("🔍 Ask") and user_question.strip():
    with st.spinner("Generating answer..."):
        result = generate_answer(user_question)

        st.markdown("### 📌 Answer")
        st.success(result["answer"])

        st.markdown("### 📖 Sources (Top 2 Complaint Excerpts Used)")
        for i, source in enumerate(result["retrieved_sources"], 1):
            st.markdown(f"**Source {i}:**")
            st.info(source["text"])
else:
    st.markdown("_Enter a question above and click Ask to begin._")

# Clear button (just clears the input box via rerun)
if st.button("🧹 Clear"):
    st.experimental_rerun()
