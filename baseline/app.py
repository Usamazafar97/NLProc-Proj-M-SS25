# app.py

import streamlit as st
import json
from datetime import datetime
from generator.generator import Generator
from retriever.retriever import Retriever

LOG_FILE = "data/logs.jsonl"
GROUP_ID = "Team_Turing"

# === Logging Function ===
def log_query(question, retrieved_chunks, prompt, generated_answer, group_id, log_file=LOG_FILE):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "group_id": group_id,
        "question": question,
        "retrieved_chunks": [chunk['chunk'] for chunk in retrieved_chunks],
        "prompt": prompt,
        "generated_answer": generated_answer
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# === Streamlit App ===
st.set_page_config(page_title="QA with Retriever + Generator", layout="wide")
st.title("🧠 Question Answering System")

# Load models
@st.cache_resource
def load_models():
    retriever = Retriever()
    retriever.load("retriever_index")
    generator = Generator()
    return retriever, generator

retriever, generator = load_models()

# Input from user
question = st.text_input("🔎 Enter your question:")

if question:
    with st.spinner("Processing..."):
        # Retrieve top 5
        retrieved_chunks = retriever.query(question, top_k=5)
        top_chunks = sorted(retrieved_chunks, key=lambda x: x['score'], reverse=True)[:3]

        # Generate answer
        prompt = generator.build_prompt(top_chunks, question)
        answer = generator.generate_answer(prompt)

        # Grounding check
        is_grounded = any(chunk['chunk'].lower() in answer.lower() for chunk in retrieved_chunks)

        # Log result
        log_query(question, retrieved_chunks, prompt, answer, GROUP_ID)

    # Display results
    st.subheader("📖 Retrieved Chunks")
    for i, chunk in enumerate(top_chunks):
        st.markdown(f"**Chunk {i+1} (Score: {chunk['score']:.2f}):**\n> {chunk['chunk']}")

    st.subheader("📜 Prompt Used (optional)")
    with st.expander("Click to view prompt"):
        st.text(prompt)

    st.subheader("🧠 Generated Answer")
    st.success(answer)

    st.markdown(f"**Grounded Answer:** {'✅ Yes' if is_grounded else '❌ No'}")
