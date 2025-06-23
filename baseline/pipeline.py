from generator.generator import Generator
from retriever.retriever import Retriever
import json
from datetime import datetime
import os
import shutil
import streamlit as st
import sys

LOG_FILE = "data/logs.jsonl"
TEST_FILE = "data/test_inputs.json"
GROUP_ID = "Team_Turing"

def log_query(question, retrieved_chunks, prompt, generated_answer, group_id, log_file=LOG_FILE):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

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

def run_tests(generator, retriever, test_file=TEST_FILE):
    with open(test_file, "r") as f:
        test_data = json.load(f)

    for i, test_case in enumerate(test_data):
        question = test_case["question"]
        retrieved_chunks = retriever.query(question, top_k=5)

        print(f"\n[{i}] Q: {question}")
        print("==== Retrieved Context Chunks ====")
        for j, chunk in enumerate(retrieved_chunks):
            print(f"[Chunk {j+1}]: {chunk['chunk']}\n")
        print("==================================")

        top_chunks = sorted(retrieved_chunks, key=lambda x: x['score'], reverse=True)[:3]

        prompt = generator.build_prompt(top_chunks, question)
        answer = generator.generate_answer(prompt)

        is_grounded = any(chunk['chunk'].lower() in answer.lower() for chunk in retrieved_chunks)

        log_query(question, retrieved_chunks, prompt, answer, GROUP_ID)

        print(f"➤ Answer: {answer}")
        print(f"✅ Grounded: {is_grounded}\n")

def run_streamlit_app(generator, retriever):
    st.set_page_config(page_title="QA System", layout="wide")
    st.title("🧠 Amazon Software Products System")

    question = st.text_input("🔎 Enter your question:")
    if question:
        with st.spinner("Processing..."):
            retrieved_chunks = retriever.query(question, top_k=5)
            top_chunks = sorted(retrieved_chunks, key=lambda x: x['score'], reverse=True)[:3]
            prompt = generator.build_prompt(top_chunks, question)
            answer = generator.generate_answer(prompt)
            is_grounded = any(chunk['chunk'].lower() in answer.lower() for chunk in retrieved_chunks)
            log_query(question, retrieved_chunks, prompt, answer, GROUP_ID)

        st.subheader("📖 Retrieved Chunks")
        for i, chunk in enumerate(top_chunks):
            st.markdown(f"**Chunk {i+1} (Score: {chunk['score']:.2f}):**\n> {chunk['chunk']}")

        st.subheader("📜 Prompt Used (optional)")
        with st.expander("Click to view prompt"):
            st.text(prompt)

        st.subheader("🧠 Generated Answer")
        st.success(answer)
        st.markdown(f"**Grounded Answer:** {'✅ Yes' if is_grounded else '❌ No'}")

if __name__ == "__main__":
    index_folder = "retriever_index"

    if os.path.exists(index_folder) and os.path.isdir(index_folder):
        shutil.rmtree(index_folder)
        print(f"Deleted existing index folder: {index_folder}")

    retriever = Retriever()
    print("❗ Index not found. Creating it from documents...")
    retriever_dir = os.path.dirname(os.path.abspath(__file__))  # Path of the current script
    software_file_path = os.path.join(retriever_dir, "retriever", "software.json")
    document_paths = [software_file_path]
    retriever.add_documents(document_paths)
    retriever.save(index_folder)
    print("✅ Index built and saved.")

    generator = Generator()

    # Decide between CLI or Streamlit based on how it's called
    # if any("streamlit" in arg for arg in sys.argv):
    run_streamlit_app(generator, retriever)
    # else:
    #     run_tests(generator, retriever)
