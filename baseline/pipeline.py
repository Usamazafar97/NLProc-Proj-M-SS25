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

# ✅ New: Metric Computation Function
def compute_metrics(retrieved_chunks, ground_truth_chunks):
    retrieved_texts = set(chunk['chunk'].strip().lower() for chunk in retrieved_chunks)
    ground_truth_set = set(gt.strip().lower() for gt in ground_truth_chunks)

    true_positives = len(retrieved_texts & ground_truth_set)
    false_positives = len(retrieved_texts - ground_truth_set)
    false_negatives = len(ground_truth_set - retrieved_texts)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1

# ✅ Updated: now accepts optional metrics
def log_query(question, retrieved_chunks, prompt, generated_answer, group_id, log_file=LOG_FILE, metrics=None):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "group_id": group_id,
        "question": question,
        "retrieved_chunks": [chunk['chunk'] for chunk in retrieved_chunks],
        "prompt": prompt,
        "generated_answer": generated_answer,
    }
    if metrics:
        log_entry["metrics"] = {
            "precision": metrics[0],
            "recall": metrics[1],
            "f1_score": metrics[2],
        }

    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# ✅ Updated: show metrics in CLI
def run_tests(generator, retriever, test_file=TEST_FILE):
    with open(test_file, "r") as f:
        test_data = json.load(f)

    for i, test_case in enumerate(test_data):
        question = test_case["question"]
        ground_truth_chunks = test_case.get("ground_truth_chunks", [])
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

        precision, recall, f1 = compute_metrics(retrieved_chunks, ground_truth_chunks)
        log_query(question, retrieved_chunks, prompt, answer, GROUP_ID, metrics=(precision, recall, f1))

        print(f"➤ Answer: {answer}")
        print(f"✅ Grounded: {is_grounded}")
        print(f"📊 Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}\n")

# ✅ Updated: metrics displayed in Streamlit
def run_streamlit_app(generator, retriever):
    st.set_page_config(page_title="QA System", layout="wide")
    st.title("🧠 Amazon's Software Review System")

    question = st.text_input("🔎 Enter your question:")
    ground_truth_input = st.text_area("✍️ (Optional) Paste expected ground truth chunks here (one per line):")

    if question:
        with st.spinner("Processing..."):
            retrieved_chunks = retriever.query(question, top_k=5)
            top_chunks = sorted(retrieved_chunks, key=lambda x: x['score'], reverse=True)[:3]
            prompt = generator.build_prompt(top_chunks, question)
            answer = generator.generate_answer(prompt)
            is_grounded = any(chunk['chunk'].lower() in answer.lower() for chunk in retrieved_chunks)

            # ✅ Metric calculation if user provides ground truth
            if ground_truth_input.strip():
                ground_truth_chunks = [line.strip() for line in ground_truth_input.split("\n") if line.strip()]
                precision, recall, f1 = compute_metrics(retrieved_chunks, ground_truth_chunks)
                metrics = (precision, recall, f1)
            else:
                metrics = None

            log_query(question, retrieved_chunks, prompt, answer, GROUP_ID, metrics=metrics)

        st.subheader("📖 Retrieved Chunks")
        for i, chunk in enumerate(top_chunks):
            st.markdown(f"**Chunk {i+1} (Score: {chunk['score']:.2f}):**\n> {chunk['chunk']}")

        st.subheader("📜 Prompt Used")
        with st.expander("Click to view prompt"):
            st.text(prompt)

        st.subheader("🧠 Generated Answer")
        st.success(answer)
        st.markdown(f"**Grounded Answer:** {'✅ Yes' if is_grounded else '❌ No'}")

        if metrics:
            precision, recall, f1 = metrics
            st.subheader("📊 Evaluation Metrics")
            st.metric("Precision", f"{precision:.2f}")
            st.metric("Recall", f"{recall:.2f}")
            st.metric("F1 Score", f"{f1:.2f}")

if __name__ == "__main__":
    index_folder = "retriever_index"

    if os.path.exists(index_folder) and os.path.isdir(index_folder):
        shutil.rmtree(index_folder)
        print(f"Deleted existing index folder: {index_folder}")

    retriever = Retriever()
    print("❗ Index not found. Creating it from documents...")
    retriever_dir = os.path.dirname(os.path.abspath(__file__))
    software_file_path = os.path.join(retriever_dir, "retriever", "software.json")
    document_paths = [software_file_path]
    retriever.add_documents(document_paths)
    retriever.save(index_folder)
    print("✅ Index built and saved.")

    generator = Generator()

    # Decide between CLI or Streamlit based on how it's called
    run_streamlit_app(generator, retriever)
    # run_tests(generator, retriever)
