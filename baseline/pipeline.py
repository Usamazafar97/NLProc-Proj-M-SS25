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

# ✅ Metric Computation Function
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

# ✅ Log Query with Metrics
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

# ✅ CLI Test Mode
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

# ✅ Streamlit Interface
def run_streamlit_app(generator, retriever):
    st.set_page_config(page_title="QA System", layout="wide")
    st.title("🧠 Amazon's Software Review System")

    uploaded_file = st.file_uploader("📂 Upload a JSON file", type="json")

    if uploaded_file:
        st.success("✅ File uploaded successfully!")

        # Read the uploaded JSON file
        uploaded_json = [json.loads(line) for line in uploaded_file]

        # Save to a temporary path if needed by your retriever
        temp_json_path = "data/temp_uploaded.json"
        with open(temp_json_path, "w") as f:
            json.dump(uploaded_json, f)

        # Rebuild retriever index from uploaded JSON
        retriever.clear_index()
        retriever.add_documents([temp_json_path])
        st.info("🔄 Index rebuilt from uploaded file.")

        # Ask question
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
    else:
        st.warning("📄 Please upload a JSON file to begin.")


# ✅ Main entry
if __name__ == "__main__":
    index_folder = "retriever_index"

    # Rebuild index for fresh start
    if os.path.exists(index_folder) and os.path.isdir(index_folder):
        shutil.rmtree(index_folder)
        print(f"🗑️ Deleted existing index folder: {index_folder}")

    retriever = Retriever()
    print("📦 Index not found. Creating it from documents...")
    retriever_dir = os.path.dirname(os.path.abspath(__file__))
    software_file_path = os.path.join(retriever_dir, "retriever", "software.json")
    document_paths = [software_file_path]
    retriever.add_documents(document_paths)
    retriever.save(index_folder)
    print("✅ Index built and saved.")

    generator = Generator()

    # Choose between CLI or Streamlit based on how the script is called
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_tests(generator, retriever)
    else:
        run_streamlit_app(generator, retriever)
