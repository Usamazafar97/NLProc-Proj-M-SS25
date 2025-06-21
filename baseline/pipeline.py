from baseline.generator.generator import Generator
from baseline.retriever.retriever import Retriever
import json
from datetime import datetime
import os

LOG_FILE = "data/logs.jsonl"
TEST_FILE = "data/test_inputs.json"
GROUP_ID = "Team_Turing"

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

def run_tests(generator, retriever, test_file=TEST_FILE):
    with open(test_file, "r") as f:
        test_data = json.load(f)

    for i, test_case in enumerate(test_data):
        question = test_case["question"]
        retrieved_chunks = retriever.query(question, top_k=3)
        prompt = generator.build_prompt(retrieved_chunks, question)
        answer = generator.generate_answer(prompt)

        log_query(question, retrieved_chunks, prompt, answer, GROUP_ID)

        print(f"[{i}] Q: {question}")
        print(f"    ➤ Answer: {answer}")
        print(f"    ✅ Grounded: {any(chunk['chunk'] in answer for chunk in retrieved_chunks)}\n")

if __name__ == "__main__":
    retriever = Retriever()

    # Check if the index already exists
    if not os.path.exists("retriever_index/data.pkl"):
        print("❗ Index not found. Creating it from documents...")

        # TODO: Replace with the actual file paths you want to index
        document_paths = ["retriever/software.json"]
        retriever.add_documents(document_paths)
        retriever.save("retriever_index")
        print("✅ Index built and saved.")
    else:
        print("✅ Index found. Loading...")
        retriever.load("retriever_index")

    generator = Generator()
    run_tests(generator, retriever)
