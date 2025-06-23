from baseline.generator.generator import Generator
from baseline.retriever.retriever import Retriever
import json
from datetime import datetime
import os
import shutil

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
        retrieved_chunks = retriever.query(question, top_k=5)  # You can tune top_k here

        # === Step 2: Print retrieved context ===
        print(f"\n[{i}] Q: {question}")
        print("==== Retrieved Context Chunks ====")
        for j, chunk in enumerate(retrieved_chunks):
            print(f"[Chunk {j+1}]: {chunk['chunk']}\n")
        print("==================================")

        # === Select Top 3 Chunks (by score) ===
        top_chunks = sorted(retrieved_chunks, key=lambda x: x['score'], reverse=True)[:3]

        # === Prompt + Answer ===
        prompt = generator.build_prompt(top_chunks, question)
        answer = generator.generate_answer(prompt)

        # === Improved Grounding Check ===
        is_grounded = any(chunk['chunk'].lower() in answer.lower() for chunk in retrieved_chunks)

        # === Logging and Output ===
        log_query(question, retrieved_chunks, prompt, answer, GROUP_ID)

        print(f"➤ Answer: {answer}")
        print(f"✅ Grounded: {is_grounded}\n")


if __name__ == "__main__":
    index_folder = "retriever_index"

    # Delete the index folder automatically if it exists
    if os.path.exists(index_folder) and os.path.isdir(index_folder):
        shutil.rmtree(index_folder)
        print(f"Deleted existing index folder: {index_folder}")

    retriever = Retriever()

    # Now the index folder doesn't exist, so always build index fresh
    print("❗ Index not found. Creating it from documents...")

    # TODO: Replace with the actual file paths you want to index
    document_paths = ["retriever/software.json"]

    retriever.add_documents(document_paths)
    retriever.save(index_folder)
    print("✅ Index built and saved.")

    generator = Generator()
    run_tests(generator, retriever)
