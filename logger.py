import json
from datetime import datetime


def log_query(question, retrieved_chunks, prompt, generated_answer, group_id, log_file="logs.jsonl"):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "group_id": group_id,
        "question": question,
        "retrieved_chunks": [retrieved_chunks],
        "prompt": prompt,
        "generated_answer": generated_answer
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")