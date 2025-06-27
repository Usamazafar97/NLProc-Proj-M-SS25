# RAG Project – Summer Semester 2025

## Overview

This project implements a basic Retrieval-Augmented Generation (RAG) system that can answer natural language questions based on provided documents. It combines a document retriever that finds relevant text chunks using semantic similarity, and a generator model (Flan-T5) that generates answers using those chunks as context. The system is modular, easy to extend, and designed to showcase the power of combining retrieval with language generation for question answering tasks.


---

# Specialization Module

**Usage Notice:**

Please use only the enhanced versions of the following components in this directory:

- `enhanced_generator.py` (Generator)
- `enhanced_pipeline.py` (Pipeline)
- `enhanced_retriever.py` (Retriever)

These files are located in their respective subdirectories under `specialization/baseline/`.

Other versions are present for reference or legacy purposes and should not be used for current work.
---

## 📦 Project Structure

```
.
├── baseline/
    ├── generator/
    │   └── generator.py        # T5 model to generate answer
    └── retriever/
        └── retriever.py        # Handles document parsing, chunking, embedding, FAISS retrieval
    ├── data/
        ├── logs.jsonl              # Stores question-answer logs
        └── test_inputs.json        # Test questions
    ├── retriever_index/            # Saved index and metadata
    ├── pipeline.py                 # Main entry point

```

---

## ⚙️ Setup

1. **Install Dependencies:**

```txt
transformers
sentence-transformers
faiss-cpu
PyMuPDF
scikit-learn
torch
```

2. **Add Documents:**

Put `.pdf`, `.txt`, or `.md` files in the root or specific folder.

```python
retriever.add_documents(["myfile.pdf", "notes.txt"])
retriever.save()
```

3. **Test Questions:**

Add your questions to `data/test_inputs.json`:

```json
[
  {"question": "What is the purpose of this system?"},
  {"question": "Boeing ramp up production to how many aircraft?"}
]
```

---

## 🚀 Run the Pipeline

```bash
python pipeline.py
```

Output:

```bash
[0] Q: What is the purpose of this system?
    ➤ Answer: The system is designed to answer questions using document context.
    ✅ Grounded: True
```

---

## 📝 Features

* ✅ Handles `.pdf`, `.txt`, and `.md` files
* 🔍 FAISS indexing for fast similarity search
* ✂️ Text chunking for efficient context management
* 🧠 T5-based generation
* 📁 Logs all interactions in `logs.jsonl`

---

## 🧪 Evaluation

Each answer is considered **grounded** if it contains a portion from any of the top-3 retrieved chunks.

---

## 📊 Sample Log Entry

```json
{
  "timestamp": "2025-05-21T15:30:00Z",
  "group_id": "Team_Turing",
  "question": "What is the main use of FAISS?",
  "retrieved_chunks": ["FAISS is used for efficient similarity search."],
  "prompt": "Context:\nFAISS is used for efficient similarity search.\n\nQuestion: What is the main use of FAISS?\nAnswer:",
  "generated_answer": "FAISS is mainly used for efficient similarity search."
}
```

---

## 📬 Future Improvements

* Switch to **LlamaIndex or LangChain** for advanced pipelines
* Add **feedback loop** to fine-tune answers
* Improve UI for non-technical users

---

Made with ❤️ by Team Turing
