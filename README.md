# 📚 Context-Aware Question Answering System

This project is a **retrieval-augmented generation (RAG)** pipeline that allows you to ask natural language questions based on custom documents (e.g., `.txt`, `.md`, `.pdf`). It uses:

* **Sentence Transformers + FAISS** for efficient document retrieval.
* **T5 (Flan-T5-Base)** for generating natural language answers.

---

## 📦 Project Structure

```
.
├── baseline/
│   ├── generator/
│   │   └── generator.py        # T5 model to generate answer
│   └── retriever/
│       └── retriever.py        # Handles document parsing, chunking, embedding, FAISS retrieval
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
