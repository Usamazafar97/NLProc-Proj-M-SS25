import os
import fitz  # PyMuPDF
import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

class Retriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.embeddings = []
        self.documents = []
        self.chunk_map = []
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.priority_fields = ['overall', 'reviewTest', 'summary', 'unixReviewTime']

    def _read_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.pdf':
            doc = fitz.open(file_path)
            return " ".join([page.get_text() for page in doc])
        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            text_chunks = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        chunk = self._extract_priority_fields(obj)
                        if chunk:
                            text_chunks.append(chunk)
                except json.JSONDecodeError as e:
                    print(f"Skipping line due to JSON error: {e}")
                    continue

            return text_chunks
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _extract_priority_fields(self, d):
        values = []
        for key in self.priority_fields:
            if key in d and isinstance(d[key], str):
                values.append(d[key])
        if values:
            return " ".join(values)
        return self._dict_to_text(d)

    def _dict_to_text(self, d):
        parts = []
        for key, value in d.items():
            if isinstance(value, dict):
                parts.append(self._dict_to_text(value))
            elif isinstance(value, list):
                parts.append(" ".join(map(str, value)))
            else:
                parts.append(str(value))
        return " ".join(parts)

    def _chunk_text(self, text, max_tokens=100):
        words = text.split()
        return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

    def _preprocess_text(self, text):
        return text.strip().replace('\n', ' ')

    def add_documents(self, file_paths):
        for path in file_paths:
            chunks_or_text = self._read_file(path)

            if isinstance(chunks_or_text, str):
                chunks = self._chunk_text(chunks_or_text)
            elif isinstance(chunks_or_text, list):
                chunks = []
                for block in chunks_or_text:
                    if len(block.split()) > 100:
                        chunks.extend(self._chunk_text(block))
                    else:
                        chunks.append(block)
            else:
                continue

            chunks = [self._preprocess_text(c) for c in chunks]

            if chunks:
                print(f"[DEBUG] Sample chunk from {path}: {chunks[0]}\n")

            embeddings = self.model.encode(chunks)

            self.embeddings.extend(embeddings)
            self.documents.extend(chunks)
            self.chunk_map.extend([path] * len(chunks))

        self._create_faiss_index()

    def _create_faiss_index(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(self.embeddings))

    def query(self, question, top_k=3):
        query_embedding = self.model.encode([question])
        D, I = self.index.search(np.array(query_embedding), top_k)

        results = []
        max_distance = np.max(D) if np.max(D) != 0 else 1.0  # avoid div-by-zero
        for i in range(top_k):
            idx = I[0][i]
            distance = float(D[0][i])
            score = 1 - (distance / max_distance)  # convert distance to similarity-like score
            results.append({
                'chunk': self.documents[idx],
                'source': self.chunk_map[idx],
                'distance': distance,
                'score': score
            })
        return results

    def save(self, path='retriever_index'):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, 'faiss.index'))
        with open(os.path.join(path, 'data.pkl'), 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings,
                'chunk_map': self.chunk_map,
                'model_name': self.model_name
            }, f)

    def load(self, path='retriever_index'):
        with open(os.path.join(path, 'data.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = data['embeddings']
            self.chunk_map = data['chunk_map']
            self.model_name = data.get('model_name', 'all-MiniLM-L6-v2')

        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.read_index(os.path.join(path, 'faiss.index'))
