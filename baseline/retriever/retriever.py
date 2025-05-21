import os
import fitz  # from PyMuPDF
import numpy as np
import faiss  # Facebook AI Similarity Search: used for efficient vector search
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    """
    A simple document retriever using SentenceTransformer for embeddings and FAISS for indexing.
    Supports loading documents from .txt, .md, and .pdf files, chunking them, embedding, and searching.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the retriever by loading the embedding model and preparing placeholders.
        """
        self.model = SentenceTransformer(model_name)  # Pretrained sentence embedding model
        self.index = None  # FAISS index for similarity search
        self.embeddings = []  # List to store all document chunk embeddings
        self.documents = []  # List to store text chunks from documents
        self.chunk_map = []  # Keeps track of the file each chunk came from
        self.dimension = self.model.get_sentence_embedding_dimension()  # Embedding vector size

    def _read_file(self, file_path):
        """
        Reads the content of a file. Supports .txt, .md, and .pdf files.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.pdf':
            doc = fitz.open(file_path)
            return " ".join([page.get_text() for page in doc])  # Combine all page texts
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _chunk_text(self, text, max_tokens=100):
        """
        Splits long text into smaller chunks of approximately ⁠ max_tokens ⁠ words.
        """
        words = text.split()
        return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

    def add_documents(self, file_paths):
        """
        Loads documents, chunks them, generates embeddings, and updates the FAISS index.
        """
        for path in file_paths:
            text = self._read_file(path)  # Read full document text
            chunks = self._chunk_text(text)  # Split into smaller text blocks
            embeddings = self.model.encode(chunks)  # Generate embeddings for each chunk

            self.embeddings.extend(embeddings)
            self.documents.extend(chunks)
            self.chunk_map.extend([path] * len(chunks))  # Map each chunk back to its source file

        self._create_faiss_index()  # Build or rebuild the FAISS index

    def _create_faiss_index(self):
        """
        Initializes and populates a FAISS index from all stored embeddings.
        """
        self.index = faiss.IndexFlatL2(self.dimension)  # L2-distance based index
        self.index.add(np.array(self.embeddings))  # Add all embeddings to the index

    def query(self, question, top_k=3):
        """
        Given a question or query string, returns the top_k most relevant text chunks.
        """
        query_embedding = self.model.encode([question])  # Embed the query
        D, I = self.index.search(np.array(query_embedding), top_k)  # Search FAISS index

        results = []
        for i in range(top_k):
            idx = I[0][i]  # Index of the matched chunk
            results.append({
                'chunk': self.documents[idx],       # Retrieved text chunk
                'source': self.chunk_map[idx],      # Source document file
                'distance': float(D[0][i])          # L2 distance from query embedding
            })
        return results

    def save(self, path='retriever_index'):
        """
        Saves the FAISS index and related metadata (documents, embeddings, mappings) to disk.
        """
        os.makedirs(path, exist_ok=True)

        # Save FAISS index to a binary file
        faiss.write_index(self.index, os.path.join(path, 'faiss.index'))

        # Save supporting metadata using pickle
        with open(os.path.join(path, 'data.pkl'), 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings,
                'chunk_map': self.chunk_map
            }, f)

    def load(self, path='retriever_index'):
        """
        Loads a previously saved FAISS index and metadata from disk.
        """
        self.index = faiss.read_index(os.path.join(path, 'faiss.index'))  # Load FAISS index
        with open(os.path.join(path, 'data.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = data['embeddings']
            self.chunk_map = data['chunk_map']


# Optional: Run a demo if the script is executed directly
if __name__ == "__main__":
    retriever = Retriever()

    # Step 1: Add your documents here (edit the filenames as needed)
    files = ["file.txt"]  # Place these files in the same directory
    retriever.add_documents(files)

    # Step 2: Save the index and data for future use
    retriever.save()

    # Step 3: Load the saved retriever (can comment out if not needed)
    retriever.load()

    # Step 4: Run a sample query
    query = "Boeing ramp up production to how many aircraft?"
    results = retriever.query(query)

    # Step 5: Display the results
    for res in results:
        print(f"\nMatch from {res['source']} (distance: {res['distance']:.4f}):\n{res['chunk']}")