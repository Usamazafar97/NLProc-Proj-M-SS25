import os
import fitz  # PyMuPDF
import json
import numpy as np
import faiss
import pickle
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple

class EnhancedRetriever:
    def __init__(self, model_name='all-mpnet-base-v2'):
        """
        Enhanced retriever with domain-specific optimizations for customer reviews
        Using all-mpnet-base-v2 which performs better on semantic similarity tasks
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.embeddings = []
        self.documents = []
        self.chunk_map = []
        self.metadata = []  # Store metadata for each chunk
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Enhanced priority fields for customer reviews
        self.priority_fields = ['reviewText', 'summary', 'overall', 'reviewerName']
        
        # Metadata fields to preserve
        self.metadata_fields = ['overall', 'reviewTime', 'reviewerID', 'asin', 
                               'reviewerName', 'verified', 'unixReviewTime']

    def _preprocess_text(self, text: str) -> str:
        """
        Enhanced preprocessing for noisy customer review text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common review artifacts
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)
        
        # Normalize common abbreviations
        text = re.sub(r'\bthx\b', 'thanks', text, flags=re.IGNORECASE)
        text = re.sub(r'\bu\b', 'you', text, flags=re.IGNORECASE)
        text = re.sub(r'\bur\b', 'your', text, flags=re.IGNORECASE)
        
        return text

    def _sentence_level_chunking(self, text: str, max_sentences: int = 3) -> List[str]:
        """
        Enhanced sentence-level chunking instead of word-based
        """
        if not text:
            return []
        
        # Split into sentences (simple approach)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk = '. '.join(sentences[i:i + max_sentences])
            if chunk:
                chunks.append(chunk)
        
        return chunks

    def _extract_review_content(self, review_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Extract and structure review content with metadata
        """
        # Extract main review text
        review_text = review_data.get('reviewText', '')
        summary = review_data.get('summary', '')
        
        # Combine review text and summary
        content = f"{summary}. {review_text}" if summary else review_text
        content = self._preprocess_text(content)
        
        # Extract metadata
        metadata = {}
        for field in self.metadata_fields:
            if field in review_data:
                metadata[field] = review_data[field]
        
        return content, metadata

    def _read_json_reviews(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Enhanced JSON reading with better error handling and preprocessing
        """
        reviews = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    review_data = json.loads(line)
                    if isinstance(review_data, dict):
                        content, metadata = self._extract_review_content(review_data)
                        if content:  # Only include non-empty reviews
                            reviews.append((content, metadata))
                except json.JSONDecodeError as e:
                    print(f"JSON error on line {line_num}: {e}")
                    continue
        
        return reviews

    def _read_file(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Enhanced file reading with support for multiple formats
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.json':
            return self._read_json_reviews(file_path)
        elif ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return [(content, {'source': file_path})]
        elif ext == '.pdf':
            doc = fitz.open(file_path)
            content = " ".join([page.get_text() for page in doc])
            return [(content, {'source': file_path})]
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def add_documents(self, file_paths: List[str]):
        """
        Add documents with enhanced processing
        """
        for path in file_paths:
            print(f"Processing {path}...")
            reviews = self._read_file(path)
            
            for content, metadata in reviews:
                # Apply sentence-level chunking
                chunks = self._sentence_level_chunking(content)
                
                for chunk in chunks:
                    if len(chunk.split()) >= 5:  # Minimum chunk size
                        self.documents.append(chunk)
                        self.chunk_map.append(path)
                        self.metadata.append(metadata)
            
            print(f"Added {len(reviews)} reviews from {path}")
        
        # Create embeddings
        print("Creating embeddings...")
        self.embeddings = self.model.encode(self.documents)
        self._create_faiss_index()
        print(f"Index created with {len(self.documents)} chunks")

    def _create_faiss_index(self):
        """Create FAISS index with enhanced configuration"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(self.embeddings))

    def query(self, question: str, top_k: int = 5, 
              min_rating: Optional[float] = None,
              max_rating: Optional[float] = None,
              date_from: Optional[str] = None,
              date_to: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Enhanced query with metadata filtering
        """
        # Check if we have data
        if not self.documents or not self.index:
            print("Warning: No documents indexed. Please add documents first.")
            return []
        
        # Encode query
        query_embedding = self.model.encode([question])
        
        # Get initial results with bounds checking
        max_search = min(top_k * 3, len(self.documents))
        if max_search == 0:
            return []
            
        D, I = self.index.search(np.array(query_embedding), max_search)
        
        results = []
        for i in range(len(I[0])):
            idx = I[0][i]
            
            # Bounds checking
            if idx >= len(self.documents) or idx >= len(self.metadata) or idx >= len(self.chunk_map):
                print(f"Warning: Index {idx} out of bounds. Skipping.")
                continue
                
            metadata = self.metadata[idx]
            
            # Apply filters
            if self._passes_filters(metadata, min_rating, max_rating, date_from, date_to):
                results.append({
                    'chunk': self.documents[idx],
                    'source': self.chunk_map[idx],
                    'distance': float(D[0][i]),
                    'metadata': metadata
                })
                
                if len(results) >= top_k:
                    break
        
        return results

    def _passes_filters(self, metadata: Dict[str, Any], 
                       min_rating: Optional[float],
                       max_rating: Optional[float],
                       date_from: Optional[str],
                       date_to: Optional[str]) -> bool:
        """
        Check if metadata passes the specified filters
        """
        # Rating filter
        if 'overall' in metadata:
            rating = float(metadata['overall'])
            if min_rating is not None and rating < min_rating:
                return False
            if max_rating is not None and rating > max_rating:
                return False
        
        # Date filter
        if 'unixReviewTime' in metadata and (date_from or date_to):
            try:
                review_time = int(metadata['unixReviewTime'])
                if date_from:
                    from_time = int(datetime.strptime(date_from, '%Y-%m-%d').timestamp())
                    if review_time < from_time:
                        return False
                if date_to:
                    to_time = int(datetime.strptime(date_to, '%Y-%m-%d').timestamp())
                    if review_time > to_time:
                        return False
            except (ValueError, TypeError):
                pass
        
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed data
        """
        if not self.metadata:
            return {}
        
        ratings = [float(m.get('overall', 0)) for m in self.metadata if 'overall' in m]
        
        stats = {
            'total_chunks': len(self.documents),
            'total_reviews': len(set(m.get('reviewerID', '') for m in self.metadata)),
            'avg_rating': np.mean(ratings) if ratings else 0,
            'rating_distribution': {}
        }
        
        # Rating distribution
        for rating in range(1, 6):
            count = sum(1 for r in ratings if r == rating)
            stats['rating_distribution'][f'{rating}_star'] = count
        
        return stats

    def save(self, path: str = 'retriever_index'):
        """Save enhanced index with metadata"""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, 'faiss.index'))
        with open(os.path.join(path, 'data.pkl'), 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings,
                'chunk_map': self.chunk_map,
                'metadata': self.metadata,
                'model_name': self.model_name
            }, f)

    def load(self, path: str = 'retriever_index'):
        """Load enhanced index with metadata"""
        with open(os.path.join(path, 'data.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = data['embeddings']
            self.chunk_map = data['chunk_map']
            self.metadata = data.get('metadata', [])
            self.model_name = data.get('model_name', 'all-mpnet-base-v2')

        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.read_index(os.path.join(path, 'faiss.index')) 