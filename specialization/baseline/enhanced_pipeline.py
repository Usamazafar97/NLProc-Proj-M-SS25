from retriever.enhanced_retriever import EnhancedRetriever
from generator.enhanced_generator import EnhancedGenerator
from sentiment.sentiment_filter import SentimentFilter
import json
from datetime import datetime
import os
from typing import List, Dict, Any, Optional

class EnhancedRAGPipeline:
    def __init__(self, 
                 retriever_model: str = 'all-mpnet-base-v2',
                 generator_model: str = 'google/flan-t5-large',
                 sentiment_model: str = 'cardiffnlp/twitter-roberta-base-sentiment-latest'):
        """
        Enhanced RAG pipeline with all improvements
        """
        self.retriever = EnhancedRetriever(model_name=retriever_model)
        self.generator = EnhancedGenerator(model_name=generator_model)
        self.sentiment_filter = SentimentFilter(model_name=sentiment_model)
        
        self.log_file = "data/enhanced_logs.jsonl"
        self.group_id = "Team_Turing_Enhanced"

    def setup_index(self, document_paths: List[str], force_rebuild: bool = False):
        """
        Setup the enhanced index
        """
        index_path = "retriever_index"
        
        if not force_rebuild and os.path.exists(os.path.join(index_path, 'data.pkl')):
            print("✅ Loading existing enhanced index...")
            self.retriever.load(index_path)
        else:
            print("🔨 Building enhanced index from documents...")
            self.retriever.add_documents(document_paths)
            self.retriever.save(index_path)
            print("✅ Enhanced index built and saved.")
        
        # Print statistics
        stats = self.retriever.get_statistics()
        print(f"📊 Index Statistics: {stats}")

    def query(self, 
              question: str, 
              top_k: int = 5,
              sentiment_filter: Optional[str] = None,
              min_rating: Optional[float] = None,
              max_rating: Optional[float] = None,
              date_from: Optional[str] = None,
              date_to: Optional[str] = None,
              min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        Enhanced query with all filtering options
        """
        print(f"🔍 Querying: {question}")
        
        # Step 1: Retrieve relevant chunks with metadata filtering
        retrieved_chunks = self.retriever.query(
            question=question,
            top_k=top_k * 2,  # Get more chunks for filtering
            min_rating=min_rating,
            max_rating=max_rating,
            date_from=date_from,
            date_to=date_to
        )
        
        print(f"📄 Retrieved {len(retrieved_chunks)} chunks")
        
        # Step 2: Apply sentiment filtering if requested
        if sentiment_filter:
            print(f"😊 Applying sentiment filter: {sentiment_filter}")
            retrieved_chunks = self.sentiment_filter.filter_by_sentiment(
                retrieved_chunks, 
                sentiment_filter=sentiment_filter,
                min_confidence=min_confidence
            )
            print(f"📄 After sentiment filtering: {len(retrieved_chunks)} chunks")
        
        # Step 3: Limit to top_k chunks
        retrieved_chunks = retrieved_chunks[:top_k]
        
        # Step 4: Generate answer with enhanced prompt engineering
        result = self.generator.generate_structured_answer(retrieved_chunks, question)
        
        # Step 5: Add metadata
        result.update({
            'question': question,
            'retrieved_chunks': retrieved_chunks,
            'query_parameters': {
                'top_k': top_k,
                'sentiment_filter': sentiment_filter,
                'min_rating': min_rating,
                'max_rating': max_rating,
                'date_from': date_from,
                'date_to': date_to,
                'min_confidence': min_confidence
            },
            'timestamp': datetime.now().isoformat(),
            'group_id': self.group_id
        })
        
        # Step 6: Log the interaction
        self._log_interaction(result)
        
        return result

    def _log_interaction(self, result: Dict[str, Any]):
        """
        Log the interaction with enhanced metadata
        """
        log_entry = {
            'timestamp': result['timestamp'],
            'group_id': result['group_id'],
            'question': result['question'],
            'answer': result['answer'],
            'query_type': result['query_type'],
            'confidence_score': result['confidence_score'],
            'average_rating': result['average_rating'],
            'rating_count': result['rating_count'],
            'context_chunks_used': result['context_chunks_used'],
            'query_parameters': result['query_parameters'],
            'retrieved_chunks': [
                {
                    'chunk': chunk['chunk'],
                    'distance': chunk['distance'],
                    'metadata': chunk.get('metadata', {}),
                    'sentiment': chunk.get('sentiment_analysis', {}).get('sentiment', 'unknown')
                }
                for chunk in result['retrieved_chunks']
            ]
        }
        
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "a", encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _precision_at_k(self, retrieved_chunks: List[Dict[str, Any]], reference_answer: str, k: int = 5) -> float:
        """
        Compute Precision@k using phrase overlap between retrieved chunks and reference answer.
        """
        if not reference_answer or not retrieved_chunks:
            return 0.0
        ref = reference_answer.lower()
        relevant = 0
        for chunk in retrieved_chunks[:k]:
            chunk_text = chunk['chunk'].lower()
            # Heuristic: consider relevant if a 5-word phrase from chunk appears in answer
            words = chunk_text.split()
            found = False
            for i in range(len(words) - 4):
                phrase = ' '.join(words[i:i+5])
                if phrase in ref:
                    found = True
                    break
            if found:
                relevant += 1
        return relevant / k

    def run_test_suite(self, test_file: str = "data/enhanced_test_inputs.json", answer_file: Optional[str] = None, precision_k: int = 5):
        """
        Run enhanced test suite with various query types and Precision@k evaluation
        """
        if not os.path.exists(test_file):
            print(f"❌ Test file {test_file} not found. Creating sample tests...")
            self._create_sample_tests(test_file)
        with open(test_file, "r", encoding='utf-8') as f:
            test_data = json.load(f)
        # Load reference answers if available
        answers = None
        if answer_file and os.path.exists(answer_file):
            with open(answer_file, "r", encoding='utf-8') as f:
                answers = {case['question']: case['answer'] for case in json.load(f)}
        print(f"🧪 Running {len(test_data)} enhanced tests...")
        precisions = []
        for i, test_case in enumerate(test_data):
            print(f"\n{'='*60}")
            print(f"Test {i+1}/{len(test_data)}")
            print(f"Question: {test_case['question']}")
            query_params = test_case.get('parameters', {})
            result = self.query(
                question=test_case['question'],
                **query_params
            )
            print(f"Answer: {result['answer']}")
            print(f"Query Type: {result['query_type']}")
            print(f"Confidence: {result['confidence_score']}")
            print(f"Chunks Used: {result['context_chunks_used']}")
            if result['average_rating']:
                print(f"Average Rating: {result['average_rating']:.1f} ({result['rating_count']} reviews)")
            is_grounded = self._check_groundedness(result['answer'], result['retrieved_chunks'])
            print(f"✅ Grounded: {is_grounded}")
            # Precision@k evaluation if reference answer is available
            ref_answer = None
            if answers and test_case['question'] in answers:
                ref_answer = answers[test_case['question']]
            elif 'answer' in test_case:
                ref_answer = test_case['answer']
            if ref_answer:
                prec = self._precision_at_k(result['retrieved_chunks'], ref_answer, k=precision_k)
                precisions.append(prec)
                print(f"Precision@{precision_k}: {prec:.2f}")
        if precisions:
            avg_prec = sum(precisions) / len(precisions)
            print(f"\nAverage Precision@{precision_k}: {avg_prec:.2f}")

    def _check_groundedness(self, answer: str, chunks: List[Dict[str, Any]]) -> bool:
        """
        Check if answer is grounded in retrieved content
        """
        answer_lower = answer.lower()
        for chunk in chunks:
            chunk_text = chunk['chunk'].lower()
            # Check if any significant part of the chunk appears in the answer
            words = chunk_text.split()
            if len(words) > 3:  # Only check chunks with meaningful content
                for i in range(len(words) - 2):
                    phrase = ' '.join(words[i:i+3])
                    if phrase in answer_lower:
                        return True
        return False

    def _create_sample_tests(self, test_file: str):
        """
        Create sample test cases for the enhanced system
        """
        sample_tests = [
            {
                "question": "What do customers think about Microsoft Office 365?",
                "parameters": {
                    "top_k": 5,
                    "sentiment_filter": None
                }
            },
            {
                "question": "What problems do users face with this software?",
                "parameters": {
                    "top_k": 5,
                    "sentiment_filter": "negative",
                    "min_confidence": 0.6
                }
            },
            {
                "question": "What positive experiences do customers have?",
                "parameters": {
                    "top_k": 5,
                    "sentiment_filter": "positive",
                    "min_rating": 4.0
                }
            },
            {
                "question": "How is the installation process?",
                "parameters": {
                    "top_k": 5,
                    "sentiment_filter": None
                }
            },
            {
                "question": "Is this software worth the money?",
                "parameters": {
                    "top_k": 5,
                    "sentiment_filter": None,
                    "min_rating": 3.0
                }
            }
        ]
        
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, "w", encoding='utf-8') as f:
            json.dump(sample_tests, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created sample test file: {test_file}")

    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics
        """
        retriever_stats = self.retriever.get_statistics()
        
        # Get sentiment distribution if we have chunks
        sentiment_stats = {}
        if hasattr(self.retriever, 'documents') and self.retriever.documents:
            chunks_for_sentiment = [{'chunk': doc} for doc in self.retriever.documents[:100]]  # Sample
            sentiment_stats = self.sentiment_filter.get_sentiment_distribution(chunks_for_sentiment)
        
        return {
            'retriever': retriever_stats,
            'sentiment': sentiment_stats,
            'model_info': {
                'retriever_model': self.retriever.model_name,
                'generator_model': 'google/flan-t5-large',
                'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest'
            }
        }

def main():
    """
    Main function to run the enhanced RAG pipeline
    """
    # Initialize enhanced pipeline
    pipeline = EnhancedRAGPipeline()
    
    # Setup index with software reviews - force rebuild to ensure synchronization
    document_paths = ["retriever/software.json"]
    pipeline.setup_index(document_paths, force_rebuild=True)
    
    # Run test suite
    pipeline.run_test_suite() 
    
    # Print system statistics
    print(f"\n{'='*60}")
    print("📊 SYSTEM STATISTICS")
    print(f"{'='*60}")
    stats = pipeline.get_system_statistics()
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
