# Enhanced RAG System - Customer Review Analysis

## 🚀 Overview

This enhanced RAG (Retrieval-Augmented Generation) system is specifically designed for analyzing customer reviews and providing intelligent answers to user queries. It builds upon the baseline system with significant improvements in retrieval, generation, and filtering capabilities.

## ✨ Key Enhancements

### 1. **Enhanced Retriever** (`baseline/retriever/enhanced_retriever.py`)
- **Domain-specific model**: Uses `all-mpnet-base-v2` for better semantic similarity
- **Sentence-level chunking**: Instead of word-based, for more coherent context
- **Metadata filtering**: Filter by rating, date range, reviewer, etc.
- **Enhanced preprocessing**: Better handling of noisy review text
- **Comprehensive statistics**: Detailed insights about indexed data

### 2. **Enhanced Generator** (`baseline/generator/enhanced_generator.py`)
- **Custom prompt engineering**: 6 different prompt templates for different query types
- **Query type detection**: Automatically detects sentiment, feature, problem, comparison, and recommendation queries
- **Structured answers**: Returns metadata with confidence scores and statistics
- **Enhanced generation parameters**: Better control over output quality

### 3. **Sentiment Filtering** (`baseline/sentiment/sentiment_filter.py`)
- **Pre-trained sentiment model**: Uses Twitter RoBERTa for accurate sentiment analysis
- **Fallback keyword system**: Works even without internet connection
- **Rating consistency checking**: Validates sentiment against numerical ratings
- **Comprehensive filtering**: Filter by positive/negative/neutral sentiment

### 4. **Enhanced Pipeline** (`baseline/enhanced_pipeline.py`)
- **Integrated system**: Combines all enhancements seamlessly
- **Advanced querying**: Multiple filtering options in one interface
- **Comprehensive logging**: Detailed logs with metadata
- **Test suite**: Automated testing with various query types

## 📦 Installation

1. **Install dependencies**:
```bash
pip install -r requirements_enhanced.txt
```

2. **Prepare your data**:
   - Place JSON review files in the `baseline/retriever/` directory
   - Each line should be a JSON object with review data

## 🚀 Quick Start

### Basic Usage

```python
from baseline.enhanced_pipeline import EnhancedRAGPipeline

# Initialize the enhanced pipeline
pipeline = EnhancedRAGPipeline()

# Setup index with your review data
document_paths = ["baseline/retriever/software.json"]
pipeline.setup_index(document_paths)

# Ask questions with various filters
result = pipeline.query(
    question="What do customers think about the installation process?",
    top_k=5,
    sentiment_filter="positive",  # Optional: filter by sentiment
    min_rating=4.0,              # Optional: filter by rating
    date_from="2023-01-01"       # Optional: filter by date
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence_score']}")
print(f"Query Type: {result['query_type']}")
```

### Advanced Querying

```python
# Find problems with negative sentiment
result = pipeline.query(
    question="What issues do users face?",
    sentiment_filter="negative",
    min_confidence=0.7
)

# Get positive experiences from high-rated reviews
result = pipeline.query(
    question="What do satisfied customers say?",
    sentiment_filter="positive",
    min_rating=4.5
)

# Compare different aspects
result = pipeline.query(
    question="How does the camera compare to other phones?",
    top_k=10
)
```

## 📊 Query Types Supported

The system automatically detects and uses appropriate prompts for:

1. **General Questions**: Basic information about products/services
2. **Sentiment Questions**: What customers think/feel about something
3. **Feature Questions**: Specific features like camera, battery, performance
4. **Problem Questions**: Issues, complaints, defects
5. **Comparison Questions**: Comparing different aspects or products
6. **Recommendation Questions**: Purchase advice, worth analysis

## 🔧 Configuration Options

### Retriever Configuration
```python
# Use different embedding model
pipeline = EnhancedRAGPipeline(retriever_model='all-MiniLM-L6-v2')

# Custom chunking parameters
retriever = EnhancedRetriever()
retriever._sentence_level_chunking(text, max_sentences=5)
```

### Generator Configuration
```python
# Use different generation model
pipeline = EnhancedRAGPipeline(generator_model='google/flan-t5-large')

# Custom generation parameters
result = generator.generate_answer(prompt, max_length=512, temperature=0.8)
```

### Sentiment Configuration
```python
# Use different sentiment model
pipeline = EnhancedRAGPipeline(sentiment_model='distilbert-base-uncased-finetuned-sst-2-english')

# Custom confidence threshold
filtered_chunks = sentiment_filter.filter_by_sentiment(chunks, min_confidence=0.8)
```

## 📈 Evaluation Metrics

The enhanced system provides comprehensive evaluation:

### Automatic Metrics
- **Confidence Score**: Based on retrieval relevance
- **Groundedness**: Whether answer contains retrieved content
- **Query Type Detection**: Automatic categorization
- **Rating Statistics**: Average ratings from retrieved reviews

### Manual Evaluation
- **Answer Relevance**: How well the answer addresses the question
- **Answer Accuracy**: Faithfulness to original reviews
- **Retrieval Effectiveness**: Precision@k for retrieved chunks
- **Answer Helpfulness**: User satisfaction ratings

## 📁 File Structure

```
baseline/
├── retriever/
│   ├── enhanced_retriever.py    # Enhanced retriever with metadata filtering
│   ├── retriever.py             # Original baseline retriever
│   └── software.json            # Sample review data
├── generator/
│   ├── enhanced_generator.py    # Enhanced generator with custom prompts
│   └── generator.py             # Original baseline generator
├── sentiment/
│   └── sentiment_filter.py      # Sentiment analysis and filtering
├── enhanced_pipeline.py         # Main enhanced pipeline
└── pipeline.py                  # Original baseline pipeline
data/
├── enhanced_logs.jsonl          # Enhanced logging
└── enhanced_test_inputs.json    # Test cases
requirements_enhanced.txt        # Enhanced dependencies
README_ENHANCED.md              # This file
```

## 🧪 Running Tests

```bash
# Run the enhanced test suite
python baseline/enhanced_pipeline.py

# Run specific tests
python -c "
from baseline.enhanced_pipeline import EnhancedRAGPipeline
pipeline = EnhancedRAGPipeline()
pipeline.setup_index(['baseline/retriever/software.json'])
pipeline.run_test_suite()
"
```

## 📊 Sample Output

```
🔍 Querying: What do customers think about Microsoft Office 365?
📄 Retrieved 10 chunks
😊 Applying sentiment filter: positive
📄 After sentiment filtering: 6 chunks

Answer: Based on customer reviews, Microsoft Office 365 receives generally positive feedback. 
Many users appreciate the subscription model, saying it's "worth the money" and provides 
good value for families or multiple devices. Customers mention the convenience of cloud 
storage and cross-device access as major benefits.

Query Type: sentiment
Confidence: 0.85
Chunks Used: 5
Average Rating: 4.2 (5 reviews)
✅ Grounded: True
```

## 🔄 Migration from Baseline

To migrate from the baseline system:

1. **Replace imports**:
```python
# Old
from baseline.retriever.retriever import Retriever
from baseline.generator.generator import Generator

# New
from baseline.enhanced_pipeline import EnhancedRAGPipeline
```

2. **Update query calls**:
```python
# Old
retriever = Retriever()
generator = Generator()
chunks = retriever.query(question, top_k=3)
prompt = generator.build_prompt(chunks, question)
answer = generator.generate_answer(prompt)

# New
pipeline = EnhancedRAGPipeline()
result = pipeline.query(question, top_k=5)
answer = result['answer']
```

## 🚀 Future Enhancements

- **Multi-domain support**: Yelp, IMDB datasets
- **Advanced filtering**: Category, price range, location
- **User personalization**: Tailored responses based on user preferences
- **Real-time updates**: Dynamic index updates
- **API interface**: REST API for web integration
- **Visualization**: Interactive dashboards for insights

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your enhancement
4. Add tests
5. Submit a pull request

## 📄 License

This project is part of the Natural Language Processing course at Summer Semester 2025.

---

**Made with ❤️ by Team Turing**
