#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

print("Testing imports...")

try:
    print("1. Testing enhanced retriever import...")
    from retriever.enhanced_retriever import EnhancedRetriever
    print("   ✅ EnhancedRetriever imported successfully")
except Exception as e:
    print(f"   ❌ EnhancedRetriever import failed: {e}")

try:
    print("2. Testing enhanced generator import...")
    from generator.enhanced_generator import EnhancedGenerator
    print("   ✅ EnhancedGenerator imported successfully")
except Exception as e:
    print(f"   ❌ EnhancedGenerator import failed: {e}")

try:
    print("3. Testing sentiment filter import...")
    from sentiment.sentiment_filter import SentimentFilter
    print("   ✅ SentimentFilter imported successfully")
except Exception as e:
    print(f"   ❌ SentimentFilter import failed: {e}")

try:
    print("4. Testing sentence-transformers...")
    from sentence_transformers import SentenceTransformer
    print("   ✅ SentenceTransformer imported successfully")
except Exception as e:
    print(f"   ❌ SentenceTransformer import failed: {e}")

try:
    print("5. Testing transformers...")
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    print("   ✅ Transformers imported successfully")
except Exception as e:
    print(f"   ❌ Transformers import failed: {e}")

try:
    print("6. Testing FAISS...")
    import faiss
    print("   ✅ FAISS imported successfully")
except Exception as e:
    print(f"   ❌ FAISS import failed: {e}")

print("\nAll import tests completed!") 