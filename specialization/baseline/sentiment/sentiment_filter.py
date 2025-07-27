import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from enum import Enum

class SentimentType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class SentimentFilter:
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize sentiment analysis model for filtering customer reviews
        Using a model fine-tuned on social media text which works well for informal reviews
        """
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name
            )
        except Exception as e:
            print(f"Warning: Could not load sentiment model {model_name}. Using fallback method.")
            self.sentiment_analyzer = None
        
        # Fallback sentiment keywords for when model is not available
        self.positive_keywords = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'perfect',
            'love', 'like', 'enjoy', 'satisfied', 'happy', 'pleased', 'impressed',
            'recommend', 'worth', 'quality', 'reliable', 'fast', 'easy', 'convenient'
        ]
        
        self.negative_keywords = [
            'bad', 'terrible', 'awful', 'horrible', 'disappointed', 'frustrated',
            'hate', 'dislike', 'problem', 'issue', 'broken', 'defective', 'poor',
            'slow', 'difficult', 'complicated', 'expensive', 'waste', 'avoid'
        ]

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a given text
        """
        if not text:
            return {'sentiment': SentimentType.NEUTRAL.value, 'confidence': 0.0, 'score': 0.0}
        
        if self.sentiment_analyzer:
            return self._analyze_with_model(text)
        else:
            return self._analyze_with_keywords(text)

    def _analyze_with_model(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using the pre-trained model
        """
        try:
            result = self.sentiment_analyzer(text[:512])[0]  # Limit text length
            
            # Map model labels to our sentiment types
            label = result['label'].lower()
            score = result['score']
            
            if 'positive' in label:
                sentiment = SentimentType.POSITIVE.value
            elif 'negative' in label:
                sentiment = SentimentType.NEGATIVE.value
            else:
                sentiment = SentimentType.NEUTRAL.value
            
            return {
                'sentiment': sentiment,
                'confidence': score,
                'score': score,
                'model_label': result['label']
            }
        except Exception as e:
            print(f"Error in model-based sentiment analysis: {e}")
            return self._analyze_with_keywords(text)

    def _analyze_with_keywords(self, text: str) -> Dict[str, Any]:
        """
        Fallback sentiment analysis using keyword matching
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in self.positive_keywords)
        negative_count = sum(1 for word in words if word in self.negative_keywords)
        
        total_words = len(words)
        if total_words == 0:
            return {'sentiment': SentimentType.NEUTRAL.value, 'confidence': 0.0, 'score': 0.0}
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        # Determine sentiment based on ratios
        if positive_ratio > negative_ratio and positive_ratio > 0.05:
            sentiment = SentimentType.POSITIVE.value
            confidence = positive_ratio
        elif negative_ratio > positive_ratio and negative_ratio > 0.05:
            sentiment = SentimentType.NEGATIVE.value
            confidence = negative_ratio
        else:
            sentiment = SentimentType.NEUTRAL.value
            confidence = 1.0 - max(positive_ratio, negative_ratio)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'score': confidence,
            'method': 'keyword_based'
        }

    def filter_by_sentiment(self, chunks: List[Dict[str, Any]], 
                           sentiment_filter: Optional[str] = None,
                           min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Filter chunks by sentiment
        """
        if not sentiment_filter:
            return chunks
        
        filtered_chunks = []
        
        for chunk in chunks:
            chunk_text = chunk.get('chunk', '')
            sentiment_result = self.analyze_sentiment(chunk_text)
            
            # Check if sentiment matches filter and meets confidence threshold
            if (sentiment_result['sentiment'] == sentiment_filter and 
                sentiment_result['confidence'] >= min_confidence):
                
                # Add sentiment information to chunk
                chunk_with_sentiment = chunk.copy()
                chunk_with_sentiment['sentiment_analysis'] = sentiment_result
                filtered_chunks.append(chunk_with_sentiment)
        
        return filtered_chunks

    def get_sentiment_distribution(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get sentiment distribution across all chunks
        """
        sentiment_counts = {
            SentimentType.POSITIVE.value: 0,
            SentimentType.NEGATIVE.value: 0,
            SentimentType.NEUTRAL.value: 0
        }
        
        total_confidence = 0.0
        sentiment_scores = []
        
        for chunk in chunks:
            chunk_text = chunk.get('chunk', '')
            sentiment_result = self.analyze_sentiment(chunk_text)
            
            sentiment = sentiment_result['sentiment']
            confidence = sentiment_result['confidence']
            
            sentiment_counts[sentiment] += 1
            total_confidence += confidence
            sentiment_scores.append(confidence)
        
        total_chunks = len(chunks)
        
        return {
            'distribution': sentiment_counts,
            'total_chunks': total_chunks,
            'average_confidence': total_confidence / total_chunks if total_chunks > 0 else 0.0,
            'sentiment_percentages': {
                sentiment: (count / total_chunks * 100) if total_chunks > 0 else 0
                for sentiment, count in sentiment_counts.items()
            }
        }

    def analyze_review_sentiment(self, review_text: str, rating: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze sentiment of a review with optional rating validation
        """
        sentiment_result = self.analyze_sentiment(review_text)
        
        result = {
            'text': review_text,
            'sentiment': sentiment_result['sentiment'],
            'confidence': sentiment_result['confidence'],
            'score': sentiment_result['score']
        }
        
        # If rating is available, check consistency
        if rating is not None:
            expected_sentiment = self._rating_to_sentiment(rating)
            result['rating_consistency'] = expected_sentiment == sentiment_result['sentiment']
            result['rating'] = rating
            result['expected_sentiment_from_rating'] = expected_sentiment
        
        return result

    def _rating_to_sentiment(self, rating: float) -> str:
        """
        Convert numerical rating to sentiment
        """
        if rating >= 4.0:
            return SentimentType.POSITIVE.value
        elif rating <= 2.0:
            return SentimentType.NEGATIVE.value
        else:
            return SentimentType.NEUTRAL.value

    def get_sentiment_summary(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get a comprehensive sentiment summary
        """
        distribution = self.get_sentiment_distribution(chunks)
        
        # Find most common sentiment
        most_common = max(distribution['distribution'].items(), key=lambda x: x[1])
        
        # Calculate overall sentiment score
        positive_ratio = distribution['sentiment_percentages'][SentimentType.POSITIVE.value] / 100
        negative_ratio = distribution['sentiment_percentages'][SentimentType.NEGATIVE.value] / 100
        overall_score = positive_ratio - negative_ratio  # Range: -1 to 1
        
        return {
            'overall_sentiment': most_common[0],
            'overall_score': overall_score,
            'distribution': distribution,
            'summary': f"Overall sentiment is {most_common[0]} ({most_common[1]} out of {distribution['total_chunks']} reviews)"
        }
