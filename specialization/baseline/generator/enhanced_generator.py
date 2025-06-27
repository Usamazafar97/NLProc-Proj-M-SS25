from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from typing import List, Dict, Any, Optional
import re

class EnhancedGenerator:
    def __init__(self, model_name="google/flan-t5-large"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Custom prompt templates for different types of customer review queries
        self.prompt_templates = {
            'general': self._general_prompt,
            'sentiment': self._sentiment_prompt,
            'feature': self._feature_prompt,
            'comparison': self._comparison_prompt,
            'problem': self._problem_prompt,
            'recommendation': self._recommendation_prompt
        }

    def _detect_query_type(self, question: str) -> str:
        """
        Detect the type of query to choose appropriate prompt template
        """
        question_lower = question.lower()
        
        # Summary and sentiment-related queries (improved detection)
        if any(word in question_lower for word in ['summarize', 'summary', 'summarise', 'what users say', 'what customers say', 'what people think']):
            return 'sentiment'
        
        # Sentiment-related queries
        if any(word in question_lower for word in ['feel', 'think', 'opinion', 'like', 'dislike', 'love', 'hate', 'saying']):
            return 'sentiment'
        
        # Feature-specific queries
        if any(word in question_lower for word in ['camera', 'battery', 'screen', 'performance', 'quality', 'durability', 'feature', 'functionality']):
            return 'feature'
        
        # Problem/complaint queries
        if any(word in question_lower for word in ['problem', 'issue', 'complaint', 'broken', 'defect', 'fault', 'trouble', 'difficulty']):
            return 'problem'
        
        # Comparison queries
        if any(word in question_lower for word in ['compare', 'difference', 'better', 'worse', 'versus', 'vs', 'against']):
            return 'comparison'
        
        # Recommendation queries
        if any(word in question_lower for word in ['recommend', 'suggest', 'worth', 'buy', 'purchase', 'should i']):
            return 'recommendation'
        
        return 'general'

    def _general_prompt(self, context_chunks: List[Dict[str, Any]], question: str) -> str:
        """
        General prompt for customer review questions
        """
        context = "\n".join([chunk['chunk'] for chunk in context_chunks])
        
        prompt = f"""Based on the following customer reviews, provide a comprehensive and detailed answer to the question. 
Be specific and include relevant details from the reviews to support your response.

Customer Reviews:
{context}

Question: {question}

Please provide a detailed answer that directly addresses the question using information from the reviews:"""
        return prompt

    def _sentiment_prompt(self, context_chunks: List[Dict[str, Any]], question: str) -> str:
        """
        Prompt for sentiment and opinion-based questions
        """
        context = "\n".join([chunk['chunk'] for chunk in context_chunks])
        
        prompt = f"""Based on the following customer reviews, provide a comprehensive and detailed summary of what customers think and feel about this topic.

Customer Reviews:
{context}

Question: {question}

Please write a detailed paragraph that summarizes the customer opinions. Include:
- Overall sentiment and general opinions
- Specific positive feedback mentioned by customers
- Specific negative feedback or concerns raised
- Key themes or patterns in the reviews
- Important examples or quotes from the reviews when relevant

Your summary should be informative, well-structured, and give a complete picture of customer opinions. Write in a natural, flowing paragraph format."""
        return prompt

    def _feature_prompt(self, context_chunks: List[Dict[str, Any]], question: str) -> str:
        """
        Prompt for feature-specific questions
        """
        context = "\n".join([chunk['chunk'] for chunk in context_chunks])
        
        prompt = f"""Based on the following customer reviews, provide detailed information about the specific feature mentioned in the question.

Customer Reviews:
{context}

Question: {question}

Please provide a comprehensive analysis covering how well the feature performs, any issues customers mention, positive aspects, and specific examples from the reviews. Focus on providing detailed, specific information about this feature."""
        return prompt

    def _problem_prompt(self, context_chunks: List[Dict[str, Any]], question: str) -> str:
        """
        Prompt for problem and complaint-related questions
        """
        context = "\n".join([chunk['chunk'] for chunk in context_chunks])
        
        prompt = f"""Based on the following customer reviews, identify and summarize the problems, issues, or complaints mentioned by customers.

Customer Reviews:
{context}

Question: {question}

Please provide a detailed analysis covering specific problems mentioned, how frequently they occur, their severity, any solutions customers mention, and specific examples from the reviews. Focus on providing comprehensive information about customer problems and issues."""
        return prompt

    def _comparison_prompt(self, context_chunks: List[Dict[str, Any]], question: str) -> str:
        """
        Prompt for comparison questions
        """
        context = "\n".join([chunk['chunk'] for chunk in context_chunks])
        
        prompt = f"""Based on the following customer reviews, provide a detailed comparison or analysis as requested.

Customer Reviews:
{context}

Question: {question}

Please provide a comprehensive comparison covering clear differences between items, advantages and disadvantages, customer preferences, performance comparisons, and specific examples from reviews. Provide a detailed, balanced comparison using information from the reviews."""
        return prompt

    def _recommendation_prompt(self, context_chunks: List[Dict[str, Any]], question: str) -> str:
        """
        Prompt for recommendation and purchase advice questions
        """
        context = "\n".join([chunk['chunk'] for chunk in context_chunks])
        
        prompt = f"""Based on the following customer reviews, provide helpful advice and recommendations.

Customer Reviews:
{context}

Question: {question}

Please provide comprehensive advice covering overall recommendations, key advantages and benefits, potential drawbacks, specific situations where this would be good or not ideal, and value for money considerations. Provide balanced, detailed advice based on customer experiences."""
        return prompt

    def build_prompt(self, context_chunks: List[Dict[str, Any]], question: str) -> str:
        """
        Build a custom prompt based on the query type
        """
        query_type = self._detect_query_type(question)
        prompt_template = self.prompt_templates.get(query_type, self.prompt_templates['general'])
        
        return prompt_template(context_chunks, question)

    def generate_answer(self, prompt: str, max_length: int = 256, temperature: float = 0.7) -> str:
        """
        Generate answer with enhanced parameters
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Enhanced generation parameters for better quality answers
        outputs = self.model.generate(
            **inputs, 
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
            length_penalty=1.2,  # Encourage longer answers
            repetition_penalty=1.1  # Reduce repetition
        )
        
        raw_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._clean_answer(raw_answer)

    def _clean_answer(self, answer: str) -> str:
        """
        Clean up the generated answer to remove prompt artifacts and improve quality
        """
        if not answer:
            return answer
        
        # Remove common prompt artifacts
        answer = answer.strip()
        
        # Remove numbered lists that might be copied from prompts
        lines = answer.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip lines that are just numbers or numbered list items
            if re.match(r'^\d+\.?\s*$', line) or re.match(r'^\d+\.\s*[A-Za-z\s]*$', line):
                continue
            # Skip lines that start with common prompt phrases
            if any(phrase in line.lower() for phrase in [
                'please provide', 'please include', 'please focus', 'please cover',
                'your response should', 'your summary should', 'provide balanced',
                'focus on providing', 'based on the following'
            ]):
                continue
            if line:
                cleaned_lines.append(line)
        
        cleaned_answer = '\n'.join(cleaned_lines).strip()
        
        # If the answer is too short or seems like a prompt artifact, try to extract meaningful content
        if len(cleaned_answer.split()) < 10:
            # Look for actual content after common prompt endings
            if ':' in answer:
                parts = answer.split(':')
                if len(parts) > 1:
                    potential_content = parts[-1].strip()
                    if len(potential_content.split()) > 5:
                        return potential_content
        
        return cleaned_answer if cleaned_answer else answer

    def generate_structured_answer(self, context_chunks: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
        """
        Generate a structured answer with metadata
        """
        prompt = self.build_prompt(context_chunks, question)
        
        # Adjust generation parameters based on query type
        query_type = self._detect_query_type(question)
        
        if query_type == 'sentiment':
            # Use longer max_length and lower temperature for sentiment/summary queries
            answer = self.generate_answer(prompt, max_length=768, temperature=0.6)
        elif query_type == 'general':
            # Use moderate parameters for general queries
            answer = self.generate_answer(prompt, max_length=512, temperature=0.7)
        else:
            # Use standard parameters for other query types
            answer = self.generate_answer(prompt, max_length=384, temperature=0.7)
        
        # Extract key information from context
        ratings = []
        for chunk in context_chunks:
            if 'metadata' in chunk and 'overall' in chunk['metadata']:
                try:
                    ratings.append(float(chunk['metadata']['overall']))
                except (ValueError, TypeError):
                    pass
        
        # Calculate statistics
        avg_rating = sum(ratings) / len(ratings) if ratings else None
        rating_count = len(ratings)
        
        return {
            'answer': answer,
            'prompt': prompt,
            'query_type': query_type,
            'context_chunks_used': len(context_chunks),
            'average_rating': avg_rating,
            'rating_count': rating_count,
            'confidence_score': self._calculate_confidence(context_chunks, answer)
        }

    def _calculate_confidence(self, context_chunks: List[Dict[str, Any]], answer: str) -> float:
        """
        Calculate a simple confidence score based on context relevance
        """
        if not context_chunks:
            return 0.0
        
        # Simple confidence based on number of chunks and their relevance
        total_distance = sum(chunk.get('distance', 1.0) for chunk in context_chunks)
        avg_distance = total_distance / len(context_chunks)
        
        # Convert distance to confidence (lower distance = higher confidence)
        confidence = max(0.0, 1.0 - (avg_distance / 2.0))
        
        return round(confidence, 2)

    def get_prompt_statistics(self, question: str) -> Dict[str, Any]:
        """
        Get statistics about the prompt generation
        """
        query_type = self._detect_query_type(question)
        
        return {
            'query_type': query_type,
            'available_templates': list(self.prompt_templates.keys()),
            'template_used': query_type
        } 