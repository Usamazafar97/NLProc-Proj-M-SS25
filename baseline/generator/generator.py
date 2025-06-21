from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class Generator:
    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def build_prompt(self, context_chunks, question):
        context = "\n".join([chunk['chunk'] for chunk in context_chunks])
        return f"""You are a helpful assistant. Use only the information provided in the context to answer the question.
    .

    Context:
    {context}

    Question:
    {question}

    Answer:"""

    def generate_answer(self, prompt, max_length=128):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
