from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class Generator:
    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def build_prompt(self, context_chunks, question):
        context = "\n".join([f"- {chunk['chunk']}" for chunk in context_chunks])

        examples = """Examples:
    Context:
    - The software is easy to install and activate.
    - Setup took just a few minutes.
    Question: Is Microsoft Office 365 easy to install?
    Answer: Yes, it's easy to install and activate. Most users found the setup process quick and hassle-free.

    Context:
    - Overpriced and buggy as hell but it mostly works.
    - Works better than some older versions.
    Question: Is Microsoft Office 365 worth the money?
    Answer: Microsoft Office 365 is considered overpriced and buggy by some users, but it still functions adequately and performs better than some older versions.

    Context:
    - The product is outdated.
    - LibreOffice offers a free and updated alternative.
    Question: Should I choose Microsoft Office over LibreOffice?
    Answer: The context suggests LibreOffice may be a better choice, as it's both free and more up-to-date than this version of Microsoft Office.

    Context:
    - Easy to use and easy to understand.
    - Might not be for the "novice".
    Question: Is Microsoft Office 365 beginner-friendly?
    Answer: Microsoft Office 365 is generally easy to use, but it might not be ideal for complete beginners without prior experience.

    """

        return f"""You are an assistant answering user questions using the provided context. 
    Follow these rules:
    - Base your answer strictly on the context.
    - Highlight both positive and negative points, if available.
    - Give a short but balanced and informative answer (2–4 lines).
    - If the context does not clearly answer the question, say: "The context does not contain a clear answer."

    {examples}

    Now use the following context to answer the next question.

    Context:
    {context}

    Question: {question}

    Answer:"""

    def generate_answer(self, prompt, max_length=128):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
