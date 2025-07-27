import json

from logger import log_query


def run_tests(generator, retriever, test_file="test_inputs.json", log_file="logs.jsonl"):
    with open(test_file, "r") as f:
        test_data = json.load(f)

    for i, test_case in enumerate(test_data):
        question = test_case["question"]
        expected = test_case["answer"]

        context = retriever.retrieve(question)
        prompt = generator.build_prompt(context, question)
        answer = generator.generate_answer(prompt)

        log_query(question, context, prompt, answer, group_id="test_run", log_file=log_file)

        print(f"[{i}] Q: {question}")
        print(f"    A: {answer}")
        print(f"    ✅ Grounded: {context in answer}")
        print()
