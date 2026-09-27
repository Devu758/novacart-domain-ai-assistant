
from unsloth import FastLanguageModel
import torch


MODEL_NAME = "Devu758/novacart-qwen-dpo"


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)


def generate_answer(question):

    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=150,
        do_sample=False,
    )

    generated_tokens = outputs[0][inputs.shape[-1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()


if __name__ == "__main__":

    while True:

        question = input("\nAsk NovaCart Assistant: ")

        if question.lower() in ["exit", "quit"]:
            break

        answer = generate_answer(question)

        print("\nAssistant:")
        print(answer)
