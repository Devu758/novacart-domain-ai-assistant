from unsloth import FastLanguageModel
import torch


MODEL_NAME = "Devu758/novacart-qwen-dpo"

# Load final DPO model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)


SYSTEM_PROMPT = """
You are NovaCart's customer-support assistant.

Answer questions using NovaCart policies.
Be concise, clear, professional, and helpful.

Do not invent policies or information that you do not know.
If the user asks something outside NovaCart customer support,
politely explain that you are a NovaCart support assistant.
"""


class NovaCartChatbot:

    def __init__(self):
        self.history = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.strip()
            }
        ]

    def generate_answer(self, question):

        # Add current user message
        self.history.append({
            "role": "user",
            "content": question
        })

        # Convert complete conversation into Qwen chat format
        inputs = tokenizer.apply_chat_template(
            self.history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        # Generate response
        with torch.no_grad():

            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=180,
                do_sample=False,
                repetition_penalty=1.05,
            )

        generated_tokens = outputs[0][inputs.shape[-1]:]

        answer = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        # Save assistant response in conversation history
        self.history.append({
            "role": "assistant",
            "content": answer
        })

        return answer


    def reset(self):

        self.history = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.strip()
            }
        ]

        print("Conversation history cleared.")


    def show_history(self):

        for message in self.history:

            role = message["role"].upper()

            print(f"\n{role}:")
            print(message["content"])


if __name__ == "__main__":

    chatbot = NovaCartChatbot()

    print("\nNovaCart AI Assistant")
    print("---------------------")
    print("Commands:")
    print("  exit    - close chatbot")
    print("  reset   - start a new conversation")
    print("  history - show conversation history")

    while True:

        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() in ["exit", "quit"]:
            print("\nNovaCart: Goodbye!")
            break

        if question.lower() == "reset":
            chatbot.reset()
            continue

        if question.lower() == "history":
            chatbot.show_history()
            continue

        answer = chatbot.generate_answer(question)

        print("\nNovaCart:")
        print(answer)
