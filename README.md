
# NovaCart Domain-Specific AI Assistant

A domain-specific customer support AI assistant built by fine-tuning **Qwen2.5-1.5B-Instruct** using **Unsloth, QLoRA, Supervised Fine-Tuning (SFT), and Direct Preference Optimization (DPO)**.

The project demonstrates an end-to-end LLM fine-tuning workflow:

```text
Qwen2.5-1.5B-Instruct
        ↓
Non-Instruction Fine-Tuning
        ↓
Domain-Adapted Model
        ↓
Instruction Fine-Tuning (SFT)
        ↓
Instruction-Following Customer Support Model
        ↓
Direct Preference Optimization (DPO)
        ↓
Final NovaCart Customer Support Assistant
```

---

## 1. Business Problem

The objective is to build a domain-specific AI assistant for a fictional e-commerce company called **NovaCart**.

The assistant should answer customer-support questions related to:

- Order cancellation
- Returns
- Refunds
- Damaged products
- Incorrect products
- Shipping and delivery
- Order tracking
- Payment failures
- Duplicate charges
- Missing items
- Warranty
- Address changes
- Coupons
- Password recovery
- Customer support

A general-purpose LLM may provide generic responses that do not follow NovaCart-specific policies.

The goal of fine-tuning is to make the model provide responses that are:

- Domain-specific
- Correct
- Helpful
- Clear
- Professional
- Less generic
- Less likely to hallucinate unrelated policy information

---

# 2. Model

Base model:

```text
Qwen2.5-1.5B-Instruct
```

Fine-tuning framework:

```text
Unsloth
```

Training approach:

```text
QLoRA
```

Quantization:

```text
4-bit
```

The small model size and QLoRA approach made it possible to perform the experiment using a Google Colab GPU.

---

# 3. Fine-Tuning Architecture

```text
                         ┌────────────────────────┐
                         │ Qwen2.5-1.5B-Instruct │
                         └───────────┬────────────┘
                                     │
                                     ▼
                      ┌───────────────────────────┐
                      │ Non-Instruction Fine-Tune│
                      │ Raw NovaCart policy text │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
                Devu758/novacart-qwen-stage1-domain
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Instruction Fine-Tuning│
                       │ 120 Question/Answer    │
                       │ examples               │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       Devu758/novacart-qwen-sft
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ DPO Preference Alignment│
                       │ 50 Preference Examples │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       Devu758/novacart-qwen-dpo
                                    │
                                    ▼
                         Final NovaCart Assistant
```

---

# 4. Dataset

Three datasets were used.

## 4.1 Non-Instruction Dataset

File:

```text
data/non_instruction_data.txt
```

Contains approximately **60 paragraphs** of NovaCart domain text.

The dataset covers topics including:

- Refunds
- Returns
- Cancellation
- Delivery
- Tracking
- Payment issues
- Damaged products
- Missing products
- Warranty
- Customer support

Example:

```text
Approved NovaCart refunds are generally processed within
5 to 7 business days after the returned product has been
reviewed. Refunds are normally returned to the original
payment method.
```

The purpose of this dataset is to expose the model to NovaCart terminology, policies, and domain information.

---

## 4.2 Instruction Dataset

File:

```text
data/instruction_dataset.jsonl
```

Number of examples:

```text
120
```

Format:

```json
{
  "instruction": "How long does NovaCart take to process a refund?",
  "response": "Approved NovaCart refunds are generally processed within 5 to 7 business days and returned to the original payment method."
}
```

Multiple question variations were created for each policy.

The purpose of this dataset is to teach the model how to respond to user questions using domain-specific knowledge.

---

## 4.3 Preference Dataset

File:

```text
data/preference_dataset.jsonl
```

Number of preference pairs:

```text
50
```

Format:

```json
{
  "prompt": "How long does NovaCart take to process a refund?",
  "chosen": "Approved NovaCart refunds are generally processed within 5 to 7 business days.",
  "rejected": "Refunds take 5 to 7 business days and then another 3 to 7 days for delivery."
}
```

The chosen response is designed to be:

- Correct
- Relevant
- Concise
- Professional
- Domain-specific

Rejected responses contain problems such as:

- Incorrect information
- Hallucinated information
- Irrelevant information
- Unsafe recommendations
- Overconfident claims
- Poor domain adherence

---

# 5. Stage 1 — Non-Instruction Fine-Tuning

The first training stage used raw domain text.

Unlike instruction fine-tuning, the dataset does not explicitly contain questions and answers.

Example:

```text
NovaCart allows customers to cancel an order before shipment.
Once an order has shipped, cancellation is no longer available.
```

The objective is to help the model learn:

```text
Domain terminology
        +
Domain facts
        +
Domain writing patterns
```

This can also be thought of as continued language-model training on domain-specific text.

### Configuration

```text
Model: Qwen2.5-1.5B-Instruct

Quantization: 4-bit

LoRA Rank: 16
LoRA Alpha: 16
LoRA Dropout: 0

Epochs: 2

Per-device batch size: 2

Gradient accumulation: 4

Effective batch size:
2 × 4 = 8

Learning rate:
2e-4

Max sequence length:
1024
```

Stage-1 Hugging Face adapter:

```text
https://huggingface.co/Devu758/novacart-qwen-stage1-domain
```

---

# 6. Stage 2 — Supervised Fine-Tuning (SFT)

After domain adaptation, the model was trained using explicit question-answer examples.

Example:

```text
User:
How long does NovaCart take to process a refund?

Assistant:
Approved NovaCart refunds are generally processed
within 5 to 7 business days.
```

SFT teaches the model:

```text
User instruction
       ↓
Appropriate response
```

### Configuration

```text
Training examples: 120

Epochs: 2

Batch size: 2

Gradient accumulation: 4

Effective batch size: 8

Learning rate: 2e-4

Maximum sequence length: 1024
```

Stage-2 Hugging Face adapter:

```text
https://huggingface.co/Devu758/novacart-qwen-sft
```

---

# 7. Stage 3 — Direct Preference Optimization

After SFT, the model could answer domain questions better but could still include unnecessary or hallucinated information.

For example, during testing the SFT model correctly learned:

```text
Refund = 5–7 business days
```

but also added unrelated delivery information:

```text
3–7 business days
```

Preference alignment was therefore performed using DPO.

Each example contains:

```text
Prompt
   │
   ├── Chosen response
   │
   └── Rejected response
```

The objective is to increase the model's preference for the chosen response relative to the rejected response.

### DPO Configuration

```text
Preference examples: 50

Epochs: 2

Batch size: 1

Gradient accumulation: 4

Effective batch size: 4

Learning rate: 5e-6

Beta: 0.1

Max length: 1024

Max prompt length: 512
```

The DPO learning rate was intentionally much smaller than the SFT learning rate because DPO performs preference refinement on an already fine-tuned model.

Final Hugging Face model:

```text
https://huggingface.co/Devu758/novacart-qwen-dpo
```

---

# 8. Base vs SFT vs DPO

The same 10 customer-support questions were used across model stages.

The models were evaluated on:

- Correctness
- Domain accuracy
- Helpfulness
- Relevance
- Clarity
- Tone
- Safety
- Hallucination reduction
- Professional response quality

Example:

```text
Question:
How long does NovaCart take to process a refund?
```

### After Non-Instruction Fine-Tuning

The model confused refund processing with delivery time and responded with information about:

```text
3–7 business day delivery
```

instead of correctly answering the refund policy.

### After SFT

The model correctly learned:

```text
Refund processing = 5–7 business days
```

but still added unrelated information about delivery time.

### After DPO

Preference training was designed to favor responses that contain only relevant refund information and reject responses mixing refund and shipping policies.

Full evaluation results are available in:

```text
reports/final_evaluation.md
```

---

# 9. Why Full Fine-Tuning Is Expensive

Full fine-tuning updates every model parameter.

For a large LLM, training requires memory for:

```text
Model weights
+
Gradients
+
Optimizer states
+
Activations
```

As model size increases, GPU-memory requirements become extremely large.

For this reason, Parameter-Efficient Fine-Tuning techniques such as LoRA are useful.

---

# 10. LoRA

LoRA stands for:

```text
Low-Rank Adaptation
```

Instead of modifying all model parameters, the original model weights remain frozen.

Small trainable matrices are added to selected transformer layers.

Conceptually:

```text
Effective Weight

W' = W + ΔW
```

where:

```text
W = original frozen model weight

ΔW = LoRA update
```

The LoRA update can be represented using two smaller matrices:

```text
ΔW ≈ B × A
```

This dramatically reduces the number of trainable parameters.

---

# 11. LoRA Rank

The LoRA rank controls the dimensionality of the low-rank matrices.

This project used:

```text
r = 16
```

Conceptually:

```text
Lower rank
→ fewer trainable parameters
→ lower memory requirement
→ lower adaptation capacity

Higher rank
→ more trainable parameters
→ greater adaptation capacity
→ greater memory requirement
```

---

# 12. LoRA Alpha

This project used:

```text
lora_alpha = 16
```

Alpha controls the scaling of the LoRA update.

A simplified representation is:

```text
Scaling ≈ alpha / rank
```

For this project:

```text
alpha = 16
rank = 16

scaling ≈ 1
```

---

# 13. LoRA Dropout

Configuration:

```text
lora_dropout = 0
```

Dropout can reduce overfitting by randomly disabling part of the LoRA path during training.

A value of zero was used because Unsloth provides optimized behavior for this configuration and the experiment was designed as a small demonstration.

---

# 14. QLoRA

QLoRA combines:

```text
Quantized base model
        +
LoRA adapters
```

In this project, the Qwen base model was loaded using:

```text
4-bit quantization
```

The large base-model weights therefore require substantially less GPU memory.

LoRA adapters remain trainable.

Conceptually:

```text
Qwen Base Model
      ↓
4-bit Quantization
      ↓
Frozen Base Weights
      +
Trainable LoRA
      ↓
QLoRA Fine-Tuning
```

This is why the model could be trained on a limited-memory Colab GPU.

---

# 15. LoRA Adapter vs Full Model

The Hugging Face repositories contain LoRA adapters instead of full copies of Qwen.

For example, the adapter is approximately tens of megabytes rather than several gigabytes.

At runtime:

```text
Qwen Base Model
      +
LoRA Adapter
      ↓
Fine-Tuned Behavior
```

The adapter configuration identifies the associated base model.

The LoRA modification is applied during model computation.

A merged model could also be created:

```text
W_merged = W_base + ΔW_LoRA
```

but keeping adapters separate reduces storage and allows multiple specialized adapters to reuse the same base model.

---

# 16. Training Loss

LLMs are trained to predict tokens.

Suppose the expected next token is:

```text
days
```

The model assigns probabilities:

```text
days   → 0.70
weeks  → 0.10
hours  → 0.05
...
```

Cross-entropy loss penalizes the model when it assigns low probability to the correct token.

Conceptually:

```text
Loss = -log(P(correct token))
```

Therefore:

```text
High probability for correct token
→ lower loss

Low probability for correct token
→ higher loss
```

Training loss is averaged across many predicted tokens.

Loss does not need to decrease at every single training step because different mini-batches have different difficulty.

---

# 17. Batch Size

Batch size determines how many training examples are processed together.

Example:

```text
Batch size = 2

Example 1 ┐
          ├── Batch
Example 2 ┘
```

Large batches generally require more GPU memory.

Small batches were therefore used in this project.

---

# 18. Gradient Accumulation

Gradient accumulation allows multiple mini-batches to contribute gradients before an optimizer update.

Example:

```text
Batch size = 2

Gradient accumulation = 4
```

Then:

```text
Batch 1 → accumulate
Batch 2 → accumulate
Batch 3 → accumulate
Batch 4 → optimizer update
```

Effective batch size:

```text
2 × 4 = 8
```

This allows the model to simulate a larger batch while using less GPU memory.

---

# 19. Number of Training Steps

Approximate optimizer steps per epoch are determined by:

```text
Dataset Size
────────────────────────────────────────
Batch Size × Gradient Accumulation
```

For Stage 1:

```text
Dataset = 60

Batch = 2

Gradient accumulation = 4

Effective batch = 8
```

Approximately:

```text
60 / 8 ≈ 8 optimizer steps per epoch
```

For two epochs:

```text
≈ 16 optimizer steps
```

When gradient accumulation was reduced to 2:

```text
effective batch = 2 × 2 = 4

60 / 4 = 15

15 × 2 epochs = 30 steps
```

This explains why decreasing gradient accumulation increased the number of optimizer steps.

---

# 20. Epoch

One epoch means:

```text
One complete pass through the training dataset.
```

For example:

```text
Dataset = 120 examples

2 epochs
```

means the training process sees approximately:

```text
120 × 2 = 240 example exposures
```

Too many epochs can cause overfitting, especially with small datasets.

---

# 21. Learning Rate

The learning rate controls how large an optimizer update is.

Conceptually:

```text
Very low learning rate
→ slow learning

Very high learning rate
→ unstable training

Appropriate learning rate
→ gradual useful adaptation
```

This project used:

```text
SFT: 2e-4

DPO: 5e-6
```

DPO uses a smaller learning rate because it is refining an already fine-tuned model.

---

# 22. Non-Instruction FT vs SFT

## Non-Instruction Fine-Tuning

Input:

```text
Raw domain text
```

Purpose:

```text
Teach the model what the domain contains.
```

---

## SFT

Input:

```text
Question → Answer
```

Purpose:

```text
Teach the model how to respond.
```

Simple mental model:

```text
Non-Instruction FT
→ What should the model know?

SFT
→ How should the model answer?
```

---

# 23. SFT vs DPO

SFT provides one target answer:

```text
Prompt
   ↓
Correct response
```

DPO provides preference information:

```text
Prompt
   ↓
Chosen > Rejected
```

Therefore:

```text
SFT
→ teaches task behavior

DPO
→ refines preferred behavior
```

---

# 24. DPO Reference Model

DPO compares the updated policy against reference behavior.

Conceptually:

```text
                    SFT Model
                       │
             ┌─────────┴─────────┐
             ↓                   ↓

       Trainable Policy      Reference
             │                   │
             └─────────┬─────────┘
                       ↓
               Preference Loss
                       ↓
                 Updated Model
```

The reference prevents preference optimization from arbitrarily moving too far from the original SFT behavior.

---

# 25. DPO Beta

Configuration:

```text
beta = 0.1
```

Beta controls the preference/reference tradeoff during DPO.

It influences how aggressively the model changes relative to its reference behavior.

---

# 26. Chat Templates

Instruction-tuned models expect conversations in specific formats.

Instead of manually concatenating:

```text
Question + Answer
```

the tokenizer's chat template was used:

```python
tokenizer.apply_chat_template(...)
```

This ensures user and assistant messages use the control tokens expected by Qwen.

Incorrect chat formatting can reduce instruction-following performance.

---

# 27. Why Deterministic Evaluation Was Used

For comparison, generation used:

```python
do_sample=False
```

instead of high-temperature sampling.

This makes model comparison more consistent because outputs are less affected by random sampling.

The same 10 prompts were used across model stages.

---

# 28. Repository Structure

```text
novacart-domain-ai-assistant/
│
├── data/
│   ├── non_instruction_data.txt
│   ├── instruction_dataset.jsonl
│   └── preference_dataset.jsonl
│
├── notebooks/
│   ├── 01_non_instruction_finetuning.ipynb
│   ├── 02_instruction_finetuning.ipynb
│   └── 03_dpo_alignment.ipynb
│
├── reports/
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   ├── final_evaluation.md
│   ├── fine_tuning_explanation.md
│   ├── stage1_training_config.json
│   ├── stage1_training_logs.json
│   ├── stage2_sft_config.json
│   ├── stage2_sft_logs.json
│   ├── stage3_dpo_config.json
│   └── stage3_dpo_logs.json
│
├── src/
│   └── inference.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

Model adapters are hosted separately on Hugging Face instead of storing large model files inside GitHub.

---

# 29. Running the Final Model

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python src/inference.py
```

Example:

```text
Ask NovaCart Assistant:
How long does a refund take?

Assistant:
Approved NovaCart refunds are generally processed
within 5 to 7 business days...
```

---

# 30. Hugging Face Models

### Domain Adaptation

```text
https://huggingface.co/Devu758/novacart-qwen-stage1-domain
```

### SFT Model

```text
https://huggingface.co/Devu758/novacart-qwen-sft
```

### Final DPO Model

```text
https://huggingface.co/Devu758/novacart-qwen-dpo
```

---

# 31. Challenges Faced

### Limited GPU Memory

Google Colab GPU memory is limited.

Solution:

```text
4-bit quantization
+
QLoRA
+
small physical batches
+
gradient accumulation
```

---

### Domain Knowledge vs Instruction Following

After non-instruction fine-tuning, the model had exposure to NovaCart policies but did not consistently map questions to the correct policy.

Instruction fine-tuning improved this behavior.

---

### Hallucination / Policy Mixing

After SFT, the model sometimes combined related but different policies.

Example:

```text
Refund:
5–7 business days

Delivery:
3–7 business days
```

The SFT model correctly mentioned the refund period but sometimes added the delivery period.

DPO preference examples were designed to penalize this type of response.

---

### DPO Adapter Training

During initial DPO training, the model showed:

```text
Trainable parameters = 0
```

which caused optimizer failure.

The issue was resolved by ensuring the LoRA policy adapter was active and trainable while keeping the reference adapter frozen.

This highlighted the importance of verifying trainable parameters before training.

---

# 32. Future Improvements

Possible improvements include:

- Use a much larger real-world customer-support dataset.
- Split datasets into training, validation, and test sets.
- Use assistant-only loss masking during SFT.
- Add automated evaluation metrics.
- Use LLM-as-a-judge evaluation.
- Evaluate hallucination rate.
- Add RAG for dynamically changing policies.
- Use a larger Qwen/Llama model.
- Experiment with larger LoRA rank.
- Tune DPO beta.
- Perform hyperparameter search.
- Merge adapter with the base model for deployment.
- Deploy the final model behind a FastAPI endpoint.
- Build a web interface using Streamlit or Gradio.

---

# 33. Key Interview Revision

Remember the entire project using:

```text
RAW TEXT
    ↓
Non-Instruction Fine-Tuning
    ↓
Domain Knowledge
    ↓
Q&A Dataset
    ↓
SFT
    ↓
Instruction Following
    ↓
Preference Dataset
    ↓
DPO
    ↓
Better Response Preference
```

### LoRA

```text
Freeze large model
+
train small adapters
```

### QLoRA

```text
Quantize base model
+
train LoRA
```

### SFT

```text
Prompt → desired answer
```

### DPO

```text
Prompt → chosen answer > rejected answer
```

### Gradient accumulation

```text
Simulates a larger batch without requiring the GPU
to hold the entire effective batch simultaneously.
```

### Adapter

```text
Small learned weight update stored separately
from the base model.
```

### Quantization

```text
Represent model weights with fewer bits to reduce
memory usage.
```

---

# 34. Interview Questions

### Why did you use QLoRA instead of full fine-tuning?

QLoRA dramatically reduces GPU-memory requirements by loading the base model in quantized form while training only a small set of LoRA parameters.

---

### Why perform non-instruction fine-tuning first?

It exposes the model to domain terminology, policies, and background knowledge before teaching explicit question-answer behavior.

---

### Why is SFT required after domain adaptation?

Knowing domain text does not automatically guarantee good instruction following. SFT explicitly teaches the mapping between user questions and desired responses.

---

### Why use DPO after SFT?

SFT teaches correct responses, while DPO can further improve which responses the model prefers by comparing better and worse alternatives.

---

### Why use gradient accumulation?

It provides a larger effective batch size without requiring enough GPU memory to process that full batch simultaneously.

---

### Why is the DPO learning rate lower than the SFT learning rate?

DPO is refining an already trained model, so smaller updates reduce the risk of destroying useful behavior learned during SFT.

---

### What is the difference between an adapter and a merged model?

An adapter stores only the learned LoRA changes and still requires the base model. A merged model permanently incorporates those changes into the base weights.

---

### Why did training loss sometimes increase from one step to the next?

Each mini-batch contains different examples and difficulty. Training loss is noisy, so the overall trend matters more than strictly decreasing loss at every step.

---

# 35. Final Outcome

This project demonstrates the complete lifecycle of creating a domain-specific LLM assistant using parameter-efficient fine-tuning:

```text
Base LLM
   ↓
Domain Adaptation
   ↓
Instruction Fine-Tuning
   ↓
Preference Alignment
   ↓
Evaluation
   ↓
Deployment-ready inference
```

The result is a reproducible NovaCart customer-support assistant with separate adapters for each stage and a clear comparison between the base, SFT, and DPO-aligned models.
