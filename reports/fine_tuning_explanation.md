
# Fine-Tuning Theory and Configuration

## 1. Why Full Fine-Tuning Is Expensive

Full fine-tuning updates all parameters of a large language model.
This requires significant GPU memory because the system must store the
model weights, gradients, optimizer states, and intermediate activations.

For large models, this can require multiple high-memory GPUs.

## 2. LoRA

LoRA stands for Low-Rank Adaptation.

Instead of updating all model parameters, LoRA freezes the original model
weights and introduces small trainable low-rank matrices into selected
layers.

Conceptually:

W_effective = W_base + Delta_W

where Delta_W is represented using two smaller matrices.

This greatly reduces the number of trainable parameters and GPU memory usage.

## 3. QLoRA

QLoRA combines quantization with LoRA.

In this project, the Qwen2.5-1.5B model was loaded using 4-bit quantization.
The original model remained quantized while small LoRA adapter parameters
were trained.

This makes fine-tuning possible on limited GPU hardware such as a Google
Colab T4.

## 4. Non-Instruction Fine-Tuning

Non-instruction fine-tuning uses raw domain text rather than question-answer
pairs.

Its goal is to expose the model to domain terminology, policies, facts,
writing style, and background knowledge.

In this project, NovaCart customer-support policy paragraphs were used for
domain adaptation.

## 5. Instruction Fine-Tuning / SFT

Supervised Fine-Tuning trains the model using instruction-response pairs.

Example:

Question:
How long does a NovaCart refund take?

Answer:
Approved refunds generally take 5 to 7 business days.

SFT teaches the model how to respond to user questions using domain-specific
knowledge.

## 6. DPO

DPO stands for Direct Preference Optimization.

Instead of providing only one target response, each training example contains:

- Prompt
- Chosen response
- Rejected response

The model learns to prefer the chosen response over the rejected response.

Example:

Prompt:
How long does a refund take?

Chosen:
Approved refunds take 5 to 7 business days.

Rejected:
Refunds take 5 to 7 days plus another 3 to 7 days for delivery.

## 7. SFT vs DPO

SFT teaches:
Question -> Correct response

DPO teaches:
Preferred response > Less preferred response

SFT primarily teaches task behavior, while DPO is used to refine response
quality, relevance, correctness, tone, and preference alignment.

## 8. Hyperparameters Used

### LoRA / QLoRA
- Rank (r): 16
- Alpha: 16
- Dropout: 0
- Quantization: 4-bit

### Non-Instruction Fine-Tuning
- Epochs: 2
- Batch size: 2
- Gradient accumulation: 4
- Effective batch size: 8
- Learning rate: 2e-4

### Instruction Fine-Tuning
- Training examples: 120
- Epochs: 2
- Batch size: 2
- Gradient accumulation: 4
- Effective batch size: 8
- Learning rate: 2e-4

### DPO
- Preference examples: 50
- Epochs: 2
- Batch size: 1
- Gradient accumulation: 4
- Effective batch size: 4
- Learning rate: 5e-6
- Beta: 0.1

## 9. Why Gradient Accumulation Was Used

Gradient accumulation allows multiple small batches to contribute gradients
before performing one optimizer update.

Effective batch size:

batch_size x gradient_accumulation_steps

For example:

2 x 4 = 8

This allows a larger effective batch size while keeping GPU memory usage low.

## 10. Training Loss

Training loss measures how different the model's predicted token probability
distribution is from the expected next tokens.

Cross-entropy loss is commonly used.

Lower loss generally indicates better fit to training data, although low
training loss alone does not guarantee good generalization.

## 11. Adapter vs Merged Model

The LoRA adapters were stored separately from the Qwen base model.

At inference time, the base model is loaded together with the trained adapter.

This keeps the saved fine-tuned artifact small.

A merged standalone model could be created later if required for deployment.
