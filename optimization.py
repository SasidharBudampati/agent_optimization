import os
from openai import OpenAI
from dotenv import load_dotenv
import os
import sys

# Try Colab Secrets first
try:
    load_dotenv(dotenv_path=".secrets")

    api_key = os.getenv('OPENAI_KEY')
    print("✅ API key loaded from .secret")
except Exception:
    api_key = input("Enter your OpenAI API key: ")
    print("✅ API key set manually")

client = OpenAI(api_key=api_key)

## Import The Dataset: Customer Support Tickets
import random
from data import tickets

random.seed(42)

# Shuffle and split 80/20
random.shuffle(tickets)
split_idx = int(len(tickets) * 0.8)
train_data = tickets[:split_idx]
test_data = tickets[split_idx:]

print(f"Total tickets: {len(tickets)}")
print(f"Training set:  {len(train_data)} tickets")
print(f"Test set:      {len(test_data)} tickets")
print(f"\nCategories: {sorted(set(cat for _, cat in tickets))}")
print(f"\nSample ticket: \"{train_data[0][0]}\" → {train_data[0][1]}")

# setup the helper function to evaluate the model's performance on the test set
import time

CATEGORIES = ["billing", "technical", "account", "shipping"]

def classify_openai(ticket_text, model="gpt-4o-mini", few_shot_examples=None):
    """Classify a support ticket using OpenAI API."""
    system_prompt = (
        "You are a customer support ticket classifier. "
        "Classify each ticket into exactly one category: billing, technical, account, or shipping. "
        "Respond with ONLY the category name, nothing else."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Add few-shot examples if provided
    if few_shot_examples:
        for example_text, example_label in few_shot_examples:
            messages.append({"role": "user", "content": example_text})
            messages.append({"role": "assistant", "content": example_label})

    messages.append({"role": "user", "content": ticket_text})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=10
    )
    return response.choices[0].message.content.strip().lower()

def evaluate(predictions, actuals):
    """Calculate accuracy and per-category metrics."""
    correct = sum(p == a for p, a in zip(predictions, actuals))
    accuracy = correct / len(actuals)

    print(f"\n{'='*50}")
    print(f"Overall Accuracy: {accuracy:.1%} ({correct}/{len(actuals)})")
    print(f"{'='*50}")

    for cat in CATEGORIES:
        cat_indices = [i for i, a in enumerate(actuals) if a == cat]
        if cat_indices:
            cat_correct = sum(predictions[i] == actuals[i] for i in cat_indices)
            print(f"  {cat:12s}: {cat_correct}/{len(cat_indices)} correct")

    # Show misclassifications
    errors = [(actuals[i], predictions[i], i) for i in range(len(actuals)) if predictions[i] != actuals[i]]
    if errors:
        print(f"\nMisclassifications ({len(errors)}):")
        for actual, predicted, idx in errors[:5]:
            print(f"  \"{test_data[idx][0][:60]}...\"")
            print(f"    Expected: {actual} → Got: {predicted}")

    return accuracy

# Stage 1: Zero-Shot Classification with GPT-4o-mini

print("Stage 1: Zero-Shot GPT-4o-mini")
print("=" * 50)

start = time.time()
predictions_zeroshot = []

for ticket_text, _ in test_data:
    pred = classify_openai(ticket_text, model="gpt-4o-mini")
    predictions_zeroshot.append(pred)

elapsed_zeroshot = time.time() - start
actuals = [label for _, label in test_data]

acc_zeroshot = evaluate(predictions_zeroshot, actuals)
print(f"\n⏱️  Time: {elapsed_zeroshot:.1f}s for {len(test_data)} tickets")
print(f"💰 Model: gpt-4o-mini (cheapest tier)")

# One example per category from training data
few_shot_examples = []
seen_categories = set()
for text, label in train_data:
    if label not in seen_categories:
        few_shot_examples.append((text, label))
        seen_categories.add(label)
    if len(seen_categories) == 4:
        break

# Setting up for the few-shot examples being used
print("Few-shot examples:")
for text, label in few_shot_examples:
    print(f"  [{label:10s}] \"{text[:60]}...\"")
print()

start = time.time()
predictions_fewshot_mini = []

for ticket_text, _ in test_data:
    pred = classify_openai(ticket_text, model="gpt-4o-mini", few_shot_examples=few_shot_examples)
    predictions_fewshot_mini.append(pred)

elapsed_fewshot_mini = time.time() - start

# Evaluate the few-shot GPT-4o-mini model

print("Stage 2: Few-Shot GPT-4o-mini")
acc_fewshot_mini = evaluate(predictions_fewshot_mini, actuals)
print(f"\n⏱️  Time: {elapsed_fewshot_mini:.1f}s for {len(test_data)} tickets")
print(f"💰 Model: gpt-4o-mini + 4 examples")

# Stage 3: Few-Shot Classification with GPT-4o (more powerful, more expensive)

print("Stage 3: Few-Shot GPT-4o")
print("=" * 50)

start = time.time()
predictions_fewshot_4o = []

for ticket_text, _ in test_data:
    pred = classify_openai(ticket_text, model="gpt-4o", few_shot_examples=few_shot_examples)
    predictions_fewshot_4o.append(pred)

elapsed_fewshot_4o = time.time() - start

acc_fewshot_4o = evaluate(predictions_fewshot_4o, actuals)
print(f"\n⏱️  Time: {elapsed_fewshot_4o:.1f}s for {len(test_data)} tickets")
print(f"💰 Model: gpt-4o + 4 examples (5-10x cost of mini)")

print("Stage 3: Few-Shot GPT-4o")
print("=" * 50)

start = time.time()
predictions_fewshot_4o = []

for ticket_text, _ in test_data:
    pred = classify_openai(ticket_text, model="gpt-4o", few_shot_examples=few_shot_examples)
    predictions_fewshot_4o.append(pred)

elapsed_fewshot_4o = time.time() - start

acc_fewshot_4o = evaluate(predictions_fewshot_4o, actuals)
print(f"\n⏱️  Time: {elapsed_fewshot_4o:.1f}s for {len(test_data)} tickets")
print(f"💰 Model: gpt-4o + 4 examples (5-10x cost of mini)")

#Stage 4: LoRA Fine-Tuning (SmolLM2-1.7B)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# 4-bit quantization config — makes the model fit in ~2GB VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

print(f"\n✅ Model loaded!")
print(f"   Parameters: {model.num_parameters():,}")
print(f"   GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

########## LoRA configuration — we only train small adapter matrices

lora_config = LoraConfig(
    r=16,                          # Rank — higher = more capacity, more memory
    lora_alpha=32,                 # Scaling factor (usually 2x rank)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable parameters: {trainable:,} / {total:,}")
print(f"Trainable percentage: {100 * trainable / total:.2f}%")
print(f"\n💡 LoRA trains only {100 * trainable / total:.2f}% of the model!")

# Format data for fine-tuning
def format_for_training(ticket_text, label):
    """Format as a chat-style prompt."""
    prompt = (
        f"Classify this customer support ticket into one of these categories: "
        f"billing, technical, account, shipping.\n\n"
        f"Ticket: {ticket_text}\n\n"
        f"Category: {label}"
    )
    return prompt

# Create training dataset
from datasets import Dataset

train_texts = [format_for_training(text, label) for text, label in train_data]
train_dataset = Dataset.from_dict({"text": train_texts})

print(f"Training examples: {len(train_dataset)}")
print(f"\nSample formatted input:")
print("-" * 50)
print(train_texts[0])

#Training the LoRA model

from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir="./lora-support-classifier",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    warmup_steps=10,
    logging_steps=5,
    save_strategy="no",
    fp16=False, # Disable fp16 to prevent the AMP gradient scaler from causing conflicts on T4
    bf16=False,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    processing_class=tokenizer,
)

print("🚀 Starting fine-tuning...")
trainer.train()
print("\n✅ Fine-tuning complete!")

 # Evaluate the fine-tuned model on the test set

def classify_finetuned(ticket_text):
    """Classify using our fine-tuned model."""
    prompt = (
        f"Classify this customer support ticket into one of these categories: "
        f"billing, technical, account, shipping.\n\n"
        f"Ticket: {ticket_text}\n\n"
        f"Category:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

    # Extract just the category
    for cat in CATEGORIES:
        if cat in result:
            return cat
    return result.split()[0] if result else "unknown"


print("Stage 4: LoRA Fine-Tuned SmolLM2-1.7B")
print("=" * 50)

start = time.time()
predictions_lora = []

for ticket_text, _ in test_data:
    pred = classify_finetuned(ticket_text)
    predictions_lora.append(pred)

elapsed_lora = time.time() - start

acc_lora = evaluate(predictions_lora, actuals)
print(f"\n⏱️  Time: {elapsed_lora:.1f}s for {len(test_data)} tickets")
print(f"💰 Model: SmolLM2-1.7B + LoRA (self-hosted, no API cost)")

# Approximate costs per 1K tickets (based on typical token usage)
# ~150 tokens per ticket (prompt + completion)
results = {
    "Zero-Shot GPT-4o-mini": {
        "accuracy": acc_zeroshot,
        "time": elapsed_zeroshot,
        "cost_per_1k": "$0.02",
        "setup_effort": "None",
    },
    "Few-Shot GPT-4o-mini": {
        "accuracy": acc_fewshot_mini,
        "time": elapsed_fewshot_mini,
        "cost_per_1k": "$0.03",
        "setup_effort": "~1 hour",
    },
    "Few-Shot GPT-4o": {
        "accuracy": acc_fewshot_4o,
        "time": elapsed_fewshot_4o,
        "cost_per_1k": "$0.40",
        "setup_effort": "~1 hour",
    },
    "LoRA SmolLM2-1.7B": {
        "accuracy": acc_lora,
        "time": elapsed_lora,
        "cost_per_1k": "~$0 (self-hosted)",
        "setup_effort": "~1 week",
    },
}

print("\n" + "=" * 75)
print(f"{'Approach':<25} {'Accuracy':>10} {'Time':>10} {'Cost/1K':>15} {'Setup':>12}")
print("=" * 75)
for name, r in results.items():
    print(f"{name:<25} {r['accuracy']:>9.1%} {r['time']:>9.1f}s {r['cost_per_1k']:>15} {r['setup_effort']:>12}")
print("=" * 75)

## Print the comparison results in a more readable format

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

names = list(results.keys())
accuracies = [results[n]["accuracy"] for n in names]
times = [results[n]["time"] for n in names]
colors = ["#3B82F6", "#06B6D4", "#8B5CF6", "#F59E0B"]

# Accuracy comparison
axes[0].barh(names, accuracies, color=colors, height=0.6)
axes[0].set_xlim(0, 1.05)
axes[0].set_xlabel("Accuracy")
axes[0].set_title("Accuracy by Approach", fontweight="bold")
for i, v in enumerate(accuracies):
    axes[0].text(v + 0.01, i, f"{v:.1%}", va="center", fontweight="bold")

# Time comparison
axes[1].barh(names, times, color=colors, height=0.6)
axes[1].set_xlabel("Time (seconds)")
axes[1].set_title("Inference Time (test set)", fontweight="bold")
for i, v in enumerate(times):
    axes[1].text(v + 0.2, i, f"{v:.1f}s", va="center", fontweight="bold")

plt.tight_layout()
plt.savefig("comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n📊 Chart saved as comparison.png")

# Let's review the LoRA model's misclassifications in more detail and the size 

# Save and measure the LoRA adapter
model.save_pretrained("./lora-adapter")

import os
adapter_size = sum(
    os.path.getsize(os.path.join("./lora-adapter", f))
    for f in os.listdir("./lora-adapter")
    if f.endswith((".bin", ".safetensors"))
)
full_model_size = 1.7e9 * 2  # 1.7B params × 2 bytes (FP16)

print(f"Full model size:   ~{full_model_size/1e9:.1f} GB")
print(f"LoRA adapter size: {adapter_size/1e6:.1f} MB")
print(f"Compression ratio: {full_model_size/adapter_size:.0f}x smaller")
print(f"\n💡 You ship the {adapter_size/1e6:.1f}MB adapter, not the {full_model_size/1e9:.1f}GB model!")
print(f"   The base model is shared across all your fine-tuned variants.")