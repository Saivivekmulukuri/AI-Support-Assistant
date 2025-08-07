import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# --- CONFIGURATION ---
BASE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_NAME = "databricks/databricks-dolly-15k"
NEW_ADAPTER_NAME = "qwen3-4b-samsung-support-adapter"

# --- MAIN SCRIPT ---

def main():
    print("Starting the fine-tuning process...")

    # 1. Configure 4-bit quantization for memory efficiency
    print("1. Configuring quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # 2. Load the base model with quantization
    print(f"2. Loading base model: {BASE_MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto", # Automatically use available GPUs
        trust_remote_code=True,
    )
    model.config.use_cache = False # Recommended for fine-tuning
    model.config.pretraining_tp = 1

    # 3. Load the tokenizer
    print("3. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Set padding token for training
    tokenizer.padding_side = "right"

    # 4. Configure LoRA (Low-Rank Adaptation)
    print("4. Configuring LoRA...")
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    
    model = get_peft_model(model, lora_config)
    print("LoRA adapter added to the model.")

    # 5. Load and prepare the dataset
    print(f"5. Loading and preparing dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train")
    
    categories_to_keep = ['open_qa', 'closed_qa', 'information_extraction', 'summarization']
    dataset = dataset.filter(lambda example: example['category'] in categories_to_keep)
    dataset = dataset.shuffle(seed=42).select(range(2000))

    print(f"Dataset prepared with {len(dataset)} samples.")

    # Define a formatting function that uses the Qwen chat template
    def format_for_qwen(example):
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": f"{example['instruction']}\n\n{example['context']}"},
            {"role": "assistant", "content": example['response']}
        ]
        # We apply the template but don't tokenize, SFTTrainer will handle it
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # 6. Configure Training Arguments
    print("6. Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir="./results_qwen3",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=50,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    # 7. Initialize the Trainer
    print("7. Initializing the SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=training_args,
        formatting_func=format_for_qwen, # Use our custom formatting function
    )

    # 8. Start Fine-Tuning
    print("\n" + "="*20)
    print("🚀 STARTING FINE-TUNING (QWEN3-4B MODEL) 🚀")
    print("="*20 + "\n")
    trainer.train()
    print("\n" + "="*20)
    print("✅ FINE-TUNING COMPLETE ✅")
    print("="*20 + "\n")

    # 9. Save the Fine-Tuned Adapter
    print(f"9. Saving adapter to '{NEW_ADAPTER_NAME}'...")
    trainer.model.save_pretrained(NEW_ADAPTER_NAME)
    print("Adapter saved successfully. You can now use it in app.py.")


if __name__ == "__main__":
    main()