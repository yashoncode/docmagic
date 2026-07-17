"""QLoRA fine-tune of Llama 3.2 3B on logistics Q&A. Runs on free Colab T4.

Usage on Colab:  !pip install unsloth  then  !python finetune_colab.py
Expects dataset.jsonl next to this file (see dataset_template.jsonl).
"""

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16)

dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
dataset = dataset.map(
    lambda ex: {
        "text": tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
    }
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        output_dir="outputs",
    ),
)
trainer.train()

model.save_pretrained("logichat-lora")
# To publish: model.push_to_hub("YOUR_HF_USERNAME/logichat-llama-3.2-3b", token="hf_...")
# To run locally via Ollama: model.save_pretrained_gguf("logichat-gguf", tokenizer)
