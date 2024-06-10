import os

import numpy as np

import torch
from datasets import  load_from_disk
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments,  AutoModelForCausalLM, AutoTokenizer
)


model = "./gpt-neo-res"
tokens = "./token-big"

context_length = 512
os.environ["WANDB_DISABLED"] = "true"


model = AutoModelForCausalLM.from_pretrained("./gpt-neo-res_big")
tokenizer = AutoTokenizer.from_pretrained("./gpt-neo-res_big")
device = torch.device("cpu")
model.to(device)


tokenized_dataset = load_from_disk(tokens)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./ev",
    num_train_epochs=20,
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
)



train_results = trainer.evaluate(tokenized_dataset["train"])
val_results = trainer.evaluate()
print("Loss")
print(f"Train: {train_results}")
print(f"Validation: {val_results}")

print("Perplexity:")

print(f"Train: {np.exp(train_results['eval_loss']):.2f}")
print(f"Validation: {np.exp(val_results['eval_loss']):.2f}")

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
)


train_results = trainer.evaluate(tokenized_dataset["train"])
val_results = trainer.evaluate()
#test_results = trainer.evaluate(tokenized_dataset["test"])
print("-----")
print("Loss")
print(f"Train: {train_results}")
print(f"Validation: {val_results}")

print("Perplexity:")

print(f"Train: {np.exp(train_results['eval_loss']):.2f}")
print(f"Validation: {np.exp(val_results['eval_loss']):.2f}")