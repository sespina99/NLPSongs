import os
import re
import numpy as np

import torch
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, pipeline
)


model = "./t5_small"
tokens = "./tokenized_dataset_t5_small"

context_length = 512
os.environ["WANDB_DISABLED"] = "true"

tokenizer = T5Tokenizer.from_pretrained(model)
model = T5ForConditionalGeneration.from_pretrained(model)
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



# train_results = trainer.evaluate(tokenized_dataset["train"])
# val_results = trainer.evaluate()
# print("Loss")
# print(f"Train: {train_results}")
# print(f"Validation: {val_results}")
#
# print("Perplexity:")
#
# print(f"Train: {np.exp(train_results['eval_loss']):.2f}")
# print(f"Validation: {np.exp(val_results['eval_loss']):.2f}")

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
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