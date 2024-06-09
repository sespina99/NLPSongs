import os
import re

import torch
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq,
    Trainer, TrainingArguments
)

model_checkpoint = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
context_length = 512
os.environ["WANDB_DISABLED"] = "true"

def clean_text(example):
    """Corrects characters according to the doc of yelp."""
    text = re.sub(r'\n', '\n', example["lyrics"])  # Real newlines
    text = re.sub(r'\"', '"', text)  # Real quotes
    example["text"] = text
    return example


def tokenize_fn(examples):
    titles = examples["title"]
    lyrics = examples["lyrics"]

    # Combine title and lyrics into a single string for each example
    inputs = [f"generate lyrics for: {title}" for title in titles]

    # Tokenize the combined strings
    model_inputs = tokenizer(
        inputs,
        truncation=True,
        max_length=context_length,
        padding="max_length"
    )

    # Set up the targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            lyrics,
            truncation=True,
            max_length=context_length,
            padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_model():
    dataset = load_dataset("csv", data_files="/Users/micacapart/Downloads/songs_english.csv")["train"]

    sorted_dataset = dataset.sort("views", reverse=True)
    sorted_dataset = sorted_dataset.filter(lambda x: x['tag'] == 'pop')
    # print(sorted_dataset[:5])

    # Calculate the new size to be 5% of the original dataset size
    total_size = len(sorted_dataset)
    new_size = 1000

    # Select the top 5% of the dataset
    reduced_dataset = sorted_dataset.select(range(new_size))

    aux = len(reduced_dataset)
    training_index = int(aux * 0.8)
    validation_index = training_index + int(aux * 0.1)

    small_dataset = DatasetDict(
        train=reduced_dataset.shuffle(seed=33).select(range(0, training_index)),
        val=reduced_dataset.shuffle(seed=33).select(range(training_index, validation_index)),
        test=reduced_dataset.shuffle(seed=33).select(range(validation_index, aux)),
    )

    print(f"Dataset size: {len(reduced_dataset)}")
    print(small_dataset)

    small_dataset = small_dataset.map(clean_text)

    tokenized_dataset_dir = './tokenized_dataset_t5_small2'

    if os.path.exists(tokenized_dataset_dir):
        tokenized_dataset = load_from_disk(tokenized_dataset_dir)
        print("Tokenized dataset loaded successfully.")
    else:
        print(f"Tokenized dataset directory '{tokenized_dataset_dir}' does not exist.")
        tokenized_dataset = small_dataset.map(
            tokenize_fn, batched=True, num_proc=4,
            remove_columns=small_dataset["train"].column_names)
        tokenized_dataset.save_to_disk(tokenized_dataset_dir)

    print(small_dataset)
    print(tokenized_dataset)

    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    pretrained_model_name = model_checkpoint.split("/")[-1]
    finetuned_model_name = f"{pretrained_model_name}-finetuned"

    save_dir = f"./t5_small"

    training_args = TrainingArguments(
        save_dir,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-4,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        do_eval=True,
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=2,
        save_steps=200,
        logging_dir='./logs',
        logging_strategy="steps",
        logging_steps=200,
        fp16=False,
        push_to_hub=False,
        report_to=None,
        use_mps_device=True
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
    )

    trainer.train()
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == '__main__':
    train_model()
