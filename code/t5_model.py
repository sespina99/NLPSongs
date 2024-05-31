import os
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, T5ForConditionalGeneration
)
# Define constants
MODEL_CHECKPOINT = "t5-base"
CONTEXT_LENGTH = 128
DATA_FILE = "/Users/micacapart/Downloads/song_lyrics_filtered 2.csv"  # Update with your dataset path
SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3


# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)



def tokenize_fn(examples):
    """Tokenizes `lyrics` from examples in a dataset."""
    titles = examples["title"]
    genres = examples["tag"]
    lyrics = examples["lyrics"]

    inputs = [f"--TITLE--: {title} --GENRE--:{genre} --LYRICS--:{lyric}" for title, genre, lyric in
              zip(titles, genres, lyrics)]
    outputs = tokenizer(
        inputs,
        truncation=True,
        max_length=CONTEXT_LENGTH,
        padding="max_length",
        return_tensors="pt"  # Return PyTorch tensors
    )
    return outputs


def prepare_dataset(data_file):
    """Loads the dataset from a CSV file and splits it into train, validation, and test sets."""
    dataset = load_dataset("csv", data_files=data_file)

    # Shuffle and split the dataset
    dataset = dataset["train"].shuffle(seed=SEED)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, total_size))

    return DatasetDict({
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    })


def main():
    # Prepare the dataset
    dataset = prepare_dataset(DATA_FILE)

    print(f"Total dataset size: {len(dataset['train']) + len(dataset['val']) + len(dataset['test'])}")
    print(f"Training set size: {len(dataset['train'])}")
    print(f"Validation set size: {len(dataset['val'])}")
    print(f"Test set size: {len(dataset['test'])}")

    # Tokenize the dataset
    # tokenized_dataset = dataset.map(
    #     tokenize_fn, batched=True, num_proc=4,
    #     remove_columns=dataset["train"].column_names
    # )
    tokenized_dataset_dir = './tokenized_dataset_t5'

    if os.path.exists(tokenized_dataset_dir):
        tokenized_dataset = load_from_disk(tokenized_dataset_dir)
        print("Tokenized dataset loaded successfully.")
    else:
        print(f"Tokenized dataset directory '{tokenized_dataset_dir}' does not exist.")
        tokenized_dataset = dataset.map(
            tokenize_fn, batched=True, num_proc=4,
            remove_columns=dataset["train"].column_names)
        tokenized_dataset.save_to_disk(tokenized_dataset_dir)

    print("Tokenized dataset:")
    print(tokenized_dataset)

    # Load the model
    model = T5ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=500,
        lr_scheduler_type="linear",
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_dir='./logs',
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"]
    )

    trainer.train()  # Start training
    trainer.save_model()  # Save the final model


if __name__ == '__main__':
    main()
