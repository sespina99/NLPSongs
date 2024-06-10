import os
import re

import torch
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)

model_checkpoint = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
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
    genres = examples["tag"]
    lyrics = examples["lyrics"]

    # Combine title, genre, and lyrics into a single string for each example
    inputs = [f"Title: {title} | Genre: {genre}\nLyrics: {lyric}" for title, genre, lyric in zip(titles, genres, lyrics)]

    # Tokenize the combined strings
    outputs = tokenizer(
        inputs,
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
        padding="max_length"
    )
    return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

def train_model():
    dataset = load_dataset("csv", data_files="/Users/micacapart/Downloads/songs_english.csv")["train"]

    sorted_dataset = dataset.sort("views", reverse=True)
    sorted_dataset = sorted_dataset.filter(lambda x: x['tag'] == 'pop')
    

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

    tokenizer.pad_token = tokenizer.eos_token
    small_dataset = small_dataset.map(clean_text)

    tokenized_dataset_dir = './token-small'

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

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, pad_token_id=tokenizer.eos_token_id)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    pretrained_model_name = model_checkpoint.split("/")[-1]
    finetuned_model_name = f"{pretrained_model_name}-finetuned"
    save = './gpt-neo-res2'

    training_args = TrainingArguments(
        output_dir=save,
        num_train_epochs=40,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-4,
        weight_decay=0.5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        do_eval=True,
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=2,
        save_steps=2000,
        logging_dir='./logs',
        logging_strategy="steps",
        logging_steps=1000,
        fp16=False,
        push_to_hub=False,
        report_to=None,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        metric_for_best_model="eval_loss"
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

    trainer.save_model(save)
    tokenizer.save_pretrained(save)

if __name__ == '__main__':
    train_model()
