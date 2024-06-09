import os
import re

import torch
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)


model_checkpoint = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
context_length = 512
os.environ["WANDB_DISABLED"] = "true"

def clean_text(example):
    """Corrects characters according to the doc of yelp."""
    text = re.sub(r'\n', '\n', example["lyrics"])  # Real newlines
    text = re.sub(r'\"', '"', text)  # Real quotes
    example["text"] = text
    return example

def preprocess_function(examples):
    titles = examples["title"]
    genres = examples["tag"]
    lyrics = examples["lyrics"]

    # Combine title, genre, and lyrics into a single string for each example
    inputs = [f"Title: {title} | Genre: {genre}\nLyrics: {lyric}" for title, genre, lyric in
              zip(titles, genres, lyrics)]
    return tokenizer(inputs,
                     truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
        padding="max_length")
block_size = 128
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def train_model():
    dataset = load_dataset("csv", data_files="/Users/micacapart/Downloads/songs_english.csv")["train"]

    sorted_dataset = dataset.sort("views", reverse=True)
    sorted_dataset = sorted_dataset.filter(lambda x: x['tag'] == 'pop')
    sorted_dataset = sorted_dataset.shuffle()
    # print(sorted_dataset[:5])

    # Calculate the new size to be 5% of the original dataset size
    total_size = len(sorted_dataset)
    new_size = 3000

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
            preprocess_function, batched=True, num_proc=4,
            remove_columns=small_dataset["train"].column_names)
        tokenized_dataset.save_to_disk(tokenized_dataset_dir)
    lm_dataset = tokenized_dataset

    print(small_dataset)
    print(tokenized_dataset)

    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2", pad_token_id=tokenizer.eos_token_id)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    pretrained_model_name = model_checkpoint.split("/")[-1]
    finetuned_model_name = f"{pretrained_model_name}-finetuned"
    save = './model'

    training_args = TrainingArguments(
        output_dir="./model",
        num_train_epochs=20,
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["val"],
    )

    trainer.train()

    trainer.save_model(save)
    tokenizer.save_pretrained(save)

if __name__ == '__main__':
    train_model()
