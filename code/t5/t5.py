import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, load_from_disk, load_metric
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
import os
import numpy as np

# Constants
MODEL_CHECKPOINT = "t5-small"
DATA_FILE = "/Users/micacapart/Downloads/songs_english.csv"
SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.02
SAVE_STEPS = 500
EVAL_STEPS = 500

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT)
model = T5ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)

max_source_length = 512
max_target_length = 512

os.environ["WANDB_DISABLED"] = "true"

# Determine device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model.to(device)


def tokenize_fn(examples):
    task_prefix = "Write a song titled '{title}' in the {genre} genre:\n"
    inputs = [f"{task_prefix.format(title=title, genre=genre)}" for title, genre in zip(examples["title"], examples["tag"])]
    encoding = tokenizer(
        inputs,
        padding="max_length",
        max_length=max_source_length,
        truncation=True,
    )
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    with tokenizer.as_target_tokenizer():
        target_encoding = tokenizer(
            examples["lyrics"],
            padding="max_length",
            max_length=max_target_length,
            truncation=True,
        )
    labels = target_encoding["input_ids"]
    labels = [[label if label != tokenizer.pad_token_id else -100 for label in label_ids] for label_ids in labels]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def prepare_dataset(data_file):
    """Loads the dataset from a CSV file and splits it into train, validation, and test sets, reducing the size to 5% of the original dataset."""
    dataset = load_dataset("csv", data_files=data_file)["train"]

    sorted_dataset = dataset.sort("views", reverse=True)
    sorted_dataset = sorted_dataset.filter(lambda x: x['tag'] == 'pop')
    # print(sorted_dataset[:5])

    # Calculate the new size to be 5% of the original dataset size
    total_size = len(sorted_dataset)
    new_size = 5000

    # Select the top 5% of the dataset
    reduced_dataset = sorted_dataset.select(range(new_size))

    # Split the dataset into train, validation, and test sets
    total_size = len(reduced_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset = reduced_dataset.select(range(train_size))
    val_dataset = reduced_dataset.select(range(train_size, train_size + val_size))
    test_dataset = reduced_dataset.select(range(train_size + val_size, total_size))

    return DatasetDict({
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    })

def main():
    # Prepare the dataset
    dataset = prepare_dataset(DATA_FILE)

    # Tokenize the dataset
    tokenized_dataset_dir = './tokenized_dataset_t5_small'

    if os.path.exists(tokenized_dataset_dir):
        tokenized_dataset = load_from_disk(tokenized_dataset_dir)
        print("Tokenized dataset loaded successfully.")
    else:
        print(f"Tokenized dataset directory '{tokenized_dataset_dir}' does not exist.")
        tokenized_dataset = dataset.map(
            tokenize_fn, batched=True, num_proc=4,
            remove_columns=dataset["train"].column_names)
        tokenized_dataset.save_to_disk(tokenized_dataset_dir)

    # DataCollator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir='./t5_song_generator_small',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=2,
        save_steps=SAVE_STEPS,
        logging_dir='./logs',
        logging_strategy="steps",
        logging_steps=1000,
        fp16=False,
        push_to_hub=False,
        save_safetensors=False,
        report_to=None,  # Disable reporting to wandb
        metric_for_best_model="accuracy",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8
    )
    metric = load_metric("accuracy")
    def compute_metrics(p):
        pred, labels = p
        pred = np.argmax(pred, axis=2)
        pred = pred.flatten()
        labels = labels.flatten()
        return metric.compute(predictions=pred, references=labels)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model("./t5_song_generator_small")
    tokenizer.save_pretrained("./t5_song_generator_small")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
