import os
import re

from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)


model_checkpoint = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
context_length = 512
def clean_text(example):
    """Corrige caracteres raros segun la doc de yelp
    """
    texto = re.sub(r'\n', '\n', example["lyrics"]) # real newlines
    texto = re.sub(r'\"', '"', texto) # comillas de verdad
    example["text"] = texto
    return example
def tokenize_fn(examples):
    titles = examples["title"]
    genres = examples["tag"]
    lyrics = examples["lyrics"]

    # Combinar título, género y letra en una sola cadena para cada ejemplo
    inputs = [f"--TITLE--: {title} --GENRE--:{genre} --LYRICS--:{lyric}" for title, genre, lyric in zip(titles, genres, lyrics)]

    # Tokenizar las cadenas combinadas
    outputs = tokenizer(
        inputs,
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
        padding="max_length"
    )
    return {"input_ids": outputs["input_ids"]}


def train_model():
    dataset = load_dataset("csv", data_files="/Users/micacapart/Downloads/song_lyrics_filtered 2.csv")
    dataset = dataset.shuffle()

    aux = dataset.shape['train'][0]
    training_index = int(aux * 0.8)
    validation_index = training_index + int(aux * 0.1)

    small_dataset = DatasetDict(
        train=dataset["train"].shuffle(seed=33).select(range(0, training_index)),
        val=dataset["train"].shuffle(seed=33).select(range(training_index, validation_index)),
        test=dataset["train"].shuffle(seed=33).select(range(validation_index, aux)),
    )

    print(dataset.shape['train'][0])
    print(*dataset["train"].features.items(), sep="\n")
    tokenizer.pad_token = tokenizer.eos_token
    small_dataset = small_dataset.map(clean_text)

    tokenized_dataset_dir = './tokenized_dataset'

    if os.path.exists(tokenized_dataset_dir):
        tokenized_dataset = load_from_disk(tokenized_dataset_dir)
        print("Tokenized dataset loaded successfully.")
    else:
        print(f"Tokenized dataset directory '{tokenized_dataset_dir}' does not exist.")
        tokenized_dataset = small_dataset.map(
            tokenize_fn, batched=True, num_proc=4,
            remove_columns=small_dataset["train"].column_names)
        tokenized_dataset.save_to_disk('./tokenized_dataset')

    print(small_dataset)

    print(tokenized_dataset)


    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint, pad_token_id=tokenizer.eos_token_id)


    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    pretrained_model_name = model_checkpoint.split("/")[-1]
    finetuned_model_name = f"{pretrained_model_name}-finetuned"

    training_args = TrainingArguments(
        finetuned_model_name,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=5e-4,
        weight_decay=0.1,  # forma de regularizacion (restringe el tamaño de updates de SGD)
        warmup_ratio=0.1,  # # warmup evita divergencia de loss en primeros steps (10%)
        lr_scheduler_type="cosine",
        do_eval=True,  # eval en validation set
        gradient_accumulation_steps=1,  # acumula gradientes por N steps --> update cada N*32 samples
        # sirve cuando batches grandes no entran en memoria y tenemos muchos samples
        evaluation_strategy="steps",  # eval en validation set
        eval_steps=50,
        save_strategy="steps",
        load_best_model_at_end=True,  # conserva mejor modelo segun eval loss
        save_total_limit=2,  # save max 2 models including best one
        save_steps=50,  # checkpoint model every N steps
        logging_dir='./logs',  # logging
        logging_strategy="steps",
        logging_steps=1,
        fp16=False,  # float16 en training (only on CUDA)
        push_to_hub=False,
        #    report_to="wandb",  # enable logging to W&B
        save_safetensors=False  # por un bug
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],  # .select(range(0, 128)),
        eval_dataset=tokenized_dataset["val"],  # .select(range(0, 128)),
    )

    trainer.save_model()

if __name__ == '__main__':
    train_model()
