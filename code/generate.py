import re, os

import numpy as np
import pandas as pd
import torch
#import wandb
from datasets import load_dataset, DatasetDict
from IPython.display import display, HTML
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)
from transformers.trainer_callback import PrinterCallback
from transformers.utils.notebook import NotebookProgressCallback

model_checkpoint = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
context_length = 128
model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint, pad_token_id=tokenizer.eos_token_id)
#device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"

def generate(
        prompt=None, max_length=100, greedy=True, model=model, tokenizer=tokenizer
):
    """Generar texto con sampling (greedy=False) o greedy search (greedy=True)

    prompt=None stands for beggining of sequence.

    NOTE si bien parece que GPT2 puede generar a partir de BOS token, la
    documentacion es poco clara. Ademas hicimos nuestro finetuning sin BOS token.
    Entonces solo vamos a usar la funcion pasandole un contexto.

    Ver:
    https://github.com/huggingface/transformers/issues/3311#issuecomment-601264426
    https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/generate_unconditional_samples.py#L60
    """
    do_sample = False if greedy else True
    # model.eval() to set dropout and batch normalization layers to evaluation mode before running inference
    model.eval()
    with torch.no_grad():
        if prompt:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            outputs = model.generate(input_ids, do_sample=do_sample, max_length=max_length,
                                     pad_token_id=tokenizer.eos_token_id)
        else:
            outputs = model.generate(do_sample=do_sample, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    # pad_token_id=tokenizer.eos_token_id to suppress warning
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

if __name__ == '__main__':
    model.load_state_dict(torch.load("./distilgpt2-finetuned/pytorch_model.bin"))
    model.eval()


    res_ = generate('dog is cute')
    print(res_)
