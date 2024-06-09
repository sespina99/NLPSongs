import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Define constants
MODEL_DIR = "./t5_small"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512

# Load the trained model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_song(title, genre):
    # Prepare the input text
    task_prefix = "Generate song from title and genre: "
    input_text = f"generate lyrics for: {title}"
    input_text = f"Generate a lyrics with titled '{title}' in the {genre} genre"

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).to(device)

    # Generate the output
    outputs = model.generate(
        input_ids=input_ids,
        max_length=MAX_TARGET_LENGTH,
        top_k=50,
        do_sample=True,
        top_p=0.92,
        num_beams=5,  # Beam search for better quality (you can adjust this)
        early_stopping=True,
        no_repeat_ngram_size=2,
    )

    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

# Example usage
title = "Summer Love"
genre = "Pop"
generated_song = generate_song(title, genre)
print("Generated Song Lyrics:")
print(generated_song)
