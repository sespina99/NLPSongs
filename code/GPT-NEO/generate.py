from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer
model_name = "./gpt-neo-res_big"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def generate_lyrics(prompt):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate the lyrics
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs['attention_mask'],
        max_length=500,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.95,
        do_sample=True,
        early_stopping=True,
        num_beams=5,
    )

    # Decode the generated tokens to text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Example prompt
prompt = "Write the first verse for a song titled 'Summer Love' in the genre of pop."
lyrics = generate_lyrics(prompt)
print()
print(lyrics)
