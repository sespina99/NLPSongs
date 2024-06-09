from transformers import pipeline
prompt = 'Title "Summer Love" | Genre "Pop" Lyrics'

generator = pipeline("text-generation", model="./model")
res = generator(prompt, max_length=300,
             truncation=True,
                no_repeat_ngram_size=2,
   )
print(res)
