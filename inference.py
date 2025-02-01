import torch
from transformers import AutoTokenizer
MODEL_NAME = "fine-tuned-gpt"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print("Inference initialized.")