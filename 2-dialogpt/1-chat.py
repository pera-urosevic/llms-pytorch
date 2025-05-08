import torch_directml
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch_directml.device() if torch_directml.is_available() else "cpu"

model_name = "microsoft/DialoGPT-medium"

model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.chat_template = """
{% for message in messages %}
{% if message['role'] == 'user' %}
User: {{ message['content'].strip() }}
{% elif message['role'] == 'bot' %}
Bot: {{ message['content'].strip() }}
{% endif %}
{% endfor %}
Bot:
"""

messages = [
    {"role": "user", "content": "Hi!"},
    {"role": "bot", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": "Tell me the capital of France."},
]

formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(formatted_prompt, return_tensors="pt")

inputs = {k: v.to(device) for k, v in inputs.items()}

output = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
