import torch_directml
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

device = torch_directml.device() if torch_directml.is_available() else "cpu"

model_path = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path)
model.to(device)

pipe = pipeline(
    "text-generation",
    model=model,
    device=device,
    tokenizer=tokenizer,
    temperature=0.7,
    max_new_tokens=512,
    do_sample=True,
    top_p=0.9,
)

system_prompt = "You are a helpful, polite, and concise assistant."


def format_prompt(messages):
    prompt = f"<|system|>\n{system_prompt}\n"
    for message in messages:
        if message["role"] == "user":
            prompt += "<|user|>\n" + message["content"] + "\n"
        elif message["role"] == "assistant":
            prompt += "<|assistant|>\n" + message["content"] + "\n"
    prompt += "<|assistant|>\n"
    return prompt


chat_history = []

while True:
    user_input = input("User: ")
    if user_input.lower() == "":
        break

    chat_history.append({"role": "user", "content": user_input})

    prompt = format_prompt(chat_history)
    result = pipe(prompt, pad_token_id=pipe.tokenizer.eos_token_id)
    response = result[0]["generated_text"]
    assistant_message = response[len(prompt) :].strip().split("<|user|>")[0].strip()
    print(f"Assistant: {assistant_message}\n")

    chat_history.append({"role": "assistant", "content": assistant_message})
