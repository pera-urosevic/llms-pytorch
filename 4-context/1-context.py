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


def ask(title, chat_history, question):
    chat_history.append({"role": "user", "content": question})
    prompt = format_prompt(chat_history)
    result = pipe(prompt, pad_token_id=pipe.tokenizer.eos_token_id)
    response = result[0]["generated_text"]
    assistant_message = response[len(prompt) :].strip().split("<|user|>")[0].strip()
    print(f"[{title}] Answer: {assistant_message}\n")


ask(
    "Without Context",
    [],
    "Who is Count Von Krammzzz?",
)

ask(
    "With Context",
    [
        {
            "role": "assistant",
            "content": "An excerpt from the book Sherlock Holmes: “You may address me as the Count Von Krammzzz, a Bohemian nobleman. I understand that this gentleman, your friend, is a man of honour and discretion, whom I may trust with a matter of the most extreme importance. If not, I should much prefer to communicate with you alone.”",
        },
    ],
    "Who is Count Von Krammzzz?",
)
