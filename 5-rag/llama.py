import torch
from device import device
from transformers import AutoModelForCausalLM, AutoTokenizer

LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

try:
    llm_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
    llm_model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_NAME)
    llm_model.to(device)
    llm_model.eval()
except Exception as e:
    print(f"Error loading LLM model/tokenizer: {e}")
    exit()


def format_llama3_prompt(query, context):
    if not context:
        print("Context is empty, exiting...")
        exit()

    system_message = "You are a helpful assistant. Answer the user's question based ONLY on the following context provided."
    context_string = "\n\n".join(context)
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}\n\nContext:\n{context_string}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|>"
    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt


def generate_response(query, context):
    prompt = format_llama3_prompt(query, context)
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
    terminators = [
        llm_tokenizer.eos_token_id,
        llm_tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    try:
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=llm_tokenizer.eos_token_id,
            )
        response_ids = outputs[0][inputs.input_ids.shape[-1] :]
        response = llm_tokenizer.decode(response_ids, skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I encountered an error while generating the response."
