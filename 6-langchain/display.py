import re


def display_result(result: str):
    content = result.replace("<|begin_of_text|>", "").replace("<|end_of_text|>", "")
    chunks = content.split("<|eot_id|>")
    for chunk in chunks:
        match = re.search(
            r"^<\|start_header_id\|>(.*)<\|end_header_id\|>\n(.*)$",
            chunk,
            flags=re.DOTALL,
        )
        if match:
            type = match.group(1)
            text = match.group(2)
            if type == "system":
                context = re.search(r"CONTEXT:(.*)$", text, flags=re.DOTALL)
                if context:
                    preview = context.group(1).strip().replace("\n", " ")[:500]
                    print(f"Context:\n{preview}...\n")
            if type == "user":
                print(f"User: {text}\n")
            if type == "assistant":
                print(f"Assistant: {text}\n")
