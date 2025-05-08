import torch_directml

device = torch_directml.device() if torch_directml.is_available() else "cpu"

chroma_path = "./~chroma/"

embedding_name = "sentence-transformers/all-MiniLM-L6-v2"

model_name = "meta-llama/Llama-3.2-1B-Instruct"

questions = [
    "who is count von count",
    "how does count von krammzzz present himself",
    "where was irene adler born",
]
