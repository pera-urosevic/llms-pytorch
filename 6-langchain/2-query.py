from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import (
    device,
    chroma_path,
    embedding_name,
    questions,
)

query = questions[1]

persist_directory = chroma_path

embedding = HuggingFaceEmbeddings(
    model_name=embedding_name,
    model_kwargs={"device": device},
)

db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
docs = db.similarity_search_with_score(query, k=3)


for doc in docs:
    print(f"{doc}\n")
