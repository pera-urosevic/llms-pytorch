from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import (
    device,
    chroma_path,
    embedding_name,
)

book_path = "../~data/the adventures of sherlock holmes.txt"

loader = TextLoader(book_path, encoding="utf-8")
pages = loader.load()

chunk_size = 1500
chunk_overlap = 150

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)

documents = splitter.split_documents(pages)

embedding = HuggingFaceEmbeddings(
    model_name=embedding_name,
    model_kwargs={"device": device},
)

db = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    persist_directory=chroma_path,
)
