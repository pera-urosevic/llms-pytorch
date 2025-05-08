import os
import time
import uuid
import torch
import chromadb
from device import device
from transformers import AutoTokenizer, AutoModel

BOOK_PATH = "../~data/the adventures of sherlock holmes.txt"

EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

CHROMA_COLLECTION_NAME = os.path.splitext(os.path.basename(BOOK_PATH))[0].replace(
    " ", "_"
)
CHROMA_PERSIST_PATH = "./~chromadb"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

TOP_K = 3

try:
    embedding_tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL_PATH,
        local_files_only=True,
    )
    embedding_model = AutoModel.from_pretrained(
        EMBEDDING_MODEL_PATH,
        local_files_only=True,
    )
    embedding_model.to(device)
    embedding_model.eval()
    EMBEDDING_DIM = embedding_model.config.hidden_size
except Exception as e:
    print(f"Error loading embedding model/tokenizer: {e}")
    exit()

try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
except Exception as e:
    print(f"Error initializing ChromaDB client: {e}")
    exit()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def load_data(data_source):
    try:
        with open(data_source, "r", encoding="utf-8") as f:
            data = f.read()
        print(f"Loaded data successfully, size: {len(data)} characters.")
        return data
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_source}")
        return ""
    except Exception as e:
        print(f"Error loading data from {data_source}: {e}")
        return ""


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            sentence_end = max(
                text.rfind(".", start, end),
                text.rfind("?", start, end),
                text.rfind("!", start, end),
            )
            if sentence_end <= start:
                sentence_end = text.rfind(" ", start, end)
            if sentence_end > start and end - sentence_end < chunk_size * 0.3:
                end = sentence_end + 1
            elif (
                end > 0
                and text[end - 1].isalnum()
                and end < text_length
                and text[end].isalnum()
            ):
                space_before_end = text.rfind(" ", start, end)
                if space_before_end > start:
                    end = space_before_end + 1

        final_chunk = text[start:end].strip()
        if final_chunk:
            chunks.append(final_chunk)

        safe_overlap = min(overlap, chunk_size - 10)
        next_start = end - safe_overlap

        next_start = max(start + 1, next_start)

        if next_start <= start:
            next_start = end
            if next_start <= start:
                break

        start = next_start

    print(f"Split text into {len(chunks)} chunks.")
    return chunks


def create_or_get_chroma_collection():
    try:
        collection = chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return collection
    except Exception as e:
        print(
            f"Error getting or creating ChromaDB collection '{CHROMA_COLLECTION_NAME}': {e}"
        )
        return None


def prepare_data_pipeline():
    start_time = time.time()

    collection = create_or_get_chroma_collection()
    if collection is None:
        print("Failed to get or create ChromaDB collection. Aborting.")
        return None

    if collection.count() > 0:
        print(f"Collection already available, skipping data preparation.")
        return collection

    print(
        f"Collection '{CHROMA_COLLECTION_NAME}' is empty. Proceeding with data preparation."
    )

    data = load_data(BOOK_PATH)
    if not data:
        print("Data loading failed. Aborting.")
        return None

    chunks = split_text(data)
    if not chunks:
        print("Text splitting failed or produced no chunks. Aborting.")
        return None

    embeddings = get_embeddings(chunks)
    if not embeddings or len(embeddings) != len(chunks):
        print(
            f"Embedding generation failed or produced incorrect number of embeddings ({len(embeddings)} vs {len(chunks)} chunks). Aborting."
        )
        return None

    store_in_chroma(collection, chunks, embeddings)

    end_time = time.time()
    print(f"Data preparation finished in {end_time - start_time:.2f} seconds")
    return collection


def search_chroma(query, collection):
    if not collection:
        print("Cannot search: ChromaDB collection is not available.")
        return []
    if collection.count() == 0:
        print("Cannot search: ChromaDB collection is empty.")
        return []

    try:
        encoded_input = embedding_tokenizer(
            [query],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=CHUNK_SIZE,
        ).to(device)

        with torch.no_grad():
            model_output = embedding_model(**encoded_input)

        query_embedding = mean_pooling(model_output, encoded_input["attention_mask"])
        query_embedding_list = query_embedding.cpu().tolist()[0]
        results = collection.query(
            query_embeddings=[query_embedding_list],
            n_results=TOP_K,
            include=["documents"],
        )

        if results and results.get("documents") and results["documents"][0]:
            relevant_chunks = results["documents"][0]
            return relevant_chunks
        else:
            print("No relevant documents found in ChromaDB.")
            return []

    except Exception as e:
        print(f"Error searching ChromaDB: {e}")
        return []


def get_embeddings(texts, batch_size=32):
    if not texts:
        return []

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        try:
            encoded_input = embedding_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=CHUNK_SIZE,
            ).to(device)

            with torch.no_grad():
                model_output = embedding_model(**encoded_input)

            sentence_embeddings = mean_pooling(
                model_output, encoded_input["attention_mask"]
            )

            all_embeddings.extend(sentence_embeddings.cpu().tolist())
            if (i // batch_size + 1) % 10 == 0:
                print(
                    f"  Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}"
                )

        except Exception as e:
            print(f"Error generating embeddings for batch starting at index {i}: {e}")
            continue

    print(f"Embeddings generated successfully for {len(all_embeddings)} chunks.")
    return all_embeddings


def store_in_chroma(collection, chunks, embeddings):
    if not chunks or not embeddings or len(chunks) != len(embeddings):
        print("Error: Invalid data provided for ChromaDB insertion. No data inserted.")
        return

    ids = [str(uuid.uuid4()) for _ in chunks]

    try:
        print(
            f"Adding {len(chunks)} records to ChromaDB collection '{collection.name}'..."
        )
        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]
            batch_documents = chunks[i : i + batch_size]
            collection.add(
                ids=batch_ids, embeddings=batch_embeddings, documents=batch_documents
            )
            print(
                f"  Added batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}"
            )

        print(f"Successfully added records. Collection count: {collection.count()}")
    except Exception as e:
        print(f"Error adding data to ChromaDB: {e}")
