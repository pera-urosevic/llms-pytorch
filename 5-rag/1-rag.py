from db import prepare_data_pipeline, search_chroma
from llama import generate_response

if __name__ == "__main__":
    rag_collection = prepare_data_pipeline()

    if rag_collection is None:
        print("Failed to prepare data or retrieve collection, exiting...")
        exit()

    print("Enter your query (or press Enter to quit)\n")

    try:
        while True:
            user_query = input("User: ")
            if not user_query:
                break

            retrieved_context = search_chroma(user_query, rag_collection)
            final_response = generate_response(user_query, retrieved_context)

            print(f"Assistant: {final_response}")
            print("\n---\n")

    except KeyboardInterrupt:
        print("\nInterrupt encountered, exiting...")
    finally:
        print("\nBye!")
