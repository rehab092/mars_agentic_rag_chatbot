
# ============================================================
# TEST BGE-M3 + CHROMADB SEMANTIC SEARCH
# ============================================================

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
import torch


BASE_DIR = Path(__file__).resolve().parent

CHROMA_PATH = BASE_DIR / "chroma_storage"
COLLECTION_NAME = "mars_rag_bge_m3"

MODEL_NAME = "BAAI/bge-m3"


def main():

    # Connect to the existing persistent database.
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None
    )

    print("Total stored chunks:", collection.count())

    if collection.count() == 0:
        print("The collection is empty.")
        return

    # Load the SAME embedding model used for documents.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    model = SentenceTransformer(
        MODEL_NAME,
        device=device
    )

    # Ask a question interactively.
    question = input("\nEnter your question: ").strip()

    if not question:
        print("No question provided.")
        return

    # Convert the question into a 1024-dimensional vector.
    query_vector = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).tolist()

    if len(query_vector) != 1024:
        raise ValueError("Unexpected query embedding size.")

    # Search ChromaDB for similar chunks.
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(10, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    print("\n========== SEARCH RESULTS ==========")

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for index, (text, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):

        print(f"\nResult {index}")
        print("Cosine distance:", round(distance, 4))
        print("Document:", metadata.get("document_id"))
        print("Page:", metadata.get("page_number"))
        print("Content type:", metadata.get("content_type"))
        print("Source:", metadata.get("source_file"))

        print("\nRetrieved content:")
        print(text)

        print("-" * 60)


if __name__ == "__main__":
    main()



