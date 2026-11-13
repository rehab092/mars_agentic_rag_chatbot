
# ============================================================
# STORE BGE-M3 EMBEDDINGS IN CHROMADB
# ============================================================
#
# INPUT:  embeddings/embeddings.jsonl
# OUTPUT: chroma_storage/
#
# The collection is REBUILT on every run, so chunks from
# earlier runs (old chunking settings, re-described images,
# removed pages) can never linger as stale duplicates.
# Records with a duplicate ID or duplicate text are skipped.
# ============================================================

import json
import math
from pathlib import Path

import chromadb


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "embeddings" / "embeddings.jsonl"

CHROMA_PATH = BASE_DIR / "chroma_storage"

COLLECTION_NAME = "mars_rag_bge_m3"

EMBEDDING_MODEL = "BAAI/bge-m3"
EXPECTED_DIMENSIONS = 1024
BATCH_SIZE = 100

# Fields stored as Chroma metadata (scalars only).
METADATA_FIELDS = [
    "record_id", "document_id", "content_type", "source_file",
    "section", "page_number", "page_start", "page_end",
    "chunk_index", "total_chunks_in_record", "token_count",
    "image_path", "image_url",
]


# ------------------------------------------------------------
# INITIALIZE CHROMADB
# ------------------------------------------------------------

def create_collection():

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    existing = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing:
        old_count = client.get_collection(COLLECTION_NAME).count()
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted old collection ({old_count} records).")

    return client.create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={
            "embedding_model": EMBEDDING_MODEL,
            "dimensions": EXPECTED_DIMENSIONS
        },
        embedding_function=None
    )


# ------------------------------------------------------------
# VALIDATE AN EMBEDDING RECORD
# ------------------------------------------------------------

def validate_record(record):

    for field in ("chunk_id", "text", "embedding"):
        if field not in record:
            raise ValueError(f"Missing required field: {field}")

    vector = record["embedding"]

    if len(vector) != EXPECTED_DIMENSIONS:
        raise ValueError(
            f"Expected {EXPECTED_DIMENSIONS} dimensions, got {len(vector)}"
        )

    if not all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in vector
    ):
        raise ValueError("Embedding contains invalid numerical values.")

    if not record["text"].strip():
        raise ValueError("Chunk text is empty.")

    if record.get("embedding_model") != EMBEDDING_MODEL:
        raise ValueError("Unexpected embedding model.")


def build_metadata(record):

    metadata = {"embedding_model": EMBEDDING_MODEL}

    for field in METADATA_FIELDS:
        value = record.get(field)
        # Chroma accepts str / int / float / bool only.
        if value is None or value == "":
            continue
        metadata[field] = value if isinstance(value, (int, float, bool)) else str(value)

    return metadata


# ------------------------------------------------------------
# STORE ONE BATCH (records are already validated)
# ------------------------------------------------------------

def store_batch(collection, records):

    collection.add(
        ids=[r["chunk_id"] for r in records],
        embeddings=[r["embedding"] for r in records],
        documents=[r["text"] for r in records],
        metadatas=[build_metadata(r) for r in records]
    )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Embeddings file not found: {INPUT_FILE}")

    collection = create_collection()

    print("Collection:", COLLECTION_NAME)

    seen_ids = set()
    seen_texts = set()
    duplicates = 0
    batch = []
    total = 0

    with INPUT_FILE.open("r", encoding="utf-8") as file:

        for line in file:

            if not line.strip():
                continue

            record = json.loads(line)
            validate_record(record)

            text_key = " ".join(record["text"].lower().split())

            if record["chunk_id"] in seen_ids or text_key in seen_texts:
                duplicates += 1
                continue

            seen_ids.add(record["chunk_id"])
            seen_texts.add(text_key)
            batch.append(record)

            if len(batch) >= BATCH_SIZE:
                store_batch(collection, batch)
                total += len(batch)
                print(f"Stored {total} records...")
                batch = []

    if batch:
        store_batch(collection, batch)
        total += len(batch)

    print("\nCHROMADB STORAGE COMPLETE")
    print("Records stored:", total)
    print("Duplicates skipped:", duplicates)
    print("Collection total:", collection.count())
    print("Database folder:", CHROMA_PATH)


if __name__ == "__main__":
    main()
