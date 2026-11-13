
# ============================================================
# BGE-M3 EMBEDDING GENERATION FOR AGENTIC RAG
# ============================================================
#
# Input:
#   chunked/chunks.jsonl
#
# Output:
#   embeddings/embeddings.jsonl
#
# Each output record contains:
# - Chunk ID
# - Original chunk text
# - Document metadata
# - 1024-dimensional BGE-M3 embedding
#
# ============================================================

import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "chunked" / "chunks.jsonl"

OUTPUT_DIR = BASE_DIR / "embeddings"
OUTPUT_FILE = OUTPUT_DIR / "embeddings.jsonl"

MODEL_NAME = "BAAI/bge-m3"

BATCH_SIZE = 8
EXPECTED_DIMENSIONS = 1024


# ------------------------------------------------------------
# STEP 1: LOAD EMBEDDING MODEL
# ------------------------------------------------------------

def load_embedding_model():

    if torch.cuda.is_available():
        device = "cuda"
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("CUDA unavailable: using CPU (much slower).")

    print("Loading BGE-M3 embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device
    )

    print("BGE-M3 loaded successfully!")

    return model


# ------------------------------------------------------------
# STEP 2: READ CHUNKS
# ------------------------------------------------------------

def load_chunks():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {INPUT_FILE}"
        )

    chunks = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            record = json.loads(line)

            # Ignore empty chunks.
            if not record.get("text", "").strip():
                continue

            chunks.append(record)

    print(f"Loaded {len(chunks)} chunks.")

    return chunks


# ------------------------------------------------------------
# STEP 3: GENERATE EMBEDDINGS
# ------------------------------------------------------------

def generate_embeddings(model, texts):

    # BGE-M3 converts each text chunk into a vector.
    #
    # normalize_embeddings=True:
    # Normalizes vectors for cosine similarity search.
    #
    # batch_size:
    # Controls the number of chunks processed together.

    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # Validate dimensions.
    if vectors.ndim != 2:
        raise ValueError("Unexpected embedding shape.")

    if vectors.shape[1] != EXPECTED_DIMENSIONS:
        raise ValueError(
            f"Expected 1024 dimensions, "
            f"got {vectors.shape[1]}"
        )

    if not np.isfinite(vectors).all():
        raise ValueError(
            "Embedding contains NaN or infinite values."
        )

    return vectors


# ------------------------------------------------------------
# STEP 4: SAVE EMBEDDINGS WITH METADATA
# ------------------------------------------------------------

def save_embeddings(chunks, vectors):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Write to a temporary file first, then replace
    # the final output after successful completion.
    temporary_file = OUTPUT_FILE.with_suffix(".tmp")

    with temporary_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        for chunk, vector in zip(chunks, vectors):

            # Keep all chunk metadata (section, pages,
            # image URL, ...) and add the vector.
            record = {
                **chunk,
                "embedding_model": MODEL_NAME,
                "embedding_dimension": EXPECTED_DIMENSIONS,
                "embedding": vector.tolist()
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    temporary_file.replace(OUTPUT_FILE)

    print(f"Embeddings saved to: {OUTPUT_FILE}")


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

def main():

    print("Starting BGE-M3 embedding pipeline...")

    chunks = load_chunks()

    if not chunks:
        print("No chunks available for embedding.")
        return

    model = load_embedding_model()

    # Extract text from structured chunk records.
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Generate numerical vectors.
    vectors = generate_embeddings(
        model,
        texts
    )

    print("Embedding shape:", vectors.shape)

    # Save vectors and metadata.
    save_embeddings(chunks, vectors)

    print("\nEMBEDDING GENERATION COMPLETED")
    print(f"Total chunks: {len(chunks)}")
    print(f"Vector dimensions: {vectors.shape[1]}")


if __name__ == "__main__":
    main()



