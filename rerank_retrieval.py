
# ============================================================
# HYBRID RETRIEVAL + RERANKING
# ============================================================
#
# 1. Dense search in ChromaDB (BGE-M3 vectors).
# 2. Keyword search (BM25) over the same chunks, so exact
#    terms such as "MOXIE", "SHERLOC" or part numbers are found
#    even when the vectors miss them.
# 3. Both lists are fused with Reciprocal Rank Fusion.
# 4. A cross-encoder reranks the fused candidates.
# ============================================================

import math
import re
from collections import Counter
from pathlib import Path

import torch
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

BASE_DIR = Path(__file__).resolve().parent

CHROMA_PATH = BASE_DIR / "chroma_storage"
COLLECTION_NAME = "mars_rag_bge_m3"

EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

DENSE_RESULTS = 40      # candidates from vector search
KEYWORD_RESULTS = 40    # candidates from BM25
RERANK_CANDIDATES = 40  # fused candidates sent to the reranker
FINAL_RESULTS = 5

RRF_K = 60

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# BM25 KEYWORD INDEX
# ------------------------------------------------------------

def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower())


class BM25:

    def __init__(self, ids, texts, k1=1.5, b=0.75):
        self.ids = ids
        self.k1 = k1
        self.b = b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = sum(self.lengths) / max(1, len(self.lengths))

        df = Counter()
        for doc in self.docs:
            df.update(doc.keys())

        n = len(self.docs)
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query, limit):
        terms = [t for t in set(tokenize(query)) if t in self.idf]
        scores = []

        for i, doc in enumerate(self.docs):
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_length)
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            if score > 0:
                scores.append((score, self.ids[i]))

        scores.sort(reverse=True)
        return [chunk_id for _score, chunk_id in scores[:limit]]


# ------------------------------------------------------------
# SETUP
# ------------------------------------------------------------

def connect_database():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_collection(name=COLLECTION_NAME, embedding_function=None)


def load_models():
    print("Using device:", DEVICE)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE, max_length=512)
    return embedding_model, reranker


def load_corpus(collection):
    """All chunks, for the BM25 index and candidate lookup."""

    data = collection.get(include=["documents", "metadatas"])

    corpus = {
        chunk_id: {"text": text, "metadata": metadata or {}}
        for chunk_id, text, metadata in zip(
            data["ids"], data["documents"], data["metadatas"]
        )
    }

    bm25 = BM25(list(corpus), [c["text"] for c in corpus.values()])

    return corpus, bm25


# ------------------------------------------------------------
# RETRIEVAL
# ------------------------------------------------------------

def retrieve_chunks(question, collection, embedding_model, corpus, bm25):

    query_vector = embedding_model.encode(
        question, normalize_embeddings=True
    ).tolist()

    dense = collection.query(
        query_embeddings=[query_vector],
        n_results=min(DENSE_RESULTS, collection.count()),
        include=["distances"]
    )

    dense_ids = dense["ids"][0]
    distances = dict(zip(dense_ids, dense["distances"][0]))
    keyword_ids = bm25.search(question, KEYWORD_RESULTS)

    # Reciprocal Rank Fusion.
    fused = Counter()
    for ranking in (dense_ids, keyword_ids):
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] += 1.0 / (RRF_K + rank)

    candidates = []

    for chunk_id, score in fused.most_common(RERANK_CANDIDATES):
        chunk = corpus[chunk_id]
        candidates.append({
            "chunk_id": chunk_id,
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "vector_distance": distances.get(chunk_id),
            "fusion_score": score,
        })

    return candidates


def rerank_chunks(question, candidates, reranker):

    if not candidates:
        return []

    scores = reranker.predict(
        [(question, c["text"]) for c in candidates],
        batch_size=8,
        show_progress_bar=False
    )

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

    return candidates[:FINAL_RESULTS]


# ------------------------------------------------------------
# INTERACTIVE TEST
# ------------------------------------------------------------

def main():

    collection = connect_database()
    print("Total stored chunks:", collection.count())

    corpus, bm25 = load_corpus(collection)
    embedding_model, reranker = load_models()

    while True:

        question = input("\nEnter your question (blank to quit): ").strip()

        if not question:
            break

        candidates = retrieve_chunks(
            question, collection, embedding_model, corpus, bm25
        )
        results = rerank_chunks(question, candidates, reranker)

        for index, result in enumerate(results, start=1):
            meta = result["metadata"]
            print(f"\nResult {index}  rerank={result['rerank_score']:.3f}")
            print("Document:", meta.get("document_id"))
            print("Pages:", meta.get("page_start"), "-", meta.get("page_end"))
            print("Type:", meta.get("content_type"))
            if meta.get("image_url"):
                print("Image:", meta["image_url"])
            print(result["text"])
            print("-" * 60)


if __name__ == "__main__":
    main()
