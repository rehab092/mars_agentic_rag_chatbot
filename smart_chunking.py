
# ============================================================
# SMART CHUNKING FOR AGENTIC RAG
# ============================================================
#
# Input:   structured/documents.jsonl
# Output:  chunked/chunks.jsonl
#
# Strategy:
# 1. Text is chunked per DOCUMENT (not per page), so
#    paragraphs that continue onto the next page stay together.
# 2. Headings ("# Heading" from extract_pdf.py, or a short-line
#    heuristic) set the current section; a chunk never spans
#    two sections and every chunk carries its section.
# 3. Neighbouring paragraphs are merged when semantically
#    similar and within the token limit (BGE-M3 tokenizer).
# 4. Consecutive chunks of one section overlap slightly.
# 5. Tiny chunks are merged into a neighbour.
# 6. Each chunk's text starts with a context line
#    ("Document › Section") so the embedding knows its place.
# 7. Exact duplicate chunks (same normalized text) are dropped.
# ============================================================

import json
import re
import hashlib
from collections import defaultdict
from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "structured" / "documents.jsonl"

OUTPUT_DIR = BASE_DIR / "chunked"
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"

# Small model used only to decide chunk boundaries.
SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Token counts use the embedding model's own tokenizer.
TOKENIZER_MODEL = "BAAI/bge-m3"

MAX_TOKENS = 450       # body tokens per chunk
MIN_TOKENS = 40        # smaller chunks are merged into a neighbour
OVERLAP_TOKENS = 60    # carried from the previous chunk
SIMILARITY_THRESHOLD = 0.45


# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------

print("Loading models...")
semantic_model = SentenceTransformer(SIMILARITY_MODEL)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
print("Models loaded!")


def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


# ------------------------------------------------------------
# SPLIT LONG TEXT
# ------------------------------------------------------------

def split_long_text(text, limit):
    """
    Split at sentence boundaries; a single sentence longer
    than the limit is split by token windows.
    """

    pieces = []
    current = ""

    for sentence in split_sentences(text):

        if count_tokens(sentence) > limit:

            if current:
                pieces.append(current)
                current = ""

            ids = tokenizer.encode(sentence, add_special_tokens=False)

            for start in range(0, len(ids), limit):
                part = tokenizer.decode(
                    ids[start:start + limit], skip_special_tokens=True
                ).strip()
                if part:
                    pieces.append(part)

            continue

        candidate = f"{current} {sentence}".strip()

        if count_tokens(candidate) <= limit:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = sentence

    if current:
        pieces.append(current)

    return pieces


def tail_overlap(text, limit):
    """Last whole sentences of text, up to `limit` tokens."""

    tail = []
    total = 0

    for sentence in reversed(split_sentences(text)):
        n = count_tokens(sentence)
        if total + n > limit:
            break
        tail.insert(0, sentence)
        total += n

    return " ".join(tail)


# ------------------------------------------------------------
# HEADINGS
# ------------------------------------------------------------

def heading_text(paragraph):
    """Return the heading text if paragraph is a heading."""

    if paragraph.startswith("# "):
        return paragraph[2:].strip()

    # Heuristic for outputs produced before headings were marked.
    words = paragraph.split()

    if (
        "\n" not in paragraph
        and 1 <= len(words) <= 8
        and not paragraph.endswith((".", "!", "?", ":", ",", ";"))
        and not re.search(r"[@/]|\d{3,}|\d+\s*(?:kg|km|cm|mm|mg|%|m\b)", paragraph)
        # Headings are mostly capitalised words.
        and sum(w[:1].isupper() for w in words) >= max(1, len(words) * 0.6)
    ):
        return paragraph.strip()

    return None


# ------------------------------------------------------------
# DOCUMENT-LEVEL TEXT CHUNKING
# ------------------------------------------------------------

def build_units(page_records):
    """
    Turn a document's pages into paragraph units:
    {"text", "page", "section"} in reading order.
    """

    units = []
    section = ""
    pending_headings = []

    for record in sorted(page_records, key=lambda r: r["page_number"] or 0):

        paragraphs = [
            p.strip() for p in re.split(r"\n\s*\n", record["content"]) if p.strip()
        ]

        for paragraph in paragraphs:

            heading = heading_text(paragraph)

            if heading:
                pending_headings.append(heading)
                continue

            if pending_headings:
                # Consecutive headings (e.g. chapter + section)
                # form one section path.
                section = " › ".join(pending_headings[-2:])
                pending_headings = []

            for piece in split_long_text(paragraph, MAX_TOKENS):
                units.append({
                    "text": piece,
                    "page": record["page_number"],
                    "section": section,
                })

    return units


def chunk_document_text(page_records):

    units = build_units(page_records)

    if not units:
        return []

    embeddings = semantic_model.encode(
        [u["text"] for u in units],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    chunks = []
    current = None

    def start_chunk(unit):
        return {
            "body": unit["text"],
            "section": unit["section"],
            "page_start": unit["page"],
            "page_end": unit["page"],
        }

    for i, unit in enumerate(units):

        if current is None:
            current = start_chunk(unit)
            continue

        same_section = unit["section"] == current["section"]
        similarity = float(embeddings[i - 1] @ embeddings[i])
        candidate = current["body"] + "\n\n" + unit["text"]
        fits = count_tokens(candidate) <= MAX_TOKENS
        too_small = count_tokens(current["body"]) < MIN_TOKENS

        if same_section and fits and (similarity >= SIMILARITY_THRESHOLD or too_small):
            current["body"] = candidate
            current["page_end"] = unit["page"]
            continue

        chunks.append(current)
        current = start_chunk(unit)

        # Overlap with the previous chunk of the same section.
        if same_section:
            overlap = tail_overlap(chunks[-1]["body"], OVERLAP_TOKENS)
            if overlap and count_tokens(overlap + "\n\n" + current["body"]) <= MAX_TOKENS + OVERLAP_TOKENS:
                current["body"] = overlap + "\n\n" + current["body"]

    chunks.append(current)

    # Merge a tiny last chunk of a section into its predecessor.
    merged = []
    for chunk in chunks:
        if (
            merged
            and count_tokens(chunk["body"]) < MIN_TOKENS
            and merged[-1]["section"] == chunk["section"]
            and count_tokens(merged[-1]["body"] + "\n\n" + chunk["body"]) <= MAX_TOKENS + OVERLAP_TOKENS
        ):
            merged[-1]["body"] += "\n\n" + chunk["body"]
            merged[-1]["page_end"] = chunk["page_end"]
        else:
            merged.append(chunk)

    return merged


# ------------------------------------------------------------
# IMAGE AND TABLE CHUNKING
# ------------------------------------------------------------

def chunk_image_description(content):
    if count_tokens(content) <= MAX_TOKENS:
        return [content]
    return split_long_text(content, MAX_TOKENS)


def chunk_table(content):
    """Keep table metadata and whole rows; repeat the header."""

    row_matches = list(re.finditer(r"(?m)^Row \d+:", content))

    if not row_matches:
        return split_long_text(content, MAX_TOKENS)

    header = content[:row_matches[0].start()].strip()

    rows = [
        content[m.start():(row_matches[i + 1].start() if i + 1 < len(row_matches) else len(content))].strip()
        for i, m in enumerate(row_matches)
    ]

    chunks = []
    current = header

    for row in rows:

        candidate = f"{current}\n\n{row}".strip()

        if count_tokens(candidate) <= MAX_TOKENS:
            current = candidate
            continue

        if current and current != header:
            chunks.append(current)

        current = f"{header}\n\n{row}".strip()

        if count_tokens(current) > MAX_TOKENS:
            chunks.extend(split_long_text(current, MAX_TOKENS))
            current = header

    if current and current != header:
        chunks.append(current)

    return chunks


# ------------------------------------------------------------
# OUTPUT HELPERS
# ------------------------------------------------------------

def context_line(document_id, section=None, label=None):
    parts = [document_id]
    if section:
        parts.append(section)
    if label:
        parts.append(label)
    return " › ".join(parts)


def normalized_key(text):
    return hashlib.sha256(
        " ".join(text.lower().split()).encode("utf-8")
    ).hexdigest()


def create_chunk_id(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


# ------------------------------------------------------------
# MAIN CHUNKING PIPELINE
# ------------------------------------------------------------

def main():

    if not INPUT_FILE.exists():
        print("Structured documents file not found.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records_by_doc = defaultdict(list)
    total_records = 0

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records_by_doc[record["document_id"]].append(record)
                total_records += 1

    seen = set()
    duplicates = 0
    total_chunks = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:

        def write(document_id, content_type, source_file, record_id,
                  index, body, header, extra):

            nonlocal duplicates, total_chunks

            key = normalized_key(body)
            if key in seen:
                duplicates += 1
                return
            seen.add(key)

            text = f"{header}\n\n{body}"

            out.write(json.dumps({
                "chunk_id": create_chunk_id(document_id, content_type, source_file, index, key),
                "record_id": record_id,
                "document_id": document_id,
                "content_type": content_type,
                "source_file": source_file,
                "chunk_index": index,
                "token_count": count_tokens(text),
                "text": text,
                **extra,
            }, ensure_ascii=False) + "\n")

            total_chunks += 1

        for document_id, records in records_by_doc.items():

            print(f"\nDocument: {document_id}")

            # --- Text: one pass over the whole document.
            text_records = [r for r in records if r["content_type"] == "text"]
            text_chunks = chunk_document_text(text_records)

            for index, chunk in enumerate(text_chunks, start=1):
                write(
                    document_id, "text",
                    f"outputs/{document_id}/text",
                    create_chunk_id(document_id, "text"),
                    index, chunk["body"],
                    context_line(document_id, chunk["section"]),
                    {
                        "section": chunk["section"],
                        "page_number": chunk["page_start"],
                        "page_start": chunk["page_start"],
                        "page_end": chunk["page_end"],
                        "total_chunks_in_record": len(text_chunks),
                    },
                )

            print(f"  Text chunks: {len(text_chunks)}")

            # --- Images and tables: one record each.
            for record in records:

                content_type = record["content_type"]

                if content_type == "image_descriptions":
                    pieces = chunk_image_description(record["content"])
                    label = f"Image on page {record['page_number']}"
                elif content_type == "tables":
                    pieces = chunk_table(record["content"])
                    label = f"Table on page {record['page_number']}"
                else:
                    continue

                extra = {
                    "section": "",
                    "page_number": record["page_number"],
                    "page_start": record["page_number"],
                    "page_end": record["page_number"],
                    "total_chunks_in_record": len(pieces),
                }

                for key in ("image_path", "image_url"):
                    if record.get(key):
                        extra[key] = record[key]

                for index, piece in enumerate(pieces, start=1):
                    write(
                        document_id, content_type, record["source_file"],
                        record["record_id"], index, piece,
                        context_line(document_id, label=label), extra,
                    )

    print("\nSMART CHUNKING COMPLETED")
    print(f"Records processed: {total_records}")
    print(f"Chunks generated: {total_chunks}")
    print(f"Duplicate chunks dropped: {duplicates}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
