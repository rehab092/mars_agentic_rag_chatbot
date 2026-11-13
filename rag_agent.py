
# ============================================================
# AGENTIC RAG PIPELINE (implements RAG.pdf)
# ============================================================
#
#  User query
#   -> Load conversation history            (app logic)
#   -> Query Rewriter                       (LLM call 1)
#   -> Orchestrator: needs RAG?             (LLM call 2)
#        No  -> Main LLM answer from history
#        Yes -> Retrieve (ChromaDB BGE-M3 + BM25)
#               -> Re-rank (RRF, then cross-encoder)
#               -> Relevance Evaluator      (LLM call 3)
#                    Yes -> grounded answer
#                    No  -> retries left?
#                             Yes -> rewrite with feedback, retry
#                             No  -> safe response
#   -> Save query and response              (memory)
#
# run() is a generator of events so the UI can show each
# step live and stream the answer token by token.
# ============================================================

import json
import os
import re
import sqlite3
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import rerank_retrieval as retrieval

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "").strip().rstrip("/")
QWEN_API_KEY = (os.getenv("QWEN_API_KEY") or "").strip()
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.5-4b")

MEMORY_DB = BASE_DIR / "conversations.db"

HISTORY_TURNS = 6          # previous user/assistant pairs used
MAX_RETRIES = 2            # extra retrieval attempts after a "No"
CONTEXT_CHUNKS = 5         # chunks given to the evaluator / answer
CHUNK_CHARS_FOR_EVAL = 900

SAFE_RESPONSE = (
    "I couldn't find enough relevant information in the mission "
    "documents to answer that reliably. Try rephrasing the question "
    "or asking about a specific instrument, mission phase or system."
)


# ------------------------------------------------------------
# LLM CLIENT
# ------------------------------------------------------------

def _client():
    base_url = QWEN_BASE_URL
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return OpenAI(
        base_url=base_url,
        api_key=QWEN_API_KEY,
        default_headers={"X-Qwen-Key": QWEN_API_KEY},
        timeout=120.0,
        max_retries=2,
    )


llm = _client()

NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def chat(messages, max_tokens=400, temperature=0.1):
    response = llm.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=NO_THINKING,
    )
    return (response.choices[0].message.content or "").strip()


def chat_stream(messages, max_tokens=1200, temperature=0.2):
    stream = llm.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
        extra_body=NO_THINKING,
    )
    for part in stream:
        if part.choices and part.choices[0].delta.content:
            yield part.choices[0].delta.content


def chat_json(messages, default, max_tokens=300):
    """Ask for JSON; fall back to `default` if it can't be parsed."""
    text = chat(messages, max_tokens=max_tokens, temperature=0.0)
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except ValueError:
            pass
    return default


# ------------------------------------------------------------
# CONVERSATION MEMORY (SQLite)
# ------------------------------------------------------------

def _db():
    conn = sqlite3.connect(MEMORY_DB)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS messages ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " session_id TEXT NOT NULL,"
        " role TEXT NOT NULL,"
        " content TEXT NOT NULL,"
        " meta TEXT,"
        " created REAL NOT NULL)"
    )
    return conn


def load_history(session_id, turns=HISTORY_TURNS):
    with _db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, turns * 2),
        ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def load_full_history(session_id):
    with _db() as conn:
        rows = conn.execute(
            "SELECT role, content, meta FROM messages WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [
        {"role": r, "content": c, "meta": json.loads(m) if m else None}
        for r, c, m in rows
    ]


def save_turn(session_id, question, answer, meta):
    now = time.time()
    with _db() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, meta, created) VALUES (?,?,?,?,?)",
            (session_id, "user", question, None, now),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, meta, created) VALUES (?,?,?,?,?)",
            (session_id, "assistant", answer, json.dumps(meta), now),
        )


def clear_history(session_id):
    with _db() as conn:
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))


def history_text(history):
    if not history:
        return "(no previous conversation)"
    return "\n".join(f"{m['role'].upper()}: {m['content'][:600]}" for m in history)


# ------------------------------------------------------------
# LLM STEPS
# ------------------------------------------------------------

def rewrite_query(question, history):
    """LLM call 1: make the question standalone."""
    if not history:
        return question
    result = chat_json([
        {"role": "system", "content":
            "You rewrite the user's latest question into a standalone search "
            "query, resolving pronouns and references using the conversation. "
            "Keep names, instruments and numbers exactly. Do not answer it. "
            "If the latest message is not a question (greeting, thanks, "
            "small talk) or is already standalone, return it unchanged. "
            'Reply only with JSON: {"query": "..."}'},
        {"role": "user", "content":
            f"Conversation:\n{history_text(history)}\n\nLatest question: {question}"},
    ], default={"query": question})
    return (result.get("query") or question).strip()


def needs_rag(question, rewritten, history):
    """LLM call 2: orchestrator."""
    result = chat_json([
        {"role": "system", "content":
            "You route questions for an assistant whose knowledge base holds "
            "NASA Mars 2020 / Perseverance mission documents (mission overview, "
            "landing, rover engineering, instruments, Ingenuity, launch, "
            "spacecraft). Decide whether answering needs a document search. "
            "Greetings, thanks, small talk, questions about this conversation "
            "itself or general requests that need no mission facts do NOT need "
            "retrieval. Any factual question about Mars, the mission, rovers, "
            "spacecraft or instruments DOES. "
            'Reply only with JSON: {"use_rag": true|false, "reason": "..."}'},
        {"role": "user", "content":
            f"Conversation:\n{history_text(history)}\n\n"
            f"Question: {question}\nRewritten: {rewritten}"},
    ], default={"use_rag": True, "reason": "default"})
    return bool(result.get("use_rag", True)), result.get("reason", "")


def format_chunks(chunks, limit_chars=None):
    blocks = []
    for i, c in enumerate(chunks, start=1):
        m = c["metadata"]
        pages = m.get("page_start")
        if m.get("page_end") and m.get("page_end") != pages:
            pages = f"{pages}-{m['page_end']}"
        text = c["text"] if not limit_chars else c["text"][:limit_chars]
        blocks.append(
            f"[{i}] ({m.get('document_id')}, page {pages}, {m.get('content_type')})\n{text}"
        )
    return "\n\n".join(blocks)


def evaluate(question, rewritten, chunks):
    """LLM call 3: are the documents relevant and sufficient?"""
    result = chat_json([
        {"role": "system", "content":
            "You judge retrieved passages. Decide whether they contain enough "
            "relevant information to answer the question accurately. If not, "
            "explain briefly what is missing so the search can be improved. "
            'Reply only with JSON: {"sufficient": true|false, '
            '"relevant_ids": [numbers], "feedback": "..."}'},
        {"role": "user", "content":
            f"Original question: {question}\nRewritten query: {rewritten}\n\n"
            f"Passages:\n{format_chunks(chunks, CHUNK_CHARS_FOR_EVAL)}"},
    ], default={"sufficient": True, "relevant_ids": [], "feedback": ""})
    return result


def improve_query(question, rewritten, feedback):
    """Query rewriter with evaluator feedback."""
    result = chat_json([
        {"role": "system", "content":
            "The previous search did not find good passages. Write a better "
            "search query for a NASA Mars 2020 document collection, using the "
            "evaluator feedback. Prefer specific technical terms and names. "
            'Reply only with JSON: {"query": "..."}'},
        {"role": "user", "content":
            f"Original question: {question}\nPrevious query: {rewritten}\n"
            f"Evaluator feedback: {feedback}"},
    ], default={"query": rewritten})
    return (result.get("query") or rewritten).strip()


def answer_messages_plain(question, history):
    return [
        {"role": "system", "content":
            "You are Perseverance Guide, a friendly assistant for NASA's Mars "
            "2020 mission. Answer conversationally and concisely. If the user "
            "asks for mission facts, say you can look them up in the mission "
            "documents."},
        *history,
        {"role": "user", "content": question},
    ]


def answer_messages_grounded(question, rewritten, chunks, history):
    return [
        {"role": "system", "content":
            "You are Perseverance Guide, an expert assistant for NASA's Mars "
            "2020 mission. Answer ONLY from the numbered sources provided. "
            "Cite sources inline like [1] or [2][3] after the facts they support. "
            "If the sources don't cover part of the question, say so plainly. "
            "Write clear, well-organised prose; use short bullet lists when "
            "listing several items. Do not invent numbers or names."},
        *history,
        {"role": "user", "content":
            f"Question: {question}\n(Search query used: {rewritten})\n\n"
            f"Sources:\n{format_chunks(chunks)}"},
    ]


# ------------------------------------------------------------
# RETRIEVAL RESOURCES (loaded once)
# ------------------------------------------------------------

class Resources:
    def __init__(self):
        self.collection = retrieval.connect_database()
        self.corpus, self.bm25 = retrieval.load_corpus(self.collection)
        self.embedding_model, self.reranker = retrieval.load_models()

    def search(self, query):
        candidates = retrieval.retrieve_chunks(
            query, self.collection, self.embedding_model, self.corpus, self.bm25
        )
        ranked = retrieval.rerank_chunks(query, candidates, self.reranker)
        return ranked[:CONTEXT_CHUNKS]


def source_card(index, chunk):
    m = chunk["metadata"]
    return {
        "id": index,
        "document": m.get("document_id"),
        "section": m.get("section"),
        "page_start": m.get("page_start"),
        "page_end": m.get("page_end"),
        "type": m.get("content_type"),
        "image_url": m.get("image_url"),
        "score": round(chunk.get("rerank_score", 0.0), 3),
        # Drop the "Document › Section" context line for display.
        "text": chunk["text"].split("\n\n", 1)[-1][:700],
    }


# ------------------------------------------------------------
# MAIN FLOW
# ------------------------------------------------------------

def run(resources, session_id, question):
    """Yield events: step / sources / token / done."""

    def step(name, detail="", status="done"):
        return {"type": "step", "name": name, "detail": detail, "status": status}

    history = load_history(session_id)
    yield step("Loaded conversation", f"{len(history) // 2} previous turns")

    yield step("Rewriting query", status="running")
    rewritten = rewrite_query(question, history)
    yield step("Rewriting query", rewritten)

    yield step("Routing", status="running")
    use_rag, reason = needs_rag(question, rewritten, history)
    yield step("Routing", ("Search documents — " if use_rag else "Answer directly — ") + reason)

    answer_parts = []
    meta = {"rewritten": rewritten, "used_rag": use_rag, "sources": [], "retries": 0}

    if not use_rag:
        for token in chat_stream(answer_messages_plain(question, history), max_tokens=600):
            answer_parts.append(token)
            yield {"type": "token", "text": token}

    else:
        query = rewritten
        retries = 0
        chunks = []
        sufficient = False

        while True:
            yield step("Retrieving documents", query, status="running")
            chunks = resources.search(query)
            yield step("Retrieving documents", f"{len(chunks)} passages · hybrid search + RRF + rerank")

            yield step("Evaluating relevance", status="running")
            verdict = evaluate(question, query, chunks) if chunks else {
                "sufficient": False, "feedback": "No passages found."}
            sufficient = bool(verdict.get("sufficient"))
            yield step(
                "Evaluating relevance",
                "Relevant and sufficient" if sufficient else f"Insufficient — {verdict.get('feedback', '')}",
            )

            if sufficient:
                break

            if retries >= MAX_RETRIES:
                yield step("Retry limit reached", f"{retries} retries used")
                break

            retries += 1
            yield step("Improving query", f"Retry {retries}/{MAX_RETRIES}", status="running")
            query = improve_query(question, query, verdict.get("feedback", ""))
            yield step("Improving query", query)

        meta["retries"] = retries
        meta["final_query"] = query

        if sufficient:
            # Keep evaluator-approved chunks first when it named them.
            ids = [i for i in verdict.get("relevant_ids", []) if isinstance(i, int) and 1 <= i <= len(chunks)]
            if ids:
                chunks = [chunks[i - 1] for i in ids] + [c for j, c in enumerate(chunks, 1) if j not in ids]

            sources = [source_card(i, c) for i, c in enumerate(chunks, start=1)]
            meta["sources"] = sources
            yield {"type": "sources", "sources": sources}

            yield step("Writing answer", status="running")
            for token in chat_stream(answer_messages_grounded(question, query, chunks, history)):
                answer_parts.append(token)
                yield {"type": "token", "text": token}
            yield step("Writing answer", "Grounded in retrieved sources")
        else:
            answer_parts.append(SAFE_RESPONSE)
            yield {"type": "token", "text": SAFE_RESPONSE}

    answer = "".join(answer_parts).strip()
    save_turn(session_id, question, answer, meta)
    yield {"type": "done", "meta": meta}
