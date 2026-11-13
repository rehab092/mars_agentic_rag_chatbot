
# ============================================================
# CHATBOT ACCURACY TEST
# ============================================================
#
# Start the app first:   .\venv\Scripts\python.exe app.py
# Then run:              .\venv\Scripts\python.exe eval_chatbot.py
#
# Each question in eval_questions.json is sent to /api/chat.
# Checks:
#   expect        every item must appear in the answer
#                 ("a|b" means a OR b; case-insensitive)
#   expect_route  "direct" = answered without document search
#   expect_safe   should return the "not enough information" reply
# Also reports: citations present, retries used, latency.
# Results are saved to eval_results.json.
# ============================================================

import json
import time
import uuid
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8000"
SAFE_MARKER = "couldn't find enough relevant information"


def ask(session_id, question):
    answer, meta = "", {}
    with requests.post(f"{URL}/api/chat",
                       json={"session_id": session_id, "message": question},
                       stream=True, timeout=600) as r:
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            ev = json.loads(line[6:])
            if ev["type"] == "token":
                answer += ev["text"]
            elif ev["type"] == "done":
                meta = ev["meta"]
            elif ev["type"] == "error":
                raise RuntimeError(ev["message"])
    return answer.strip(), meta


def check(item, answer, meta):
    low = answer.lower()
    problems = []

    for alt in item.get("expect", []):
        if not any(a.strip().lower() in low for a in alt.split("|")):
            problems.append(f"missing '{alt}'")

    if item.get("expect_route") == "direct" and meta.get("used_rag"):
        problems.append("searched documents (expected direct answer)")

    is_safe = SAFE_MARKER in low
    if item.get("expect_safe") and not is_safe:
        problems.append("answered instead of safe response")
    if item.get("expect") and is_safe:
        problems.append("gave safe response (retrieval failed)")

    if item.get("expect") and meta.get("used_rag") and "[" not in answer:
        problems.append("no citations")

    return problems


def main():
    questions = json.loads((BASE_DIR / "eval_questions.json").read_text(encoding="utf-8"))
    sessions = {}
    results = []
    run_id = uuid.uuid4().hex[:8]

    for item in questions:
        conv = item.get("conversation") or item["id"]
        session = sessions.setdefault(conv, f"eval-{run_id}-{conv}")

        start = time.time()
        try:
            answer, meta = ask(session, item["question"])
            problems = check(item, answer, meta)
        except Exception as error:
            answer, meta, problems = "", {}, [f"error: {error}"]
        seconds = time.time() - start

        ok = not problems
        results.append({**item, "pass": ok, "problems": problems, "answer": answer,
                        "used_rag": meta.get("used_rag"), "retries": meta.get("retries"),
                        "seconds": round(seconds, 1)})

        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {item['id']:3} {item['category']:13} {seconds:5.1f}s  {item['question'][:60]}")
        if problems:
            print(f"        -> {'; '.join(problems)}")
            print(f"        answer: {answer[:200]!r}")

    # Clean up eval conversations from memory.
    for session in sessions.values():
        requests.delete(f"{URL}/api/history/{session}")

    passed = sum(r["pass"] for r in results)
    print("\n" + "=" * 60)
    print(f"ACCURACY: {passed}/{len(results)} ({100 * passed / len(results):.0f}%)")
    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r["pass"])
    for cat, vals in by_cat.items():
        print(f"  {cat:14} {sum(vals)}/{len(vals)}")
    print(f"Average latency: {sum(r['seconds'] for r in results) / len(results):.1f}s")

    (BASE_DIR / "eval_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Details saved to eval_results.json")


if __name__ == "__main__":
    main()
