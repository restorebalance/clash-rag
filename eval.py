"""Measure retrieval quality against eval_questions.json.

Each entry is {"question": ..., "expect": "Page Title"}, where expect may
also be a list of titles, any one of which counts as a hit. A null expect
means no page should answer it: that scores as correct only when the best
match falls below ABSTAIN_MAX_SCORE, which is how we catch a retriever
that confidently returns something for an unanswerable question.
"""

import json
from pathlib import Path

import retrieve

QUESTIONS_PATH = Path("eval_questions.json")
K = 5
ABSTAIN_MAX_SCORE = 0.35


def load_questions(path=QUESTIONS_PATH):
    """Return the eval entries, accepting a bare list or {"questions": [...]}."""
    if not path.exists():
        raise SystemExit(f"{path} not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


def judge(entry, results):
    """Return (correct, pages) for one question's results."""
    pages = [result["title"] for result in results]
    expected = entry.get("expect")
    if expected is None:
        top = results[0]["score"] if results else 0.0
        return top < ABSTAIN_MAX_SCORE, pages
    if isinstance(expected, list):
        return any(title in pages for title in expected), pages
    return expected in pages, pages


def run(entries):
    """Score every entry, returning the list of misses."""
    misses = []
    correct = 0
    for entry in entries:
        results = retrieve.search(entry["question"], k=K)
        is_correct, pages = judge(entry, results)
        correct += is_correct
        if not is_correct:
            misses.append((entry, results, pages))
    return correct, misses


def report(entries, correct, misses):
    """Print the recall figure and every miss in detail."""
    total = len(entries)
    print(f"recall@{K}: {correct}/{total} = {100 * correct / total:.1f}%")

    answerable = [e for e in entries if e.get("expect") is not None]
    abstain = total - len(answerable)
    if abstain:
        print(f"  ({len(answerable)} answerable, {abstain} expecting no match)")

    if not misses:
        print("\nno misses")
        return

    print(f"\n{len(misses)} misses")
    for entry, results, pages in misses:
        print(f"\n  Q: {entry['question']}")
        if entry.get("expect") is None:
            top = results[0]["score"] if results else 0.0
            print(f"  expected: no match (top score {top:.3f} >= {ABSTAIN_MAX_SCORE})")
        else:
            expected = entry["expect"]
            if isinstance(expected, list):
                expected = " or ".join(expected)
            print(f"  expected: {expected}")
        print(f"  got:")
        for rank, (page, result) in enumerate(zip(pages, results), start=1):
            print(f"    {rank}. [{result['score']:.3f}] {page} - {result['section']}")


def main():
    entries = load_questions()
    correct, misses = run(entries)
    report(entries, correct, misses)


if __name__ == "__main__":
    main()
