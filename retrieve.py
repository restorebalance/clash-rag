"""Search the chunk store.

Embeds a question as a query vector and hands it to the store. Cohere
returns a different vector depending on input_type, so questions must be
embedded as search_query even though the chunks were search_document.
"""

import sys

import ingest
import store

DEFAULT_K = 5


def search(question, k=DEFAULT_K):
    """Return the k stored chunks most relevant to a question."""
    vector = ingest.embed_texts([question], input_type="search_query")[0]
    return store.search(vector, k=k)


def body(result):
    """The chunk text without the "Title - Section" line ingest prepends."""
    return result["text"].split("\n", 1)[-1].strip()


def format_result(result, index):
    """Render one result for the command line."""
    header = f"{index}. [{result['score']:.3f}] {result['title']} - {result['section']}"
    return f"{header}\n{body(result)}\n"


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 retrieve.py <question>")
    question = " ".join(sys.argv[1:])
    for index, result in enumerate(search(question), start=1):
        print(format_result(result, index))


if __name__ == "__main__":
    main()
