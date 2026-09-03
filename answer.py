"""Answer a question from retrieved chunks.

Retrieves the top chunks with retrieve.search(), numbers them [1]-[5] in the
prompt, and streams Cohere's answer. The model is told to work only from
those chunks and to cite them by number, so an answer the corpus does not
support comes back as "I don't know" rather than a plausible invention.

Generation uses the same Cohere key as the embeddings, called over the REST
API with requests as ingest.py does. answer() is a generator so app.py can
hand it straight to a streaming response.
"""

import json
import sys

import requests

import config
import retrieve

CHAT_URL = "https://api.cohere.com/v2/chat"
# Not command-a-plus-05-2026: it thinks before answering, and on this prompt
# it spent a 900-token budget entirely on thinking and returned no answer.
# Disabling its thinking makes the same request fail with INVALID_TOOL_GENERATION.
CHAT_MODEL = "command-a-03-2025"
K = 5

# A grounded answer over five chunks runs ~350 tokens; this is a ceiling for
# a runaway response, not a target.
MAX_TOKENS = 1024

SYSTEM = """You answer questions about Clash Royale strategy using only the \
numbered excerpts provided with each question.

Rules:
- Use only what the excerpts state. Do not add card knowledge of your own.
- Cite the excerpts you used by number, like [2] or [1][3], next to the claim \
they support.
- If the excerpts do not contain the answer, say you don't know. Do not guess \
or fill the gap from memory."""


def build_prompt(question, results):
    """Render the retrieved chunks as a numbered block above the question."""
    excerpts = []
    for index, result in enumerate(results, start=1):
        excerpts.append(
            f"[{index}] {result['title']} — {result['section']}\n{retrieve.body(result)}"
        )
    joined = "\n\n".join(excerpts)
    return f"Excerpts:\n\n{joined}\n\nQuestion: {question}"


def stream_text(payload, key):
    """Yield the text pieces of a streaming Cohere chat response.

    The stream is server-sent events, terminated by a [DONE] sentinel. Text
    arrives as content-delta events carrying delta.message.content.text; a
    delta with no text is a non-text block, such as a thinking block from a
    reasoning model, and is skipped.
    """
    response = requests.post(
        CHAT_URL,
        json=payload,
        headers={"Authorization": f"Bearer {key}"},
        stream=True,
        timeout=120,
    )
    if response.status_code != 200:
        raise SystemExit(f"chat failed [{response.status_code}]: {response.text[:300]}")

    # text/event-stream carries no charset, and requests falls back to
    # latin-1 for text/*, which mangles every curly quote the model writes.
    response.encoding = "utf-8"

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        if event.get("type") == "content-delta":
            content = event["delta"]["message"]["content"]
            if "text" in content:
                yield content["text"]
        elif event.get("type") == "message-end":
            reason = event.get("delta", {}).get("finish_reason") or ""
            if reason.startswith("ERROR"):
                raise SystemExit(f"generation failed: {reason}")


def answer(question, k=K, results=None):
    """Yield the answer to a question, token by token, as it arrives.

    Pass results to reuse chunks the caller already retrieved: app.py sends
    them to the client before the answer starts, and searching again here
    would embed the question a second time.
    """
    if results is None:
        results = retrieve.search(question, k=k)
    if not results:
        yield "The store is empty, so I don't know."
        return

    payload = {
        "model": CHAT_MODEL,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(question, results)},
        ],
    }
    yield from stream_text(payload, config.require("COHERE_API_KEY"))


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 answer.py <question>")
    question = " ".join(sys.argv[1:])
    for piece in answer(question):
        print(piece, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
