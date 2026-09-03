"""HTTP API over the pipeline.

/ask streams a generated answer as server-sent events, sending the chunks it
was grounded in first so a client can render its sources before the first
token arrives. /chunks exposes retrieval on its own, with no generation and
no Cohere chat call. index.html in the project root is served at /.

Run with: uvicorn app:app --reload
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import answer
import retrieve

INDEX_PATH = Path(__file__).resolve().parent / "index.html"

app = FastAPI(title="clash-rag")


class Question(BaseModel):
    question: str
    k: int = answer.K


def public_chunks(results):
    """Shape retrieval results for a client.

    n is the citation number: chunk n is what [n] in the answer refers to.
    """
    return [
        {
            "n": index,
            "title": result["title"],
            "section": result["section"],
            "score": round(result["score"], 4),
            "text": retrieve.body(result),
        }
        for index, result in enumerate(results, start=1)
    ]


def search(question, k):
    """Retrieve, turning a failed embedding call into a 502 rather than a crash."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        return retrieve.search(question, k=k)
    except SystemExit as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


def sse(event, data):
    """Format one server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def ask_events(question, results):
    """Yield the SSE stream for one question: chunks, then tokens, then done.

    The response has already started by the time generation runs, so a
    failure there arrives as an error event rather than an HTTP status.
    """
    yield sse("chunks", {"question": question, "chunks": public_chunks(results)})
    try:
        for piece in answer.answer(question, results=results):
            yield sse("token", {"text": piece})
    except SystemExit as error:
        yield sse("error", {"message": str(error)})
        return
    yield sse("done", {})


@app.post("/ask")
def ask(payload: Question):
    """Stream an answer to a question, grounded in the retrieved chunks."""
    results = search(payload.question, payload.k)
    return StreamingResponse(
        ask_events(payload.question, results),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/chunks")
def chunks(q: str = Query(...), k: int = answer.K):
    """Return retrieval results for a query, without generating an answer."""
    return {"question": q, "chunks": public_chunks(search(q, k))}


@app.get("/")
def index():
    """Serve the frontend."""
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=404, detail="no index.html in the project root")
    return FileResponse(INDEX_PATH)
