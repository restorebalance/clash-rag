# Royale Codex

A retrieval-augmented question answering system over Clash Royale strategy
prose from the community wiki. Ask a question, get an answer written only
from retrieved wiki passages, with a numbered citation on every claim.

Built without LangChain, LlamaIndex, or a vector database. The retrieval,
the chunking, the SQLite store and the streaming API client are all written
directly, which keeps the moving parts visible and swappable.

## How it works

```
scrape.py    Fandom MediaWiki API  ->  raw/*.wikitext
clean.py     raw/                  ->  docs/*.txt + *.json
ingest.py    docs/                 ->  chunks.db   (chunk, embed, store)
retrieve.py  question              ->  top-k chunks by cosine similarity
answer.py    question + chunks     ->  streamed answer with [n] citations
app.py       HTTP API + web UI
eval.py      recall@5 over eval_questions.json
```

Cards are scraped through the five rarity categories, since no single
category holds every card and each card has exactly one rarity. Cleaning
keeps prose sections and drops stat tables, galleries and translation
lists. Chunking splits on section and bullet boundaries so each chunk is a
self-contained piece of advice, targeting ~1,500 characters.

The current corpus is **1,554 chunks from 197 wiki pages**.

### Models

Embeddings are Cohere `embed-v4.0` at 1024 dimensions. Documents are
embedded with `input_type=search_document` and questions with
`search_query` — the same string yields a different vector for each, so
they are cached under separate keys. v4.0 rather than `embed-english-v3.0`
because the latter caps at 512 tokens while our longest chunks reach ~525,
which would silently truncate them.

Generation is Cohere `command-a-03-2025` over the streaming `/v2/chat`
endpoint. Both APIs are called over REST with `requests`; no SDK.

### Storage

SQLite, one row per chunk, embeddings stored as float32 blobs. Search loads
the matrix and does cosine similarity in NumPy. At 1,554 chunks this is
fast enough that an index would be premature. `store.py` exposes only
`replace_chunks()` and `search()`, so the backend can be swapped without
touching anything else.

## Setup

Developed on Python 3.14.6. Needs a Cohere API key.

```bash
pip install -r requirements.txt

echo "COHERE_API_KEY=your-key-here" > .env
```

`.env` is gitignored. A shell export overrides it.

Then build the corpus — this is the slow part, and it costs API calls:

```bash
python3 scrape.py     # ~200 pages, 1 req/sec, polite to the wiki
python3 clean.py
python3 ingest.py     # embeds 1,554 chunks in batches of 96
```

`ingest.py` paces batches to stay under the trial key's 100k tokens/minute
limit and caches every vector in `.embed_cache.npz`, so a rate-limit stall
never costs you vectors you already paid for. Re-running is cheap.

## Running

```bash
uvicorn app:app --reload
```

Then open http://127.0.0.1:8000, or pass `--port` to move it.

Command line, without the web UI:

```bash
python3 retrieve.py "how do I counter mini pekka"   # retrieval only
python3 answer.py   "how do I counter mini pekka"   # streamed answer
```

## HTTP API

**`POST /ask`** — `{"question": "...", "k": 5}`, responds as
`text/event-stream`:

```
event: chunks
data: {"question": "...", "chunks": [{"n": 1, "title": "...", "section": "...",
                                      "score": 0.6529, "text": "..."}]}

event: token
data: {"text": "To counter a Mini"}

event: done
data: {}
```

The retrieved chunks arrive before generation starts, so a client can show
its sources while tokens are still streaming. `n` is the citation number:
chunk `n` is what `[n]` in the answer refers to. A generation failure
arrives as `event: error` with a `message`, since the response is already
committed to 200 by then. Failures before the stream opens are ordinary
HTTP: 400 empty question, 422 missing `q`, 502 retrieval failed.

**`GET /chunks?q=...&k=5`** — the same chunk array, no generation, no chat
API call.

## Web UI

`index.html`, served at `/`. Vanilla JavaScript, no build step; Tailwind
comes from the Play CDN. Answers stream in token by token, `[n]` markers
are clickable and expand the chunk they cite, and a collapsible panel lists
every retrieved chunk with its similarity score. Conversations are kept in
`localStorage` and can be reopened from the sidebar without re-querying.

Answer text reaches the DOM only as text nodes — never `innerHTML` — so
nothing the model or the corpus produces can inject markup.

## Evaluation

```bash
python3 eval.py
```

Scores recall@5 over `eval_questions.json`: a question passes when its
expected page title appears in the top 5. Entries with `"expect": null` are
questions the corpus cannot answer, and pass only when the best match falls
below `ABSTAIN_MAX_SCORE`, which catches a retriever that confidently
returns something for an unanswerable question.

## Notes

- `raw/`, `docs/`, `chunks.db` and `.embed_cache.npz` are gitignored. A
  fresh clone has no data until the pipeline above has been run.
- The corpus counts shown in the web UI header are hardcoded in
  `index.html`; they will drift if you re-ingest a different corpus.
- `/ask` has no authentication and no rate limiting. Every question spends
  your Cohere key, so think before exposing it beyond localhost.
- Wiki content is licensed CC BY-SA by Fandom. None of it is committed
  here; the scraper fetches it on your machine.
