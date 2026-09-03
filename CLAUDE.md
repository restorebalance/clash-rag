# Project
RAG pipeline over Clash Royale strategy prose. Backend FastAPI,
SQLite store, Cohere embeddings, Claude for generation.

# Rules
- No LangChain, LlamaIndex, or vector DB libraries. Write it directly.
- Ask before adding any dependency.
- Standard library plus requests/numpy/fastapi only unless discussed.
- Keep functions short and readable over clever.
- The vector store is a swappable interface: store.py exposes
  replace_chunks(), search(), and nothing else leaks SQLite details.

# Embeddings
Cohere embed-v4.0 at 1024 dims, called over the REST API with requests
(no SDK). Documents embed with input_type=search_document, queries with
search_query -- the same string yields a different vector for each, so
they are cached under separate keys.

Not embed-english-v3.0: it caps at 512 tokens and our chunks reach ~525,
so the longest strategy bullets would be silently truncated. v4.0 allows
8192.

COHERE_API_KEY comes from .env in the project root (gitignored), loaded
by config.py; a shell export overrides it. The trial key's binding limit
is 100k tokens/minute, so ingest.py paces batches and caches vectors in
.embed_cache.npz to survive a stall.

# Layout
scrape.py   fetch wiki pages -> raw/
clean.py    raw/ -> docs/ as plain text + metadata
ingest.py   docs/ -> SQLite
store.py    storage interface
config.py   .env loading and required settings
retrieve.py search
app.py      FastAPI
eval.py     recall@5 over eval_questions.json