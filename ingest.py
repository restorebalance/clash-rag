"""Turn cleaned documents into embeddable chunks.

Reads docs/*.txt with their .json metadata, drops documents too short to
carry prose, and splits the rest on section and bullet boundaries so each
chunk is a self-contained piece of strategy advice.
"""

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import requests

import config
import store

DOCS_DIR = Path("docs")

EMBED_URL = "https://api.cohere.com/v2/embed"
EMBED_MODEL = "embed-v4.0"
EMBED_DIMS = 1024

# Cohere accepts at most 96 texts per embed call. The trial key's binding
# limit is 100,000 tokens per minute: a 96-chunk batch is roughly 27,000
# tokens, so a ~22s pause keeps us near three batches a minute. The cache
# means a rate-limit stall never costs us vectors we already paid for.
EMBED_BATCH = 96
EMBED_PAUSE = 22.0
MAX_RETRIES = 6
CACHE_PATH = Path(".embed_cache.npz")

# Pages under this are stat tables with no prose left after cleaning.
# The corpus has a clear gap here: the largest reject is 753 chars, the
# smallest keeper 1,755.
MIN_DOC_CHARS = 1000

# A chunk shorter than this is a stray fragment, not an idea.
MIN_CHUNK_CHARS = 120

# A soft target, not a hard cap: a bullet with one long child cannot be split
# further without cutting mid-sentence, so ~5% of chunks run over, up to ~2,100.
MAX_CHUNK_CHARS = 1500

SECTION_RE = re.compile(r"^== (.+) ==$")


def load_documents():
    """Yield (metadata, body) for every cleaned document."""
    for meta_path in sorted(DOCS_DIR.glob("*.json")):
        text_path = meta_path.with_suffix(".txt")
        if not text_path.exists():
            continue
        yield json.loads(meta_path.read_text(encoding="utf-8")), text_path.read_text(
            encoding="utf-8"
        )


def is_long_enough(metadata, body):
    """True if the document has enough prose to be worth embedding."""
    return len(body.strip()) >= MIN_DOC_CHARS


def split_sections(body):
    """Yield (section_name, section_text) pairs from a cleaned document."""
    name, lines = "Overview", []
    for line in body.splitlines():
        match = SECTION_RE.match(line)
        if match:
            if lines:
                yield name, "\n".join(lines).strip()
            name, lines = match.group(1), []
        else:
            lines.append(line)
    if lines:
        yield name, "\n".join(lines).strip()


def split_blocks(text):
    """Split a section into blocks: a top-level bullet plus its children."""
    blocks, current = [], []
    for line in text.splitlines():
        starts_block = line.startswith("- ") or (not line.strip() and current)
        if starts_block and current and not line.startswith("  "):
            blocks.append("\n".join(current).strip())
            current = []
        if line.strip():
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [block for block in blocks if block]


def explode(block):
    """Split an over-long bullet group, repeating its parent line for context."""
    if len(block) <= MAX_CHUNK_CHARS:
        return [block]
    parent, *children = block.splitlines()
    if not children:
        return [block]  # a single very long line; nothing to split on
    parts, current = [], []
    for child in children:
        too_long = len(parent) + sum(len(c) for c in current) + len(child) > MAX_CHUNK_CHARS
        if current and too_long:
            parts.append("\n".join([parent] + current))
            current = []
        current.append(child)
    if current:
        parts.append("\n".join([parent] + current))
    return parts


def pack(blocks):
    """Group blocks into chunks that stay under MAX_CHUNK_CHARS."""
    chunks, current = [], []
    for block in blocks:
        candidate = current + [block]
        if current and sum(len(b) for b in candidate) > MAX_CHUNK_CHARS:
            chunks.append("\n".join(current))
            current = [block]
        else:
            current = candidate
    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_document(metadata, body):
    """Return the chunk records for one document."""
    records = []
    for section, text in split_sections(body):
        blocks = [part for block in split_blocks(text) for part in explode(block)]
        for chunk in pack(blocks):
            if len(chunk) < MIN_CHUNK_CHARS:
                continue
            records.append(
                {
                    "title": metadata["title"],
                    "card": metadata["card"],
                    "section": section,
                    "rarity": metadata.get("rarity"),
                    "type": metadata.get("type"),
                    "text": f"{metadata['title']} — {section}\n{chunk}",
                }
            )
    return records


def build_chunks():
    """Return (chunks, skipped_titles) for the whole corpus."""
    chunks, skipped = [], []
    for metadata, body in load_documents():
        if not is_long_enough(metadata, body):
            skipped.append((metadata["title"], len(body.strip())))
            continue
        chunks.extend(chunk_document(metadata, body))
    return chunks, skipped


def batched(items, size):
    """Yield successive slices of items."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


def embed_batch(texts, input_type, key):
    """Embed one batch, retrying on rate limits and transient failures."""
    payload = {
        "model": EMBED_MODEL,
        "input_type": input_type,
        "embedding_types": ["float"],
        "output_dimension": EMBED_DIMS,
        "texts": texts,
    }
    for attempt in range(MAX_RETRIES):
        response = requests.post(
            EMBED_URL,
            json=payload,
            headers={"Authorization": f"Bearer {key}"},
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()["embeddings"]["float"]
        if response.status_code not in (429, 500, 502, 503, 504):
            raise SystemExit(f"embed failed [{response.status_code}]: {response.text[:300]}")
        wait = float(response.headers.get("retry-after", 2 ** attempt))
        print(f"    {response.status_code}, waiting {wait:.0f}s", flush=True)
        time.sleep(wait)
    raise SystemExit(
        f"embed still failing after {MAX_RETRIES} attempts: "
        f"[{response.status_code}] {response.text[:300]}"
    )


def text_key(text, input_type="search_document"):
    """Cache key for one embedding.

    Documents are keyed by their text alone so that vectors cached by
    earlier runs stay valid; other input types are namespaced, since Cohere
    returns a different vector for the same string embedded as a query.
    """
    prefix = "" if input_type == "search_document" else f"{input_type}\0"
    return hashlib.sha1(f"{prefix}{text}".encode("utf-8")).hexdigest()


def load_cache():
    """Return {text_key: vector} for embeddings computed on earlier runs."""
    if not CACHE_PATH.exists():
        return {}
    data = np.load(CACHE_PATH, allow_pickle=False)
    return dict(zip(data["keys"].tolist(), data["vectors"]))


def save_cache(cache):
    np.savez(
        CACHE_PATH,
        keys=np.array(list(cache), dtype="U40"),
        vectors=np.array(list(cache.values()), dtype=np.float32),
    )


def embed_texts(texts, input_type="search_document"):
    """Embed texts, reusing cached vectors and batching to respect the API."""
    cache = load_cache()
    pending = [
        text for text in dict.fromkeys(texts)
        if text_key(text, input_type) not in cache
    ]
    if cache:
        print(f"  reusing {len(texts) - len(pending)} cached vectors")

    if pending:
        key = config.require("COHERE_API_KEY")
        done = 0
        for batch in batched(pending, EMBED_BATCH):
            for text, vector in zip(batch, embed_batch(batch, input_type, key)):
                cache[text_key(text, input_type)] = vector
            save_cache(cache)
            done += len(batch)
            print(f"  embedded {done}/{len(pending)}", flush=True)
            if done < len(pending):
                time.sleep(EMBED_PAUSE)

    return [cache[text_key(text, input_type)] for text in texts]


def main():
    chunks, skipped = build_chunks()
    sizes = sorted(len(c["text"]) for c in chunks)
    print(f"{len(chunks)} chunks from {len(set(c['title'] for c in chunks))} documents")
    if sizes:
        print(f"chunk chars: min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}")
    print(f"skipped {len(skipped)} documents under {MIN_DOC_CHARS} chars:")
    for title, size in sorted(skipped, key=lambda item: item[1]):
        print(f"  {size:5}  {title}")

    print(f"\nembedding with {EMBED_MODEL} ({EMBED_DIMS} dims)")
    vectors = embed_texts([chunk["text"] for chunk in chunks])
    stored = store.replace_chunks(chunks, vectors)
    print(f"stored {stored} chunks")


if __name__ == "__main__":
    main()
