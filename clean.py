"""Turn raw wikitext into plain-text documents plus metadata.

Reads raw/*.wikitext and writes, for each page, docs/<slug>.txt holding the
prose and docs/<slug>.json holding its infobox fields. Only prose sections
survive: stat tables, galleries and translation lists are dropped.
"""

import json
import re
from pathlib import Path

RAW_DIR = Path("raw")
DOCS_DIR = Path("docs")

# Sections that hold no strategy prose worth embedding.
DROP_SECTIONS = {
    "gallery",
    "history",
    "trivia",
    "in other languages",
    "sound effects",
}

# Templates whose last argument is prose we want to keep inline.
KEEP_TEMPLATE_ARG = {"quote", "rarity"}

INFOBOX_RE = re.compile(r"\{\{(?:evolved|heroic)?[ _]*card[ _]infobox", re.I)
HEADING_RE = re.compile(r"^=={1,4}(.+?)==+\s*$")


def decode_title(path):
    """Invert scrape.raw_path: ___ -> _, __ -> /, _ -> space."""
    table = {"___": "_", "__": "/", "_": " "}
    return re.sub(r"___|__|_", lambda m: table[m.group()], path.stem)


def find_close(text, start, open_tok, close_tok):
    """Return the index just past the close token matching the one at start."""
    depth = 0
    i = start
    while i < len(text):
        if text.startswith(open_tok, i):
            depth += 1
            i += len(open_tok)
        elif text.startswith(close_tok, i):
            depth -= 1
            i += len(close_tok)
            if depth == 0:
                return i
        else:
            i += 1
    return len(text)


def split_top_level(text, sep="|"):
    """Split on sep, ignoring separators nested inside {{ }} or [[ ]]."""
    parts, depth, current = [], 0, []
    i = 0
    while i < len(text):
        if text.startswith(("{{", "[[") , i):
            depth += 1
            current.append(text[i:i + 2])
            i += 2
        elif text.startswith(("}}", "]]"), i):
            depth -= 1
            current.append(text[i:i + 2])
            i += 2
        elif text[i] == sep and depth == 0:
            parts.append("".join(current))
            current = []
            i += 1
        else:
            current.append(text[i])
            i += 1
    parts.append("".join(current))
    return parts


def render_template(inner):
    """Reduce one template body to the prose it contributes, if any."""
    parts = split_top_level(inner)
    name = parts[0].strip().lower().replace("_", " ")
    if name in KEEP_TEMPLATE_ARG and len(parts) > 1:
        return strip_templates(parts[-1])
    return ""


def strip_balanced(text, open_tok, close_tok, render):
    """Replace every balanced open/close span using render(inner_text)."""
    out, i = [], 0
    while True:
        start = text.find(open_tok, i)
        if start < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:start])
        end = find_close(text, start, open_tok, close_tok)
        out.append(render(text[start + len(open_tok):end - len(close_tok)]))
        i = end


def strip_templates(text):
    return strip_balanced(text, "{{", "}}", render_template)


def strip_tables(text):
    return strip_balanced(text, "{|", "|}", lambda inner: "")


def strip_links(text):
    """[[a|b]] -> b, [[a]] -> a, [url text] -> text, dropping interwiki links."""
    def wiki_link(match):
        target = match.group(1)
        parts = target.split("|")
        if re.match(r"^\s*:?(file|image):", parts[0], re.I):
            return ""  # embedded image, not prose
        if len(parts) == 1 and re.match(r"^\s*category:", parts[0], re.I):
            return ""  # bare category tag, not prose
        if len(parts) == 1 and re.match(r"^[a-z-]{2,5}:", parts[0]):
            return ""  # interwiki language link
        return parts[-1].lstrip(":").split(":")[-1] if len(parts) == 1 else parts[-1]

    text = re.sub(r"\[\[([^\[\]]+)\]\]", wiki_link, text)
    return re.sub(r"\[https?://\S+?(?: ([^\]]*))?\]", lambda m: m.group(1) or "", text)


def strip_markup(text):
    """Reduce wikitext to plain prose."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S | re.I)
    text = re.sub(r"<ref[^>]*/>", "", text, flags=re.I)
    text = re.sub(r"<gallery.*?</gallery>", "", text, flags=re.S | re.I)
    text = strip_tables(text)
    text = strip_templates(text)
    text = strip_links(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"'{2,5}", "", text)
    return text


def clean_line(line):
    """Normalise one line, turning wiki list markers into dashes."""
    bullet = re.match(r"^([*#:;]+)\s*(.*)$", line)
    if bullet:
        indent = "  " * (len(bullet.group(1)) - 1)
        return f"{indent}- {bullet.group(2).strip()}"
    return line.strip()


def normalise_heading(raw):
    """Strip bold markers and spacing from a heading for comparison."""
    return re.sub(r"'{2,5}", "", raw).strip().lower()


def split_sections(text):
    """Yield (heading, lines) pairs; the lead section has an empty heading."""
    sections, heading, lines = [], "", []
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            sections.append((heading, lines))
            heading, lines = re.sub(r"'{2,5}", "", match.group(1)).strip(), []
        else:
            lines.append(line)
    sections.append((heading, lines))
    return sections


def render_sections(text):
    """Build the document body, dropping non-prose sections."""
    chunks, kept = [], []
    for heading, lines in split_sections(text):
        if normalise_heading(heading) in DROP_SECTIONS:
            continue
        body = "\n".join(clean_line(line) for line in lines)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if not body:
            continue
        kept.append(heading or "Overview")
        chunks.append(f"== {heading} ==\n{body}" if heading else body)
    return "\n\n".join(chunks), kept


def extract_infobox(text):
    """Return the card infobox fields as a dict, or {} if absent."""
    match = INFOBOX_RE.search(text)
    if not match:
        return {}
    end = find_close(text, match.start(), "{{", "}}")
    inner = text[match.start() + 2:end - 2]
    fields = {}
    for part in split_top_level(inner)[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        clean = strip_links(strip_templates(value)).strip()
        if clean:
            fields[key.strip().lower()] = clean
    return fields


def clean_page(path):
    """Return (metadata, body_text) for one raw wikitext file."""
    raw = path.read_text(encoding="utf-8")
    title = decode_title(path)
    fields = extract_infobox(raw)
    body, sections = render_sections(strip_markup(raw))
    base, _, variant = title.partition("/")
    metadata = {
        "title": title,
        "slug": path.stem,
        "card": base,
        "variant": variant or None,
        "rarity": fields.get("rarity"),
        "type": fields.get("type"),
        "cost": fields.get("cost"),
        "arena": fields.get("arena"),
        "release_date": fields.get("releasedate"),
        "sections": sections,
        "source": str(path),
        "chars": len(body),
    }
    return metadata, body


def clean_all():
    """Clean every raw page into docs/. Returns the list of metadata dicts."""
    DOCS_DIR.mkdir(exist_ok=True)
    written = []
    for path in sorted(RAW_DIR.glob("*.wikitext")):
        metadata, body = clean_page(path)
        (DOCS_DIR / f"{path.stem}.txt").write_text(body, encoding="utf-8")
        (DOCS_DIR / f"{path.stem}.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written.append(metadata)
    return written


def main():
    written = clean_all()
    empty = [m["title"] for m in written if m["chars"] == 0]
    total = sum(m["chars"] for m in written)
    print(f"{len(written)} documents written to {DOCS_DIR}/")
    print(f"{total:,} characters, median {sorted(m['chars'] for m in written)[len(written) // 2]:,}")
    if empty:
        print(f"empty: {', '.join(empty)}")


if __name__ == "__main__":
    main()
