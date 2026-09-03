"""Fetch Clash Royale strategy content from the Fandom MediaWiki API.

Card pages are scraped via the rarity categories: no single category holds
every card, but each card has exactly one rarity, so the five together
cover the set without duplicates.
"""

import sys
import time
from pathlib import Path

import requests

API_URL = "https://clashroyale.fandom.com/api.php"
USER_AGENT = "clash-rag-scraper/0.1 (personal RAG project; contact: jmaxd94@gmail.com)"
REQUEST_DELAY = 1.0

RAW_DIR = Path("raw")
TITLES_PER_FETCH = 50

CARD_CATEGORIES = [
    "Category:Champion Cards",
    "Category:Common Cards",
    "Category:Epic Cards",
    "Category:Legendary Cards",
    "Category:Rare Cards",
]

PROBE_PAGE = "Hog_Rider"


def api_get(params):
    """Call the MediaWiki API and return the parsed JSON body."""
    params = dict(params, format="json")
    response = requests.get(
        API_URL,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return response.json()


def page_categories(title):
    """Return the category titles a page belongs to."""
    data = api_get(
        {
            "action": "query",
            "prop": "categories",
            "titles": title,
            "cllimit": "max",
        }
    )
    pages = data.get("query", {}).get("pages", {})
    categories = []
    for page in pages.values():
        if "missing" in page:
            raise SystemExit(f"page not found: {title}")
        for category in page.get("categories", []):
            categories.append(category["title"])
    return sorted(categories)


def category_pages(category):
    """Return every article title in a category, following cmcontinue."""
    titles = []
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "page",
            "cmlimit": "max",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = api_get(params)
        for member in data["query"]["categorymembers"]:
            titles.append(member["title"])
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            return titles


def fetch_wikitext(titles):
    """Return {title: wikitext} for a batch of titles."""
    data = api_get(
        {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "titles": "|".join(titles),
        }
    )
    pages = data.get("query", {}).get("pages", {})
    contents = {}
    for page in pages.values():
        if "missing" in page or "revisions" not in page:
            continue
        slot = page["revisions"][0]["slots"]["main"]
        contents[page["title"]] = slot.get("*", "")
    return contents


def raw_path(title):
    """Map a page title to its file in raw/, keeping the title recoverable."""
    safe = title.replace("_", "___").replace("/", "__").replace(" ", "_")
    return RAW_DIR / f"{safe}.wikitext"


def chunked(items, size):
    """Yield successive slices of items."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


def scrape_cards():
    """Fetch every card page's wikitext into raw/. Returns the file count."""
    titles = set()
    for category in CARD_CATEGORIES:
        found = category_pages(category)
        print(f"{category}: {len(found)} pages")
        titles.update(found)

    titles = sorted(titles)
    print(f"{len(titles)} unique card pages")

    RAW_DIR.mkdir(exist_ok=True)
    written = 0
    for batch in chunked(titles, TITLES_PER_FETCH):
        for title, wikitext in fetch_wikitext(batch).items():
            raw_path(title).write_text(wikitext, encoding="utf-8")
            written += 1
        print(f"  fetched {written}/{len(titles)}")
    return written


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--categories":
        title = sys.argv[2] if len(sys.argv) > 2 else PROBE_PAGE
        for category in page_categories(title):
            print(f"  {category}")
        return
    written = scrape_cards()
    print(f"{written} pages saved to {RAW_DIR}/")


if __name__ == "__main__":
    main()
