"""Seed the persistent 'kb' collection with logistics basics from Wikipedia.

Run once (re-run to refresh): python kb_ingest.py
Chat then answers definitional questions ("what is cross-docking?") even when
the uploaded documents don't cover them. Uploaded docs still rank first.
"""

import time

import requests
from langchain_core.documents import Document

from app import resources, splitter

ARTICLES = [
    "Logistics",
    "Supply chain management",
    "Warehouse",
    "Warehouse management system",
    "Inventory",
    "Enterprise resource planning",
    "Customer relationship management",
    "Incoterms",
    "Third-party logistics",
    "Freight forwarder",
    "Bill of lading",
    "Cross-docking",
    "Less-than-truckload shipping",
    "Cold chain",
    "Pallet",
    "Last mile (transportation)",
]

API = "https://en.wikipedia.org/w/api.php"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "DocMagic/1.0 (portfolio RAG project)"


def fetch(title: str, attempts: int = 3) -> str:
    """Full plain-text extract of one article (redirects followed)."""
    for _ in range(attempts):
        r = SESSION.get(
            API,
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "redirects": 1,
                "format": "json",
                "titles": title,
            },
            timeout=30,
        )
        if r.status_code == 429:  # rate limited: honor the server's backoff
            time.sleep(int(r.headers.get("Retry-After", 30)))
            continue
        r.raise_for_status()
        pages = r.json()["query"]["pages"]
        return next(iter(pages.values())).get("extract", "")
    raise RuntimeError(f"still rate-limited after {attempts} attempts: {title}")


def main():
    _, _, kb_store = resources()
    kb_store.reset_collection()  # re-running refreshes instead of duplicating
    docs = []
    for title in ARTICLES:
        time.sleep(1)  # stay under wikipedia's rate limit
        text = fetch(title)
        if not text:
            print(f"skip (no extract): {title}")
            continue
        docs.append(Document(page_content=text, metadata={"source": "Wikipedia", "loc": title}))
        print(f"fetched: {title} ({len(text)} chars)")
    chunks = splitter.split_documents(docs)
    kb_store.add_documents(chunks)  # NVIDIAEmbeddings batches 50/call itself
    print(f"indexed {len(chunks)} chunks into 'kb'")


if __name__ == "__main__":
    main()
