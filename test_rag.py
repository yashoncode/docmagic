"""Smallest check that fails if chunking or retrieval breaks. No API key needed."""

import chromadb

from app import chunk_text


def test_chunking():
    words = " ".join(f"w{i}" for i in range(600))
    chunks = chunk_text(words, size=250, overlap=50)
    assert len(chunks) == 3
    # overlap: last 50 words of chunk 1 == first 50 of chunk 2
    assert chunks[0].split()[-50:] == chunks[1].split()[:50]
    assert chunk_text("") == []


def test_retrieval_roundtrip():
    db = chromadb.EphemeralClient()
    col = db.get_or_create_collection("test")
    col.add(
        ids=["1", "2"],
        documents=[
            "Warehouse storage rate is 12 rupees per square foot per month.",
            "The CRM lead pipeline has four stages: new, contacted, quoted, won.",
        ],
        metadatas=[{"source": "rates.pdf", "page": 1}, {"source": "crm.pdf", "page": 2}],
    )
    res = col.query(query_texts=["how much does storage cost?"], n_results=1)
    assert res["metadatas"][0][0]["source"] == "rates.pdf"


if __name__ == "__main__":
    test_chunking()
    test_retrieval_roundtrip()
    print("ok")
