"""Smallest check that fails if loading, splitting or retrieval breaks. No API key needed."""

import os
import tempfile

import chromadb
from langchain_core.documents import Document
from openpyxl import Workbook

from app import load_units, splitter


def test_splitting():
    doc = Document(
        page_content=" ".join(f"w{i}" for i in range(600)),
        metadata={"source": "x.pdf", "loc": "p.1"},
    )
    chunks = splitter.split_documents([doc])
    assert len(chunks) > 1
    # metadata (the citation label) must survive splitting
    assert all(c.metadata == doc.metadata for c in chunks)
    assert splitter.split_documents([]) == []


def test_retrieval_roundtrip():
    # framework-free sanity check: Chroma's built-in local embeddings
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


def test_excel_loader():
    wb = Workbook()
    ws = wb.active
    ws.title = "Rates"
    ws.append(["City", "Rate"])
    ws.append(["Chennai", 12])
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        path = f.name
    wb.save(path)
    try:
        docs = load_units(path)
        assert len(docs) == 1
        assert docs[0].page_content == "City | Rate\nChennai | 12"
        assert docs[0].metadata == {"source": os.path.basename(path), "loc": "Rates"}
    finally:
        os.remove(path)


if __name__ == "__main__":
    test_splitting()
    test_retrieval_roundtrip()
    test_excel_loader()
    print("ok")
