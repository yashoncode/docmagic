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


def test_header_detection():
    # mimic a real ERP export: title/metadata block, blank row, then the table
    from app import excel_frames

    wb = Workbook()
    ws = wb.active
    ws.append(["Inventory Report", "As on: 12 June, 26"])
    ws.append(["Client", "BP Oil Mills Ltd."])
    ws.append(["Contract", "#1251"])
    ws.append([])
    ws.append(["Item", "Qty", "Rate"])
    ws.append(["Oil drum", 40, 1200])
    ws.append(["Pallet", 15, 300])
    import io

    buf = io.BytesIO()
    wb.save(buf)
    frames = excel_frames(buf.getvalue())
    df = next(iter(frames.values()))
    assert list(df.columns) == ["Item", "Qty", "Rate"]
    assert len(df) == 2
    assert df["Qty"].sum() == 55  # numeric dtype restored


if __name__ == "__main__":
    test_splitting()
    test_retrieval_roundtrip()
    test_excel_loader()
    test_header_detection()
    print("ok")
