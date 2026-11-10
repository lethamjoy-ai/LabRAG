"""Local-file parsing, conservative cleaning, provenance, and chunking."""
import hashlib
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_BYTES = 20 * 1024 * 1024


def clean_text(text: str) -> str:
    # Preserve line breaks, signs, decimals, units, and reference intervals.
    text = text.replace("\x00", "").replace("\u00a0", " ")
    text = re.sub(r"[^\S\n]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_document(data: bytes, filename: str, metadata: dict | None = None) -> list[Document]:
    if len(data) > MAX_BYTES:
        raise ValueError("Document exceeds the 20 MB limit.")
    suffix = Path(filename).suffix.lower()
    base = dict(metadata or {})
    base.update(filename=Path(filename).name, document_id=hashlib.sha256(data).hexdigest())
    base.setdefault("source", Path(filename).name)
    pages = []
    if suffix == ".pdf":
        import pymupdf
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise ValueError("Password-protected PDFs are unsupported; upload an unlocked copy.")
            for number, page in enumerate(pdf, 1):
                # Sorted extraction preserves readable lines; tables remain best-effort text.
                pages.append((number, page.get_text("text", sort=True)))
    elif suffix in {".html", ".htm"}:
        soup = BeautifulSoup(data.decode("utf-8-sig"), "html.parser")
        for node in soup(["script", "style", "nav", "footer", "noscript"]):
            node.decompose()
        for row in soup.find_all("tr"):
            row.replace_with(" | ".join(cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])) + "\n")
        pages = [(1, soup.get_text("\n", strip=True))]
    elif suffix == ".txt":
        pages = [(1, data.decode("utf-8-sig"))]
    else:
        raise ValueError("Supported formats: PDF, HTML, and UTF-8 TXT.")
    docs = [Document(page_content=clean_text(text), metadata={**base, "page": n})
            for n, text in pages if clean_text(text)]
    if not docs:
        raise ValueError("No extractable text. Scanned PDFs require OCR before upload.")
    return docs


def chunk_documents(documents: list[Document], size: int = 1000, overlap: int = 150) -> list[Document]:
    if size < 100 or not 0 <= overlap < size:
        raise ValueError("Chunk size must be >=100 and overlap must be smaller than size.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                                              add_start_index=True)
    chunks = splitter.split_documents(documents)
    for doc in chunks:
        identity = json.dumps({"metadata": doc.metadata, "text": doc.page_content}, sort_keys=True)
        doc.metadata["chunk_id"] = hashlib.sha256(identity.encode()).hexdigest()
    return chunks


def load_manifest(data: bytes) -> list[dict]:
    records = json.loads(data)
    if not isinstance(records, list):
        raise ValueError("Manifest must be a JSON list.")
    for record in records:
        if not isinstance(record, dict) or not all(isinstance(record.get(k), str) and record[k].strip()
                for k in ("filename", "test_name", "provider", "source_url")):
            raise ValueError("Each manifest entry needs filename, test_name, provider, source_url strings.")
        if not record["source_url"].startswith("https://"):
            raise ValueError("Manifest source_url must use HTTPS.")
        if "metadata" in record and not isinstance(record["metadata"], dict):
            raise ValueError("Manifest metadata must be an object.")
    names = [r["filename"] for r in records]
    if len(names) != len(set(names)):
        raise ValueError("Manifest filenames must be unique.")
    return records
