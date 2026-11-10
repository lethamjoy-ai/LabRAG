import hashlib
import re
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from src.ingestion.text_cleaner import clean_laboratory_document
from src.ingestion.test_metadata import annotate_documents


def identify_document_type(filename, data):
    extension = Path(filename).suffix.lower()
    types = {".pdf": "pdf", ".html": "html", ".htm": "html", ".txt": "txt"}
    if extension not in types:
        raise ValueError("Upload PDF, HTML, or UTF-8 TXT.")
    if not data:
        raise ValueError("The document is empty.")
    if extension == ".pdf" and not data.startswith(b"%PDF-"):
        raise ValueError("Invalid PDF signature.")
    if extension != ".pdf" and (data.startswith(b"%PDF-") or b"\x00" in data):
        raise ValueError("File content does not match the text format.")
    return types[extension]


def normalize_whitespace(text):
    return clean_laboratory_document(text)


def load_document(data, filename, metadata=None, max_file_bytes=20 * 1024 * 1024):
    if len(data) > max_file_bytes:
        raise ValueError("Document exceeds the configured file-size limit.")
    kind = identify_document_type(filename, data)
    content_hash = hashlib.sha256(data).hexdigest()
    base = {**(metadata or {}), "document_type": kind, "content_hash": content_hash,
            "document_id": content_hash, "filename": Path(filename).name}
    try:
        if kind == "pdf":
            import pymupdf
            with pymupdf.open(stream=data, filetype="pdf") as pdf:
                if pdf.needs_pass:
                    raise ValueError("Password-protected PDFs must be unlocked before upload.")
                pages = [(i, page.get_text("text", sort=True)) for i, page in enumerate(pdf, 1)]
        else:
            text = data.decode("utf-8-sig")
            if kind == "html":
                soup = BeautifulSoup(text, "html.parser")
                for node in soup(["script", "style", "nav", "footer", "noscript"]):
                    node.decompose()
                for row in soup.find_all("tr"):
                    row.replace_with(" | ".join(c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])) + "\n")
                text = soup.get_text("\n", strip=True)
            pages = [(None, text)]
    except Exception as exc:
        raise ValueError("Could not extract document. Check encoding, corruption, or PDF password protection.") from exc
    documents = []
    for page, text in pages:
        original_text = text
        text = normalize_whitespace(text)
        if text:
            heading = text.splitlines()[0] if kind != "pdf" else ""
            documents.append(Document(page_content=text, metadata={**base, "page_number": page,
                "section_heading": heading, "original_text": original_text}))
    if not documents:
        raise ValueError("No extractable text; scanned PDFs require OCR outside this application.")
    return annotate_documents(documents)
