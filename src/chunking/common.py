"""Shared deterministic metadata, exact source spans, and token-aware splitting."""
import hashlib
import json
import re
from collections import defaultdict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.token_counter import count_tokens
from src.ingestion.test_metadata import annotate_documents, test_metadata_for_span

REQUIRED = ("document_id", "document_title", "test_name", "laboratory_provider", "source_url", "accessed_date", "content_hash")


def check_parameters(size, overlap):
    if type(size) is not int or type(overlap) is not int or size <= 0 or not 0 <= overlap < size:
        raise ValueError("Token size must be a positive integer; overlap must be an integer >=0 and smaller than size.")


def has_table(text):
    lines = text.splitlines()
    return any("|" in line or "\t" in line for line in lines) or sum(bool(re.search(r"\S {2,}\S", line)) for line in lines) >= 2


def text_spans(text, size, overlap):
    check_parameters(size, overlap)
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
        length_function=count_tokens, separators=["\n\n", "\n", " ", ""],
        keep_separator=True, strip_whitespace=False)
    previous_start = -1
    previous_end = 0
    for piece in splitter.split_text(text):
        if not piece.strip():
            continue
        # Use exact character spans rather than token counts as character offsets.
        start = text.find(piece, max(0, previous_start + 1))
        while previous_start >= 0 and start >= 0 and start < previous_end and count_tokens(text[start:previous_end]) > overlap:
            start = text.find(piece, start + 1)
        if start < 0 or start > previous_end and text[previous_end:start].strip():
            raise ValueError("Splitter could not preserve contiguous source spans; review the extracted text.")
        end = start + len(piece)
        previous_start, previous_end = start, end
        yield start, end, piece


def chunk_identity(metadata, text):
    keys = ["document_id", "source_content_hash", "source_part", "page_number", "chunking_strategy",
            "chunk_index", "start_char", "end_char", "section_heading", "chunk_size", "chunk_overlap"]
    identity = {k: metadata.get(k) for k in keys}
    identity["text"] = text
    if "test_metadata_source" in metadata:
        identity.update({k: metadata.get(k) for k in ("test_name", "test_code", "test_metadata_source")})
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def make_chunks(documents, strategy, size, overlap, parts):
    check_parameters(size, overlap)
    if any(not isinstance(doc, Document) for doc in documents):
        raise ValueError("Chunking expects LangChain Document objects.")
    documents = annotate_documents(documents)
    sequences = defaultdict(int)
    hashes = {}
    grouped = defaultdict(list)
    for doc in documents:
        grouped[doc.metadata.get("document_id", "")].append(doc.page_content)
    for key, texts in grouped.items():
        hashes[key] = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    chunks = []
    for part_index, doc in enumerate(documents):
        if not isinstance(doc, Document):
            raise ValueError("Chunking expects LangChain Document objects.")
        for start, end, heading, extra in parts(doc):
            text = extra.get("heading_prefix", "") + doc.page_content[start:end]
            if not text.strip():
                continue
            identity = doc.metadata.get("document_id", "")
            meta = {k: v for k, v in doc.metadata.items() if k != "original_text"}
            meta.update(source_part=part_index, start_char=start, end_char=end,
                source_content_hash=hashes[identity], chunk_index=sequences[identity],
                chunking_strategy=strategy, token_count=count_tokens(text), tokenizer="cl100k_base",
                chunk_size=size, chunk_overlap=overlap, section_heading=heading, **extra)
            meta.update(test_metadata_for_span(doc, start, end))
            meta["chunk_content_hash"] = hashlib.sha256(text.encode()).hexdigest()
            meta["chunk_id"] = chunk_identity(meta, text)
            sequences[identity] += 1
            chunks.append(Document(page_content=text, metadata=meta))
    return chunks
