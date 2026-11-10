"""Conservative text normalization; never infer clinical information."""
import re
from collections import Counter
from bs4 import BeautifulSoup
from langchain_core.documents import Document


def clean_laboratory_document(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        raise ValueError("Document text must be a string or None.")
    if re.search(r"<(?:html|body|h[1-6]|p|table|nav|script)\b", text, re.I):
        soup = BeautifulSoup(text, "html.parser")
        for node in soup(["script", "style", "nav", "footer", "noscript"]):
            node.decompose()
        for row in soup.find_all("tr"):
            row.replace_with(" | ".join(c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])) + "\n")
        text = soup.get_text("\n", strip=True)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ").replace("\x00", "")
    # Retain tabs and multi-space columns rather than collapsing table layout.
    pages = text.split("\f")
    edges = Counter()
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        edges.update(set(lines[:1] + lines[-1:]))
    boilerplate = re.compile(r"^(?:https://\S+|copyright\b.*|©.*|printed on\b.*)$", re.I)
    output = []
    for page in pages:
        lines = page.splitlines()
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        boundary = set(nonempty[:1] + nonempty[-1:])
        for i, line in enumerate(lines):
            value = line.strip()
            if i in boundary and len(pages) >= 3:
                counter = re.fullmatch(r"page\s+\d+\s+(?:of|/)\s+\d+", value, re.I)
                if counter or (edges[value] >= 3 and boilerplate.fullmatch(value)):
                    continue
            output.append(line.rstrip())
        output.append("")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()


def clean_documents(documents):
    """Apply page-aware cleaning per source; preserve uncleaned text for review."""
    documents = list(documents)
    groups = {}
    for index, doc in enumerate(documents):
        groups.setdefault(doc.metadata.get("document_id", f"anonymous-{index}"), []).append(doc)
    repeated = {}
    for identity, group in groups.items():
        edges = Counter()
        for doc in group:
            lines = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
            edges.update(set(lines[:1] + lines[-1:]))
        repeated[identity] = {line for line, n in edges.items() if n >= 3 and
            re.fullmatch(r"(?:https://\S+|copyright\b.*|©.*|printed on\b.*)", line, re.I)}
    result = []
    for index, doc in enumerate(documents):
        original = doc.metadata.get("original_text", doc.page_content)
        identity = doc.metadata.get("document_id", f"anonymous-{index}")
        lines = doc.page_content.splitlines()
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        edges = set(nonempty[:1] + nonempty[-1:])
        if len(groups[identity]) >= 3:
            lines = [line for i, line in enumerate(lines) if not (i in edges and
                (line.strip() in repeated[identity] or re.fullmatch(r"page\s+\d+\s+(?:of|/)\s+\d+", line.strip(), re.I)))]
        result.append(Document(page_content=clean_laboratory_document("\n".join(lines)),
                               metadata={**doc.metadata, "original_text": original}))
    return result
