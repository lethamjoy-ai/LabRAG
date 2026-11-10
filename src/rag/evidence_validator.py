"""Conservative lexical gate. Similarity is never treated as confidence."""
import re
from langchain_core.documents import Document
from src.ingestion.test_metadata import detect_test_entries

STOP = set("what which is are the a an for of in to be can how long does do it its required please tell me specimen collection requirements test serum laboratory previous user question current and".split())
IDENTITY_STOP = STOP | set("frozen refrigerated storage stability stored preparation patient fast fasting biotin ordered orderable billing bill only minimum volume much preferred acceptable tube container types accepted instructions between difference transport temperature days hours blood".split())
IDENTITY_STOP -= {"serum"}
IDENTITY_STOP |= {"as"}
IDENTITY_STOP |= {"type"}
IDENTITY_STOP |= {"submission", "submit", "shipping", "shipment", "send", "sending"}
IDENTITY_STOP |= set("sample samples requirement required requirements processing process handle handling transport transportation stability store stored storage condition conditions temperature temperatures result results mean meaning need needs needed amount collect collection collection collected instruction instructions time duration limit limits acceptable accept accepted how when where why show list details tell also other more about concerning regarding patient preparations preparation prepare container containers tubes tube top tops cap caps recommend recommended preferred minimum maximum volume volumes information refrigerated refrigerate frozen freeze freezes freezing freezing freezing specimen specimens centrifuge centrifuged centrifugation aliquot aliquoted".split())
IDENTITY_STOP |= set("should prepare apply use needed need collected processed handled handling processing processing collect required recommendation recommendations explain compared compare versus vs about refrigerate freeze receive before after with at my please give information details tell".split())
FACETS = {
    "stability": (r"frozen|refrigerat|stability|stored|storage|how long", r"frozen|refrigerat|stability|storage|days|hours"),
    "preparation": (r"prepar|fast|biotin", r"prepar|fast|biotin"),
    "orderability": (r"orderable|ordered|bill only", r"orderable|bill only|billing"),
    "volume": (r"volume|minimum|how much", r"volume|\bmL\b"),
    "collection": (r"specimen|collect|tube|container", r"specimen|collect|tube|container"),
}


def policy_blocked(query):
    return bool(re.search(r"\b(diagnos\w*|treat\w*|prescrib\w*|medication|dosage)\b", query, re.I))


def requested_facets(query):
    current = query.split("Current question:")[-1]
    if current.startswith("Regarding ") and ": " in current:
        current = current.split(": ",1)[1]
    selected = [(name,pattern) for name,(question_pattern,pattern) in FACETS.items()
                if re.search(question_pattern,current,re.I)]
    # 'Specimen' in a storage/volume question is context, not a second request
    # for collection instructions. Explicit collection/container requests remain.
    if any(name!="collection" for name,_ in selected) and not re.search(r"collect|tube|container|specimen\s+(?:type|requirements?)",current,re.I):
        selected = [(name,pattern) for name,pattern in selected if name!="collection"]
    return selected


def identity_query(query):
    """Deterministic heading query derived only from user-supplied words."""
    codes = list(dict.fromkeys(re.findall(r"\b[A-Z][A-Z0-9]{1,14}\b", query)))
    codes = [code for code in codes if code not in {"PDF", "MAYO", "PCR", "DNA", "RNA"}]
    if codes:
        return " ".join(codes)
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]*", query)
    return " ".join(dict.fromkeys(word.lower() for word in words
        if word.lower() not in IDENTITY_STOP and len(word) > 1))[:500]


def source_regions(doc):
    """Use only identity labels present in this retrieved chunk, never another page.

    The Mayo catalog layout is CODE / numeric internal ID / title. Split at each
    observed header so a neighboring test's instructions cannot become its evidence.
    Body-only chunks with unresolved metadata remain unusable for clinical answers.
    """
    text = doc.page_content
    if doc.metadata.get("identity_context"):
        return [(0, len(text), doc.metadata["test_name"], doc.metadata["test_code"])]
    headers = list(re.finditer(r"(?m)^[ \t]*([A-Z][A-Z0-9]{1,14})\s*\n\d{3,10}\s*\n([^\n]{4,180})", text))
    if headers:
        return [(m.start(), headers[i + 1].start() if i + 1 < len(headers) else len(text), m[3].strip(), m[1])
                for i, m in enumerate(headers)]
    entries = detect_test_entries(text)
    if entries:
        return [(e.start, e.end, e.test_name, e.test_code) for e in entries]
    meta = doc.metadata
    if meta.get("test_name") or meta.get("test_code"):
        return [(0, len(text), meta.get("test_name", ""), meta.get("test_code", ""))]
    # Compact text fixtures / explicit inline header; not arbitrary prose acronyms.
    match = re.match(r"^([A-Z][A-Z0-9]{1,9})\s+([^\n.]{3,100})[.\n]", text)
    return [(0, len(text), match[2], match[1])] if match else []


def scope_sources(query, sources):
    """Retain only source-backed test regions matching the question identity."""
    codes = set(re.findall(r"\b[A-Z][A-Z0-9]{1,14}\b", query)) - {"PDF", "MAYO", "PCR", "DNA", "RNA"}
    words = set(re.findall(r"[a-z][a-z0-9]+", query.lower())) - IDENTITY_STOP
    words = {"feces" if word == "fecal" else word for word in words}
    comparison = bool(re.search(r"\b(compare|between|difference|versus|vs)\b", query, re.I))
    normalized_query = " ".join(re.findall(r"[a-z0-9]+", query.lower()))
    scoped = {}
    identities = set()
    matched_codes = set()
    for key, doc in sources.items():
        candidates = []
        for start, end, name, code in source_regions(doc):
            name_words = set(re.findall(r"[a-z][a-z0-9]+", name.lower())) - IDENTITY_STOP
            # All identity words are required: total/ionized, urine/serum remain distinct.
            full_name = " ".join(re.findall(r"[a-z0-9]+", name.lower()))
            qualifies = code in codes if codes else bool(words) and words <= (name_words | {code.lower()})
            # A literal complete test name establishes identity even when the
            # question contains other instruction words. Explicit codes must match.
            if not codes and full_name and re.search(r"(?:^| )" + re.escape(full_name) + r"(?: |$)", normalized_query):
                qualifies = True
            if not qualifies:
                continue
            fragment = doc.page_content[start:end]
            meta = {**doc.metadata, "test_name": name, "test_code": code}
            candidates.append(Document(page_content=fragment, metadata=meta))
            identities.add((name.casefold(), code))
            matched_codes.add(code)
        for index, candidate in enumerate(candidates):
            scoped[key if len(candidates) == 1 else f"{key}_{index + 1}"] = candidate
    if codes and not codes <= matched_codes:
        return {}, "Some requested test codes are missing from retrieved evidence."
    if not scoped:
        return {}, "The requested test identity could not be established from the retrieved passages."
    if comparison and len(identities) < 2:
        return {}, "The comparison requires independently identified evidence for both tests."
    if len(identities) > 1 and not comparison and len(codes) < 2:
        return {}, "Multiple test identities match. Specify the exact test code and specimen type."
    return scoped, "Test identities established from retrieved labels."


def validate_evidence(query, sources):
    if policy_blocked(query):
        return False, "Patient-specific diagnosis and treatment are outside this reference assistant's scope."
    if not sources:
        return False, "No matching passages were retrieved from the indexed partial catalog."
    words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9-]+", query)} - STOP
    # A literal test code/name token must be present; high scores alone do not qualify.
    relevant = []
    for doc in sources.values():
        text = doc.page_content + " " + str(doc.metadata.get("test_name", "")) + " " + str(doc.metadata.get("test_code", ""))
        tokens = set(re.findall(r"[a-z0-9-]+", text.lower()))
        if words & tokens:
            relevant.append(text)
    if not relevant:
        return False, "Retrieved passages do not establish the requested test identity."
    combined = "\n".join(relevant)
    if re.search(r"exact.*temperature|temperature.*(?:celsius|degrees|°c)|numeric.*temperature", query, re.I) and not re.search(r"[-+]?\d+(?:\.\d+)?\s*(?:°\s*[CF]|degrees?\s*(?:Celsius|Fahrenheit))", combined, re.I):
        return False, "Relevant test evidence was retrieved, but an exact numeric transport temperature is not specified in those passages."
    for _, evidence_pattern in requested_facets(query):
        if not re.search(evidence_pattern, combined, re.I):
            return False, "Retrieved passages do not contain the requested instruction."
    return True, "Lexical evidence gate passed; generation must still select supported quotes."
