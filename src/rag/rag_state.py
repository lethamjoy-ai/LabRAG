from typing import TypedDict, Any


class RAGState(TypedDict, total=False):
    answer_generation_mode: str
    extractive_answer: str
    display_answer: str
    synthesis_diagnostics: dict
    selection_diagnostics: dict
    candidate_matches: list
    policy_blocked: bool
    evidence_status: str
    retrieval_details: dict
    question: str
    history: list[dict]
    query: str
    matches: list
    sources: dict
    selection: Any
    answer: str
    sufficient: bool
    reason: str
    citations: list[dict]
    missing_information: list[str]
    conflicting_evidence: list[str]
    trace: list[dict]
    metrics: dict
    namespace: str
    status: str
