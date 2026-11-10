"""Transparent deterministic metrics; these are proxies, not clinical validation."""
import math
import re
import time
import pandas as pd
from labrag.graph import valid_citations, ABSTENTION


def evaluate(graph, cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation requires a nonempty JSON list of cases.")
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError("Every case requires a nonempty question.")
        for key in ("expected_sources", "expected_terms"):
            if not isinstance(case.get(key), list) or not case[key] or not all(isinstance(v, str) and v for v in case[key]):
                raise ValueError(f"Every case requires a nonempty {key} string list.")
    rows = []
    for case in cases:
        started = time.perf_counter()
        result = graph.invoke({"question": case["question"]})
        sources = [d.metadata.get("source", "") for d in result["sources"]]
        expected = set(case["expected_sources"])
        relevance = [int(s in expected) for s in sources]
        hit = next((i for i, value in enumerate(relevance, 1) if value), None)
        dcg = sum(v / math.log2(i + 2) for i, v in enumerate(relevance))
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(expected), len(sources))))
        rows.append({"question": case["question"],
            "recall_at_4": len(expected.intersection(sources)) / len(expected),
            "mrr": 1 / hit if hit else 0,
            "ndcg_at_4": dcg / ideal if ideal else 0,
            "term_coverage": sum(bool(re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", result["answer"], re.I))
                                 for t in case["expected_terms"]) / len(case["expected_terms"]),
            "citation_format_valid": float(valid_citations(result["answer"], len(sources))),
            "abstained": result["answer"] == ABSTENTION,
            "latency_seconds": time.perf_counter() - started,
            "answer": result["answer"]})
    return pd.DataFrame(rows)
