"""Session-only context; generated answers are never retrieval evidence."""
import re


def resolve_question(question, history, max_turns=5):
    from src.rag.evidence_validator import identity_query
    question = question.strip()
    recent = history[-max_turns:]
    identity = identity_query(question)
    if identity and re.match(r"what about\b", question, re.I):
        return f"What are the specimen collection and handling requirements for {identity}?"
    if recent and not identity:
        # Read only user questions/resolved queries, never generated answers.
        context = identity_query(recent[-1].get("query", recent[-1].get("question", "")))
        if context:
            return f"Regarding {context}: {question}"
    return question


def append_turn(history, turn, max_turns=5):
    return [*history, turn][-max_turns:]
