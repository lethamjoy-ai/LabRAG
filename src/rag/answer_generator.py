import json
from langchain_core.messages import SystemMessage, HumanMessage
from src.rag.prompt_templates import SYSTEM_PROMPT
from src.rag.citation_manager import restore_source_whitespace
from src.rag.excerpt_spans import excerpt_spans


def generate_answer(service, query, sources):
    evidence = [{"source_id": key, "metadata": doc.metadata, "text": doc.page_content,
                 "verified_excerpts": excerpt_spans(key, doc.page_content)}
                for key, doc in sources.items()]
    selection = service.select([SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=json.dumps({"question": query, "retrieved_evidence": evidence}, ensure_ascii=False, default=str))])
    from src.llm.groq_service import AnswerSelection,InvalidStructuredGeneration
    if not isinstance(selection,AnswerSelection):
        raise InvalidStructuredGeneration('Generator returned an invalid structured selection.') from None
    return restore_source_whitespace(selection, sources)
