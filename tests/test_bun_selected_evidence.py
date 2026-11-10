"""Replay public BUN evidence; do not infer specimen type from a tube or title."""
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from config.settings import Settings
from src.llm.groq_service import AnswerSelection
from src.rag.citation_manager import validate_selection
from src.rag.conversational_answer import extract_facts, synthesize, CONVERSATIONAL
from src.rag.rag_graph import build_rag_graph
from src.retrieval.semantic_retriever import SearchResult

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/bun_selected_evidence.json').read_text(encoding='utf-8'))
QUESTION = FIXTURE['provenance']['question']


@pytest.fixture(params=FIXTURE['cases'], ids=lambda c: c['strategy'])
def case(request):
    value = request.param
    sources = {}
    for sid, source in value['sources'].items():
        assert hashlib.sha256(source['text'].encode()).hexdigest() == source['text_sha256']
        assert source['metadata']['test_code'] == 'BUN'
        sources[sid] = Document(page_content=source['text'], metadata=source['metadata'])
    return value, sources


def test_exact_openai_selection_is_missing_explicit_specimen_type(case):
    value, sources = case
    selection = AnswerSelection(excerpts=value['selected_openai_excerpts'])
    assert len(selection.excerpts) == 1 and selection.excerpts[0].source_id == 'S1'
    assert validate_selection(selection, sources)
    kinds = {f.kind for f in extract_facts(selection, sources)}
    assert kinds == {'preferred_tube', 'acceptable_tube', 'submission_container', 'specimen_volume'}
    assert 'Serum' in sources['S3'].page_content
    result = synthesize(QUESTION, selection, sources)
    assert not result['validated'] and result['answer'] == ''
    assert all(c['source_id'] == 'S1' for c in result['claims'])
    # Available but unselected S3, the title suffix Serum, and Serum gel are
    # insufficient to add an unsupported specimen-type claim to this selection.
    assert not any(c['kind'] == 'specimen_type' for c in result['claims'])


def test_previous_two_excerpt_selection_supports_conversational_specimen_answer(case):
    value, sources = case
    selection = AnswerSelection(excerpts=value['selected_previous_extractive_excerpts'])
    assert validate_selection(selection, sources)
    result = synthesize(QUESTION, selection, sources)
    assert result['validated']
    assert 'serum as the specimen type' in result['answer']
    specimen = [c for c in result['claims'] if c['kind'] == 'specimen_type']
    assert len(specimen) == 1 and specimen[0]['source_id'] == 'S3'
    assert specimen[0]['data']['value'] == 'Serum'
    assert '7 days' not in result['answer'] and '365 days' not in result['answer']


def test_bun_stability_table_parses_carry_forward_specimen_correctly(case):
    _, sources = case
    selection = AnswerSelection(excerpts=[{'source_id': 'S3', 'quote': sources['S3'].page_content}])
    result = synthesize('How long can BUN serum be refrigerated or frozen?', selection, sources)
    assert result['validated']
    storage = [c for c in result['claims'] if c['kind'] == 'storage']
    assert {(c['data']['temperature'], c['data']['duration'], c['data']['unit']) for c in storage} == {
        ('Refrigerated (preferred)', '7', 'days'), ('Frozen', '365', 'days')}
    assert all(c['source_id'] == 'S3' and c['data']['specimen'] == 'Serum' for c in storage)


@pytest.mark.parametrize('selected,expected', [
    ('selected_openai_excerpts', 'abstained'), ('selected_previous_extractive_excerpts', 'answered')])
def test_complete_graph_preserves_quote_validation_and_citation_support(case, selected, expected):
    value, sources = case
    retriever, llm = Mock(), Mock()
    retriever.search.return_value = SearchResult(QUESTION, value['strategy'], value['strategy']+'-chunks',
        [{'document': doc, 'score': .8} for doc in sources.values()], .01)
    llm.select.return_value = AnswerSelection(excerpts=value[selected])
    result = build_rag_graph(retriever, llm, Settings(), strategy=value['strategy'],
        raw_retrieval=True, answer_generation_mode=CONVERSATIONAL).invoke({'question': QUESTION})
    assert result['status'] == expected
    assert result['selection_diagnostics']['quote_validation'] == 'passed'
    if expected == 'abstained':
        assert result['citations'] == []
        assert result['selection_diagnostics']['failure_category'] == 'Synthesis facts unresolved'
    else:
        assert {c['source_id'] for c in result['citations']} == {'S1', 'S3'}
    llm.select.assert_called_once()
    retriever.search.assert_called_once()
