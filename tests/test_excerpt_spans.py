import copy
import json
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from config.settings import Settings
from src.llm.groq_service import AnswerSelection, GroqService, InvalidStructuredGeneration
from src.rag.answer_generator import generate_answer
from src.rag.citation_manager import validate_selection, selection_diagnostics
from src.rag.excerpt_spans import excerpt_spans


def request(text='Minimum volume:\n  0.25 mL.', sid='S1'):
    source = {'source_id': sid, 'text': text, 'verified_excerpts': excerpt_spans(sid, text)}
    return [HumanMessage(content=json.dumps({'retrieved_evidence': [source]}))]


def service(response):
    backend = Mock()
    backend.invoke.return_value = response
    return GroqService(Settings(), backend)


def test_ids_resolve_to_original_whitespace_and_validate():
    text = 'Minimum volume:\n  0.25 mL.'
    selected = service({'selected_excerpt_ids': ['S1_E1']}).select(request(text))
    assert selected.excerpts[0].quote == text
    sources = {'S1': Document(page_content=text)}
    assert validate_selection(selected, sources)
    assert selected.selection_protocol == 'verified_excerpt_ids_v1'
    assert selection_diagnostics(selected, sources)['selection_protocol'] == 'verified_excerpt_ids_v1'


@pytest.mark.parametrize('response', [
    {'selected_excerpt_ids': ['S2_E1']},  # Wrong source
    {'selected_excerpt_ids': ['S1_E999']},  # Invented span
    {'selected_excerpt_ids': ['S1_E1+S2_E1']},  # Cross-source invented ID
    {'selected_excerpt_ids': ['S1_E1'], 'start': -1, 'end': 9999},
    {'selected_excerpt_ids': ['S1_E1'], 'quote': 'Minimum 0.25 mL.'},  # Omission
    {'selected_excerpt_ids': ['S1_E1'], 'quote': 'Minimum volume: exactly 0.25 mL.'},
    {'excerpts': [{'source_id': 'S1', 'quote': 'Minimum volume: 0.25 mL.'}]},
])
def test_untrusted_text_offsets_and_identifiers_are_rejected(response):
    with pytest.raises(InvalidStructuredGeneration):
        service(response).select(request())


def test_invalid_supplied_offset_rejected_before_backend():
    messages = request()
    data = json.loads(messages[0].content)
    data['retrieved_evidence'][0]['verified_excerpts'][0]['end'] += 1
    messages[0].content = json.dumps(data)
    engine = service({'selected_excerpt_ids': ['S1_E1']})
    with pytest.raises(InvalidStructuredGeneration):
        engine.select(messages)
    engine.backend.invoke.assert_not_called()


def test_spans_cover_original_and_never_cross_sources_or_merge():
    text = ('Specimen requirements:\nPreferred: Serum gel\n\n' * 100)
    spans = excerpt_spans('S1', text)
    assert ''.join(s['text'] for s in spans) == text
    assert all(s['text'] == text[s['start']:s['end']] and len(s['text']) <= 1200 for s in spans)
    sources = {'S1': Document(page_content=text), 'S2': Document(page_content='Different test')}
    engine = service({'selected_excerpt_ids': ['S1_E1', 'S1_E2', 'S2_E1']})
    selected = generate_answer(engine, 'Compare requirements', sources)
    assert [e.source_id for e in selected.excerpts] == ['S1', 'S1', 'S2']
    assert selected.excerpts[0].quote == spans[0]['text']
    assert validate_selection(selected, sources)


def test_empty_selection_duplicate_ids_and_legacy_compatibility():
    assert not service({'selected_excerpt_ids': []}).select(request()).excerpts
    assert len(service({'selected_excerpt_ids': ['S1_E1', 'S1_E1']}).select(request()).excerpts) == 1
    historical = {'excerpts': [{'source_id': 'S1', 'quote': 'Original quote'}]}
    before = copy.deepcopy(historical)
    loaded = AnswerSelection.model_validate(historical)
    assert loaded.selection_protocol == 'exact_quote_v1'
    assert service(historical).select([]).excerpts == loaded.excerpts
    assert historical == before


def test_legacy_altered_quotes_still_rejected():
    sources = {'S1': Document(page_content='Preferred collection container: serum gel tube.')}
    for quote in ['Preferred container: serum gel tube.',
                  'Preferred collection container: special serum gel tube.',
                  'Preferred collection container: serum gel tube. Different source text']:
        selection = AnswerSelection(excerpts=[{'source_id': 'S1', 'quote': quote}])
        assert not validate_selection(selection, sources)


def test_graph_validates_resolved_id_selection_without_changing_retrieval():
    from src.rag.rag_graph import build_rag_graph
    from src.retrieval.semantic_retriever import SearchResult
    text = 'Specimen Minimum Volume:\n  0.25 mL'
    doc = Document(page_content=text, metadata={'test_name': 'Calcium, Total, Serum',
        'test_code': 'CA', 'chunk_id': 'fixture', 'corpus_type': 'synthetic'})
    retriever = Mock()
    retriever.search.return_value = SearchResult('q', 'fixed', 'fixed-chunks',
        [{'document': doc, 'score': .8}], 0)
    graph = build_rag_graph(retriever, service({'selected_excerpt_ids': ['S1_E1']}),
                            Settings(), raw_retrieval=True)
    result = graph.invoke({'question': 'What is the minimum volume for CA?'})
    assert result['status'] == 'answered'
    assert result['answer'] == text + ' [S1]'
    assert result['selection_diagnostics']['quote_validation'] == 'passed'
    assert result['selection_diagnostics']['selection_protocol'] == 'verified_excerpt_ids_v1'
    assert result['matches'][0]['document'].metadata['chunk_id'] == 'fixture'
