import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from config.settings import Settings
from src.llm.groq_service import AnswerSelection
from src.rag.citation_manager import validate_selection
from src.rag.conversational_answer import extract_facts, synthesize, CONVERSATIONAL_V2
from src.rag.evidence_coverage import supplement_selection
from src.rag.rag_graph import build_rag_graph
from src.retrieval.semantic_retriever import SearchResult

DATA=json.loads((Path(__file__).parent/'fixtures/calprotectin_bordetella_evidence.json').read_text())
ANSWERABLE=[c for c in DATA['cases'] if c['selected_excerpts']]


def sources_for(case):
    for s in case['sources'].values():assert hashlib.sha256(s['text'].encode()).hexdigest()==s['text_sha256']
    return {sid:Document(page_content=s['text'],metadata=s['metadata']) for sid,s in case['sources'].items()}


@pytest.mark.parametrize('case',ANSWERABLE,ids=lambda c:c['question_id']+c['configuration'])
def test_saved_evidence_is_answered_with_exact_sources_and_one_generation(case):
    sources=sources_for(case)
    selection=AnswerSelection(excerpts=case['selected_excerpts'])
    assert validate_selection(selection,sources)
    effective,diagnostic,valid=supplement_selection(case['question'],selection,sources)
    assert valid and validate_selection(effective,sources)
    result=synthesize(case['question'],effective,sources,extended_layouts=True)
    assert result['validated']
    if case['question_id']=='TC09.1':
        for text in ['fresh random fecal specimen, no preservatives','send immediately after collection',
                     'cannot be sent immediately, freeze specimen','7 days','72 hours']:
            assert text in result['answer']
        assert {f['data']['specimen'] for f in result['claims'] if f['kind']=='storage'}=={'Fecal'}
    else:
        assert 'Nasopharyngeal swab' in result['answer']
        assert 'Nasopharyngeal (not throat) aspirate/wash or nasal aspirate/wash' in result['answer']
        types=[f for f in result['claims'] if f['kind']=='specimen_type']
        assert {f['data']['specimen_role'] for f in types}=={'preferred','acceptable'}
        for f in types:
            text=sources[f['source_id']].page_content[f['start']:f['end']]
            assert f['data']['specimen_role'].lower() in text.lower()
        assert not diagnostic['conflict']
    retriever,llm=Mock(),Mock()
    retriever.search.return_value=SearchResult(case['question'],'section','test',
        [{'document':doc,'score':.8} for doc in sources.values()],.01)
    def select(messages):
        # A fresh retrieval invocation assigns its own source IDs. Rebind the
        # saved exact quotations to the actual supplied passages, never by score.
        evidence=json.loads(messages[-1].content)['retrieved_evidence']
        return AnswerSelection(excerpts=[{'source_id':next(s['source_id'] for s in evidence if e.quote in s['text']),
                                         'quote':e.quote} for e in selection.excerpts])
    llm.select.side_effect=select
    final=build_rag_graph(retriever,llm,Settings(),raw_retrieval=True,
        answer_generation_mode=CONVERSATIONAL_V2).invoke({'question':case['question']})
    assert final['status']=='answered' and final['selection_diagnostics']['quote_validation']=='passed'
    assert {c['source_id'] for c in final['citations']}=={f['source_id'] for f in final['synthesis_diagnostics']['claims']}
    llm.select.assert_called_once();retriever.search.assert_called_once()


def test_preferred_only_bordetella_pool_remains_incomplete():
    case=next(c for c in DATA['cases'] if c['question_id']=='TC10.1' and c['configuration']=='A')
    sources=sources_for(case)
    assert case['selected_excerpts']==[]  # Historical empty selection is preserved.
    original=AnswerSelection(excerpts=[])
    _,diag,valid=supplement_selection(case['question'],original,sources)
    assert not valid and diag['status']=='invalid_or_conflicting_original_selection'
    # A simulated valid preferred excerpt still cannot supply the absent alternative.
    preferred=AnswerSelection(excerpts=[{'source_id':'S5','quote':sources['S5'].page_content}])
    _,diag,valid=supplement_selection(case['question'],preferred,sources)
    assert not valid and any(p['qualifier']=='acceptable' for p in diag['missing_after'])


def test_bordetella_coverage_adds_missing_acceptable_role_from_retrieved_evidence():
    case=next(c for c in DATA['cases'] if c['question_id']=='TC10.1' and c['configuration']=='B')
    sources=sources_for(case)
    initial=AnswerSelection(excerpts=[case['selected_excerpts'][0]])
    effective,diag,valid=supplement_selection(case['question'],initial,sources)
    assert valid and diag['missing_before'][0]['qualifier']=='acceptable'
    assert any(e['source_id']=='S3' for e in diag['supplemental_excerpts'])
    assert validate_selection(effective,sources)


def test_collection_is_required_when_collection_and_storage_are_requested():
    case=next(c for c in DATA['cases'] if c['question_id']=='TC09.1' and c['configuration']=='B')
    source=sources_for(case)['S2']
    sources={'S1':source}
    _,diag,valid=supplement_selection(case['question'],AnswerSelection(excerpts=[{'source_id':'S1','quote':source.page_content}]),sources)
    assert not valid and any(p['kind']=='collection_instruction' for p in diag['missing_after'])


def test_unsupported_respiratory_layout_and_altered_quotes_stay_rejected():
    case=next(c for c in DATA['cases'] if c['question_id']=='TC10.1' and c['configuration']=='B')
    source=sources_for(case)['S2'].model_copy(deep=True)
    source.page_content='Specimen Type: Nasopharyngeal swab'  # Role heading absent.
    sources={'S1':source}
    initial=AnswerSelection(excerpts=[{'source_id':'S1','quote':source.page_content}])
    assert extract_facts(initial,sources,extended_layouts=True)==[]
    initial.excerpts[0].quote='Specimen Type: throat swab'
    assert not validate_selection(initial,sources)


def test_v1_parser_behavior_is_unchanged():
    case=next(c for c in DATA['cases'] if c['question_id']=='TC09.1' and c['configuration']=='B')
    result=synthesize(case['question'],AnswerSelection(excerpts=case['selected_excerpts']),sources_for(case))
    assert not result['validated']


def test_conflicting_values_within_one_specimen_role_are_still_rejected():
    case=next(c for c in DATA['cases'] if c['question_id']=='TC10.1' and c['configuration']=='B')
    original=sources_for(case)
    other=original['S3'].model_copy(deep=True)
    other.page_content=other.page_content.replace('Acceptable:', 'Preferred:')
    sources={'S1':original['S2'],'S2':other}
    selection=AnswerSelection(excerpts=[{'source_id':'S1','quote':sources['S1'].page_content}])
    _,diag,valid=supplement_selection(case['question'],selection,sources)
    assert not valid and diag['conflict']
