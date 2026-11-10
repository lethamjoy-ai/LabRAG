import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from config.settings import Settings
from src.llm.groq_service import AnswerSelection
from src.llm.provider_factory import create_llm
from src.rag.citation_manager import validate_selection
from src.rag.conversational_answer import synthesize, CONVERSATIONAL, CONVERSATIONAL_V2
from src.rag.evidence_coverage import supplement_selection
from src.rag.rag_graph import build_rag_graph
from src.retrieval.semantic_retriever import SearchResult

SAVED = json.loads((Path(__file__).parent/'fixtures/bun_selected_evidence.json').read_text())


def doc(text, code='BUN', name='Blood Urea Nitrogen (BUN), Serum', **extra):
    return Document(page_content=text, metadata={'test_code':code, 'test_name':name,
        'corpus_type':'synthetic', **extra})


def selected(sources, sid='S1'):
    return AnswerSelection(excerpts=[{'source_id':sid,'quote':sources[sid].page_content}])


@pytest.mark.parametrize('value', SAVED['cases'], ids=lambda c:c['strategy'])
@pytest.mark.parametrize('provider', ['groq','openai'])
def test_bun_v2_adds_verified_serum_table_with_one_provider_call(value, provider):
    sources={sid:Document(page_content=s['text'], metadata=s['metadata']) for sid,s in value['sources'].items()}
    original=AnswerSelection(excerpts=value['selected_openai_excerpts'])
    before=original.model_dump()
    effective,diagnostic,valid=supplement_selection(SAVED['provenance']['question'], original, sources)
    assert valid and validate_selection(effective,sources)
    assert original.model_dump()==before
    assert diagnostic['missing_before'][0]['kind']=='specimen_type' and diagnostic['missing_after']==[]
    assert diagnostic['supplemental_excerpts'][0]['source_id']=='S3'
    assert 'quote' not in diagnostic['supplemental_excerpts'][0]
    result=synthesize(SAVED['provenance']['question'],effective,sources)
    assert result['validated'] and 'serum as the specimen type' in result['answer']
    assert '0.5 mL' in result['answer'] and 'Serum gel' in result['answer'] and 'Plastic vial' in result['answer']
    assert 'Serum' not in json.dumps(diagnostic)
    retriever=Mock();retriever.search.return_value=SearchResult('q',value['strategy'],'test',
        [{'document':d,'score':.8} for d in sources.values()],.01)
    backend=Mock();backend.invoke.return_value={'selected_excerpt_ids':['S1_E1'],
        'missing_information':[], 'conflicting_evidence':[]}
    service=create_llm(Settings(llm_provider=provider),backend)
    result=build_rag_graph(retriever,service,Settings(),raw_retrieval=True,
        answer_generation_mode=CONVERSATIONAL_V2).invoke({'question':SAVED['provenance']['question']})
    assert result['status']=='answered' and result['answer_generation_mode']==CONVERSATIONAL_V2
    assert {c['source_id'] for c in result['citations']}=={'S1','S3'}
    assert result['selection'].model_dump()['excerpts']==before['excerpts']
    assert result['synthesis_diagnostics']['evidence_coverage']['missing_after']==[]
    backend.invoke.assert_called_once();retriever.search.assert_called_once()


@pytest.mark.parametrize('question, sources, expected', [
    ('How long can BUN serum be refrigerated or frozen?',
     {'S1':doc('Specimen Type: Serum'), 'S2':doc('Specimen Type  Temperature  Time\nSerum  Refrigerated  7 days\nFrozen  365 days')}, ['7 days','365 days']),
    ('What is the minimum specimen volume for CA?',
     {'S1':doc('Specimen Volume: 0.5 mL','CA','Calcium, Total, Serum'),
      'S2':doc('Specimen Minimum Volume: 0.25 mL','CA','Calcium, Total, Serum')}, ['0.25 mL']),
    ('What preparation is required for CPR?',
     {'S1':doc('Fasting: 8 hours, required','CPR','C-Peptide, Serum'),
      'S2':doc('For 12 hours before specimen collection, patient should not take multivitamins or dietary supplements containing biotin.','CPR','C-Peptide, Serum')}, ['8 hours','12 hours']),
    ('Compare specimen types between CA and CAI',
     {'S1':doc('Specimen Type: Serum','CA','Calcium, Total, Serum'),
      'S2':doc('Specimen Type: Serum','CAI','Calcium, Ionized, Serum')}, ['Calcium, Total','Calcium, Ionized']),
])
def test_explicit_required_fields_are_supplemented(question,sources,expected):
    effective,diagnostic,valid=supplement_selection(question,selected(sources),sources)
    assert valid and validate_selection(effective,sources)
    result=synthesize(question,effective,sources)
    assert result['validated'] and all(text in result['answer'] for text in expected)
    if 'minimum' in question:assert '0.5 mL' not in result['answer']


@pytest.mark.parametrize('extra', [
    {},
    {'S2':doc('The blood sample is suitable.')},
    {'S2':doc('Specimen Type: Serum', corpus_type='private', patient_name='PRIVATE')},
    {'S2':doc('Specimen Type: Plasma','CA','Calcium, Total, Serum')},
])
def test_missing_unsupported_untrusted_or_other_test_evidence_does_not_fill_gap(extra):
    sources={'S1':doc('Collection Container/Tube:\nPreferred: Serum gel\nSpecimen Volume: 0.5 mL'),**extra}
    effective,diagnostic,valid=supplement_selection('What specimen is required for BUN?',selected(sources),sources)
    assert not valid and not diagnostic['supplemental_excerpts']
    assert not synthesize('What specimen is required for BUN?',effective,sources)['validated']
    assert 'PRIVATE' not in json.dumps(diagnostic)


def test_conflict_is_not_resolved_by_source_order():
    sources={'S1':doc('Specimen Volume: 0.5 mL'), 'S2':doc('Specimen Type: Serum'), 'S3':doc('Specimen Type: Plasma')}
    _,diagnostic,valid=supplement_selection('What specimen is required for BUN?',selected(sources),sources)
    assert not valid and diagnostic['conflict'] and not diagnostic['supplemental_excerpts']


def test_invalid_original_quote_cannot_be_repaired():
    sources={'S1':doc('Specimen Type: Serum')}
    original=AnswerSelection(excerpts=[{'source_id':'S1','quote':'Specimen Type: Plasma'}])
    effective,_,valid=supplement_selection('What specimen is required for BUN?',original,sources)
    assert not valid and effective==original


def test_v1_stays_fail_closed_and_version_is_distinct():
    assert CONVERSATIONAL != CONVERSATIONAL_V2
    value=SAVED['cases'][0]
    sources={sid:Document(page_content=s['text'],metadata=s['metadata']) for sid,s in value['sources'].items()}
    assert not synthesize(SAVED['provenance']['question'],AnswerSelection(excerpts=value['selected_openai_excerpts']),sources)['validated']


def test_missing_storage_condition_remains_insufficient():
    sources={'S1':doc('Specimen Type  Temperature  Time\nSerum  Frozen  365 days')}
    _,diagnostic,valid=supplement_selection('How long can BUN serum be refrigerated or frozen?',selected(sources),sources)
    assert not valid and diagnostic['missing_after'][0]['qualifier']=='refrigerated'


def test_missing_preparation_component_is_not_invented():
    sources={'S1':doc('Fasting: 8 hours, required','CPR','C-Peptide, Serum')}
    _,diagnostic,valid=supplement_selection('What preparation is required for CPR?',selected(sources),sources)
    assert not valid and {p['kind'] for p in diagnostic['missing_after']}=={'biotin'}


def test_conflict_rejects_v2_graph_even_when_original_selection_is_valid():
    sources={'S1':doc('Specimen Type: Serum'), 'S2':doc('Specimen Type: Plasma')}
    retriever,llm=Mock(),Mock()
    retriever.search.return_value=SearchResult('q','section','test',
        [{'document':d,'score':.8} for d in sources.values()],.01)
    llm.select.return_value=selected(sources)
    result=build_rag_graph(retriever,llm,Settings(),raw_retrieval=True,
        answer_generation_mode=CONVERSATIONAL_V2).invoke({'question':'What specimen is required for BUN?'})
    assert result['status']=='abstained' and result['citations']==[]
    assert result['synthesis_diagnostics']['evidence_coverage']['conflict']
    llm.select.assert_called_once()
