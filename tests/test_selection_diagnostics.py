import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from langchain_core.documents import Document
from config.settings import Settings
from src.llm.groq_service import AnswerSelection,GroqService,InvalidStructuredGeneration
from src.rag.citation_manager import validate_selection,selection_diagnostics
from src.rag.rag_graph import build_rag_graph
from src.retrieval.semantic_retriever import SearchResult


def sources():
    return {'S1':Document(page_content='Specimen Minimum Volume: 0.25 mL',metadata={
        'test_name':'Calcium, Total, Serum','test_code':'CA','chunk_id':'fixture','corpus_type':'synthetic'})}


@pytest.mark.parametrize('excerpts,reason,valid',[
    ([], 'No excerpts selected',False),
    ([{'source_id':'S99','quote':'Specimen Minimum Volume: 0.25 mL'}],'Unknown source identifier',False),
    ([{'source_id':'S1','quote':'Specimen Minimum Volume: 0.5 mL'}],'Quote does not match the cited passage',False),
    ([{'source_id':'S1','quote':'Specimen Minimum Volume: 0.25 mL'}],None,True),
    ([{'source_id':'S1','quote':'   '}],'Other validation failure',False),
])
def test_diagnostics_preserve_existing_exact_quote_decision(excerpts,reason,valid):
    selection=AnswerSelection(excerpts=excerpts)
    details=selection_diagnostics(selection,sources())
    assert validate_selection(selection,sources())==valid
    assert details['excerpt_count']==len(excerpts) and details['failure_category']==reason
    assert details['quote_validation']==('passed' if valid else 'rejected')
    if valid:assert details['excerpts'][0]['quote']==excerpts[0]['quote']


def test_sensitive_unmatched_payload_is_not_retained():
    secret='gsk_SECRET patient Alice Smith DOB 2000-01-01'
    selection=AnswerSelection(excerpts=[{'source_id':secret,'quote':secret}])
    serialized=json.dumps(selection_diagnostics(selection,sources()))
    assert secret not in serialized and 'Alice' not in serialized and 'gsk_' not in serialized
    private=sources();private['S1'].metadata['corpus_type']='private'
    detail=selection_diagnostics(AnswerSelection(excerpts=[{'source_id':'S1','quote':private['S1'].page_content}]),private)
    assert detail['excerpts'][0]['quote'] is None and detail['quote_validation']=='passed'


@pytest.mark.parametrize('answer,status,reason',[
    ({'excerpts':[]},'abstained','No excerpts selected'),
    ({'excerpts':[{'source_id':'S99','quote':'anything'}]},'abstained','Unknown source identifier'),
    ({'excerpts':[{'source_id':'S1','quote':'Specimen Minimum Volume: 0.5 mL'}]},'abstained','Quote does not match the cited passage'),
    ({'excerpts':[{'source_id':'S1','quote':'Specimen Minimum Volume: 0.25 mL'}]},'answered',None),
])
def test_graph_reports_post_generation_rejection(answer,status,reason):
    retriever=Mock();retriever.search.return_value=SearchResult('q','fixed','fixed-chunks',
        [{'document':sources()['S1'],'score':.8}],0)
    # Exercise downstream legacy quote validation independently of the new
    # Groq ID-only wire schema (covered in test_excerpt_spans).
    generator=Mock();generator.select.return_value=AnswerSelection.model_validate(answer)
    result=build_rag_graph(retriever,generator,Settings(),raw_retrieval=True).invoke({'question':'What is the minimum volume for CA?'})
    details=result['selection_diagnostics']
    assert result['status']==status and details['final_answer_status']==status
    assert details['generation_executed'] and details['evidence_gate_passed']
    assert details['failure_category']==reason


def test_legitimate_evidence_abstention_does_not_generate():
    retriever=Mock();retriever.search.return_value=SearchResult('q','fixed','fixed-chunks',[],0)
    service=Mock()
    result=build_rag_graph(retriever,service,Settings(),raw_retrieval=True).invoke({'question':'What is the minimum volume for CA?'})
    assert result['selection_diagnostics']['failure_category']=='Evidence insufficient'
    assert not result['selection_diagnostics']['generation_executed']
    service.select.assert_not_called()


def test_malformed_response_sanitized_and_recorded_as_failed(tmp_path):
    from tests.test_evaluation_runner import runner
    backend=Mock();backend.invoke.return_value={'excerpts':[{'quote':'gsk_SECRET patient Alice'}]}
    engine=runner(tmp_path);engine.service=GroqService(Settings(),backend)
    result=engine.run(mode='rag',configurations=['A'])
    row=result['records'][0]
    assert row['status']=='Failed'
    assert row['selection_diagnostics']['failure_category']=='Invalid structured generation response'
    assert row['selection_diagnostics']['final_answer_status']=='failed'
    assert 'gsk_SECRET' not in json.dumps(result) and 'Alice' not in json.dumps(result)


def test_parser_failure_never_exports_provider_payload():
    from langchain_core.exceptions import OutputParserException
    backend=Mock();backend.invoke.side_effect=OutputParserException('gsk_SECRET patient Alice')
    with pytest.raises(InvalidStructuredGeneration) as caught:
        GroqService(Settings(),backend).select([])
    assert 'gsk_SECRET' not in str(caught.value)
    assert caught.value.diagnostics['generation_executed']


def test_runner_persists_selection_and_no_longer_calls_it_incorrect_refusal(tmp_path):
    from tests.test_evaluation_runner import runner
    engine=runner(tmp_path);engine.service=Mock();engine.service.select.return_value=AnswerSelection(excerpts=[])
    result=engine.run(mode='rag',configurations=['A'])
    row=result['records'][0]
    assert row['selection_diagnostics']['excerpt_count']==0
    assert row['raw_selection_diagnostics']['excerpt_count']==0
    assert row['answer_quality']['abstention']['classification'].startswith('Post-generation abstention')
    assert row['answer_quality']['correctness']['score'] is None


def test_historical_reports_are_not_backfilled():
    from src.evaluation.dashboard_reporting import comparison_rows
    path=Path('evaluation/results/20261106T002751_d39ad45bfa.json')
    if not path.exists():pytest.skip('Optional saved pilot report')
    before=path.read_bytes();run=json.loads(before)
    comparison_rows(run)
    assert path.read_bytes()==before
    assert all('selection_diagnostics' not in r for r in run['records'])
