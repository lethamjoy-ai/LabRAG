import copy
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from langchain_core.documents import Document
from src.rag.quote_mismatch import diagnose_quote
from src.rag.citation_manager import selection_diagnostics,validate_selection,restore_source_whitespace
from src.llm.groq_service import AnswerSelection


def docs(text='alpha beta gamma delta'):
    return {'S1':Document(page_content=text,metadata={'corpus_type':'synthetic','document_id':'fixture','source_content_hash':'hash',
        'test_code':'CA','page_number':1,'start_char':0,'end_char':len(text)})}


@pytest.mark.parametrize('text,quote,category',[
    ('alpha beta gamma delta','alpha new beta gamma delta','Quote contains inserted words'),
    ('alpha extra beta gamma delta','alpha beta gamma delta','Quote omits source words'),
    ('alpha beta gamma delta','alpha changed gamma delta','Quote substitutes words'),
    ('Specimen Volume: 0.5 mL','Specimen Volume - 0.5 mL','Quote changes punctuation'),
    ('Preferred: Serum gel\nAcceptable: Red top','Collection Container/Tube:\nPreferred: Serum gel\nAcceptable: Red top','Quote includes a heading absent from the cited chunk'),
])
def test_redacted_edit_classification_does_not_accept_quote(text,quote,category):
    sources=docs(text);selection=AnswerSelection(excerpts=[{'source_id':'S1','quote':quote}])
    before=copy.deepcopy(selection)
    detail=diagnose_quote(quote,'S1',sources)
    assert detail['category']==category
    assert sum(detail[k] for k in ('token_insertions','token_deletions','token_substitutions'))>0
    assert detail['offsets']
    assert not validate_selection(selection,sources) and selection==before
    assert quote not in json.dumps(detail)


def test_whitespace_repair_and_wrong_source_remain_different():
    sources=docs('alpha\n    beta gamma')
    selection=AnswerSelection(excerpts=[{'source_id':'S1','quote':'alpha beta gamma'}])
    assert diagnose_quote(selection.excerpts[0].quote,'S1',sources)['category']=='Whitespace-only differences'
    assert not validate_selection(selection,sources)
    assert validate_selection(restore_source_whitespace(selection,sources),sources)
    sources['S2']=Document(page_content='something unrelated')
    result=diagnose_quote('alpha beta gamma','S2',sources)
    assert result['alternate_source_ids']==['S1']
    assert result['category']=='Quote matches a different supplied source'
    bad=AnswerSelection(excerpts=[{'source_id':'S2','quote':'alpha beta gamma'}])
    assert not validate_selection(restore_source_whitespace(bad,sources),sources)


def test_cross_chunk_is_diagnostic_only_and_requires_same_source():
    sources=docs('alpha beta')
    sources['S2']=Document(page_content='gamma delta',metadata={**sources['S1'].metadata,'start_char':11,'end_char':22})
    quote='alpha beta gamma delta'
    assert diagnose_quote(quote,'S1',sources)['cross_chunk_source_ids']==['S1','S2']
    assert diagnose_quote(quote,'S1',sources)['category']=='Quote spans multiple chunk boundaries'
    selection=AnswerSelection(excerpts=[{'source_id':'S1','quote':quote}])
    assert not validate_selection(selection,sources)
    sources['S2'].metadata['test_code']='OTHER'
    assert not diagnose_quote(quote,'S1',sources)['cross_chunk_source_ids']


def test_paraphrase_unknown_cause_and_private_text_are_not_exported():
    sources=docs('The patient must fast for eight hours')
    quote='Fasting should last eight hours for the patient'
    selection=AnswerSelection(excerpts=[{'source_id':'S1','quote':quote}])
    assert not validate_selection(restore_source_whitespace(selection,sources),sources)
    secret='gsk_SECRET patient Alice Smith DOB 2000-01-01'
    diagnostic=selection_diagnostics(AnswerSelection(excerpts=[{'source_id':'S1','quote':secret}]),sources)
    serialized=json.dumps(diagnostic)
    assert 'gsk_SECRET' not in serialized and 'Alice' not in serialized and '2000-01-01' not in serialized
    assert diagnostic['excerpts'][0]['quote'] is None
    assert diagnose_quote('entirely disjoint vocabulary','S1',sources)['category']=='Mismatch cause cannot be determined'


def test_one_invalid_excerpt_still_rejects_whole_selection():
    sources=docs()
    selection=AnswerSelection(excerpts=[{'source_id':'S1','quote':'alpha beta'}, {'source_id':'S1','quote':'invented words'}])
    assert not validate_selection(selection,sources)
    assert selection_diagnostics(selection,sources)['quote_validation']=='rejected'


def test_runner_records_whitespace_restoration_success(tmp_path):
    from tests.test_evaluation_runner import runner
    from src.retrieval.semantic_retriever import SearchResult
    engine=runner(tmp_path)
    doc=Document(page_content='Minimum specimen volume:\n   0.25 mL.',metadata={'test_code':'CA','test_name':'Synthetic Calcium','chunk_id':'fixed-ca','document_id':'fixture','corpus_type':'synthetic'})
    engine.retriever=Mock();engine.retriever.search.return_value=SearchResult('q','fixed','fixed-chunks',[{'document':doc,'score':.8}],0)
    engine.service=Mock();engine.service.select.return_value=AnswerSelection(excerpts=[{'source_id':'S1','quote':'Minimum specimen volume: 0.25 mL.'}])
    row=engine.run(configurations=['A'],mode='rag')['records'][0]
    assert row['status']=='Completed'
    assert row['raw_selection_diagnostics']['excerpts'][0]['mismatch_diagnostics']['category']=='Whitespace-only differences'
    assert row['selection_diagnostics']['excerpts'][0]['mismatch_diagnostics']['whitespace_restoration_succeeded'] is True


def test_historical_quotes_are_not_reconstructed_or_modified():
    from src.evaluation.dashboard_reporting import comparison_rows
    path=Path('evaluation/results/20261106T010928_423a4f7e6d.json')
    if not path.exists():pytest.skip('Optional saved report')
    before=path.read_bytes();report=json.loads(before)
    comparison_rows(report)
    assert path.read_bytes()==before
    assert all('mismatch_diagnostics' not in e for row in report['records'] for e in (row.get('selection_diagnostics') or {}).get('excerpts',[]))
