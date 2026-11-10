from unittest.mock import Mock
import os
import pytest
from langchain_core.documents import Document
from src.retrieval.reranker import Reranker,RerankerError


def matches():return [{"document":Document(page_content='source '+str(i),metadata={"chunk_id":str(i),"test_code":"DEMO"}),"score":.9-i*.1} for i in range(3)]


def test_order_pairs_metadata_scores_stability_and_latency():
    backend=Mock();backend.predict.return_value=[1,3,3]
    original=matches();r=Reranker(backend=backend).rerank('Question',original,2)
    assert [m['document'].metadata['chunk_id'] for m in r.matches]==['1','2']
    assert r.matches[0]['score']==original[1]['score'] and r.matches[0]['document'] is original[1]['document']
    assert 'reranker_score' not in original[1] and r.latency_seconds>=0
    assert backend.predict.call_args.args[0]==[('Question',m['document'].page_content) for m in original]


def test_empty_and_long_source_not_modified():
    backend=Mock();backend.predict.return_value=[1]
    reranker=Reranker(backend=backend)
    assert reranker.rerank('q',[]).matches==[];backend.predict.assert_not_called()
    long=Document(page_content='long '*3000,metadata={'chunk_id':'long'})
    assert len(reranker.rerank('q',[{'document':long,'score':.5}],5).matches)==1
    assert len(long.page_content)>10000


@pytest.mark.parametrize('query,candidates,k',[('',[],5),('q',[],0),('q',[{}],5),('q','bad',5)])
def test_invalid_inputs(query,candidates,k):
    with pytest.raises(ValueError):Reranker().rerank(query,candidates,k)


def test_initialization_failure_and_nonfinite_scores(monkeypatch):
    import src.retrieval.reranker as module
    monkeypatch.setattr(module,'load_cross_encoder',Mock(side_effect=RuntimeError('private')))
    with pytest.raises(RerankerError,match='No fallback'):Reranker().rerank('q',matches())
    backend=Mock();backend.predict.return_value=[float('nan')]*3
    with pytest.raises(RerankerError):Reranker(backend=backend).rerank('q',matches())


@pytest.mark.integration
def test_optional_real_local_cross_encoder():
    if os.getenv('LABRAG_RUN_LOCAL_RERANKER')!='1':pytest.skip('Explicit local model opt-in required')
    result=Reranker().rerank('What is the volume?',matches())
    assert len(result.matches)==3
    doc=Document(page_content='long source '*2000,metadata={'chunk_id':'long'})
    assert Reranker().rerank('What is the volume?',[{'document':doc,'score':.5}]).matches[0]['document'].page_content==doc.page_content


def test_cache_constructor_cpu_limit_and_duplicate_ids(monkeypatch):
    import sys
    from types import SimpleNamespace
    from src.retrieval.reranker import load_cross_encoder
    load_cross_encoder.cache_clear()
    factory=Mock()
    monkeypatch.setitem(sys.modules,'sentence_transformers',SimpleNamespace(CrossEncoder=factory))
    assert load_cross_encoder('test-model') is load_cross_encoder('test-model')
    assert factory.call_count==1 and factory.call_args.kwargs['device']=='cpu' and factory.call_args.kwargs['max_length']==512
    load_cross_encoder.cache_clear()
    backend=Mock();backend.predict.return_value=[1,2,3]
    candidates=matches()
    result=Reranker(backend=backend).rerank('q',[candidates[0],*candidates])
    assert len(result.matches)==3
