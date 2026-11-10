from copy import deepcopy
import pytest
from langchain_core.documents import Document
from src.rag.citation_manager import citation
from src.evaluation.citation_evaluator import evaluate_citations


def source():return Document(page_content='Minimum specimen volume: 0.25 mL.',metadata={'chunk_id':'a','document_id':'doc','test_code':'CA','printed_catalog_page':590,'page_number':91})


@pytest.mark.parametrize('key,value',[('source_id','S99'),('document_id','wrong'),('test_code','CAI'),('printed_catalog_page',591),('chunk_id','wrong')])
def test_wrong_source_metadata(key,value):
    doc=source();c=citation('S1',doc);c[key]=value
    result=evaluate_citations('Minimum specimen volume: 0.25 mL. [S1]',[c],{'S1':doc},['CA'])
    assert result['validity_rate']<1


def test_structural_validity_is_not_claim_support_and_duplicates_do_not_inflate():
    doc=source();c=citation('S1',doc)
    result=evaluate_citations('Minimum specimen volume: 2.5 mL. [S1]',[c,deepcopy(c)],{'S1':doc},['CA'])
    assert result['validity_rate']==1 and result['support_rate']==0 and len(result['citations'])==1
    assert result['claim_support'][0]['classification']=='Contradicted'


def test_missing_citations_unknown_pages_and_multiple_sources():
    doc=source();doc.metadata.pop('printed_catalog_page');c=citation('S1',doc)
    assert evaluate_citations('Minimum specimen volume: 0.25 mL.',[],{'S1':doc},['CA'])['missing_citations']==1
    sources={'S1':doc,'S2':Document(page_content='Frozen 30 days.',metadata={'chunk_id':'b','document_id':'doc','test_code':'CA'})}
    result=evaluate_citations('Minimum specimen volume: 0.25 mL. [S1]\nFrozen 30 days. [S2]',[c,citation('S2',sources['S2'])],sources,['CA'])
    assert result['validity_rate']==result['support_rate']==1
