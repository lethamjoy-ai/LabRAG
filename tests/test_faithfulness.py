from unittest.mock import Mock
import pytest
from langchain_core.documents import Document
from config.settings import Settings
from src.evaluation.faithfulness_evaluator import evaluate_faithfulness,GroqFaithfulnessJudge


@pytest.mark.parametrize('claim,status',[
    ('The minimum specimen volume is 0.25 mL.','Supported'),
    ('The minimum specimen volume is 2.5 mL.','Contradicted'),
    ('The minimum specimen volume is 0.25 mg.','Contradicted'),
    ('The minimum specimen volume is 0.25 mL and cures cancer.','Unverifiable'),
    ('The specimen is magical.','Unverifiable'),
])
def test_numeric_fields_and_ambiguous_claims(claim,status):
    sources={'S1':Document(page_content='Minimum specimen volume: 0.25 mL.',metadata={'test_code':'CA'})}
    result=evaluate_faithfulness('Volume?',claim+' [S1]',sources,['CA'])
    assert result['claims'][0]['classification']==status


def test_wrong_test_missing_evidence_multiple_claims_and_abstention():
    sources={'S1':Document(page_content='Minimum specimen volume: 0.25 mL.',metadata={'test_code':'CA'})}
    assert evaluate_faithfulness('q','Minimum specimen volume: 0.25 mL. [S1]',sources,['CAI'])['claims'][0]['classification']=='Unsupported'
    assert evaluate_faithfulness('q','Minimum volume: 0.25 mL.',{})['claims'][0]['classification']=='Unverifiable'
    result=evaluate_faithfulness('q','Minimum specimen volume: 0.25 mL. [S1]\n\nMinimum specimen volume: 2.5 mL. [S1]',sources,['CA'])
    assert result['supported_claim_rate']==.5 and result['contradicted_claims']==1
    assert evaluate_faithfulness('q','I cannot answer',{},abstained=True)['supported_claim_rate'] is None
    assert evaluate_faithfulness('q','',{})['supported_claim_rate'] is None


def test_requested_and_minimum_are_not_substituted_and_judge_errors_are_reviewable():
    sources={'S1':Document(page_content='Minimum specimen volume: 0.25 mL.',metadata={'test_code':'CA'})}
    assert evaluate_faithfulness('q','Specimen volume: 0.25 mL. [S1]',sources,['CA'])['claims'][0]['classification']!='Supported'
    backend=Mock();backend.invoke.return_value={'broken':'judge'}
    result=evaluate_faithfulness('q','Minimum specimen volume: 0.25 mL. [S1]',sources,['CA'],judge=GroqFaithfulnessJudge(Settings(),backend))
    assert result['human_review'] and 'error' in result['judge']
    backend.invoke.return_value={'claims':[{'claim_index':0,'classification':'Contradicted','reason':'Synthetic judge disagreement'}]}
    result=evaluate_faithfulness('q','Minimum specimen volume: 0.25 mL. [S1]',sources,['CA'],judge=GroqFaithfulnessJudge(Settings(),backend))
    assert result['human_review'] and result['judge']['disagreement_with_deterministic_check']
    assert result['supported_claim_rate']==1
