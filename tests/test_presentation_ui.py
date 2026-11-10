import copy
import json
from pathlib import Path
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest
from src.ui_presentation import question_views


def test_question_views_include_missing_pairs_without_invented_metrics():
    run={'parameters':{'question_ids':['q1','q2'],'configurations':['A','B'],'mode':'rag'},
        'records':[{'question_id':'q1','configuration':'A','question':'Recorded question',
            'status':'Completed','answer':'Recorded answer','answer_quality':{'correctness':{'score':.5},'faithfulness':{'score':1}},
            'retrieval_metrics':{'hit_rate':1,'mrr':.5}}]}
    reference={'q2':{'question':'Pending question','expected_facts':['Reference fact']}}
    before=copy.deepcopy(run)
    views=question_views(run,reference)
    assert len(views)==4 and run==before
    assert views[0]['Correctness']==.5 and views[0]['Faithfulness']==1
    pending=next(r for r in views if r['Question ID']=='q2' and r['Variant']=='B')
    assert pending['Evaluation status']=='Pending / Missing'
    assert pending['Question']=='Pending question' and pending['Expected answer']=='Reference fact'
    assert pending['Generated answer']=='Not executed — pending'
    assert pending['Correctness'] is None and pending['Faithfulness'] is None
    assert 'retrieved' not in pending['_record']


def test_historical_view_does_not_invent_selection_or_expected_answer():
    run={'parameters':{'mode':'retrieval'},'records':[{'question_id':'old','configuration':'A','status':'Completed'}]}
    views=question_views(run)
    assert len(views)==1 and views[0]['Expected answer']=='Not recorded'
    assert views[0]['Generated answer']=='Not generated — retrieval-only mode'
    assert views[0]['Correctness'] is None


def test_presentation_navigation_controls_and_saved_metrics_are_read_only(tmp_path,monkeypatch):
    import src.ui_day5 as dashboard
    from tests.test_evaluation_runner import runner
    from tests.day5_helpers import synthetic_dataset
    from src.evaluation.evaluation_runner import EvaluationRunner
    from src.llm.groq_service import GroqService
    from src.vectorstore.pinecone_client import PineconeConnection
    root=tmp_path/'evaluation';root.mkdir()
    run=runner(root).run(mode='rag')
    (root/'synthetic_ground_truth.json').write_text(json.dumps(synthetic_dataset()),encoding='utf-8')
    path=root/'results'/(run['run_id']+'.json');before=path.read_bytes()
    monkeypatch.setattr(dashboard,'ROOT',tmp_path)
    forbidden=Mock(side_effect=AssertionError('No automatic execution'))
    monkeypatch.setattr(EvaluationRunner,'run',forbidden)
    monkeypatch.setattr(GroqService,'select',forbidden)
    monkeypatch.setattr(PineconeConnection,'connect',forbidden)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    assert not app.exception and not app.error
    assert [tab.label for tab in app.tabs]==['Ask LabRAG','Evaluation Results','About the Project']
    assert app.title[0].value=='LabRAG — AI-Powered Laboratory Test Assistant'
    assert any(b.label=='Send' for b in app.button)
    assert any(w.label=='Sample Questions' for w in app.selectbox)
    assert next(e for e in app.main.get('expander') if e.proto.label=='Advanced Semantic Retrieval Explorer').proto.expanded is False
    assert any(s.label=='LLM Provider' for s in app.sidebar.selectbox)
    assert next(m for m in app.metric if m.label=='Total evaluation pairs').value=='4'
    assert next(m for m in app.metric if m.label=='Completed pairs').value=='4'
    assert next(m for m in app.metric if m.label=='Missing pairs').value=='0'
    technical={'Advanced Evaluation Settings','Chat Settings','Detailed comparison & validation',
        'Knowledge Base & Document Tools','Additional Validation Reports','Advanced Strategy Comparison'}
    seen=set()
    for element in app.get('expander'):
        if element.proto.label in technical:
            seen.add(element.proto.label)
            assert not element.proto.expanded
    assert seen==technical
    assert path.read_bytes()==before
    assert any(s.value=='Evaluated Questions' for s in app.subheader)
    assert any('Future Enhancements' in s.value for s in app.subheader)
    forbidden.assert_not_called()


def test_sample_questions_use_verified_successful_saved_results(tmp_path):
    from src.ui_presentation import sample_questions
    from src.evaluation.dataset_validator import fingerprint
    from tests.day5_helpers import synthetic_dataset
    root=tmp_path/'evaluation';(root/'results').mkdir(parents=True)
    data=synthetic_dataset();data.pop('fingerprint')
    data['fingerprint']=fingerprint(data)
    (root/'sample_ground_truth.json').write_text(json.dumps(data),encoding='utf-8')
    run={'source':'live','dataset_fingerprint':data['fingerprint'],'records':[
        {'question_id':'SYN01.1','status':'Completed','answer_status':'answered','citations':[{'source_id':'S1'}]},
        {'question_id':'unknown','status':'Completed','answer_status':'answered','citations':[{'source_id':'S1'}]}]}
    (root/'results/1.json').write_text(json.dumps(run),encoding='utf-8')
    assert sample_questions(tmp_path)==[data['questions'][0]['question']]
    run['records'][0]['answer_status']='abstained'
    (root/'results/2.json').write_text(json.dumps(run),encoding='utf-8')
    # An earlier supported reference may still be offered; unknown text never is.
    assert sample_questions(tmp_path)==[data['questions'][0]['question']]
