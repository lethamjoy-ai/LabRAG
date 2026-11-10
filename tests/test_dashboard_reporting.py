import copy
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from src.evaluation.dashboard_reporting import coverage,comparison_rows,matching_questions,reciprocal_rank_heatmap


def fixture_run():
    records=[]
    for c in 'ABCD':
        for i,status in enumerate(['Completed','Excluded','Incompatible','Failed']):
            row={'question_id':f'Q{i}','question':f'Question {i}','configuration':c,'status':status}
            if status=='Completed':row.update(retrieval_metrics={'scored':True,'hit_rate':0,'mrr':0,'precision':0,'first_relevant_rank':None})
            records.append(row)
    return {'parameters':{'question_ids':['Q0','Q1','Q2','Q3'],'configurations':list('ABCD')},'records':records}


def test_unique_coverage_and_record_counts():
    counts,configs,details=coverage(fixture_run())
    assert counts=={'Selected questions':4,'Scored questions':1,'Excluded questions':1,'Incompatible questions':1,'Unverified questions':0,'Failed questions':1,'Coverage percentage':25}
    assert all(c['Records']==4 and c['Scored']==1 and c['Excluded']==1 for c in configs)
    assert details['same_scored_set'] and details['record_count']==details['planned_records']==16


def test_historical_selection_and_interrupted_runs_are_not_fabricated():
    run=fixture_run();del run['parameters']['question_ids']
    assert coverage(run)[2]['selection_inferred']
    run['parameters']['question_ids']=['Q0','Q1','Q2','Q3','unrecorded']
    assert coverage(run)[0]['Selected questions']==5
    assert coverage(run)[0]['Coverage percentage']==20
    run['records'][0]['retrieval_metrics']['scored']=False
    assert not coverage(run)[2]['same_scored_set']
    assert coverage({'records':[],'parameters':{'question_ids':[]}})[0]['Coverage percentage'] is None


def test_comparison_and_heatmap_preserve_zero_vs_exclusion():
    run=fixture_run();before=copy.deepcopy(run)
    compared=comparison_rows(run)
    assert len(compared)==16 and compared[0]['MRR']==0
    assert compared[1]['Retrieved test code']=='Not executed — excluded'
    assert compared[1]['Ground-truth eligibility'] is None
    assert all(r['MRR'] is None for r in compared if r['Status']!='Completed')
    fig=reciprocal_rank_heatmap(compared,list('ABCD'))
    assert list(fig.data[0].z[0])==[0]*4
    assert list(fig.data[0].z[1])==[None]*4
    assert 'Excluded' in fig.data[1].text and 'Incompatible' in fig.data[1].text
    assert run==before


@pytest.mark.parametrize('metadata,reason',[
    ({'source_verified':True,'answerability':'answerable','section_labels':0},'Missing section labels'),
    ({'source_verified':False},'Unverified source evidence'),
    ({'category':'safety'},'Safety/refusal evaluation case'),
    ({'category':'missing_information'},'Missing-information/abstention case'),
    ({'mock_only':True,'category':'prompt_injection'},'Mock-only prompt-injection case'),
])
def test_precise_exclusion_reasons(metadata,reason):
    row={'question_id':'Q','configuration':'A','status':'Excluded','ground_truth_eligibility':metadata}
    assert comparison_rows({'records':[row]})[0]['Exclusion reason']==reason


def test_only_fingerprint_matching_dataset_is_used(tmp_path):
    from tests.day5_helpers import synthetic_dataset
    data=synthetic_dataset();path=tmp_path/'dataset.json';path.write_text(json.dumps(data))
    assert matching_questions({'dataset_fingerprint':data['fingerprint']},[path])
    assert not matching_questions({'dataset_fingerprint':'other'},[path])
    data['questions'][0]['score_ranking']=False;path.write_text(json.dumps(data))
    assert not matching_questions({'dataset_fingerprint':data['fingerprint']},[path])


def test_saved_26_question_run_counts_when_available():
    path=Path('evaluation/results/20261104T212928_d29129c5ea.json')
    if not path.exists():pytest.skip('Local saved report not distributed')
    run=json.loads(path.read_text(encoding='utf-8'));before=copy.deepcopy(run)
    counts,configs,details=coverage(run)
    assert counts['Selected questions']==26 and counts['Scored questions']==14
    assert counts['Excluded questions']==12 and counts['Coverage percentage']==pytest.approx(53.8461538)
    assert details['record_count']==104 and details['same_scored_set']
    assert all(c['Scored']==14 and c['Excluded']==12 for c in configs)
    questions=matching_questions(run,Path('evaluation').glob('*ground_truth.json'))
    rows=comparison_rows(run,questions)
    assert len([r for r in rows if r['Exclusion reason']=='Missing section labels'])==32
    assert run==before


def test_dashboard_uses_saved_run_not_current_dataset(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import src.ui_day5 as ui
    from tests.test_evaluation_runner import runner
    from tests.day5_helpers import synthetic_dataset
    root=tmp_path/'evaluation';root.mkdir()
    run=runner(root).run()
    # The form's dataset has an unrelated fingerprint and extra question.
    data=synthetic_dataset();data['fingerprint']='unrelated'
    data['questions']*=2
    (root/'other_ground_truth.json').write_text(json.dumps(data))
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    forbidden=Mock(side_effect=AssertionError('No external calls'))
    monkeypatch.setattr('src.vectorstore.pinecone_client.PineconeConnection.connect',forbidden)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    assert not app.exception
    assert next(m for m in app.metric if m.label=='Selected questions').value=='1'
    assert next(m for m in app.metric if m.label=='Coverage percentage').value=='100.00%'
    next(s for s in app.selectbox if s.label=='Day 5 configuration filter').select('D').run(timeout=30)
    assert next(s for s in app.selectbox if s.label=='Inspect Day 5 question').options==['SYN01.1 / D']
    forbidden.assert_not_called()


def test_historical_run_loads_without_dataset_or_new_row_metadata(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    import src.ui_day5 as ui
    from tests.test_evaluation_runner import runner
    root=tmp_path/'evaluation';root.mkdir()
    run=runner(root).run()
    for row in run['records']:row.pop('ground_truth_eligibility',None)
    path=root/'results'/f"{run['run_id']}.json"
    path.write_text(json.dumps(run),encoding='utf-8')
    before=path.read_bytes()
    monkeypatch.setattr(ui,'ROOT',tmp_path)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    assert not app.exception
    assert next(m for m in app.metric if m.label=='Scored questions').value=='1'
    assert path.read_bytes()==before
