import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from config.settings import Settings
from src.llm.groq_service import GroqService,LLMError,InvalidStructuredGeneration
from src.evaluation.checkpoint_safety import run_lock
from src.evaluation.dataset_validator import fingerprint
from tests.test_evaluation_runner import runner


class HttpFailure(Exception):
    def __init__(self,status,headers=None):
        super().__init__('SECRET authorization bearer patient Alice raw prompt')
        self.status_code=status
        self.response=Mock(headers=headers or {})
        self.body={'error':{'message':'SECRET patient Alice'}}


@pytest.mark.parametrize('status,category,retryable,calls',[
    (429,'rate_limit',True,3),(401,'authentication_or_permission',False,1),
    (403,'authentication_or_permission',False,1),(400,'invalid_request',False,1),
    (413,'invalid_request',False,1),(503,'transient_http',True,3),
])
def test_safe_provider_categories_and_retry_bound(status,category,retryable,calls):
    backend=Mock();backend.invoke.side_effect=HttpFailure(status)
    sleep=Mock()
    with pytest.raises(LLMError) as caught:
        GroqService(Settings(),backend,sleep=sleep).select([])
    detail=caught.value.error_diagnostics
    assert detail['http_status']==status and detail['provider_error_category']==category
    assert detail['retryable']==retryable and detail['retry_count']==calls-1
    assert detail['failure_timestamp'] and detail['execution_stage']=='generate_answer'
    assert backend.invoke.call_count==calls and sleep.call_count==calls-1
    assert all(word not in json.dumps(detail) for word in ['SECRET','Alice','authorization','raw prompt'])


def test_verified_rate_limit_retry_after_and_success():
    backend=Mock();backend.invoke.side_effect=[HttpFailure(429,{'retry-after':'3'}),{'excerpts':[]}]
    sleep=Mock()
    assert not GroqService(Settings(),backend,sleep=sleep).select([]).excerpts
    sleep.assert_called_once_with(3)


def test_non_rate_limit_header_ignored_and_long_wait_deferred():
    for status,wait,expected in [(503,'9',2),(429,'999',0),(429,'nonsense',0)]:
        backend=Mock();backend.invoke.side_effect=HttpFailure(status,{'retry-after':wait})
        sleep=Mock()
        with pytest.raises(LLMError):GroqService(Settings(),backend,sleep=sleep).select([])
        assert sleep.call_count==expected
        if status==503:assert [c.args[0] for c in sleep.call_args_list]==[.5,1]


def test_connection_and_schema_failures():
    backend=Mock();backend.invoke.side_effect=ConnectionError('SECRET patient Alice')
    with pytest.raises(LLMError) as caught:
        GroqService(Settings(),backend,sleep=Mock()).select([])
    assert caught.value.error_diagnostics['retry_count']==2
    assert caught.value.error_diagnostics['provider_error_category']=='connectivity_or_timeout'
    backend=Mock();backend.invoke.return_value={'excerpts':[{'quote':'SECRET patient Alice'}]}
    with pytest.raises(InvalidStructuredGeneration) as caught:
        GroqService(Settings(),backend,sleep=Mock()).select([])
    assert backend.invoke.call_count==1 and not caught.value.error_diagnostics['retryable']
    assert caught.value.error_diagnostics['provider_error_category']=='schema_validation'


def test_resume_backup_lock_signature_and_skip(tmp_path):
    engine=runner(tmp_path)
    original=engine.run(configurations=['A','B'],cancel=lambda:len(engine.retriever.calls)>0)
    path=tmp_path/'results'/(original['run_id']+'.json');before=path.read_bytes()
    with pytest.raises(ValueError,match='changed'):
        engine.run(configurations=['A','B'],resume=original['run_id'],top_k=3)
    assert path.read_bytes()==before and not list((tmp_path/'backups').glob('**/*.json'))
    with run_lock(tmp_path,original['run_id']):
        with pytest.raises(ValueError,match='locked'):
            engine.run(configurations=['A','B'],resume=original['run_id'])
    assert path.read_bytes()==before
    result=engine.run(configurations=['A','B'],resume=original['run_id'])
    assert result['status']=='Completed' and len(engine.retriever.calls)==2
    backup=list((tmp_path/'backups').glob('**/*.json'))
    assert len(backup)==1 and backup[0].read_bytes()==before
    assert any(e.get('event')=='resume' for e in result['events'])


def test_failed_rows_retried_history_retained_and_completed_skipped(tmp_path):
    engine=runner(tmp_path)
    backend=Mock();backend.invoke.side_effect=HttpFailure(401)
    engine.service=GroqService(Settings(),backend,sleep=Mock())
    first=engine.run(configurations=['A'],mode='rag')
    assert first['status']=='Completed with failures'
    row=first['records'][0]
    assert row['status']=='Failed' and row['error_diagnostics']['http_status']==401
    from tests.day5_helpers import SyntheticGenerator
    engine.service=SyntheticGenerator()
    result=engine.run(configurations=['A'],mode='rag',resume=first['run_id'])
    assert result['records'][0]['status']=='Completed'
    assert result['events'][-1]['previous_error_diagnostics']['http_status']==401
    calls=len(engine.retriever.calls)
    engine.run(configurations=['A'],mode='rag',resume=first['run_id'])
    assert len(engine.retriever.calls)==calls


def test_interruption_and_outer_failure_status(tmp_path):
    class StopExecution(BaseException):pass
    engine=runner(tmp_path)
    with pytest.raises(StopExecution):
        engine.run(progress=lambda *args:(_ for _ in ()).throw(StopExecution()))
    path=next((tmp_path/'results').glob('*.json'))
    saved=json.loads(path.read_text())
    assert saved['status']=='Interrupted' and len(saved['records'])==1
    with pytest.raises(RuntimeError):
        engine.run(progress=lambda *args:(_ for _ in ()).throw(RuntimeError('SECRET')))
    saved=[json.loads(p.read_text()) for p in (tmp_path/'results').glob('*.json')]
    assert any(r['status']=='Failed' for r in saved)
    assert all('SECRET' not in json.dumps(r) for r in saved)


def test_narrow_migration_requires_opt_in_and_valid_original_signature(tmp_path):
    engine=runner(tmp_path)
    original=engine.run(configurations=['A'])
    path=tmp_path/'results'/(original['run_id']+'.json')
    original['models']['pipeline_fingerprint']='e39863e7a927d249068780b6fed870cb06d917ecb7f0638fcfbc050987575875'
    original['signature']=fingerprint({'dataset':original['dataset_fingerprint'],'parameters':original['parameters'],
        'models':original['models'],'source':original['source'],'index':original['index_name']})
    path.write_text(json.dumps(original));before=path.read_bytes()
    with pytest.raises(ValueError,match='changed'):engine.run(configurations=['A'],resume=original['run_id'])
    assert path.read_bytes()==before
    result=engine.run(configurations=['A'],resume=original['run_id'],allow_recovery_migration=True)
    assert result['signature']==original['signature']
    assert result['events'][-1]['recovery_migration']
    result['signature']='tampered';path.write_text(json.dumps(result));before=path.read_bytes()
    with pytest.raises(ValueError,match='invalid'):
        engine.run(configurations=['A'],resume=original['run_id'],allow_recovery_migration=True)
    assert path.read_bytes()==before


def test_historical_checkpoint_is_readable_and_unchanged():
    from src.evaluation.report_generator import csv_text,summarize
    path=Path('evaluation/results/20261106T042020_94e13d55d1.json')
    if not path.exists():pytest.skip('Optional local checkpoint')
    before=path.read_bytes();run=json.loads(before)
    summarize(run);csv_text(run)
    assert path.read_bytes()==before


def test_retry_after_http_date_and_parser_failure_are_safe():
    from datetime import datetime,timezone,timedelta
    from email.utils import format_datetime
    from src.llm.error_diagnostics import classify,retry_delay
    from langchain_core.exceptions import OutputParserException
    exc=HttpFailure(429,{'retry-after':format_datetime(datetime.now(timezone.utc)+timedelta(seconds=10))})
    assert 8 <= retry_delay(exc,classify(exc),0) <= 10
    backend=Mock();backend.invoke.side_effect=OutputParserException('SECRET patient Alice')
    sleep=Mock()
    with pytest.raises(InvalidStructuredGeneration) as caught:
        GroqService(Settings(),backend,sleep=sleep).select([])
    assert backend.invoke.call_count==1 and caught.value.error_diagnostics['retry_count']==0
    sleep.assert_not_called()
    assert 'SECRET' not in json.dumps(caught.value.error_diagnostics)


def test_backup_failure_leaves_checkpoint_unchanged_and_releases_lock(tmp_path,monkeypatch):
    engine=runner(tmp_path);result=engine.run(configurations=['A'])
    path=tmp_path/'results'/(result['run_id']+'.json');before=path.read_bytes()
    monkeypatch.setattr('src.evaluation.evaluation_runner.backup_checkpoint',Mock(side_effect=OSError('Backup failed')))
    with pytest.raises(OSError):engine.run(configurations=['A'],resume=result['run_id'])
    assert path.read_bytes()==before
    assert not list((tmp_path/'locks').glob('*.lock'))


def test_excluded_checkpoint_pair_is_not_executed_again(tmp_path):
    engine=runner(tmp_path);result=engine.run(configurations=['A'])
    path=tmp_path/'results'/(result['run_id']+'.json')
    # Fixture represents a historically excluded pair; signatures cover config.
    result['records'][0]['status']='Excluded'
    path.write_text(json.dumps(result));calls=len(engine.retriever.calls)
    resumed=engine.run(configurations=['A'],resume=result['run_id'])
    assert resumed['records'][0]['status']=='Excluded' and len(engine.retriever.calls)==calls
