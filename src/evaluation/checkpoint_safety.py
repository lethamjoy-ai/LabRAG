"""Exclusive local run writers and byte-preserving recovery snapshots."""
import functools
import json
import os
import re
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone


class ResumeDeferred(ValueError):
    """Safe fixed message; no provider payload."""


@contextmanager
def run_lock(root, run_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id):
        raise ValueError('Invalid run identifier')
    folder = root / 'locks'
    folder.mkdir(parents=True,exist_ok=True)
    path = folder / (run_id + '.lock')
    try:
        handle = path.open('x',encoding='utf-8')
    except FileExistsError:
        raise ValueError('Run is locked. Another writer may be active; verify before recovering a stale lock.') from None
    try:
        with handle:
            json.dump({'pid':os.getpid(),'acquired_at':datetime.now(timezone.utc).isoformat()},handle)
        yield
    finally:
        path.unlink(missing_ok=True)


def guarded_run(function):
    @functools.wraps(function)
    def wrapped(self,*args,**kwargs):
        # Bind positional callers too; private checkpoint IDs cannot be supplied.
        import inspect
        bound = inspect.signature(function).bind(self,*args,**kwargs)
        resume = bound.arguments.get('resume')
        run_id = resume or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:10]
        with run_lock(self.root,run_id):
            bound.arguments['_checkpoint_id'] = run_id
            from src.llm.request_pacing import pacing_scope
            with pacing_scope(bound.arguments.get('min_request_interval',0),bound.arguments.get('account_requests_per_minute')):
                return function(*bound.args,**bound.kwargs)
    return wrapped


def backup_checkpoint(root,run_id):
    """Unique write-once directory; original checkpoint bytes remain untouched."""
    destination = root/'backups'/run_id/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex)
    destination.mkdir(parents=True,exist_ok=False)
    for source in (root/'results'/(run_id+'.json'),root/'results'/(run_id+'.csv'),root/'reports'/(run_id+'_summary.md')):
        if source.exists():
            target = destination/source.name
            with source.open('rb') as src,target.open('xb') as dst:
                shutil.copyfileobj(src,dst)
            target.chmod(0o444)
    return str(destination.relative_to(root))
