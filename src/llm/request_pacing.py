"""One request-start clock shared by generation, retries and optional judges."""
import time
import math
from contextvars import ContextVar
from contextlib import contextmanager

ACTIVE=ContextVar('labrag_request_pacer',default=None)


class RequestPacer:
    def __init__(self,minimum=0,requests_per_minute=None,clock=time.monotonic,sleep=time.sleep):
        if isinstance(minimum,bool) or not math.isfinite(float(minimum)) or not 0<=minimum<=60:
            raise ValueError('Minimum request interval must be 0–60 seconds')
        if requests_per_minute is not None and (isinstance(requests_per_minute,bool) or not math.isfinite(float(requests_per_minute)) or requests_per_minute<1):
            raise ValueError('Account requests per minute must be at least 1 or unset')
        self.interval=max(minimum,60/requests_per_minute if requests_per_minute else 0)
        self.clock,self.sleep,self.last=clock,sleep,None

    def wait(self):
        if self.last is not None:
            delay=self.interval-(self.clock()-self.last)
            if delay>0:self.sleep(delay)
        self.last=self.clock()


def pace_request():
    pacer=ACTIVE.get()
    if pacer:pacer.wait()


@contextmanager
def pacing_scope(minimum=0,requests_per_minute=None):
    token=ACTIVE.set(RequestPacer(minimum,requests_per_minute))
    try:yield
    finally:ACTIVE.reset(token)
