"""Allowlisted provider diagnostics; never serialize exception text or bodies."""
import math
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime


def details(category, message, stage='generate_answer', status=None, retryable=False, retries=0):
    return {'execution_stage':stage, 'provider_error_category':category,
            'http_status':status, 'retryable':retryable, 'retry_count':retries,
            'sanitized_message':message, 'failure_timestamp':datetime.now(timezone.utc).isoformat()}


def classify(exc, retries=0, provider='groq'):
    status = getattr(exc, 'status_code', None)
    if type(status) is not int or not 100 <= status <= 599:
        status = None
    body = getattr(exc, 'body', {})
    error = body.get('error', body) if isinstance(body, dict) else {}
    code = error.get('code') if isinstance(error, dict) else None
    if status in (401,403):
        category, message, retryable = 'authentication_or_permission', 'Groq rejected authentication or permissions. Check the configured API key.', False
    elif status == 429:
        category, message, retryable = 'rate_limit', 'Groq rate limit reached.', True
    elif status in (408,409) or status is not None and status >= 500:
        category, message, retryable = 'transient_http', 'Groq returned a transient HTTP error.', True
    elif status is not None:
        category, message, retryable = 'invalid_request', 'Groq rejected the request format or parameters.', False
        if status == 404 or code in {'model_not_found','model_decommissioned'}:
            message = 'The configured Groq model is unavailable or inaccessible. Check LLM_MODEL.'
        elif status == 413 or code == 'context_length_exceeded':
            message = "The request exceeds Groq's input limit. Review prompt size and token settings."
    elif isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__name__ in {
            'APIConnectionError','APITimeoutError','ConnectError','ConnectTimeout','ReadTimeout','WriteTimeout','PoolTimeout'}:
        category, message, retryable = 'connectivity_or_timeout', 'Groq connection or request timed out.', True
    else:
        category, message, retryable = 'unknown_provider_error', 'Groq request failed; provider details withheld.', False
    result=details(category,message,status=status,retryable=retryable,retries=retries)
    result['provider']=provider
    if provider=='openai':
        result['sanitized_message']=message.replace('Groq','OpenAI').replace('GROQ_API_KEY','OPENAI_API_KEY')
        if code in {'insufficient_quota','billing_hard_limit_reached','billing_not_active'}:
            result.update(provider_error_category='billing_or_quota',retryable=False,
                sanitized_message='OpenAI API request could not be completed. Check API billing, available credits, and model access.')
    if status==429 and result['provider_error_category']=='rate_limit':
        result.update(rate_limit_details(exc))
    return result


def numeric_delay(value):
    if value is None:return None
    try:delay=float(value)
    except (TypeError,ValueError):
        try:delay=(parsedate_to_datetime(value)-datetime.now(timezone.utc)).total_seconds()
        except (TypeError,ValueError,OverflowError):return None
    return max(0,delay) if math.isfinite(delay) and 0<=delay<=31536000 else None


def rate_limit_details(exc):
    import re
    headers=getattr(getattr(exc,'response',None),'headers',{}) or {}
    delay=numeric_delay(headers.get('retry-after'))
    quotas=[];resets=[]
    for kind in ('requests','tokens'):
        try:empty=float(headers.get('x-ratelimit-remaining-'+kind))==0
        except (TypeError,ValueError):empty=False
        if empty:
            quotas.append(kind)
            reset=headers.get('x-ratelimit-reset-'+kind)
            if isinstance(reset,str) and re.fullmatch(r'(?:\d+(?:\.\d+)?(?:ms|s|m|h|d))+',reset):
                seconds=sum(float(n)*{'ms':.001,'s':1,'m':60,'h':3600,'d':86400}[u]
                    for n,u in re.findall(r'(\d+(?:\.\d+)?)(ms|s|m|h|d)',reset))
                if math.isfinite(seconds) and seconds<=31536000:resets.append(seconds)
    recovery=max(([delay] if delay is not None else [])+resets,default=None)
    return {'retry_after_seconds':delay,'quota_type':'+'.join(quotas) or None,
        'quota_reset_seconds':max(resets,default=None),
        'resume_after':(datetime.now(timezone.utc)+timedelta(seconds=recovery)).isoformat() if recovery is not None else None,
        'backoff_reason':'provider_delay_exceeds_inline_budget' if recovery is not None and recovery>60 else
            'provider_delay' if recovery is not None else 'missing_or_invalid_retry_after'}


def retry_delay(exc, diagnostic, attempt):
    """Respect Retry-After only on verified 429; never sleep beyond 60 seconds."""
    if not diagnostic['retryable']:
        return None
    if diagnostic['http_status'] == 429:
        known=[diagnostic[k] for k in ('quota_reset_seconds','retry_after_seconds') if diagnostic.get(k) is not None]
        if known:return max(known) if max(known)<=60 else None
        headers = getattr(getattr(exc, 'response', None), 'headers', {})
        value = headers.get('retry-after')
        if value is not None:
            delay=numeric_delay(value)
            if delay is None or delay > 60:
                return None  # Defer to an explicit later resume rather than retry early.
            return max(0, delay)
    return min(.5 * 2 ** attempt, 60)
