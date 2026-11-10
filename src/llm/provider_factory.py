"""Lazy provider construction; selection never contacts a provider."""
from src.llm.groq_service import GroqService,LLMError

OPENAI_MODELS={'gpt-4.1','gpt-4.1-mini','gpt-4.1-nano','gpt-4.1-2025-04-14',
    'gpt-4.1-mini-2025-04-14','gpt-4.1-nano-2025-04-14','gpt-4o','gpt-4o-mini',
    'gpt-4o-2024-08-06','gpt-4o-2024-11-20','gpt-4o-mini-2024-07-18'}


def strict_schema(schema):
    """OpenAI strict schemas require every property and closed objects."""
    result=schema.model_json_schema()
    def normalize(value):
        if isinstance(value,dict):
            value.pop('default',None)
            if value.get('type')=='object':
                value['required']=list(value.get('properties',{}))
                value['additionalProperties']=False
            for item in list(value.values()):normalize(item)
        elif isinstance(value,list):
            for item in value:normalize(item)
    normalize(result)
    return result


def validate_credentials(settings,provider=None):
    provider=provider or settings.llm_provider
    key=settings.groq_api_key if provider=='groq' else settings.openai_api_key
    if not key.strip():
        from src.llm.error_diagnostics import details
        name='GROQ_API_KEY' if provider=='groq' else 'OPENAI_API_KEY'
        message=f'Set {name} in the project .env before requesting an answer.'
        diag=details('missing_credentials',message);diag['provider']=provider
        raise LLMError(message,diag)
    return key


def build_backend(settings,schema,provider=None,temperature=None):
    provider=provider or settings.llm_provider
    key=validate_credentials(settings,provider)
    common=dict(api_key=key,model=settings.llm_model,temperature=settings.llm_temperature if temperature is None else temperature,
        timeout=30,max_retries=0)
    if provider=='groq':
        from langchain_groq import ChatGroq
        return ChatGroq(**common,max_tokens=settings.llm_max_tokens).with_structured_output(schema,method='function_calling',include_raw=True)
    if provider!='openai':raise ValueError('Unknown LLM provider')
    if settings.llm_model not in OPENAI_MODELS:
        from src.llm.error_diagnostics import details
        message='OpenAI model is not enabled for strict structured output. Choose a supported GPT-4.1 or GPT-4o model.'
        diag=details('unsupported_model',message);diag['provider']='openai'
        raise LLMError(message,diag)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(**common,max_completion_tokens=settings.llm_max_tokens,
        base_url=settings.openai_base_url).with_structured_output(strict_schema(schema),
        method='json_schema',strict=True,include_raw=True)


class OpenAIService(GroqService):
    provider='openai'


def create_llm(settings,backend=None):
    if settings.llm_provider=='groq':return GroqService(settings,backend)
    if settings.llm_provider=='openai':return OpenAIService(settings,backend)
    raise ValueError('Unknown LLM provider')
