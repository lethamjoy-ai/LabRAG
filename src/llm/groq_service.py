"""Groq is constructed only when sufficient evidence reaches generation."""
import json
import time
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.exceptions import OutputParserException


class SupportedExcerpt(BaseModel):
    source_id: str
    quote: str = Field(min_length=1, max_length=3000)


class AnswerSelection(BaseModel):
    selection_protocol: Literal['exact_quote_v1', 'verified_excerpt_ids_v1'] = 'exact_quote_v1'
    excerpts: list[SupportedExcerpt] = Field(default_factory=list, max_length=12)
    missing_information: list[str] = Field(default_factory=list, max_length=10)
    conflicting_evidence: list[str] = Field(default_factory=list, max_length=10)


class ExcerptIdSelection(BaseModel):
    model_config = ConfigDict(extra='forbid')
    selected_excerpt_ids: list[str] = Field(max_length=12)
    missing_information: list[str] = Field(default_factory=list, max_length=10)
    conflicting_evidence: list[str] = Field(default_factory=list, max_length=10)


class LLMError(RuntimeError):
    def __init__(self, message, error_diagnostics=None):
        super().__init__(message)
        from src.llm.error_diagnostics import details
        self.error_diagnostics = error_diagnostics or details('configuration_or_llm_error',
            'Generation failed; check model and application configuration.')


class InvalidStructuredGeneration(LLMError):
    """Sanitized schema failure; never retain raw provider payloads."""
    diagnostics={'stage':'structured_generation','structured_response':False,'excerpt_count':None,'source_ids':[],
        'quote_validation':'not_attempted','failure_category':'Invalid structured generation response','final_answer_status':'failed','generation_executed':True}

    def __init__(self, message, retries=0, status=None):
        from src.llm.error_diagnostics import details
        super().__init__(message, details('schema_validation', 'Structured generation response was invalid.',stage='structured_generation',retries=retries,status=status))


class GroqService:
    provider='groq'
    def __init__(self, settings, backend=None, sleep=time.sleep):
        self.settings, self.backend = settings, backend
        self.sleep = sleep
        self.last_usage=None

    def _invoke(self, backend, messages):
        from src.llm.error_diagnostics import classify, retry_delay
        for attempt in range(3):  # Same bound as the former SDK max_retries=2.
            try:
                from src.llm.request_pacing import pace_request
                pace_request()
                response=backend.invoke(messages)
                if isinstance(response,dict) and 'parsed' in response and 'raw' in response:
                    if response.get('parsing_error') or response['parsed'] is None:
                        raise InvalidStructuredGeneration('Provider returned invalid structured output.',retries=attempt)
                    usage=getattr(response['raw'],'usage_metadata',None)
                    if isinstance(usage,dict):
                        counts={k:usage[k] for k in ('input_tokens','output_tokens','total_tokens') if type(usage.get(k)) is int and usage[k]>=0}
                        self.last_usage={**counts,'provider':self.provider,'model':self.settings.llm_model} if counts else None
                    response=response['parsed']
                return response, attempt
            except LLMError:
                raise
            except OutputParserException:
                raise InvalidStructuredGeneration('Groq output parsing failed.', retries=attempt) from None
            except Exception as exc:
                diagnostic = classify(exc, attempt,provider=self.provider)
                # A tool/schema failure is not a retryable transport failure.
                body = getattr(exc, 'body', {})
                error = body.get('error', body) if isinstance(body, dict) else {}
                if isinstance(error, dict) and error.get('code') == 'tool_use_failed':
                    raise InvalidStructuredGeneration('Groq structured tool response was invalid.',retries=attempt,status=diagnostic['http_status']) from None
                delay = retry_delay(exc, diagnostic, attempt)
                if attempt == 2 or delay is None:
                    raise LLMError(diagnostic['sanitized_message'], diagnostic) from None
                self.sleep(delay)

    def select(self, messages):
        self.last_usage=None
        try:
            menu = None
            if messages:
                try:
                    payload = json.loads(messages[-1].content)
                    evidence = payload.get('retrieved_evidence', [])
                    if any('verified_excerpts' in source for source in evidence):
                        from src.rag.excerpt_spans import verified_menu
                        menu = verified_menu(evidence)
                except (ValueError, KeyError, TypeError, AttributeError):
                    raise InvalidStructuredGeneration('Invalid supplied excerpt menu.') from None
            if self.backend is None:
                from src.llm.provider_factory import build_backend
                backend=build_backend(self.settings,ExcerptIdSelection if menu is not None else AnswerSelection,provider=self.provider)
            else:
                backend = self.backend
            response, retry_count = self._invoke(backend, messages)
            try:
                if menu is not None:
                    selected = ExcerptIdSelection.model_validate(response)
                    if any(key not in menu for key in selected.selected_excerpt_ids):
                        raise ValueError('Unknown excerpt identifier.')
                    return AnswerSelection(selection_protocol='verified_excerpt_ids_v1',
                        excerpts=[SupportedExcerpt(source_id=menu[key][0], quote=menu[key][1])
                                  for key in dict.fromkeys(selected.selected_excerpt_ids)],
                        missing_information=selected.missing_information,
                        conflicting_evidence=selected.conflicting_evidence)
                return AnswerSelection.model_validate(response)
            except Exception:
                raise InvalidStructuredGeneration('Groq could not return a valid structured answer. Schema validation failed.',retries=retry_count) from None
        except LLMError as exc:
            exc.error_diagnostics['provider']=self.provider
            raise
        except OutputParserException:
            raise InvalidStructuredGeneration('Groq could not return a valid structured answer. Output parsing failed.') from None
        except Exception as exc:
            if self.provider=='openai':
                from src.llm.error_diagnostics import classify
                diagnostic=classify(exc,provider='openai')
                raise LLMError(diagnostic['sanitized_message'],diagnostic) from None
            status = getattr(exc, "status_code", None)
            body = getattr(exc, "body", {})
            error = body.get("error", body) if isinstance(body, dict) else {}
            code = error.get("code") if isinstance(error, dict) else None
            if status == 401:
                message = "Groq rejected authentication. Check GROQ_API_KEY in .env."
            elif status == 404 or code in {"model_not_found", "model_decommissioned"}:
                message = "The configured Groq model is unavailable or inaccessible. Set LLM_MODEL to an available model and update Chat model in the sidebar."
            elif code == "tool_use_failed":
                message = "Groq failed to generate a valid structured tool response. Retry the question; if it persists, use another tool-capable model."
                raise InvalidStructuredGeneration(message) from None
            elif status == 413 or code == "context_length_exceeded":
                message = "The request exceeds Groq's input limit. Reduce retrieval top-k and retry."
            elif status == 400:
                message = "Groq rejected the request format or parameters. Check the model's structured-output support and token settings."
            elif status == 429:
                message = "Groq rate limit reached after bounded retries. Try again later."
            else:
                message = "Groq could not return a valid structured answer. Check connectivity/model availability and retry."
            # Provider exception bodies can contain credentials or request text.
            raise LLMError(message) from None
