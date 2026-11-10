"""Offline checks for the pending-status to committed-history handoff."""
import copy
import time
from unittest.mock import Mock

import pytest

from src.llm.groq_service import LLMError
from tests.test_modern_chat_ui import chat, submit


@pytest.mark.parametrize('outcome', ['answered', 'abstained', 'provider_error'])
def test_loading_survives_delayed_generation_until_saved_response_is_rendered(chat, monkeypatch, outcome):
    import src.ui_day4 as ui
    app, graph, _, _ = chat
    events = []
    current = {'question': '', 'loading': False}
    original_status, original_show = ui.st.status, ui.show_assistant

    def status(label, **kwargs):
        assert kwargs['state'] == 'running'
        assert not current['loading']
        current['loading'] = True
        events.append('loading')
        return original_status(label, **kwargs)

    def show(turn, tracing):
        if turn['question'] == current['question']:
            assert current['loading']
            assert ui.st.session_state['rag_history'][-1] == turn
            events.append('final')
            current['loading'] = False
        original_show(turn, tracing)

    monkeypatch.setattr(ui.st, 'status', status)
    monkeypatch.setattr(ui, 'show_assistant', show)
    forced_rerun = Mock(side_effect=AssertionError('No full-app rerun before final display'))
    monkeypatch.setattr(ui.st, 'rerun', forced_rerun)

    def invoke(payload):
        assert current['loading'] and events[-1] == 'loading'
        events.append('generation')
        # A short mocked provider delay; the native running status must stay alive.
        time.sleep(.01)
        assert current['loading']
        events.append('generation_finished')
        if outcome == 'provider_error':
            raise LLMError('private provider body', {'http_status': 429})
        return {'question': payload['question'], 'status': outcome,
                'answer': 'Validated final answer' if outcome == 'answered' else 'Insufficient retrieved evidence',
                'citations': [{'source_id': 'S1', 'test_code': 'BUN'}] if outcome == 'answered' else [],
                'sources': {}, 'matches': []}

    graph.invoke.side_effect = invoke
    questions = ['What specimen is required for BUN?',
                 'What is the minimum specimen volume for Calcium, Total, Serum?']
    first = None
    for question in questions:
        current['question'] = question
        events.clear()
        submit(app, question)
        assert events == ['loading', 'generation', 'generation_finished', 'final']
        assert not current['loading']
        turn = app.session_state['rag_history'][-1]
        assert turn['chat_display_answer'] == turn['answer']
        assert any(m.value == turn['answer'] for m in app.markdown)
        assert not any('private provider body' in m.value for m in app.markdown)
        if first is None:
            first = copy.deepcopy(turn)
        else:
            assert app.session_state['rag_history'][0] == first
    assert len(app.chat_message) == 4  # Loading bubbles were replaced, not appended.
    assert len(app.session_state['rag_history']) == 2
    assert graph.invoke.call_count == 2
    assert next(w for w in app.text_input if w.label == 'Laboratory question')
    if outcome == 'answered':
        assert app.session_state['rag_history'][0]['citations'] == [{'source_id': 'S1', 'test_code': 'BUN'}]
    # Ordinary UI reruns render stored answers without reentering loading/generation.
    monkeypatch.setattr(ui, 'show_assistant', original_show)
    events.clear()
    before = copy.deepcopy(app.session_state['rag_history'])
    app.run(timeout=30)
    assert not app.exception and not events
    assert app.session_state['rag_history'] == before and graph.invoke.call_count == 2
    forced_rerun.assert_not_called()
