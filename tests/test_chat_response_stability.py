import copy

import pytest
from streamlit.testing.v1 import AppTest

from tests.test_modern_chat_ui import chat, submit


@pytest.mark.parametrize('tracing', [False, True])
@pytest.mark.parametrize('provider', ['groq', 'openai'])
def test_final_answer_is_committed_then_rendered_once(chat, monkeypatch, tracing, provider):
    import src.ui_day4 as ui
    _, graph, _, _ = chat
    rendered = []
    original = ui.show_assistant
    def show(turn, tracing):
        rendered.append(copy.deepcopy(turn))
        original(turn, tracing)
    monkeypatch.setattr(ui, 'show_assistant', show)
    detailed = 'BUN requires serum.\n\nUse the catalog collection instructions and verify the cited source.'
    citation = {'source_id': 'S1', 'test_name': 'Blood Urea Nitrogen (BUN), Serum',
                'test_code': 'BUN', 'printed_catalog_page': '1'}
    def invoke(payload):
        # No response for the pending question is shown before graph completion.
        assert not any(t['question'] == payload['question'] for t in rendered)
        return {'question': payload['question'], 'answer': 'Validated source quotations [S1]',
            'extractive_answer': 'Validated source quotations [S1]', 'display_answer': detailed,
            'status': 'answered', 'citations': [citation], 'sources': {}, 'matches': [],
            'synthesis_diagnostics': {'validated': True}}
    graph.invoke.side_effect = invoke
    app = AppTest.from_string('from src.ui_day4 import render_chat\nfrom config.settings import Settings\n'
        f'render_chat(Settings(langsmith_tracing={tracing},langsmith_api_key="mock-key"))').run(timeout=30)
    if provider == 'openai':
        next(s for s in app.selectbox if s.label == 'LLM Provider').select(provider).run(timeout=30)
    submit(app, 'What specimen is required for BUN?')
    assert len(rendered) == 1  # Previously rendered once before, then after st.rerun.
    saved = copy.deepcopy(app.session_state['rag_history'][0])
    assert saved['chat_display_answer'] == detailed
    assert rendered[0]['chat_display_answer'] == detailed
    assert any(m.value == detailed for m in app.markdown)
    assert saved['citations'] == [citation]
    assert graph.invoke.call_count == 1
    app.run(timeout=30)
    assert app.session_state['rag_history'][0] == saved
    assert rendered[-1]['chat_display_answer'] == detailed
    assert any(c.value == detailed for c in app.code)
    assert graph.invoke.call_count == 1
    submit(app, 'What is the minimum specimen volume for Calcium, Total, Serum?')
    assert graph.invoke.call_count == 2
    assert app.session_state['rag_history'][0] == saved
    assert len(app.session_state['rag_history']) == 2
    assert app.session_state['rag_history'][0]['citations'] == [citation]


def test_final_text_uses_only_validated_conversational_output():
    from src.ui_day4 import final_chat_text
    turn = {'answer': 'Final abstention', 'display_answer': 'Unvalidated draft',
            'synthesis_diagnostics': {'validated': False}}
    assert final_chat_text(turn) == 'Final abstention'
    turn['synthesis_diagnostics']['validated'] = True
    assert final_chat_text(turn) == 'Unvalidated draft'
    turn['chat_display_answer'] = 'Stored final answer'
    turn['display_answer'] = 'Later different presentation field'
    assert final_chat_text(turn) == 'Stored final answer'
