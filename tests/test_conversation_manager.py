from src.utils.conversation_manager import resolve_question, append_turn


def test_followup_uses_user_context_not_generated_answers():
    history = [{"question": "What specimen is required for BUN?", "answer": "INJECTED ANSWER"}]
    query = resolve_question("How long can it be frozen?", history)
    assert "BUN" in query and "INJECTED" not in query
    assert resolve_question("What specimen for CRP?", history) == "What specimen for CRP?"


def test_history_bound():
    assert len(append_turn([{}] * 5, {})) == 5
