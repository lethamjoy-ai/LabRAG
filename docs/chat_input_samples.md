# Integrated chat input and sample dropdown

Ask LabRAG uses a native three-column row directly below its scrollable
conversation: Laboratory question, Sample Questions, Send. A text input is used
instead of `st.chat_input` so an adjacent dropdown can reliably populate an
editable draft without custom CSS or JavaScript. All existing verified samples
come from the same saved-report helper; none are added, removed or generated.
If none are available, the dropdown remains visible but disabled.

Selecting a sample fills the draft only. Edit the text freely, then press Send.
Typing, pressing Enter in the text field, changing providers/models, selecting a
sample, or rerunning the page does not submit a request. Send is the sole explicit
submission control. The dropdown stays beside the composer after every response.

## UI-only state

- `labrag_draft`: editable text; persists across ordinary reruns/provider changes.
- `labrag_sample`: selected example or placeholder. Resets on Send so the same
  example can be selected again deliberately.
- `_labrag_pending_question`: one queued Send event, consumed before backend work.
  The handler never submits merely because a draft or selection is nonempty.
- `_labrag_restore_draft`: restores text after a failed request on the next render,
  without automatically retrying. Successful submissions leave an empty draft.
- `_labrag_empty_submission`: one-time empty-input warning, never a provider call.

The existing `rag_history`, `append_turn`, follow-up resolver, provider/model
switcher, spinner, answer rendering, citations and retrieval pipeline remain
unchanged. New Chat/Clear Conversation clear these additional UI draft/event keys
alongside chat history. They do not clear provider settings or evaluation data.
The input row follows the bounded history pane, whose native auto-scroll keeps
the latest response visible. Native columns adapt to the available display width.

## Manual validation

1. Refresh the idle app and open Ask LabRAG. Confirm the question field, Sample
   Questions dropdown and Send are together below the conversation.
2. Select an example. It should fill the input without a request or user bubble.
3. Edit it. When ready to authorize a live question, click Send. Verify one user
   bubble, a spinner, one assistant response and an empty draft after success.
4. Choose a different sample. The dropdown must remain accessible and must not
   submit automatically. The same sample can be selected again after sending.
5. Enter an unsent draft, switch Groq/OpenAI or advanced settings, and verify the
   text/history persist without requests. Inspect source citations and evidence.
6. Test New Chat/Clear Conversation. Verify chat draft/context clear while saved
   evaluation results remain. No evaluation is started by these controls.

Offline tests mock all external services and cover custom questions, sample
population/editing, dropdown availability, input clearing, repeated selection,
provider persistence, follow-up history, error restoration and duplicate prevention.
The existing configured history limit and in-flight interruption limitations still
apply. No live requests or evaluations are initiated by implementation or testing.
