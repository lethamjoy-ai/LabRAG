# Native conversational Ask LabRAG

The composer has since been updated to an integrated native text-input / Sample
Questions / Send row. Samples populate an editable draft and no longer submit
on selection. See [current input behavior](chat_input_samples.md). The original
chat bubbles, scrollable history, source display and backend handling described
below remain; the earlier chat_input/Enter/sample-button description is historical.

Ask LabRAG now uses native user/assistant bubbles and `st.chat_input` with
placeholder **Ask a question about laboratory tests...**. Enter submits. There
is no separate question form. A centered native container limits content width
to 960 pixels; history uses a 480-pixel scrollable pane, while an empty chat is
compact. Native auto-scroll keeps the composer below messages. No custom
JavaScript, clipboard library or new dependency is introduced.

An explicit chat-input event or sample-button click goes through one handler.
The user bubble is rendered before services are invoked; an assistant spinner
appears during retrieval/generation. The result is appended once to existing
`rag_history` via the original `append_turn`, then the UI rerenders the stored
turn. Settings changes and ordinary reruns do not invoke that handler. Repeating
the same question explicitly is allowed, rather than incorrectly deduplicating
by text. Existing graph invocation, follow-up resolution and history retention
limit (`rag_max_history_turns`) remain unchanged. No new memory system is added.

New Chat and Clear Conversation clear only chat history/context and draft keys.
They preserve evaluations, selected run IDs, documents, chunks, indexing data,
model settings and provider connection status. Advanced Chat Settings and Try a
sample question are collapsed. Samples keep the existing verified saved-report
provenance; clicking one explicitly submits it through the chat flow.

Each assistant bubble retains its own cited test identity, source links,
limitations/conflicts, retrieved passages, metadata, similarity scores and
optional trace timings. Technical details remain collapsed. Copy answer exposes
a plain native code block with Streamlit's copy icon; clipboard behavior depends
on the browser. Insufficient-evidence responses remain the backend's abstentions,
not invented answers.

HTTP 429 displays: "Groq's API usage limit has been reached. Please try again
after the quota resets." The question and earlier turns remain in history.
Other errors use existing sanitized diagnostic messages or fixed safe UI text,
not raw provider exception strings. The UI adds no retry loop; existing bounded
Groq handling remains unchanged. The user may submit again later explicitly.

## Manual review

1. With no active evaluation, load the updated app and open Ask LabRAG.
2. Confirm New Chat, Clear Conversation and collapsed Advanced Chat Settings.
3. Without making a request, open samples and verify there is only one composer.
4. When ready to authorize a live question, type a catalog question and press
   Enter. Check the immediate user message, loading indicator, assistant answer,
   source links and collapsed View Retrieval Details / Copy answer sections.
5. Ask a follow-up, then change a setting or refresh the app. Messages should
   remain chronological and no additional request should execute.
6. Start New Chat and verify the context is empty while saved evaluation runs
   remain available. Verify layout at the presentation screen and mobile width.

Offline tests use mocked services and cover message ordering, persistence,
follow-up context, reset scope, sample submission, duplicate prevention, rate
limit display, safe errors, citations and the existing graph compatibility.
No live evaluation or provider request is initiated by implementation/testing.
An in-flight request interrupted by browser/runtime termination cannot provide
remote exactly-once guarantees. Earlier turns beyond the configured existing
history limit are intentionally not retained by this UI.
