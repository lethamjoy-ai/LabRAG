# Week 2 presentation dashboard

The UI now uses three primary tabs: Ask LabRAG (default), Evaluation Results,
and About the Project. Day labels remain only inside retained technical views;
they are not main navigation. This change is limited to presentation code.

## Ask LabRAG

Enter a laboratory question in the compact input row and press Send. The existing graph invocation,
conversation handling, evidence gating, generation and exact citation checks are
unchanged. Supporting citations and source links are readable directly below
the answer. Retrieved passages, metadata, scores and trace details are available
under the collapsed View Retrieval Details section.

The sample selector reads up to five unique, source-verified questions with
successful cited answers from saved live reports whose dataset fingerprint matches.
Reading examples does not call Pinecone or Groq or synthesize examples.
Selecting an entry in the adjacent Sample Questions dropdown fills an editable
draft. Only Send submits it through the shared chat handler.
Saved evidence does not guarantee identical current retrieval. If no verified
saved examples exist, the selector is omitted rather than fabricating examples.
New Chat and Clear Conversation reset only chat state. Model/strategy/reranking
controls remain in Advanced Chat Settings. The Semantic Retrieval Explorer remains
in its own collapsed section.

## Evaluation Results

Select a saved run. Overview cards show that run's total, completed, failed,
missing and remaining pairs and saved status. Unknown historical selection counts
remain unavailable. Remaining includes retries and excludes Completed/Excluded.

Existing aggregate charts, metrics, quality tables, exclusions, matched-set
comparisons, reciprocal-rank heatmap, failure diagnostics, question inspection and
JSON/CSV/Markdown exports remain available. Main quality and retrieval charts are
visible by default; secondary latency/support charts and technical tables collapse.
No backend scoring function or saved metric is changed.

Evaluated Questions expands the saved question selection across its selected
variants. Recorded results are shown as recorded. Missing pairs show Pending /
Missing and unmeasured scores, with no generated answer or evidence. Search and
variant/status filters affect only this view. Question details include full text,
available reference facts, answer, evidence, metrics, citations and validation.
Reference facts/text are joined only from a fingerprint-verified matching dataset;
missing historical information is explicitly unavailable.

Advanced Evaluation Settings retains dataset/configuration, live opt-in, batching,
checkpoint recovery and request pacing controls. All expanders default collapsed.
Expanding a section does not initiate execution. All prior initiation gates,
writer locks, backups, signature checks, rate-limit handling and run IDs remain
under the existing backend's control.

## About the Project

The project objective, problem, source, stack, capabilities and native Graphviz
architecture show the implemented LangGraph workflow. Autonomous agents beyond
that workflow, ElevenLabs voice, scheduling and appointment lookup are clearly
future enhancements. Knowledge Base & Document Tools retains upload, extraction,
metadata, removal, chunk inspection and indexing, collapsed by default.

## Manual presentation check

1. With no evaluation active, load the updated app and verify Ask LabRAG is first.
2. Review sample questions without submitting them; inspect collapsed controls.
3. Open Evaluation Results and select the desired saved presentation run.
4. Review overview cards and charts. Search a question, filter a variant/status,
   and expand Question details. Pending rows must remain unmeasured.
5. Expand Detailed comparison & validation for original tables and diagnostics;
   Advanced Evaluation Settings for all execution/recovery controls.
6. Open About the Project to review architecture and future enhancements.
7. Confirm technical sections start collapsed and all three tabs remain usable
   at the presentation window size. No live evaluation is needed for this review.

Offline UI tests explicitly forbid provider calls and cover navigation, collapsed
controls, form submission with mocks, saved metrics, pending rows, sample provenance
and historical report preservation. Browser visual review was blocked because
Computer Use could not verify the browser's current URL; native layout still
requires final visual review on the presentation screen.
