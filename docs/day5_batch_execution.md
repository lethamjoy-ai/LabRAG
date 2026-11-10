# Resumable Day 5 question batches

The Streamlit control **Maximum new questions per session** defaults to 5.
The runner's `max_new_questions_per_session` argument has the same default.
The limit applies to distinct remaining question IDs, including failed-pair
retries and partially completed questions. It does not mean five API calls:
five questions across A/B/C/D can attempt up to twenty pairs, plus bounded
transport retries and optional judges.

Questions retain their original dataset order. Completed and Excluded pairs
are skipped and do not consume slots. Once a question enters a session, all its
pending selected variants are attempted before the next question. A non-rate
failure stays retryable under the existing policy; other variants can proceed.
A terminal HTTP 429 immediately saves and returns Paused — Rate Limited, even
partway through a question. Interruptions also preserve per-pair progress.

Before starting a sixth remaining question at the default limit, the runner
atomically saves **Batch Completed — Ready to Resume**. It records the batch's
question IDs and limit in the checkpoint. The next explicit resume starts at
the earliest incomplete question. Persistent failures can therefore consume
slots again; batches do not silently abandon failed questions.

If all planned pairs have been visited, the original final statuses remain
Completed or Completed with failures. No unnecessary extra batch resume is
required. Completion of a batch is not a claim that the full evaluation finished.

## Signatures and checkpoints

The original dataset, ordered question selection, configurations, retrieval
parameters, model settings, signature and run ID remain unchanged. The batch
limit is an operational control, not a signed benchmark parameter; it may be
changed between resumes without changing the dataset or signature. Each resume
still validates the original signature, matching settings, rate-limit cooldown,
and writer lock, and makes the existing read-only backup before updates.

Installation changes the runner pipeline fingerprint. Explicit recovery-only
migration remains required for the investigated historical run. Matching all
other signed fields is mandatory; neither signatures nor question selection are
rewritten. The migration allowlist also retains the prior investigated pacing
pipeline fingerprint `2aa75a26206722e11477bd34bb45b33f60999a7e8c81197105dbd3886fdf36f5`.
Historical files are not changed by installing this feature or its offline tests.

## Resume 20261106T075303_624a0c1d52 safely

1. Confirm the previous evaluation has stopped and use Streamlit with updated code.
2. Select `mayo_day5_v3_overlap15_ground_truth.json`, Manual selection, category All,
   all 26 original questions (empty selection means all), and A/B/C/D. Do not
   manually select only five questions; that would alter the signed selection.
3. Keep RAG, Test-Context Retrieval, Top K 5, initial candidates 15, judge off,
   `openai/gpt-oss-120b`, temperature 0, max output 1200, and original index/model
   settings. Day 5 uses configured settings, not just the chat-model widget.
4. Enter the same run ID in Resume Day 5 run ID and enable recovery-only migration.
5. Leave Maximum new questions per session at 5. Preserve or increase request
   pacing based on verified Groq account quotas; batching does not replace pacing.
6. Wait for any saved resume-after time or verified quota reset. Only when ready
   to authorize live requests, enable explicit initiation and submit once.
7. At Batch Completed — Ready to Resume, explicitly submit the same run ID and
   full signed selection again. Completed/excluded pairs will be skipped, and
   failed pairs retain their previous history. Repeat until fully completed.
8. If Paused — Rate Limited appears, honor cooldown and inspect diagnostics before
   another explicit resume. No automatic continuation occurs.

The existing run is not resumed by implementation. Backups are created only
when an authorized resume is submitted. A crash can leave a stale lock; never
clear one without verifying its writer has stopped. Uncheckpointed in-flight
remote requests may repeat. Source-grounded scoring, retrieval, prompts,
reranking, indexed data and ground-truth labels are unchanged.

Offline tests: `.venv\Scripts\python.exe -m pytest -q -m "not integration"`.
