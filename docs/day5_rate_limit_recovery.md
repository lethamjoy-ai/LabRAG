# Rate-limit-aware Day 5 recovery

The evaluator pauses as **Paused — Rate Limited** after any terminal Groq HTTP
429, including exhausted bounded retries. It atomically checkpoints the current
failed pair and its diagnostics, then returns without executing another pair.
Completed/excluded records and prior retry history are preserved. No automatic
resume is scheduled. Paused is distinct from Failed and Interrupted.

## Diagnostics and backoff

In addition to existing safe error fields, 429 diagnostics record
`retry_after_seconds`, `quota_type`, `quota_reset_seconds`, `resume_after`, and
`backoff_reason`. Retry-After accepts finite nonnegative seconds or an HTTP date.
Safe numeric reset durations are read only from allowlisted rate-limit headers
when their corresponding requests/tokens remaining count is zero. Quota type
does not establish whether a limit is per minute, per day or another window.
Raw header values, response bodies, prompts and model text are not retained in
these fields. Unknown values remain null, not guessed.

Valid provider waits up to 60 seconds can use the existing bounded retry budget.
Longer waits or invalid supplied Retry-After defer immediately. With no header,
bounded exponential retries remain available; terminal 429 still pauses.
Resume-after uses the latest valid delay/reset and is enforced before backup,
checkpoint modification or API calls. A future cooldown blocks submission without
sleeping through the quota window. Unknown cooldown requires human quota review.

## Request pacing

The Streamlit configuration exposes minimum seconds between Groq request starts
and optional verified account/model requests per minute. Effective spacing is the
larger of the entered interval and 60/RPM. Zero/unknown does not assume a quota.
A run-wide context shares the clock across questions, configurations, retries,
generation and optional judges. Judges propagate terminal 429 to the evaluator;
their prompts and scoring rules are unchanged. Pacing settings are operational
controls recorded on the run and resume event, separate from signed benchmark
parameters, so recovery may increase pacing without changing retrieval settings.

Pacing is per execution, not a cross-process/account quota coordinator. It cannot
predict TPM/day consumption, token costs, traffic from other applications, or
quota resets absent provider evidence. Account RPM is user supplied, not fetched.
At most three transport attempts and a 30-second timeout remain. No automatic
retry is added for authentication, invalid requests or schema failures.

## Manual recovery of 20261106T075303_624a0c1d52

1. Confirm the old execution has stopped and start Streamlit with updated code.
2. Check your Groq console's actual account/model quota and wait for its reset.
   The historical run did not record cooldown headers; none can be reconstructed.
3. Choose `mayo_day5_v3_overlap15_ground_truth.json`, Manual selection, all 26
   questions, A/B/C/D, RAG, Test-Context Retrieval, Top K 5, candidates 15 and judge
   off. Preserve model `openai/gpt-oss-120b`, temperature 0, max output 1200 and
   other original signed settings. The chat-model widget does not configure Day 5.
4. Enter the original run ID in Resume Day 5 run ID. Enable recovery-only pipeline
   migration. Its allowlist now includes the investigated fingerprint
   `1e98cfbcc29f6ac5a1b82536fb93a4720a366de3e6a65139539abb73f094369f`.
   All other signed fields must match; the original signature is retained and
   the recovery execution signature is logged. Earlier investigated migration
   support remains available.
5. Set pacing using verified quota information. Select the explicit live-initiation
   checkbox and submit only when ready to authorize requests. This document and
   implementation do not initiate live execution.
6. Resume holds the existing writer lock, validates the original signature and
   cooldown, and takes a unique read-only backup before updates. It skips the 11
   completed pairs and retries failed/missing pairs. The current dashboard shows
   completed, failed, excluded, missing and remaining-to-process pair counts.
7. If paused again, review the sanitized diagnostic panel and resume-after time.
   Do not repeatedly submit new runs or remove active locks.

Historical reports and datasets remain unchanged by installation. On explicitly
authorized resume, updated results are written only after backup, with prior
failures retained in events. An in-flight uncheckpointed request may repeat;
remote exactly-once execution is not guaranteed. Optional-judge failures can
require repeating generation for that failed pair. Read-only local backups are
not tamper-proof storage. Hard kills can leave stale locks requiring verification.

Offline verification: `.venv\Scripts\python.exe -m pytest -q -m "not integration"`.
