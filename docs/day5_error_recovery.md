# Day 5 Groq diagnostics and checkpoint recovery

No live run is started by these changes. The original
`20261106T042020_94e13d55d1` checkpoint remains unchanged.

## Error reporting and retries

`error_diagnostics` records execution stage, allowlisted provider error category,
HTTP status when available, retryability, actual application retry count, a fixed
sanitized message, and UTC failure timestamp. Unknown retry counts stay null for
non-provider errors. Exception text, provider bodies, headers, credentials, prompts,
generated text and patient details are not copied into these fields. Historical
errors cannot be reconstructed.

Groq still has at most three transport attempts and a 30-second request timeout.
SDK retries are disabled to prevent nested retries; the application owns the same
two-retry budget and counts it. Connections/timeouts, HTTP 408/409/429 and 5xx may
retry. Authentication/permission, invalid model/request, malformed structured
responses, parser/schema and unknown errors do not automatically retry.

Only HTTP 429 enables Retry-After-aware scheduling. Numeric seconds and HTTP dates
are accepted. A valid delay up to 60 seconds is respected; invalid or longer values
defer to a later explicit retry rather than retrying early. Without that header,
bounded exponential delays are 0.5 and 1 seconds. Error categories are not proof
that a retry will succeed. A request timeout is not a whole-evaluation timeout.

## Resume guarantees

All runner executions use an exclusive local lock in `evaluation/locks`. A
second writer is rejected before checkpoint changes. Locks contain PID and UTC
acquisition time. They are removed on ordinary exits; a hard kill can leave a
stale lock. Never remove a lock until its owning execution is confirmed stopped.
Locks protect this updated runner, not old processes or external file editors.

Resume validates the checkpoint's original signature and requested configuration
before any checkpoint write or live request. A unique backup under
`evaluation/backups/<run_id>/<timestamp_uuid>/` preserves existing JSON, CSV and
Markdown bytes. Backup files are read-only and are never overwritten by the
application; this is filesystem protection, not tamper-proof archival storage.
If backup fails, execution stops before modifying the checkpoint.

Completed and Excluded pairs are skipped. Failed and missing pairs execute;
prior failure diagnostics are retained in retry events and original snapshots.
Final runs with failed records say `Completed with failures`. Caught runtime
stops/reruns say `Interrupted`; uncaught ordinary execution failures say `Failed`.
Termination records contain a fixed reason and timestamp. Hard termination or
power loss may still leave Running; a saved state is not a liveness signal.

## Manual resume of the investigated run (separate live authorization required)

1. Confirm the old evaluation has stopped. Restart Streamlit with the updated
   code before recovery; do not run two browser submissions concurrently.
2. Open Evaluation > Day 5 > Configure Day 5 evaluation. Use Manual selection.
3. Select `mayo_day5_v3_overlap15_ground_truth.json`, configurations A/B/C/D,
   category All and all 26 questions (an empty question selection means all).
4. Use RAG mode, Test-Context Retrieval, final Top K 5, initial candidates 15,
   optional judge off and no filters. The configured model must remain
   `openai/gpt-oss-120b`, temperature 0, max output tokens 1200. Keep embeddings,
   namespaces and other original settings unchanged. The runner uses application
   settings; changing only the chat-model widget does not configure Day 5.
5. Enter `20261106T042020_94e13d55d1` in Resume Day 5 run ID.
6. Enable the recovery-only migration checkbox. It allows only the investigated
   original pipeline fingerprint
   `e39863e7a927d249068780b6fed870cb06d917ecb7f0638fcfbc050987575875`.
   The stored signature must validate and every other signed field must match.
   The original signature is retained; the new execution signature is recorded
   in a resume event. This explicitly records the recovery-code change.
7. Only when ready to authorize live execution, select the explicit initiation
   checkbox and submit. A backup is made before any checkpoint update or request.
   Do not change the selection to failed questions only; that changes the signature.
8. Verify the backup and resume event, then review failed/missing records. The 35
   completed pairs are skipped. Failures retry full per-pair retrieval/generation.
   An uncheckpointed in-flight request can be repeated; remote exactly-once
   execution cannot be guaranteed.

Signature mismatches fail closed. Do not edit a signature or dataset to bypass
the check. Further pipeline changes require a separately reviewed migration.
Previously unknown provider errors will become identifiable only if they recur.
