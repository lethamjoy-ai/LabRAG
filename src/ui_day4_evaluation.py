"""Read-only report viewer at startup; mocked execution requires a button click."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
DATASET=ROOT/"evaluation/mayo_day4_cases.json"
RESULTS=ROOT/"evaluation/day4_results.json"


def render_day4_evaluation():
    st.subheader("Day 4 — RAG Chatbot Validation")
    st.caption("LabRAG is an educational laboratory reference assistant. Verify specimen collection and handling instructions against the current official laboratory test catalog before clinical use.")
    try:
        cases=json.loads(DATASET.read_text(encoding="utf-8"))["cases"]
    except (OSError,ValueError,KeyError):
        st.info("Day 4 case dataset is unavailable. Generate it with scripts/validate_day4_corpus.py.")
        return
    if st.button("Run mocked Day 4 validation"):
        from src.evaluation.day4_runner import run_cases,export_report
        with st.spinner("Running credential-free mocked workflows…"):
            report=run_cases(cases=cases,mode="mock")
            export_report(report,RESULTS)
    report=None
    if RESULTS.exists():
        try:
            report=json.loads(RESULTS.read_text(encoding="utf-8"))
        except (OSError,ValueError):
            st.warning("The saved report cannot be read. Rerun validation to regenerate it.")
    measured={r["test_id"]:r for r in (report or {}).get("records",[])}
    rows=[{**case,**measured.get(case["test_id"],{}),"result":measured.get(case["test_id"],{}).get("result","Not Tested")} for case in cases]
    if report:
        st.write(f"Report mode: **{report['mode']}** · Executed: {report['generated_at']}")
    st.info("Mock results validate workflow mechanics with synthetic fixtures. They are not live retrieval scores or evidence of clinical accuracy. Live testing is CLI-only and requires explicit opt-in.")
    columns=st.columns(5)
    columns[0].metric("Total test cases",len(rows))
    for column,status in zip(columns[1:],["Passed","Failed","Skipped","Not Tested"]):
        column.metric(status,sum(r["result"]==status for r in rows))
    st.write({"Citation validation":{s:sum(r.get("citation_status")==s for r in rows) for s in ["Valid","Invalid","Missing","Not applicable"]},
        "Follow-up results":{s:sum(r["result"]==s and r["category"].startswith("conversation") for r in rows) for s in ["Passed","Failed","Skipped","Not Tested"]},
        "Insufficient-evidence results":{s:sum(r["result"]==s and r["category"] in {"insufficient_evidence","missing_information"} for r in rows) for s in ["Passed","Failed","Skipped","Not Tested"]}})
    category=st.selectbox("Day 4 category",["All"]+sorted({r["category"] for r in rows}))
    code=st.selectbox("Day 4 test code",["All"]+sorted({c for r in rows for c in r["test_code"].split(" / ")}))
    status=st.selectbox("Day 4 result",["All","Passed","Failed","Skipped","Not Tested"])
    visible=[r for r in rows if (category=="All" or r["category"]==category) and
             (code=="All" or code in r["test_code"].split(" / ")) and (status=="All" or r["result"]==status)]
    st.dataframe(pd.DataFrame([{k:r.get(k) for k in ["test_id","category","test_code","result","citation_status","latency_seconds"]} for r in visible]),hide_index=True,use_container_width=True)
    if visible:
        ids={r["test_id"]:r for r in visible}
        selected=st.selectbox("Inspect Day 4 case",list(ids))
        row=ids[selected]
        with st.expander("Day 4 case details",expanded=True):
            st.write("Original conversation",row["conversation"])
            st.write("Expected source-verified facts",row["expected_facts"])
            st.write("Expected behavior",row["expected_behavior"])
            st.write("Source verification",row.get("verification",row.get("absence_verification","Not applicable")))
            st.write("Failure or skip reason",row.get("failure_reason") or "None recorded")
            if not row.get("turns"):
                st.caption("No measured turns recorded for this case.")
            for i,turn in enumerate(row.get("turns",[]),1):
                st.markdown(f"**Turn {i}**")
                st.write("Resolved question",turn["resolved_question"])
                st.write("Generated answer",turn["actual_answer"])
                st.json({k:turn[k] for k in ["retrieved_evidence","retrieved_chunk_ids","citations","expected","result","failure_reason","latency_seconds"]})
    if report:
        st.download_button("Download Day 4 JSON report",json.dumps(report,indent=2),file_name="day4_results.json",mime="application/json")
        if RESULTS.with_suffix(".csv").exists():
            st.download_button("Download Day 4 CSV report",RESULTS.with_suffix(".csv").read_bytes(),file_name="day4_results.csv",mime="text/csv")
