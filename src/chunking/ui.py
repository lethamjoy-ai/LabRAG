"""Knowledge Base chunking panel; operates only on saved local documents."""
import pandas as pd
import hashlib
import streamlit as st
from src.chunking.chunk_manager import ChunkManager, load_processed_document, export_jsonl
from src.chunking.chunk_validator import comparison_metrics, validate_chunks
from src.chunking.pipeline import process_document
from src.ingestion.test_metadata import detected_test_rows


@st.fragment
def render_chunk_inspector(chunks, strategy):
    """Refresh selection and details together without reparsing the uploaded catalog."""
    # A different document or regenerated snapshot must not reuse a stale selection.
    snapshot = hashlib.sha256("\n".join(c.metadata["chunk_id"] for c in chunks).encode()).hexdigest()
    index = st.selectbox("Inspect chunk", range(len(chunks)),
        key=f"inspect_{strategy}_{snapshot}",
        format_func=lambda i: f"Chunk {i + 1} · Page {chunks[i].metadata.get('page_number') or '—'}"
                             f" · {chunks[i].metadata.get('test_code') or chunks[i].metadata.get('test_name') or 'Unassigned'}")
    chunk = chunks[index]
    st.caption(f"Viewing chunk {index + 1} of {len(chunks)}")
    st.text(chunk.page_content)
    with st.expander("Chunk metadata", expanded=False):
        st.json(chunk.metadata)


def render_chunking(frame, settings):
    st.divider()
    st.subheader("Document Chunking")
    st.caption("Local processing only. cl100k_base token counts approximate embedding tokens. "
               "First use may download the public tokenizer vocabulary; no model or API inference runs.")
    if frame.empty:
        st.info("Save a validated document above to preview and compare chunks.")
        return
    records = {r["document_id"]: r for r in frame.to_dict("records")}
    selected = st.selectbox("Document to process", list(records),
        format_func=lambda key: f"{records[key]['document_title']} · {records[key]['laboratory_provider']} · {key[:8]}")
    strategy = st.selectbox("Chunking strategy", ["Fixed-size chunking", "Section-aware chunking", "Compare both strategies"])
    default_size = settings.section_max_tokens if strategy == "Section-aware chunking" else settings.chunk_size
    size = st.number_input("Chunk size / section maximum (tokens)", min_value=1, max_value=10000, value=default_size, step=25)
    overlap = st.number_input("Overlap (tokens)", min_value=0, max_value=9999, value=settings.chunk_overlap)
    strategies = ["fixed", "section"] if strategy == "Compare both strategies" else ["fixed" if strategy == "Fixed-size chunking" else "section"]
    signature = (selected, strategy, size, overlap)
    process = st.button("Process Document")
    reload_saved = st.button("Load saved chunks")
    if process or reload_saved:
        try:
            if overlap >= size:
                raise ValueError("Overlap must be smaller than the token size.")
            documents = load_processed_document(selected, metadata=records[selected])
            st.session_state["chunk_detected_tests"] = detected_test_rows(documents)
            outputs = {}
            with st.spinner("Cleaning and checking source spans…"):
                for name in strategies:
                    if process:
                        cleaned, chunks, validation = process_document(documents, name, int(size), int(overlap))
                        if not validation.errors:
                            ChunkManager().save(chunks)
                    else:
                        from src.ingestion.text_cleaner import clean_documents
                        cleaned = clean_documents(documents)
                        chunks = ChunkManager().load(selected, name)
                        validation = validate_chunks(chunks, cleaned, chunks[0].metadata["chunk_size"])
                    outputs[name] = {"cleaned": cleaned, "chunks": chunks, "validation": validation}
            st.session_state["chunk_outputs"] = (signature, outputs)
        except (ValueError, OSError, TypeError) as exc:
            st.error(f"Document processing failed: {exc}")
            st.session_state.pop("chunk_outputs", None)
        except Exception:
            st.error("Processing failed. Check tokenizer download connectivity and review the source format, then retry.")
            st.session_state.pop("chunk_outputs", None)
    saved = st.session_state.get("chunk_outputs")
    if not saved or saved[0] != signature:
        return
    outputs = saved[1]
    with st.expander("Detected tests in selected document"):
        detected = st.session_state.get("chunk_detected_tests", [])
        if detected:
            st.dataframe(pd.DataFrame(detected), use_container_width=True, hide_index=True)
        else:
            st.caption("No supported test labels found. Review optional fallback metadata.")
    stats = {name: out["validation"] for name, out in outputs.items()}
    st.dataframe(pd.DataFrame(comparison_metrics(stats)).rename(columns={"fixed": "Fixed-size", "section": "Section-aware"}),
                 use_container_width=True)
    columns = st.columns(len(outputs))
    for column, (name, output) in zip(columns, outputs.items()):
        with column:
            st.subheader("Fixed-size" if name == "fixed" else "Section-aware")
            chunks, validation = output["chunks"], output["validation"]
            if validation.errors:
                st.error("Validation failed. These results were not saved or made available for export.")
            else:
                st.success("Loaded/processed local snapshot. Review warnings before Day 3 indexing.")
            with st.expander("Validation details", expanded=False):
                st.json(validation.to_dict())
            table = [{"index": d.metadata["chunk_index"], "tokens": d.metadata["token_count"],
                      "test_name": d.metadata.get("test_name", ""), "test_code": d.metadata.get("test_code", ""),
                      "test_detection": d.metadata.get("test_metadata_source", "legacy"),
                      "page": d.metadata.get("page_number"), "section": d.metadata.get("section_heading", ""),
                      "text": d.page_content} for d in chunks]
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
            if chunks:
                render_chunk_inspector(chunks, name)
            if not validation.errors:
                st.download_button("Export processed chunks to JSONL", export_jsonl(chunks),
                    file_name=f"{selected}-{name}.jsonl", mime="application/x-ndjson", key=f"export_{name}")
    with st.expander("Original extracted text versus cleaned text"):
        for doc in next(iter(outputs.values()))["cleaned"]:
            st.caption(f"Page {doc.metadata.get('page_number') or 'not applicable'}")
            original, cleaned = st.columns(2)
            original.text(doc.metadata.get("original_text", doc.page_content))
            cleaned.text(doc.page_content)
