from datetime import date
import streamlit as st
from config.settings import Settings
from src.ingestion.document_loader import load_document
from src.ingestion.metadata_manager import MetadataManager, save_document
from src.chunking.ui import render_chunking
from src.ingestion.test_metadata import detected_test_rows
from src.ingestion.document_store import remove_document
from src.ui_day3 import render_search, render_indexing, render_comparison
from src.ui_day4 import render_chat
from src.ui_day4_evaluation import render_day4_evaluation
from src.ui_day5 import render_day5


def render_document_removal(frame):
    if frame.empty:
        return
    with st.expander("Remove a saved document", expanded=False):
        records = {row["document_id"]: row for row in frame.to_dict("records")}
        selected = st.selectbox("Document to remove", list(records), key="document_to_remove",
            format_func=lambda key: f"{records[key]['document_title']} · {records[key]['laboratory_provider']} · {key[:8]}")
        st.caption("Removes the local uploaded copy, extracted text, both chunk snapshots, and manifest record. "
                   "Your original source file is kept. Any Pinecone vectors require separate scoped cloud removal.")
        if st.button("Remove document", type="primary"):
            try:
                removed = remove_document(selected)
            except (ValueError, OSError) as exc:
                st.error(f"Could not remove document: {exc}")
                return
            for key in ["chunk_outputs", "chunk_detected_tests", *[key for key in st.session_state if key.startswith("inspect_")]]:
                st.session_state.pop(key, None)
            st.session_state["upload_generation"] = st.session_state.get("upload_generation", 0) + 1
            st.session_state["removed_document_message"] = f"Removed {removed['document_title']}. Knowledge Base refreshed."
            st.rerun()


def render_knowledge(settings):
    st.subheader("Local knowledge base")
    if message := st.session_state.pop("removed_document_message", None):
        st.success(message)
    render_document_removal(MetadataManager().list_records())
    generation = st.session_state.get("upload_generation", 0)
    upload = st.file_uploader("Upload a permitted reference", type=["pdf", "html", "htm", "txt"], key=f"upload_{generation}")
    docs = None
    if upload:
        try:
            docs = load_document(upload.getvalue(), upload.name, max_file_bytes=settings.max_file_bytes)
            st.success(f"Validated · {len(docs)} text sections/pages")
            with st.expander("Extracted text preview", expanded=False):
                for doc in docs:
                    st.caption(f"Page: {doc.metadata['page_number'] or 'not applicable'}")
                    st.text(doc.page_content[:10000])
            st.subheader("Detected test names and codes")
            rows = detected_test_rows(docs)
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
                st.caption("Extracted from source labels; review the page and evidence columns. "
                           "Catalog entries are assigned separately to their own chunks.")
            else:
                st.info("No supported test labels were found. You can still save the document. "
                        "Unrecognized names/codes remain unresolved rather than guessed.")
        except ValueError as exc:
            st.error(str(exc))
    with st.form(f"metadata_{generation}"):
        st.caption("Document-level metadata only. Test names and codes are extracted from individual entries in the source.")
        title = st.text_input("Document title")
        provider = st.text_input("Laboratory provider")
        source = st.text_input("Source URL (HTTPS; blank for local synthetic examples)")
        revision = st.text_input("Source revision date (optional YYYY-MM-DD)")
        accessed = st.date_input("Accessed date", value=date.today())
        permitted = st.checkbox("I confirm this is permitted public reference content or synthetic data, without PHI.")
        submitted = st.form_submit_button("Save validated document locally")
    if submitted:
        if docs is None or not permitted:
            st.error("Upload a valid document and confirm the content declaration before saving.")
        else:
            try:
                save_document(upload.getvalue(), docs, {"document_title": title, "test_scope": "catalog",
                    "laboratory_provider": provider, "source_url": source,
                    "source_revision_date": revision, "accessed_date": accessed.isoformat()})
                st.success("Document and metadata saved locally.")
            except ValueError as exc:
                st.error(str(exc))
            except OSError:
                st.error("Unable to save files. Check local directory permissions and disk space.")
    frame = MetadataManager().list_records()
    left, right = st.columns(2)
    left.metric("Documents", len(frame))
    right.metric("Sources / providers", frame.laboratory_provider.nunique() if len(frame) else 0)
    search = st.text_input("Search title, test, or provider")
    providers = st.multiselect("Filter providers", sorted(frame.laboratory_provider.unique().tolist()))
    view = frame
    if search:
        view = view[view.apply(lambda row: search.casefold() in " ".join(row.astype(str)).casefold(), axis=1)]
    if providers:
        view = view[view.laboratory_provider.isin(providers)]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.caption("Planned collection topics are tracked separately in data/planned_topics.csv; they are not ingested documents.")
    render_chunking(frame, settings)
    render_indexing(settings)


def main():
    st.set_page_config(page_title="LabRAG | Laboratory Test Assistant", page_icon=":test_tube:", layout="wide")
    st.markdown("<style>.block-container{padding-top:1.25rem}h1{letter-spacing:-.025em}div[data-testid='stMetric']{padding:.75rem;border:1px solid #80808035;border-radius:.6rem}</style>",unsafe_allow_html=True)
    st.title("LabRAG — AI-Powered Laboratory Test Assistant")
    st.caption("Evidence-grounded laboratory information · Gen Academy Week 2")
    st.caption("Educational reference only. Verify current official laboratory instructions before clinical use. Public references and synthetic data only; no patient information.")
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        st.warning(f"Configuration: {exc} Using development defaults for local ingestion.")
        settings = Settings()
    with st.sidebar:
        st.header("LabRAG")
        st.caption("Laboratory knowledge, backed by sources.")
        with st.expander("Configuration & connection details",expanded=False):
            for warning in settings.warnings:st.warning(warning)
            st.write(f"Embedding dimension: {settings.embedding_dimension}")
            st.write("Index similarity: cosine")
    ask, evaluation, about = st.tabs(["Ask LabRAG", "Evaluation Results", "About the Project"])
    with ask:
        render_chat(settings)
        with st.expander("Advanced Semantic Retrieval Explorer",expanded=False):
            render_search(settings)
    with evaluation:
        render_day5(settings)
        with st.expander("Additional Validation Reports",expanded=False):
            render_day4_evaluation()
        with st.expander("Advanced Strategy Comparison",expanded=False):
            render_comparison(settings)
    with about:
        from src.ui_presentation import render_about
        render_about()
        with st.expander("Knowledge Base & Document Tools",expanded=False):
            render_knowledge(settings)
