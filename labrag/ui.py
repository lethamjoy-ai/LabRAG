import json
import streamlit as st
import plotly.express as px
from labrag.config import Settings, ConfigurationError
from labrag.ingestion import parse_document, chunk_documents, load_manifest
from labrag.retrieval import Embedder, PineconeStore, Reranker
from labrag.graph import build_graph, make_llm
from labrag.evaluation import evaluate


@st.cache_resource(show_spinner="Loading embedding model and connecting to Pinecone…")
def resources(settings):
    embedder = Embedder(settings.embedding_model)
    store = PineconeStore(settings, embedder)
    return store, build_graph(store, Reranker(settings.reranker_model), make_llm(settings))


def main():
    st.set_page_config(page_title="LabRAG", page_icon="🧪", layout="wide")
    st.title("🧪 LabRAG")
    st.caption("Clinical Laboratory Knowledge Assistant · Gen Academy · Week 2")
    st.info("Educational reference prototype. Verify answers against laboratory policies and original sources. "
            "Upload public reference material only; do not enter patient identifiers.")
    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        st.error(str(exc))
        st.code("Copy-Item .env.example .env\n# Edit .env with your API keys and index name")
        st.stop()
    with st.sidebar:
        st.subheader("Workspace")
        st.write(f"Provider: {settings.provider}")
        st.write(f"Model: {settings.llm_model}")
        st.write(f"Index: {settings.index_name}")
        st.write(f"Namespace: {settings.namespace}")
        st.caption("Namespaces are not authentication boundaries. Run only for trusted users.")
        connect = st.checkbox("Connect to configured services", value=False)
    if not connect:
        st.write("Configure .env, then enable the connection in the sidebar to begin.")
        return
    try:
        store, graph = resources(settings)
    except Exception:
        st.error("Service initialization failed. Check API keys, index readiness, cosine metric, embedding "
                 "dimension, network access, and model availability. No secrets are displayed.")
        return
    ask, ingest, evaluation = st.tabs(["Ask references", "Ingest documents", "Evaluate"])
    with ingest:
        st.subheader("Upload reference documents")
        st.caption("PDF, HTML, UTF-8 TXT · 20 MB per file · files are parsed in memory")
        manifest_file = st.file_uploader("Optional source manifest", type=["json"])
        files = st.file_uploader("Laboratory references", type=["pdf", "html", "htm", "txt"], accept_multiple_files=True)
        size = st.slider("Chunk size (characters)", 300, 1800, 1000, 100)
        if st.button("Parse, embed, and index", disabled=not files):
            try:
                manifest = load_manifest(manifest_file.getvalue()) if manifest_file else []
                lookup = {r["filename"]: r for r in manifest}
            except (ValueError, UnicodeError):
                st.error("Invalid manifest. Use the schema documented in README.md.")
                return
            for upload in files:
                try:
                    entry = lookup.get(upload.name, {})
                    metadata = {**entry.get("metadata", {}), **{k: v for k, v in entry.items() if k != "metadata"}}
                    metadata["source"] = entry.get("source_url", upload.name)
                    docs = parse_document(upload.getvalue(), upload.name, metadata)
                    chunks = chunk_documents(docs, size=size)
                    with st.spinner(f"Indexing {upload.name}…"):
                        count = store.upsert(chunks)
                    st.success(f"{upload.name}: indexed {count} chunks from {len(docs)} text pages. "
                               "Pinecone queries may take a moment to reflect writes.")
                except Exception:
                    st.error(f"Could not ingest {upload.name}. Check file format, UTF-8 encoding, size, "
                             "extractable text, and Pinecone access. A partial upload can be retried.")
    with ask:
        test_name = st.text_input("Optional exact laboratory test filter", placeholder="e.g. Hemoglobin A1c")
        question = st.text_area("Ask a laboratory reference question", max_chars=4000,
                                placeholder="What specimen and handling requirements are documented?")
        if st.button("Find evidence and answer", disabled=not question.strip()):
            try:
                with st.spinner("Retrieving, reranking, and checking citations…"):
                    result = graph.invoke({"question": question, "test_name": test_name.strip() or None})
                st.session_state["last_result"] = result
            except Exception:
                st.error("Request failed. Check provider quota, model availability, and service connectivity.")
        if result := st.session_state.get("last_result"):
            st.caption(f"Question: {result['question']}")
            st.markdown(result["answer"])
            st.caption("Citation validation checks label existence, not factual or clinical correctness.")
            for i, doc in enumerate(result["sources"], 1):
                with st.expander(f"[{i}] {doc.metadata.get('filename', 'Reference')} · page {doc.metadata.get('page', '?')}"):
                    st.text(doc.metadata.get("source", ""))
                    st.text(doc.page_content)
                    st.json(doc.metadata)
    with evaluation:
        st.subheader("Evaluate against a reference dataset")
        st.caption("Upload JSON cases with question, expected_sources, and expected_terms. "
                   "Metrics are diagnostic proxies; factual support requires human review. Each case calls the LLM.")
        dataset = st.file_uploader("Evaluation cases", type=["json"], key="evaluation")
        if st.button("Run evaluation", disabled=dataset is None):
            try:
                with st.spinner("Running evaluation…"):
                    frame = evaluate(graph, json.loads(dataset.getvalue()))
                st.session_state["evaluation"] = frame
            except Exception:
                st.error("Evaluation failed. Validate the dataset schema and check service connectivity.")
        if "evaluation" in st.session_state:
            frame = st.session_state["evaluation"]
            metrics = ["recall_at_4", "mrr", "ndcg_at_4", "term_coverage", "citation_format_valid"]
            st.plotly_chart(px.bar(frame[metrics].mean().rename_axis("metric").reset_index(name="score"),
                                   x="metric", y="score", range_y=[0, 1]), use_container_width=True)
            st.dataframe(frame)
            st.download_button("Download evaluation CSV", frame.to_csv(index=False), "labrag-evaluation.csv", "text/csv")
