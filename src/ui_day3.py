"""Explicit Day 3 actions. Import/render never connects or loads an embedding model."""
import json
from pathlib import Path
from urllib.parse import urlparse, quote
import streamlit as st
from src.embeddings.embedding_service import get_configured_embedding_service, EmbeddingError
from src.vectorstore.pinecone_client import PineconeConnection, PineconeError, field, namespace_for
from src.vectorstore.index_manager import IndexManager
from src.vectorstore.indexing_manifest import IndexingManifest
from src.retrieval.semantic_retriever import SemanticRetriever


def services(settings):
    connection = PineconeConnection(settings).connect()
    embeddings = get_configured_embedding_service(settings)
    return connection, embeddings


def show_results(result):
    st.caption(f"{len(result.matches)} results · {result.latency_seconds:.3f} seconds · {result.namespace}")
    if not result.matches:
        st.info("No matching evidence found. Check indexed documents, filters, namespace, and score threshold.")
    for index, match in enumerate(result.matches, 1):
        doc, score = match["document"], match["score"]
        meta = doc.metadata
        with st.container(border=True):
            st.markdown(f"**[{index}] {meta.get('document_title', 'Reference')}**")
            st.write(f"Cosine similarity: {score:.4f}")
            st.caption(f"Provider: {meta.get('laboratory_provider', 'Unknown')} · "
                       f"Test: {meta.get('test_name') or 'Unassigned'} · Code: {meta.get('test_code') or 'Unassigned'}")
            st.caption(f"Page: {meta.get('original_page_number', meta.get('page_number', 'Unknown'))} · "
                       f"Section: {meta.get('section_heading') or 'Unassigned'} · Chunk: {meta.get('chunk_id', '')}")
            st.text(doc.page_content)
            url = meta.get("source_url", "")
            parsed = urlparse(url)
            if parsed.scheme == "https" and parsed.netloc:
                # A source can appear in several results/tabs; links need no widget state.
                safe_url = quote(url, safe=":/?&=%#@+;,~")
                st.markdown(f"[Open source document]({safe_url})")
            with st.expander("Source metadata"):
                st.json(meta)


def render_search(settings):
    # A rerun can retain old raw results after deploying context-aware retrieval.
    if st.session_state.get("semantic_search_version") != 2:
        st.session_state.pop("semantic_result", None)
        st.session_state["semantic_search_version"] = 2
    st.info("Inspect semantic retrieval directly, without generating an answer.")
    st.caption("Similarity scores rank passages; they are not probabilities of clinical correctness.")
    st.caption("Test context searches the test name and surrounding indexed chunks, then shows matching requirements. Disable it to inspect raw similarity results.")
    if settings.embedding_provider == "openai":
        st.caption("Questions are embedded through the paid OpenAI API. Only the resulting vectors are sent to Pinecone for search.")
    with st.form("semantic_search"):
        query = st.text_area("Your laboratory question", max_chars=4000,
                             placeholder="What specimen is required for an HbA1c test?")
        strategy = st.selectbox("Search chunking strategy", ["section", "fixed"])
        corpus = st.selectbox("Search corpus", ["public", "synthetic"])
        top_k = st.number_input("Search top-k", 1, 100, settings.retrieval_top_k)
        include_context = st.checkbox("Retrieve test context", value=True,
            help="Find connected requirements across small chunks. Makes additional read-only searches; never indexes or calls Groq.")
        provider = st.text_input("Laboratory provider filter (optional exact match)")
        test_name = st.text_input("Test name filter (optional exact match)")
        document = st.text_input("Document ID filter (optional exact match)")
        revision = st.text_input("Source revision date filter (optional exact match)")
        submitted = st.form_submit_button("Search laboratory references")
    if submitted:
        st.session_state.pop("semantic_result", None)
        try:
            filters = {k: v.strip() for k, v in {"laboratory_provider": provider, "test_name": test_name,
                       "document_id": document, "source_revision_date": revision}.items() if v.strip()}
            if not query.strip():
                raise ValueError("Enter a laboratory question first.")
            with st.spinner("Embedding question and retrieving evidence…"):
                connection, embeddings = services(settings)
                retriever = SemanticRetriever(settings, embeddings, connection)
                search = retriever.search_with_context if include_context else retriever.search
                st.session_state["semantic_result"] = search(query, strategy, int(top_k), filters, corpus)
        except (ValueError, PineconeError) as exc:
            st.error(str(exc))
        except Exception:
            st.error("Search could not complete. Check Day 3 dependencies, model availability, and Pinecone connectivity.")
    if "semantic_result" in st.session_state:
        result = st.session_state["semantic_result"]
        st.caption(f"Results for: {result.query}")
        if result.diagnostics.get("mode")=="test_context":
            st.caption("Test-context results: only identified test regions matching the requested instruction are shown.")
            with st.expander("Search diagnostics"):
                st.json(result.diagnostics)
            if not result.matches:
                st.info("No verified instruction chunks were found for this test. Inspect Search diagnostics or disable Retrieve test context to examine raw matches.")
        else:
            st.warning("Raw similarity results: test identity is not validated. For test-specific requirements, enable Retrieve test context and click Search laboratory references again.")
        show_results(result)


@st.cache_data(show_spinner=False)
def inventory(signatures):
    result = []
    for filename, _, _ in signatures:
        try:
            with Path(filename).open(encoding="utf-8") as handle:
                first = json.loads(next(handle))
                count = 1 + sum(1 for _ in handle)
            result.append({"path": filename, "document_id": first["metadata"]["document_id"],
                "title": first["metadata"]["document_title"], "strategy": first["metadata"]["chunking_strategy"], "chunks": count})
        except (ValueError, OSError, StopIteration, KeyError, TypeError):
            result.append({"path": filename, "title": "Invalid chunk file; reprocess source", "strategy": "invalid", "chunks": 0})
    return result


def render_indexing(settings):
    st.divider()
    st.subheader("Pinecone Vector Indexing")
    st.write(f"Index: {settings.pinecone_index_name} · Provider: {settings.embedding_provider} · Model: {settings.embedding_model} · Dimensions: {settings.embedding_dimension}")
    if settings.embedding_provider == "openai":
        st.info("OpenAI embedding actions send text to the OpenAI API and incur usage charges. The model/dimension-specific namespaces keep these vectors separate from local embeddings.")
    st.caption("No connection or uploads occur until you use the buttons below. Indexing uploads chunk text and source metadata to Pinecone.")
    if st.button("Validate OpenAI Embedding API (paid request)" if settings.embedding_provider == "openai" else "Validate Local Embedding Model"):
        try:
            with st.spinner("Calling OpenAI embedding API…" if settings.embedding_provider == "openai" else "Loading local embedding model…"):
                service = get_configured_embedding_service(settings)
                vector = service.embed_query("Synthetic software verification query.")
            st.success(f"Embedding provider generated {len(vector)} finite dimensions. No Pinecone request was made.")
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    if st.button("Validate Pinecone Connection"):
        try:
            connection = PineconeConnection(settings).connect()
            stats = connection.stats()
            st.session_state["pinecone_status"] = {"index": settings.pinecone_index_name, "namespaces": field(stats, "namespaces", {})}
        except (ValueError, PineconeError) as exc:
            st.error(str(exc))
        except Exception:
            st.error("Pinecone validation failed. Check index configuration, API permissions, and network access.")
    if status := st.session_state.get("pinecone_status"):
        st.success(f"Last validated connection: {status['index']}")
        st.json(status)
    else:
        st.caption("Connection status: not validated in this session.")
    paths = sorted(Path("data/processed").glob("*/*.jsonl"))
    items = inventory(tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths))
    st.write(f"Locally processed chunks: {sum(item['chunks'] for item in items)}")
    if items:
        st.dataframe([{k: v for k, v in item.items() if k != "path"} for item in items], hide_index=True)
    try:
        manifest = IndexingManifest().read()
        summaries = [{k: entry.get(k, "") for k in ("document_id", "chunking_strategy", "namespace", "status", "successful_indexing_timestamp")}
                     for entry in manifest["entries"].values()]
        if summaries:
            st.dataframe(summaries, hide_index=True)
        else:
            st.caption("Indexing status: no successful indexing recorded yet.")
    except ValueError as exc:
        st.error(str(exc))
        return
    strategy = st.selectbox("Select Chunking Strategy for indexing", ["section", "fixed"])
    corpus = st.selectbox("Indexing corpus", ["public", "synthetic"])
    available = {item["path"]: item for item in items if item["strategy"] == strategy}
    selected = st.multiselect("Processed documents to index", list(available),
        format_func=lambda path: f"{available[path]['title']} · {available[path]['document_id'][:8]} · {available[path]['chunks']} chunks")
    st.caption(f"Destination namespace: {namespace_for(settings, strategy, corpus)}. Use the same approved documents in each strategy for a fair comparison.")
    tables = st.checkbox("I reviewed detected tables against the source and approve their non-oversized chunks")
    cleanup = st.checkbox("Remove only obsolete vectors owned by these document snapshots after successful reindexing")
    permitted = st.checkbox("I approve uploading these permitted references without patient information to Pinecone")
    if st.button("Index Processed Documents", disabled=not selected or not permitted):
        bar = st.progress(0.0)
        results = []
        try:
            with st.spinner("Validating and indexing processed chunks…"):
                connection, embeddings = services(settings)
                manager = IndexManager(settings, embeddings, connection)
                for path in selected:
                    try:
                        result = manager.index_file(path, strategy, corpus, tables, cleanup,
                            progress=lambda done, total: bar.progress(done / total, text=f"{done}/{total} chunks acknowledged"))
                        results.append(result)
                    except (ValueError, PineconeError) as exc:
                        results.append({"document": available[path]["document_id"], "status": "failed", "error": str(exc)})
            st.session_state["indexing_summary"] = results
        except (ValueError, PineconeError) as exc:
            st.error(str(exc))
        except Exception:
            st.error("Indexing could not start. Check model dependencies, Pinecone access, and local manifest permissions.")
    if "indexing_summary" in st.session_state:
        st.json(st.session_state["indexing_summary"])


def render_comparison(settings):
    st.subheader("Compare semantic retrieval")
    if settings.embedding_provider == "openai":
        st.caption("Comparison uses one shared paid OpenAI query embedding for both namespaces.")
    with st.form("retrieval_comparison_form"):
        question = st.text_area("Comparison question", max_chars=4000)
        top_k = st.number_input("Comparison top-k", 1, 100, settings.retrieval_top_k)
        corpus = st.selectbox("Comparison corpus", ["public", "synthetic"])
        submitted = st.form_submit_button("Compare Strategies")
    if submitted:
        st.session_state.pop("retrieval_comparison", None)
        try:
            if not question.strip():
                raise ValueError("Enter a comparison question first.")
            state = IndexingManifest().read()
            sets = {strategy: {e["document_id"]: e.get("source_content_hash") for e in state["entries"].values()
                    if e.get("namespace") == namespace_for(settings, strategy, corpus)
                    and e.get("index_name") == settings.pinecone_index_name and e.get("status") == "complete"}
                    for strategy in ("fixed", "section")}
            if sets["fixed"] != sets["section"]:
                st.warning("The indexed document collections differ. Index the same sources in both strategies before judging retrieval quality.")
            with st.spinner("Searching both strategy namespaces…"):
                connection, embeddings = services(settings)
                st.session_state["retrieval_comparison"] = SemanticRetriever(settings, embeddings, connection).compare_retrieval(question, int(top_k), corpus=corpus)
        except (ValueError, PineconeError) as exc:
            st.error(str(exc))
        except Exception:
            st.error("Comparison failed. Check model availability and Pinecone connectivity.")
    if results := st.session_state.get("retrieval_comparison"):
        for column, (strategy, result) in zip(st.columns(2), results.items()):
            with column:
                st.subheader("Fixed-size" if strategy == "fixed" else "Section-aware")
                show_results(result)
        st.caption("Similarity scores alone cannot determine which strategy is better. Labeled retrieval evaluation is planned for Day 5.")
