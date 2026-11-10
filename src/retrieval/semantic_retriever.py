"""Evidence retrieval only: cosine similarities are not calibrated confidence."""
import math
import time
from dataclasses import dataclass, field
from src.vectorstore.pinecone_client import namespace_for, retry

FILTERS = {"laboratory_provider", "test_name", "document_id", "source_revision_date"}


@dataclass
class SearchResult:
    query: str
    strategy: str
    namespace: str
    matches: list
    latency_seconds: float
    diagnostics: dict = field(default_factory=dict)


class SemanticRetriever:
    def __init__(self, settings, embeddings, connection):
        self.settings, self.embeddings, self.connection = settings, embeddings, connection

    def search(self, query, strategy="section", top_k=5, filters=None, corpus="public", _vector=None):
        started = time.perf_counter()
        if not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise ValueError("Enter a question between 1 and 4000 characters.")
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError("Top-k must be an integer between 1 and 100.")
        filters = filters or {}
        if not isinstance(filters, dict) or any(k not in FILTERS or not isinstance(v, str) or not v.strip() for k, v in filters.items()):
            raise ValueError("Filters support exact nonempty provider, test_name, document_id, and source_revision_date strings.")
        namespace = namespace_for(self.settings, strategy, corpus)
        clauses = [{"embedding_model": {"$eq": self.settings.embedding_model}}, {"corpus_type": {"$eq": corpus}}]
        clauses.extend({k: {"$eq": v}} for k, v in filters.items())
        embedding_started=time.perf_counter()
        vector = _vector if _vector is not None else self.embeddings.embed_query(query)
        embedding_seconds=time.perf_counter()-embedding_started
        search_started=time.perf_counter()
        store = self.connection.vector_store(self.embeddings, namespace)
        results = retry(lambda: store.similarity_search_by_vector_with_score(vector, k=top_k, filter={"$and": clauses}))
        matches = [{"document": doc, "score": float(score)} for doc, score in results
                   if math.isfinite(score) and score >= self.settings.retrieval_score_threshold]
        return SearchResult(query, strategy, namespace, matches, time.perf_counter() - started,
            {"mode":"raw_similarity","embedding_seconds":embedding_seconds,"search_seconds":time.perf_counter()-search_started})

    def compare_retrieval(self, question, top_k=5, filters=None, corpus="public"):
        vector = self.embeddings.embed_query(question)
        return {strategy: self.search(question, strategy, top_k, filters, corpus, _vector=vector)
                for strategy in ("fixed", "section")}

    def search_for_chat(self, query, strategy="section", top_k=5, filters=None, corpus="public"):
        """Search the instruction and the test identity separately before expansion.

        Small requirement chunks frequently lack the test name. A second, bounded
        identity search supplies heading-page anchors without changing stored vectors.
        """
        from src.rag.evidence_validator import identity_query
        started = time.perf_counter()
        initial = self.search(query, strategy=strategy, top_k=top_k, filters=filters, corpus=corpus)
        anchor_query = identity_query(query)
        anchors = self.search(anchor_query, strategy=strategy, top_k=max(top_k, 5), filters=filters, corpus=corpus).matches if anchor_query and anchor_query.casefold() != query.casefold() else []
        matches, seen = [], set()
        for match in [*anchors, *initial.matches]:
            doc = match["document"]
            key = (doc.metadata.get("chunk_id"), doc.page_content)
            if key not in seen:
                seen.add(key)
                matches.append(match)
        return SearchResult(query, strategy, initial.namespace, matches, time.perf_counter() - started,
            {"instruction_matches": len(initial.matches), "identity_query": anchor_query,
             "identity_matches": len(anchors), "seed_top_k": top_k,
             "identity_search_ids":[m['document'].metadata.get('chunk_id') for m in anchors],
             "instruction_search_ids":[m['document'].metadata.get('chunk_id') for m in initial.matches]})

    def expand_context(self, result, filters=None, corpus="public"):
        """Bounded read-only page neighbors from the same namespace/model/corpus.

        Only chat uses this. Initial semantic top-k remains visible separately.
        No local files or vectors from another namespace provide answer evidence.
        """
        from src.rag.indexed_context import enrich_indexed_matches
        from src.rag.indexed_context import headers
        from src.rag.evidence_validator import identity_query
        import re
        identity = set(re.findall(r"[a-z0-9]+", identity_query(result.query).lower()))
        def anchor_strength(match):
            doc = match["document"]
            labels = [(doc.metadata.get("test_name", ""), doc.metadata.get("test_code", ""))]
            labels.extend((name, code) for _, name, code in headers(doc.page_content) if name)
            return max((len(identity & set(re.findall(r"[a-z0-9]+", f"{name} {code}".lower())))
                        for name, code in labels), default=0)
        seeds = []
        for match in sorted(result.matches, key=anchor_strength, reverse=True):
            meta = match["document"].metadata
            page = meta.get("page_number")
            if meta.get("document_id") and page:
                seed = (meta["document_id"], int(page))
                if seed not in seeds:
                    seeds.append(seed)
        matches = list(result.matches)
        seen = {(m["document"].metadata.get("chunk_id"), m["document"].page_content) for m in matches}
        if seeds:
            vector = self.embeddings.embed_query(result.query)
            store = self.connection.vector_store(self.embeddings, result.namespace)
            pages = sorted({(doc, p + offset) for doc, p in seeds[:3] for offset in (-1, 0, 1) if p + offset > 0})
            for document_id, page in pages:
                clauses = [{"embedding_model": {"$eq": self.settings.embedding_model}},
                           {"corpus_type": {"$eq": corpus}}, {"document_id": {"$eq": document_id}},
                           {"page_number": {"$eq": page}}]
                clauses.extend({key: {"$eq": value}} for key, value in (filters or {}).items())
                neighbors = retry(lambda: store.similarity_search_by_vector_with_score(vector, k=100, filter={"$and": clauses}))
                for doc, score in neighbors:
                    key = (doc.metadata.get("chunk_id"), doc.page_content)
                    if math.isfinite(score):
                        # Page queries use the full question, unlike identity anchors.
                        # Replace anchor-query scores before displaying/ranking details.
                        for match in matches:
                            existing = match["document"]
                            if (existing.metadata.get("chunk_id"), existing.page_content) == key:
                                match["score"] = float(score)
                    if key not in seen and math.isfinite(score):
                        matches.append({"document": doc, "score": float(score)})
                        seen.add(key)
        return enrich_indexed_matches(matches)

    def search_with_context(self, query, strategy="section", top_k=5, filters=None, corpus="public"):
        """Explorer results: test-scoped instruction passages with original cosine scores.

        This filters expanded indexed context and sorts by full-question similarity;
        it does not add a learned reranker or substitute another namespace's content.
        """
        import re
        from src.rag.citation_manager import make_sources
        from src.rag.evidence_validator import scope_sources, requested_facets
        started = time.perf_counter()
        seeds = self.search_for_chat(query, strategy, top_k, filters, corpus)
        expanded = self.expand_context(seeds, filters, corpus)
        scoped, reason = scope_sources(query, make_sources(expanded))
        patterns = [pattern for _,pattern in requested_facets(query)]
        candidates = []
        for doc in scoped.values():
            if patterns and not any(re.search(pattern, doc.page_content, re.I) for pattern in patterns):
                continue
            parent = next((match for match in expanded if
                match["document"].metadata.get("chunk_id") == doc.metadata.get("chunk_id") and
                doc.page_content in match["document"].page_content), None)
            if parent:
                candidates.append({"document": doc, "score": parent["score"]})
        candidates.sort(key=lambda match: match["score"], reverse=True)
        return SearchResult(query, strategy, seeds.namespace, candidates[:top_k], time.perf_counter() - started,
            {**seeds.diagnostics, "mode": "test_context", "expanded_chunks": len(expanded),
             "test_scoped_chunks": len(scoped), "instruction_candidates": len(candidates), "identity_check": reason,
             "candidate_stage":"final returned context candidates after test/facet filtering and cap",
             "configuration_evidence":[{k:m['document'].metadata[k] for k in
                 ('chunk_id','document_id','source_content_hash','chunking_strategy','chunk_size','chunk_overlap')
                 if k in m['document'].metadata} for m in expanded],
             "intermediate_seed_ids":[m['document'].metadata.get('chunk_id') for m in seeds.matches],
             "intermediate_expanded_ids":[m['document'].metadata.get('chunk_id') for m in expanded],
             "pre_cap_instruction_ids":[m['document'].metadata.get('chunk_id') for m in candidates],
             "returned_context_ids":[m['document'].metadata.get('chunk_id') for m in candidates[:top_k]]})
