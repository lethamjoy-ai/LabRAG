"""Lazy CPU cross-encoder. Scores are rankings, never clinical probabilities."""
import math
import time
from dataclasses import dataclass
from functools import lru_cache

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


class RerankerError(RuntimeError):
    pass


@lru_cache(maxsize=2)
def load_cross_encoder(model_name=MODEL):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name,device="cpu",max_length=512,
        cache_folder=".model-cache/rerankers",trust_remote_code=False,
        revision=MODEL_REVISION if model_name==MODEL else None)


@dataclass
class RerankResult:
    matches: list
    latency_seconds: float
    candidate_count: int


class Reranker:
    def __init__(self,model_name=MODEL,backend=None):
        self.model_name,self.backend=model_name,backend
        self.model_revision=MODEL_REVISION if model_name==MODEL else None

    def rerank(self,question,candidates,top_k=5):
        if not isinstance(question,str) or not question.strip() or type(top_k) is not int or not 1<=top_k<=100:
            raise ValueError("Reranking requires a question and final top-k between 1 and 100.")
        if not isinstance(candidates,list) or any(not isinstance(m,dict) or "document" not in m or not hasattr(m["document"],"page_content") for m in candidates):
            raise ValueError("Reranking expects retrieved document/score dictionaries.")
        try:
            if any(not math.isfinite(float(m["score"])) for m in candidates):raise ValueError
        except (KeyError,TypeError,ValueError):
            raise ValueError("Each reranking candidate requires a finite Pinecone score.") from None
        started=time.perf_counter()
        unique=[];seen=set()
        for match in candidates:
            doc=match["document"]
            identity=doc.metadata.get("chunk_id") or (doc.metadata.get("document_id"),doc.page_content)
            if identity not in seen:
                seen.add(identity);unique.append(match)
        if not unique:
            return RerankResult([],time.perf_counter()-started,0)
        try:
            if self.backend is None:
                self.backend=load_cross_encoder(self.model_name)
            # The cross-encoder tokenizes/truncates pairs at its supported 512-token
            # limit. Full source documents/metadata remain unchanged for citations.
            scores=self.backend.predict([(question,m["document"].page_content) for m in unique],
                batch_size=16,show_progress_bar=False,convert_to_numpy=True)
            if len(scores)!=len(unique) or any(not math.isfinite(float(s)) for s in scores):
                raise ValueError("Invalid cross-encoder scores")
            scored=[{**m,"reranker_score":float(s)} for m,s in zip(unique,scores)]
            scored.sort(key=lambda m:-m["reranker_score"])
            return RerankResult(scored[:top_k],time.perf_counter()-started,len(unique))
        except Exception:
            raise RerankerError("Local cross-encoder failed. Check Day 5 dependencies/model availability. No fallback retrieval configuration was used.") from None
