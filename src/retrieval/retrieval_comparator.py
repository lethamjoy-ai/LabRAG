"""Compare A/B/C/D within an explicit raw or Explorer test-context protocol."""
import time
from dataclasses import dataclass
from src.retrieval.reranker import Reranker


@dataclass(frozen=True)
class RetrievalConfiguration:
    key: str
    strategy: str
    rerank: bool
    label: str


CONFIGURATIONS={c.key:c for c in [RetrievalConfiguration("A","fixed",False,"Fixed"),
    RetrievalConfiguration("B","section",False,"Section"),RetrievalConfiguration("C","fixed",True,"Fixed + reranking"),
    RetrievalConfiguration("D","section",True,"Section + reranking")]}


class RetrievalComparator:
    def __init__(self,retriever,reranker=None):self.retriever,self.reranker=retriever,reranker or Reranker()

    def run(self,question,configuration,top_k=5,candidate_count=15,filters=None,protocol="raw"):
        if protocol not in {"raw","test_context"}:raise ValueError("Unknown retrieval protocol")
        if configuration not in CONFIGURATIONS:raise ValueError("Unknown retrieval configuration")
        config=CONFIGURATIONS[configuration]
        if type(top_k) is not int or not 1<=top_k<=100 or type(candidate_count) is not int or not top_k<=candidate_count<=100:
            raise ValueError("Require 1 <= final top-k <= candidate count <= 100")
        start=time.perf_counter()
        search=self.retriever.search if protocol=="raw" else self.retriever.search_with_context
        result=search(question,strategy=config.strategy,top_k=candidate_count if config.rerank else top_k,filters=filters)
        search_seconds=time.perf_counter()-start
        selected=result.matches;reranking_seconds=0.0
        if config.rerank:
            reranked=self.reranker.rerank(question,selected,top_k)
            selected=reranked.matches;reranking_seconds=reranked.latency_seconds
        return {"configuration":configuration,"strategy":config.strategy,"namespace":result.namespace,
            "diagnostics":result.diagnostics,
            "candidate_matches":result.matches,"matches":selected,"reranking_applied":config.rerank,
            "embedding_seconds":result.diagnostics.get("embedding_seconds"),"search_seconds":result.diagnostics.get("search_seconds"),
            "retrieval_seconds":search_seconds,"reranking_seconds":reranking_seconds,"pipeline_seconds":time.perf_counter()-start}
