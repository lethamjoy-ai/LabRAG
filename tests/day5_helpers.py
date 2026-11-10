from langchain_core.documents import Document
from src.evaluation.dataset_validator import fingerprint
from src.retrieval.semantic_retriever import SearchResult


def synthetic_dataset():
    labels={s:{"namespace":s+"-chunks","chunk_ids":[s+"-ca"],"complete":True,"sections":[]} for s in ["fixed","section"]}
    data={"schema_version":1,"synthetic":True,"source_sha256":"synthetic-fixture","embedding_model":"sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimension":384,"questions":[{"question_id":"SYN01.1","test_id":"SYN01","turn":1,"question":"What is the minimum volume for CA?",
            "resolved_question":"What is the minimum volume for CA?","history":[],"category":"single_fact_retrieval","expected_test_codes":["CA"],
            "expected_facts":["Minimum specimen volume: 0.25 mL."],"source_catalog_pages":[],"labels":labels,"source_verified":True,
            "answerability":"answerable","score_ranking":True,"expected_outcome":"answered"}]}
    data["fingerprint"]=fingerprint(data)
    return data


class SyntheticRetriever:
    def __init__(self):self.calls=[]
    def search(self,query,strategy="section",top_k=5,filters=None):
        self.calls.append((query,strategy,top_k,filters))
        wrong=Document(page_content="Unrelated synthetic test.",metadata={"chunk_id":strategy+"-other","document_id":"synthetic-other","test_code":"OTHER"})
        correct=Document(page_content="Minimum specimen volume: 0.25 mL.",metadata={"chunk_id":strategy+"-ca","document_id":"synthetic-ca","test_code":"CA","test_name":"Synthetic Calcium, Total, Serum"})
        return SearchResult(query,strategy,strategy+"-chunks",[{"document":wrong,"score":.9},{"document":correct,"score":.7}][:top_k],0)


class SyntheticCrossEncoder:
    def predict(self,pairs,**kwargs):return [1 if '0.25' in text else -1 for _,text in pairs]


class SyntheticGenerator:
    def select(self,messages):
        import json
        from src.llm.groq_service import AnswerSelection
        data=json.loads(messages[-1].content)
        return AnswerSelection(excerpts=[{"source_id":x["source_id"],"quote":x["text"]} for x in data["retrieved_evidence"] if "0.25" in x["text"]])
