"""Optional exhaustive read-only namespace audit for TC17; never top-k evidence."""
import hashlib
import os
import re
from datetime import datetime,timezone
from src.vectorstore.pinecone_client import field,namespace_for,retry


def audit_absence(connection,settings,case,max_vectors=100000):
    if os.getenv("LABRAG_RUN_DAY4_LIVE")!="1":
        raise ValueError("Absence audit requires explicit LABRAG_RUN_DAY4_LIVE=1.")
    namespace=namespace_for(settings,settings.rag_default_strategy)
    index=connection.index
    stats=retry(lambda:index.describe_index_stats())
    count=int(field(field(stats,"namespaces",{}).get(namespace,{}),"vector_count",0))
    if count>max_vectors:
        raise ValueError("Namespace exceeds the bounded audit limit; no absence claim made.")

    def identifiers():
        found=[]
        for page in index.list(namespace=namespace,limit=100):
            if not isinstance(page,list) or not all(isinstance(i,str) for i in page):
                raise ValueError("Unsupported list response; no absence claim made.")
            found.extend(page)
            if len(found)>max_vectors:
                raise ValueError("Audit exceeded its bound; no absence claim made.")
        return set(found)

    ids=identifiers()
    if len(ids)!=count:
        raise ValueError("Listed IDs do not match namespace count; no absence claim made.")
    hits=[]
    code=re.compile(r"\b"+re.escape(case["test_code"])+r"\b",re.I)
    name=" ".join(case["test_name"].lower().split())
    ordered=sorted(ids)
    for offset in range(0,len(ordered),100):
        batch=ordered[offset:offset+100]
        vectors=field(retry(lambda:index.fetch(ids=batch,namespace=namespace)),"vectors",{})
        if set(vectors)!=set(batch):
            raise ValueError("Some listed vectors could not be fetched; no absence claim made.")
        for identifier,vector in vectors.items():
            metadata=field(vector,"metadata",{})
            text=str(metadata.get("text",""))
            if not text:
                raise ValueError("A vector lacks source text; no absence claim made.")
            if code.search(text) or str(metadata.get("test_code","")).upper()==case["test_code"].upper() or name in " ".join(text.lower().split()):
                hits.append(identifier)
    if identifiers()!=ids:
        raise ValueError("Namespace changed during audit; no absence claim made.")
    return {"cloud_absence_verified":not hits,"method":"Exhaustive ID listing and text/code/name inspection; IDs rechecked after fetch",
        "index":settings.pinecone_index_name,"namespace":namespace,"vectors_scanned":len(ids),
        "listed_ids_sha256":hashlib.sha256("\n".join(ordered).encode()).hexdigest(),
        "matching_vector_ids":hits,"audited_at":datetime.now(timezone.utc).isoformat(),
        "scope":"Current selected namespace only; not the full Mayo catalog. Literal code/name audit, not semantic proof of all aliases."}
