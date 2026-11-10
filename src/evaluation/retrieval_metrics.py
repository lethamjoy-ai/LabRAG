"""ID-based ranking metrics with explicit relevance completeness."""
def ranking_metrics(retrieved,relevant,k=5,verified=True,complete=False):
    if type(k) is not int or k<1:
        raise ValueError("k must be positive")
    empty={"hit_rate":None,"precision":None,"mrr":None,"recall":None,"judged_pool_recall":None,"first_relevant_rank":None}
    if not verified or relevant is None or not relevant:
        return {**empty,"scored":False,"reason":"Unverified or missing positive relevance labels"}
    ranked=list(dict.fromkeys(retrieved))[:k]
    relevant=set(relevant)
    ranks=[i for i,identifier in enumerate(ranked,1) if identifier in relevant]
    hits=len(ranks)
    return {"scored":True,"hit_rate":int(bool(hits)),"precision":hits/len(ranked) if ranked else 0.0,
        "mrr":1/ranks[0] if ranks else 0.0,"recall":hits/len(relevant) if complete else None,
        "judged_pool_recall":hits/len(relevant) if not complete else None,
        "first_relevant_rank":ranks[0] if ranks else None,"returned_count":len(ranked),
        "known_relevant_count":len(relevant),"missed_relevant_ids":sorted(relevant-set(ranked)),
        "label_scope":"exhaustive" if complete else "positive judged pool; unjudged results may also be relevant",
        "precision_denominator":"unique actual returned items up to k"}
