import pytest
from src.evaluation.retrieval_metrics import ranking_metrics


@pytest.mark.parametrize("rank,expected",[(1,1),(3,1/3),(5,.2),(6,0)])
def test_first_relevant_rank_and_cutoff(rank,expected):
    ids=[f"other-{i}" for i in range(6)];ids[rank-1]="relevant"
    m=ranking_metrics(ids,{"relevant"},complete=True)
    assert m["mrr"]==expected and m["hit_rate"]==int(rank<=5)
    assert m["recall"]==int(rank<=5)


def test_recall_precision_duplicates_and_empty():
    m=ranking_metrics(["a","a","b","wrong"],{"a","b","c"},complete=True)
    assert m["precision"]==2/3 and m["recall"]==2/3
    assert ranking_metrics([],{'a'},complete=True)["precision"]==0
    assert ranking_metrics(['wrong'],{'a'})["judged_pool_recall"]==0
    assert ranking_metrics(['a'],{'a'})["recall"] is None


@pytest.mark.parametrize("relevant,verified",[(None,True),(set(),True),({'a'},False)])
def test_missing_unverified_labels_are_not_scored(relevant,verified):
    result=ranking_metrics(['a'],relevant,verified=verified)
    assert not result["scored"] and result["hit_rate"] is None


def test_namespace_chunk_ids_are_not_interchangeable():
    assert ranking_metrics(['fixed-a'],{'section-a'})['hit_rate']==0
