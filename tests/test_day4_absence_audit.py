from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from config.settings import Settings
from src.evaluation.absence_audit import audit_absence


def test_exhaustive_audit_is_gated_and_read_only(monkeypatch):
    index=Mock()
    case={"test_code":"DHVD","test_name":"1,25-Dihydroxyvitamin D, Serum"}
    connection=SimpleNamespace(index=index)
    monkeypatch.delenv("LABRAG_RUN_DAY4_LIVE",raising=False)
    with pytest.raises(ValueError,match="explicit"):
        audit_absence(connection,Settings(),case)
    index.list.assert_not_called()
    monkeypatch.setenv("LABRAG_RUN_DAY4_LIVE","1")
    index.describe_index_stats.return_value={"namespaces":{"section-chunks":{"vector_count":2}}}
    index.list.return_value=[["a","b"]]
    index.fetch.return_value={"vectors":{"a":{"metadata":{"text":"BUN Serum"}},"b":{"metadata":{"text":"CA Total Serum"}}}}
    result=audit_absence(connection,Settings(),case)
    assert result["cloud_absence_verified"] and result["vectors_scanned"]==2
    assert index.list.call_count==2
    index.upsert.assert_not_called();index.delete.assert_not_called()
    index.fetch.return_value={"vectors":{"a":{"metadata":{"text":"DHVD Test entry"}},"b":{"metadata":{"text":"CA Serum"}}}}
    assert not audit_absence(connection,Settings(),case)["cloud_absence_verified"]


def test_incomplete_namespace_audit_cannot_certify_absence(monkeypatch):
    monkeypatch.setenv("LABRAG_RUN_DAY4_LIVE","1")
    index=Mock()
    index.describe_index_stats.return_value={"namespaces":{"section-chunks":{"vector_count":2}}}
    index.list.return_value=[["a"]]
    with pytest.raises(ValueError,match="count"):
        audit_absence(SimpleNamespace(index=index),Settings(),{"test_code":"DHVD","test_name":"Vitamin D"})
