import os
import time
import httpx
import pytest

BASE = os.environ.get("CHATBOT_BASE_URL", "http://127.0.0.1:8000")

@pytest.mark.parametrize("msg,expect_intent", [
    ("Total sales in 2008", "sql_query"),
    ("Quarterly performance Q2 2008", "sql_query"),
    ("Top 5 products by revenue in 2008", "sql_query"),
    ("Filter dashboard to 2008", "visual_update"),
])
def test_intent_and_no_500(msg, expect_intent):
    with httpx.Client(timeout=30.0) as c:
        r = c.post(f"{BASE}/api/chat", json={"user_id": "pytest_user", "message": msg})
        assert r.status_code == 200, r.text
        data = r.json()
        assert "intent" in data, data
        assert data["intent"].get("intent_type") == expect_intent, data["intent"]
        # no backend error message
        assert "error" not in (data.get("response") or "").lower()

def test_topn_and_year_filter_present_in_logs(monkeypatch):
    """
    Optional: if you expose last-sql in a debug endpoint, call it here and assert YEAR(...) and TOP N exist.
    If not available, just ensure the response contains a 'Top 5' phrasing as a proxy.
    """
    with httpx.Client(timeout=30.0) as c:
        r = c.post(f"{BASE}/api/chat", json={"user_id": "pytest_user", "message": "Top 5 products by revenue in 2008"})
        assert r.status_code == 200
        data = r.json()
        # basic sanity on NL text
        assert "Top 5" in (data.get("response") or "") or "top five" in (data.get("response") or "")
