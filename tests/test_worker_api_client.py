import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = (REPO_ROOT / "worker").resolve()
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))


from api_client import APIClient  # noqa: E402


def _response(status_code: int) -> httpx.Response:
    req = httpx.Request("GET", "http://example")
    return httpx.Response(status_code=status_code, request=req)


def test_headers_include_signature_when_secret_and_body_present() -> None:
    c = APIClient(base_url="http://example", token="t")
    c.set_credentials(token="t", signing_secret="s")
    headers = c._headers(body='{"a":1}', path="/p")
    assert headers["X-Worker-Token"] == "t"
    assert "X-Signature" in headers
    assert "X-Timestamp" in headers


def test_request_with_retry_retries_on_5xx_and_succeeds() -> None:
    c = APIClient(base_url="http://example")

    calls = {"n": 0}

    def fake_request(method, url, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            r = _response(503)
            raise httpx.HTTPStatusError("x", request=r.request, response=r)
        return httpx.Response(200, request=httpx.Request(method, url), json={"ok": True})

    with patch.object(c.client, "request", side_effect=fake_request), patch.object(time, "sleep", return_value=None):
        resp = c._request_with_retry("GET", "http://example/x")
        assert resp.status_code == 200
        assert calls["n"] == 3


def test_fetch_next_job_handles_204() -> None:
    c = APIClient(base_url="http://example")
    with patch.object(c.client, "get", return_value=httpx.Response(204, request=httpx.Request("GET", "http://example"))):
        assert c.fetch_next_job("w") is None


def test_fetch_next_job_handles_404() -> None:
    c = APIClient(base_url="http://example")
    r = httpx.Response(404, request=httpx.Request("GET", "http://example"))
    with patch.object(c.client, "get", return_value=r):
        assert c.fetch_next_job("w") is None


def test_request_with_retry_does_not_retry_on_4xx() -> None:
    c = APIClient(base_url="http://example")
    r = httpx.Response(400, request=httpx.Request("GET", "http://example"))
    with patch.object(c.client, "request", side_effect=httpx.HTTPStatusError("x", request=r.request, response=r)):
        with pytest.raises(httpx.HTTPStatusError):
            c._request_with_retry("GET", "http://example/x")


def test_verify_credentials_returns_false_on_exception() -> None:
    c = APIClient(base_url="http://example")
    with patch.object(c.client, "post", side_effect=RuntimeError("boom")):
        assert c.verify_credentials("w", "t") is False


def test_get_config_returns_none_on_non_200() -> None:
    c = APIClient(base_url="http://example")
    r = httpx.Response(500, request=httpx.Request("GET", "http://example"))
    with patch.object(c.client, "get", return_value=r):
        assert c.get_config("w") is None
