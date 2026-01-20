import sys
import time
from pathlib import Path
from unittest.mock import patch

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
SDK_DIR = (REPO_ROOT / "sdk" / "python").resolve()
if str(SDK_DIR) not in sys.path:
    sys.path.insert(0, str(SDK_DIR))


from inference_client import InferenceClient  # noqa: E402


def test_request_with_fallback_uses_fallback_after_timeout() -> None:
    client = InferenceClient(base_url="http://primary", fallback_urls=["http://backup"], max_retries=2)

    calls = []

    def fake_request(method, url, **kwargs):
        calls.append(url)
        if url.startswith("http://primary"):
            raise httpx.TimeoutException("timeout")
        return httpx.Response(200, request=httpx.Request(method, url), json={"ok": True})

    with patch.object(client.client, "request", side_effect=fake_request), patch.object(time, "sleep", return_value=None):
        resp = client._request_with_fallback("GET", "/ping")
        assert resp.status_code == 200
        assert any(u.startswith("http://backup") for u in calls)


def test_request_with_fallback_raises_on_4xx() -> None:
    client = InferenceClient(base_url="http://primary", max_retries=2)

    def fake_request(method, url, **kwargs):
        r = httpx.Response(400, request=httpx.Request(method, url))
        raise httpx.HTTPStatusError("bad", request=r.request, response=r)

    with patch.object(client.client, "request", side_effect=fake_request):
        try:
            client._request_with_fallback("GET", "/bad")
        except httpx.HTTPStatusError as e:
            assert e.response.status_code == 400
        else:
            raise AssertionError("Expected HTTPStatusError")


def test_headers_include_api_key_when_set() -> None:
    c = InferenceClient(base_url="http://x", api_key="k")
    h = c._headers()
    assert h["X-API-Key"] == "k"


def test_chat_builds_correct_endpoint_for_sync_and_async() -> None:
    c = InferenceClient(base_url="http://x")

    calls = []

    def fake_request(method, path, **kwargs):
        calls.append((method, path, kwargs.get("json")))
        return httpx.Response(200, request=httpx.Request(method, "http://x" + path), json={"ok": True})

    with patch.object(c, "_request_with_fallback", side_effect=fake_request):
        c.chat(messages=[{"role": "user", "content": "hi"}], sync=True)
        c.chat(messages=[{"role": "user", "content": "hi"}], sync=False)

    assert calls[0][1] == "/api/v1/jobs/sync"
    assert calls[1][1] == "/api/v1/jobs"


def test_generate_image_uses_direct_mode_short_circuit() -> None:
    c = InferenceClient(base_url="http://x")
    with patch.object(c, "_direct_inference", return_value={"direct": True}) as d:
        out = c.generate_image(prompt="p", use_direct=True)
        assert out["direct"] is True
        d.assert_called_once()


def test_get_nearest_worker_uses_cache_within_ttl() -> None:
    c = InferenceClient(base_url="http://x")

    resp = httpx.Response(
        200,
        request=httpx.Request("GET", "http://x/api/v1/jobs/direct/nearest"),
        json={"direct_url": "http://w", "region": "asia-east"},
    )

    with patch.object(c, "_request_with_fallback", return_value=resp), patch("inference_client.time.time", side_effect=[1000.0, 1000.0, 1001.0]):
        w1 = c._get_nearest_worker("llm")
        w2 = c._get_nearest_worker("llm")
        assert w1 == w2

