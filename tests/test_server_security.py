import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = (REPO_ROOT / "server").resolve()
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))


from app.services.security import TokenManager, RequestSigner, SecuritySettings, SecurityService  # noqa: E402


def test_token_hash_verify_roundtrip() -> None:
    token = "test-token"
    token_hash = TokenManager.hash_token(token)
    assert TokenManager.verify_token_hash(token, token_hash) is True
    assert TokenManager.verify_token_hash("wrong", token_hash) is False


def test_request_signer_verify_ok_and_invalid() -> None:
    secret = "secret"
    ts = int(datetime.utcnow().timestamp())
    sig = RequestSigner.sign_request("POST", "/x", '{"a":1}', ts, secret)

    ok, err = RequestSigner.verify_signature("POST", "/x", '{"a":1}', ts, sig, secret)
    assert ok is True
    assert err == ""

    ok, err = RequestSigner.verify_signature("POST", "/x", '{"a":2}', ts, sig, secret)
    assert ok is False
    assert err == "invalid_signature"


def test_request_signer_expired() -> None:
    secret = "secret"
    now = int(datetime.utcnow().timestamp())
    expired_ts = now - (SecuritySettings.SIGNATURE_VALIDITY_SECONDS + 1)
    sig = RequestSigner.sign_request("GET", "/x", None, expired_ts, secret)
    ok, err = RequestSigner.verify_signature("GET", "/x", None, expired_ts, sig, secret)
    assert ok is False
    assert err == "signature_expired"


def test_security_service_should_refresh_token_threshold() -> None:
    db = AsyncMock()
    svc = SecurityService(db)

    class W:
        token_expires_at = None

    w = W()
    assert svc.should_refresh_token(w) is False

    w.token_expires_at = datetime.utcnow()
    assert svc.should_refresh_token(w) is True


def test_security_service_verify_request_signature_requires_secret() -> None:
    db = AsyncMock()
    svc = SecurityService(db)

    class W:
        signing_secret = None

    ok, err = asyncio_run(svc.verify_request_signature(W(), "GET", "/x", None, int(datetime.utcnow().timestamp()), "sig"))
    assert ok is False
    assert err == "no_signing_secret"


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


def test_security_service_verify_worker_auth_branches() -> None:
    import asyncio
    from datetime import timedelta

    class _Result:
        def __init__(self, worker):
            self._worker = worker

        def scalar_one_or_none(self):
            return self._worker

    class _Worker:
        def __init__(self):
            self.id = "00000000-0000-0000-0000-000000000000"
            self.locked_until = None
            self.auth_token_hash = TokenManager.hash_token("good")
            self.refresh_token_hash = None
            self.token_expires_at = None
            self.failed_auth_attempts = 1
            self.last_failed_auth = None

    async def run_cases():
        db = AsyncMock()
        svc = SecurityService(db)

        # worker_not_found
        db.execute.return_value = _Result(None)
        w, err = await svc.verify_worker_auth("00000000-0000-0000-0000-000000000000", "x")
        assert w is None and err == "worker_not_found"

        # account_locked
        worker = _Worker()
        worker.locked_until = datetime.utcnow() + timedelta(minutes=1)
        db.execute.return_value = _Result(worker)
        w, err = await svc.verify_worker_auth(worker.id, "good")
        assert w is None and err == "account_locked"

        # invalid_token
        worker = _Worker()
        db.execute.return_value = _Result(worker)
        w, err = await svc.verify_worker_auth(worker.id, "bad")
        assert w is None and err == "invalid_token"

        # token_expired
        worker = _Worker()
        worker.token_expires_at = datetime.utcnow() - timedelta(seconds=1)
        db.execute.return_value = _Result(worker)
        w, err = await svc.verify_worker_auth(worker.id, "good")
        assert w is None and err == "token_expired"

        # success resets failed attempts
        worker = _Worker()
        worker.failed_auth_attempts = 2
        db.execute.return_value = _Result(worker)
        w, err = await svc.verify_worker_auth(worker.id, "good")
        assert w is worker and err == ""
        assert worker.failed_auth_attempts == 0

    asyncio.run(run_cases())
