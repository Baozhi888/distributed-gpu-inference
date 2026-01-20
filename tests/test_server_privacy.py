import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = (REPO_ROOT / "server").resolve()
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))


from app.services.privacy import DataAnonymizer, DataEncryptor  # noqa: E402


def test_data_anonymizer_preserve_email_domain() -> None:
    a = DataAnonymizer(salt="s")
    out = a.anonymize_string("user@example.com", preserve_format=True)
    assert out.endswith("@example.com")
    assert "***@" in out


def test_data_anonymizer_masks_digits() -> None:
    a = DataAnonymizer(salt="s")
    assert a.anonymize_string("1234", preserve_format=True) == "****"
    masked = a.anonymize_string("1234567890", preserve_format=True)
    assert masked.startswith("12")
    assert masked.endswith("90")
    assert "*" in masked


def test_data_anonymizer_anonymize_ip() -> None:
    a = DataAnonymizer(salt="s")
    assert a.anonymize_ip("1.2.3.4") == "1.2.xxx.xxx"
    assert a.anonymize_ip("2001:db8:abcd:0012::1").startswith("2001:db8::")


def test_data_anonymizer_removes_pii_in_preview() -> None:
    a = DataAnonymizer(salt="s")
    content = "contact me at test@example.com and call 13800138000"
    out = a.anonymize_content(content, max_preview=200)
    assert "[EMAIL]" in out
    assert "[PHONE_CN]" in out


def test_data_anonymizer_dict_redaction_recursive() -> None:
    a = DataAnonymizer(salt="s")
    data = {
        "prompt": "secret text",
        "nested": {"token": "abc", "keep": 1},
        "items": [{"password": "p"}],
    }
    out = a.anonymize_dict(data)
    assert out["prompt"].startswith("[")
    assert out["nested"]["token"] == "[REDACTED]" or out["nested"]["token"].startswith("[")
    assert out["nested"]["keep"] == 1
    assert out["items"][0]["password"] == "[REDACTED]" or out["items"][0]["password"].startswith("[")


def test_data_encryptor_roundtrip_and_wrong_key() -> None:
    enc1 = DataEncryptor(encryption_key="k1")
    ciphertext = enc1.encrypt("hello")
    assert enc1.decrypt(ciphertext) == "hello"

    enc2 = DataEncryptor(encryption_key="k2")
    assert enc2.decrypt(ciphertext) == "[DECRYPTION_FAILED]"

