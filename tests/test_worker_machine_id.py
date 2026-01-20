import json
import sys
from pathlib import Path
from unittest.mock import patch

from tests._helpers_fs import make_temp_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = (REPO_ROOT / "worker").resolve()
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))


from machine_id import MachineFingerprint  # noqa: E402


def test_get_or_create_creates_and_reuses_when_hash_matches() -> None:
    tmp_dir = make_temp_dir("machine_id_reuse")
    fp_path = tmp_dir / "fp.json"

    first = {
        "machine_id": "m1",
        "hardware_hash": "h1",
        "details": {},
        "generated_at": "t1",
    }
    with patch.object(MachineFingerprint, "generate", return_value=first):
        created = MachineFingerprint.get_or_create(storage_path=str(fp_path))
        assert created["machine_id"] == "m1"

    saved = json.loads(fp_path.read_text(encoding="utf-8"))
    assert saved["hardware_hash"] == "h1"

    with patch.object(MachineFingerprint, "generate", return_value=first):
        reused = MachineFingerprint.get_or_create(storage_path=str(fp_path))
        assert reused["machine_id"] == "m1"


def test_get_or_create_regenerates_when_hash_changes() -> None:
    tmp_dir = make_temp_dir("machine_id_regen")
    fp_path = tmp_dir / "fp.json"
    old = {"machine_id": "m1", "hardware_hash": "h1", "details": {}, "generated_at": "t1"}
    fp_path.write_text(json.dumps(old), encoding="utf-8")

    new = {"machine_id": "m2", "hardware_hash": "h2", "details": {}, "generated_at": "t2"}
    with patch.object(MachineFingerprint, "generate", return_value=new):
        out = MachineFingerprint.get_or_create(storage_path=str(fp_path))
        assert out["machine_id"] == "m2"


def test_generate_is_deterministic_given_fixed_inputs() -> None:
    with (
        patch("machine_id.platform.system", return_value="Linux"),
        patch("machine_id.platform.release", return_value="r"),
        patch("machine_id.platform.version", return_value="v"),
        patch("machine_id.platform.machine", return_value="x"),
        patch("machine_id.platform.processor", return_value="p"),
        patch("machine_id.platform.node", return_value="n"),
        patch("machine_id.uuid.getnode", return_value=0xAABBCCDDEEFF),
        patch.object(MachineFingerprint, "_get_machine_id", return_value="mid"),
        patch.object(MachineFingerprint, "_get_gpu_info", return_value=None),
        patch.object(MachineFingerprint, "_get_timestamp", return_value="t"),
    ):
        fp1 = MachineFingerprint.generate()
        fp2 = MachineFingerprint.generate()
        assert fp1["hardware_hash"] == fp2["hardware_hash"]
        assert len(fp1["machine_id"]) == 32


def test_get_machine_id_windows_branch_parses_wmic_output() -> None:
    class _Result:
        returncode = 0
        stdout = "UUID\nABCDEF\n"

    with (
        patch("machine_id.os.path.exists", return_value=False),
        patch("machine_id.platform.system", return_value="Windows"),
        patch("machine_id.subprocess.run", return_value=_Result()),
    ):
        assert MachineFingerprint._get_machine_id() == "ABCDEF"
