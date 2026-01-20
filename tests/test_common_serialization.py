import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from common.serialization import (  # noqa: E402
    TensorSerializer,
    serialize_tensor,
    deserialize_tensor,
    StreamingTensorBuffer,
)


def test_tensor_serializer_roundtrip_numpy() -> None:
    data = (np.arange(12, dtype=np.int32) * 2).reshape(3, 4)
    data_bytes, shape, dtype_str = TensorSerializer.serialize(data, compression="none")
    restored = TensorSerializer.deserialize(data_bytes, shape, dtype_str, compression="none", device="numpy")
    assert isinstance(restored, np.ndarray)
    assert restored.shape == (3, 4)
    assert restored.dtype == np.int32
    assert np.array_equal(restored, data)


def test_serialize_tensor_dict_roundtrip() -> None:
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    payload = serialize_tensor(data, compression="none")
    restored = deserialize_tensor(payload, device="numpy")
    assert np.array_equal(restored, data)


def test_streaming_tensor_buffer_reassembles_payload() -> None:
    data = np.arange(100, dtype=np.int64)
    data_bytes, shape, dtype_str = TensorSerializer.serialize(data, compression="none")

    buf = StreamingTensorBuffer(chunk_size=13)
    header = buf.write_header(shape=shape, dtype_str=dtype_str)
    parsed = buf.read_header(header)
    assert parsed["shape"] == shape
    assert parsed["dtype"] == dtype_str

    for chunk in buf.iter_chunks(data_bytes):
        buf.write_chunk(chunk)

    restored = buf.finalize(device="numpy")
    assert np.array_equal(restored, data)


def test_streaming_tensor_buffer_finalize_requires_header() -> None:
    buf = StreamingTensorBuffer()
    try:
        buf.finalize(device="numpy")
    except ValueError as e:
        assert "Header not received" in str(e)
    else:
        raise AssertionError("Expected ValueError")


def test_tensor_serializer_rejects_unsupported_type() -> None:
    try:
        TensorSerializer.serialize({"not": "a tensor"})
    except TypeError as e:
        assert "Unsupported type" in str(e)
    else:
        raise AssertionError("Expected TypeError")


def test_tensor_serializer_compression_fallback_does_not_crash() -> None:
    data = np.arange(10, dtype=np.float32)
    TensorSerializer.serialize(data, compression="lz4")
    TensorSerializer.serialize(data, compression="zstd")


def test_tensor_serializer_torch_roundtrip_when_available() -> None:
    try:
        import torch
    except Exception:
        return

    t = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    data_bytes, shape, dtype_str = TensorSerializer.serialize(t, compression="none")
    restored = TensorSerializer.deserialize(data_bytes, shape, dtype_str, compression="none", device="cpu")
    assert isinstance(restored, torch.Tensor)
    assert restored.dtype == torch.float16
    assert restored.shape == (2, 3)
    assert torch.equal(restored, t)


def test_tensor_serializer_bfloat16_roundtrip_when_available() -> None:
    try:
        import torch
    except Exception:
        return

    if not hasattr(torch, "bfloat16"):
        return

    t = torch.ones((1, 4), dtype=torch.bfloat16)
    data_bytes, shape, dtype_str = TensorSerializer.serialize(t, compression="none")
    assert dtype_str == "bfloat16"
    restored = TensorSerializer.deserialize(data_bytes, shape, dtype_str, compression="none", device="cpu")
    assert restored.dtype == torch.bfloat16
