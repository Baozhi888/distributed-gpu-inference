"""
Tensor 序列化/反序列化工具

用于跨 Worker 高效传输 Tensor 数据
"""
import io
import struct
from typing import Tuple, Optional, Dict, Any
import numpy as np

# 可选的 torch 依赖
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# 数据类型映射
DTYPE_TO_ID = {
    "float16": 0,
    "float32": 1,
    "bfloat16": 2,
    "int8": 3,
    "int32": 4,
    "int64": 5,
}

ID_TO_DTYPE = {v: k for k, v in DTYPE_TO_ID.items()}

# NumPy dtype 映射
DTYPE_TO_NUMPY = {
    "float16": np.float16,
    "float32": np.float32,
    "int8": np.int8,
    "int32": np.int32,
    "int64": np.int64,
}

if HAS_TORCH:
    DTYPE_TO_TORCH = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    TORCH_TO_DTYPE = {v: k for k, v in DTYPE_TO_TORCH.items()}


class TensorSerializer:
    """高效的 Tensor 序列化器"""

    @staticmethod
    def serialize(
        data: Any,
        compression: str = "none"
    ) -> Tuple[bytes, Tuple[int, ...], str]:
        """
        序列化 tensor 数据

        Args:
            data: torch.Tensor 或 numpy.ndarray
            compression: 压缩方式 ("none", "lz4", "zstd")

        Returns:
            (data_bytes, shape, dtype_str)
        """
        # 转换为 numpy
        if HAS_TORCH and isinstance(data, torch.Tensor):
            # 处理 bfloat16
            if data.dtype == torch.bfloat16:
                # bfloat16 需要特殊处理，先转为 float16
                array = data.to(torch.float16).cpu().numpy()
                dtype_str = "bfloat16"  # 保留原始类型信息
            else:
                array = data.cpu().numpy()
                dtype_str = TORCH_TO_DTYPE.get(data.dtype, "float32")
        elif isinstance(data, np.ndarray):
            array = data
            dtype_str = str(array.dtype)
        else:
            raise TypeError(f"Unsupported type: {type(data)}")

        shape = array.shape
        data_bytes = array.tobytes()

        # 可选压缩
        if compression == "lz4":
            try:
                import lz4.frame
                data_bytes = lz4.frame.compress(data_bytes)
            except ImportError:
                pass  # fallback to uncompressed
        elif compression == "zstd":
            try:
                import zstandard
                cctx = zstandard.ZstdCompressor()
                data_bytes = cctx.compress(data_bytes)
            except ImportError:
                pass

        return data_bytes, shape, dtype_str

    @staticmethod
    def deserialize(
        data_bytes: bytes,
        shape: Tuple[int, ...],
        dtype_str: str,
        compression: str = "none",
        device: str = "cpu"
    ) -> Any:
        """
        反序列化 tensor 数据

        Args:
            data_bytes: 序列化的字节数据
            shape: tensor 形状
            dtype_str: 数据类型字符串
            compression: 压缩方式
            device: 目标设备 ("cpu", "cuda", "cuda:0", etc.)

        Returns:
            torch.Tensor 或 numpy.ndarray
        """
        # 解压缩
        if compression == "lz4":
            try:
                import lz4.frame
                data_bytes = lz4.frame.decompress(data_bytes)
            except ImportError:
                pass
        elif compression == "zstd":
            try:
                import zstandard
                dctx = zstandard.ZstdDecompressor()
                data_bytes = dctx.decompress(data_bytes)
            except ImportError:
                pass

        # 转换 numpy dtype
        if dtype_str == "bfloat16":
            np_dtype = np.float16  # 先用 float16 读取
        else:
            np_dtype = DTYPE_TO_NUMPY.get(dtype_str, np.float32)

        # 反序列化为 numpy
        array = np.frombuffer(data_bytes, dtype=np_dtype).reshape(shape)

        # 如果有 torch，返回 tensor
        if HAS_TORCH and device != "numpy":
            tensor = torch.from_numpy(array.copy())
            if dtype_str == "bfloat16":
                tensor = tensor.to(torch.bfloat16)
            if device.startswith("cuda") and torch.cuda.is_available():
                tensor = tensor.to(device)
            return tensor

        return array


def serialize_tensor(
    data: Any,
    compression: str = "none"
) -> Dict[str, Any]:
    """
    序列化 tensor 为字典格式（便于 JSON/msgpack 传输）

    Returns:
        {
            "data": bytes (base64 编码后的字符串),
            "shape": list,
            "dtype": str,
            "compression": str
        }
    """
    import base64

    data_bytes, shape, dtype_str = TensorSerializer.serialize(data, compression)

    return {
        "data": base64.b64encode(data_bytes).decode("ascii"),
        "shape": list(shape),
        "dtype": dtype_str,
        "compression": compression,
    }


def deserialize_tensor(
    serialized: Dict[str, Any],
    device: str = "cpu"
) -> Any:
    """
    从字典格式反序列化 tensor
    """
    import base64

    data_bytes = base64.b64decode(serialized["data"])
    shape = tuple(serialized["shape"])
    dtype_str = serialized["dtype"]
    compression = serialized.get("compression", "none")

    return TensorSerializer.deserialize(
        data_bytes, shape, dtype_str, compression, device
    )


class StreamingTensorBuffer:
    """流式 Tensor 缓冲区 - 用于分块传输大 tensor"""

    def __init__(self, chunk_size: int = 1024 * 1024):  # 1MB chunks
        self.chunk_size = chunk_size
        self.buffer = io.BytesIO()
        self.metadata: Optional[Dict[str, Any]] = None

    def write_header(self, shape: Tuple[int, ...], dtype_str: str) -> bytes:
        """写入头部信息"""
        # 格式: [ndim(4B)] [shape...] [dtype_id(1B)]
        header = struct.pack("I", len(shape))
        for dim in shape:
            header += struct.pack("Q", dim)  # 8 bytes per dim
        header += struct.pack("B", DTYPE_TO_ID.get(dtype_str, 0))

        self.metadata = {"shape": shape, "dtype": dtype_str}
        return header

    def read_header(self, header_bytes: bytes) -> Dict[str, Any]:
        """读取头部信息"""
        offset = 0
        ndim = struct.unpack_from("I", header_bytes, offset)[0]
        offset += 4

        shape = []
        for _ in range(ndim):
            dim = struct.unpack_from("Q", header_bytes, offset)[0]
            shape.append(dim)
            offset += 8

        dtype_id = struct.unpack_from("B", header_bytes, offset)[0]
        dtype_str = ID_TO_DTYPE.get(dtype_id, "float32")

        return {"shape": tuple(shape), "dtype": dtype_str}

    def iter_chunks(self, data_bytes: bytes):
        """分块迭代器"""
        for i in range(0, len(data_bytes), self.chunk_size):
            yield data_bytes[i:i + self.chunk_size]

    def write_chunk(self, chunk: bytes) -> None:
        """写入一个块"""
        self.buffer.write(chunk)

    def finalize(self, device: str = "cpu") -> Any:
        """完成接收，返回 tensor"""
        if self.metadata is None:
            raise ValueError("Header not received")

        data_bytes = self.buffer.getvalue()
        return TensorSerializer.deserialize(
            data_bytes,
            self.metadata["shape"],
            self.metadata["dtype"],
            device=device
        )
