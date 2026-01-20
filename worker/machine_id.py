"""
机器码/设备指纹生成器
生成唯一的硬件指纹用于识别Worker节点
"""
import hashlib
import platform
import uuid
import os
import json
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MachineFingerprint:
    """机器指纹生成器"""

    FINGERPRINT_FILE = ".gpu_worker_fingerprint"

    @classmethod
    def generate(cls) -> Dict[str, Any]:
        """
        生成机器指纹
        包含硬件信息的哈希值，用于唯一标识设备
        """
        fingerprint_data = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "mac_address": cls._get_mac_address(),
            "machine_id": cls._get_machine_id(),
        }

        # 添加GPU信息
        gpu_info = cls._get_gpu_info()
        if gpu_info:
            fingerprint_data["gpu"] = gpu_info

        # 生成指纹哈希
        fingerprint_string = json.dumps(fingerprint_data, sort_keys=True)
        fingerprint_hash = hashlib.sha256(fingerprint_string.encode()).hexdigest()

        return {
            "machine_id": fingerprint_hash[:32],  # 32字符的机器码
            "hardware_hash": fingerprint_hash,
            "details": fingerprint_data,
            "generated_at": cls._get_timestamp()
        }

    @classmethod
    def _get_mac_address(cls) -> str:
        """获取MAC地址"""
        try:
            mac = uuid.getnode()
            return ':'.join(('%012X' % mac)[i:i+2] for i in range(0, 12, 2))
        except Exception:
            return "unknown"

    @classmethod
    def _get_machine_id(cls) -> str:
        """获取系统机器ID"""
        # Linux
        if os.path.exists('/etc/machine-id'):
            with open('/etc/machine-id', 'r') as f:
                return f.read().strip()

        # macOS
        if platform.system() == 'Darwin':
            try:
                import subprocess
                result = subprocess.run(
                    ['ioreg', '-rd1', '-c', 'IOPlatformExpertDevice'],
                    capture_output=True, text=True
                )
                for line in result.stdout.split('\n'):
                    if 'IOPlatformUUID' in line:
                        return line.split('"')[-2]
            except Exception:
                pass

        # Windows
        if platform.system() == 'Windows':
            try:
                import subprocess
                result = subprocess.run(
                    ['wmic', 'csproduct', 'get', 'UUID'],
                    capture_output=True, text=True
                )
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    return lines[1].strip()
            except Exception:
                pass

        # 回退：使用MAC地址
        return str(uuid.getnode())

    @classmethod
    def _get_gpu_info(cls) -> Optional[Dict[str, Any]]:
        """获取GPU信息"""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "count": torch.cuda.device_count(),
                    "name": torch.cuda.get_device_name(0),
                    "uuid": cls._get_gpu_uuid()
                }
        except ImportError:
            pass
        return None

    @classmethod
    def _get_gpu_uuid(cls) -> Optional[str]:
        """获取GPU UUID"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except Exception:
            pass
        return None

    @classmethod
    def _get_timestamp(cls) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.utcnow().isoformat() + 'Z'

    @classmethod
    def get_or_create(cls, storage_path: str = None) -> Dict[str, Any]:
        """
        获取或创建机器指纹
        首次运行时生成并保存，后续读取已保存的指纹
        """
        if storage_path is None:
            storage_path = Path.home() / cls.FINGERPRINT_FILE

        storage_path = Path(storage_path)

        # 尝试读取已存在的指纹
        if storage_path.exists():
            try:
                with open(storage_path, 'r') as f:
                    saved = json.load(f)

                # 验证指纹仍然有效（硬件未更换）
                current = cls.generate()
                if saved.get('hardware_hash') == current['hardware_hash']:
                    return saved

                logger.warning("Hardware changed, regenerating fingerprint")
            except Exception as e:
                logger.warning(f"Failed to read fingerprint: {e}")

        # 生成新指纹
        fingerprint = cls.generate()

        # 保存指纹
        try:
            with open(storage_path, 'w') as f:
                json.dump(fingerprint, f, indent=2)
            logger.info(f"Machine fingerprint saved: {fingerprint['machine_id']}")
        except Exception as e:
            logger.warning(f"Failed to save fingerprint: {e}")

        return fingerprint

    @classmethod
    def get_machine_id(cls) -> str:
        """获取机器码（简化接口）"""
        fingerprint = cls.get_or_create()
        return fingerprint['machine_id']


def get_machine_id() -> str:
    """获取当前机器的唯一标识码"""
    return MachineFingerprint.get_machine_id()


def get_full_fingerprint() -> Dict[str, Any]:
    """获取完整的机器指纹信息"""
    return MachineFingerprint.get_or_create()


if __name__ == "__main__":
    # 测试
    fingerprint = get_full_fingerprint()
    print(f"Machine ID: {fingerprint['machine_id']}")
    print(f"Hardware Hash: {fingerprint['hardware_hash']}")
    print(f"Details: {json.dumps(fingerprint['details'], indent=2)}")
