"""
安全服务 - Token管理、请求签名、审计日志
实现安全最佳实践
"""
from datetime import datetime, timedelta
from typing import Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
import hashlib
import hmac
import secrets
import base64
import logging
import json
from pydantic import BaseModel

from app.models.models import Worker

logger = logging.getLogger(__name__)


# ==================== 安全配置 ====================

class SecuritySettings:
    """安全配置常量"""
    # Token配置
    TOKEN_LENGTH = 32
    TOKEN_VALIDITY_HOURS = 24
    TOKEN_REFRESH_BEFORE_HOURS = 4  # 过期前4小时可刷新

    # 签名配置
    SIGNATURE_ALGORITHM = "sha256"
    SIGNATURE_VALIDITY_SECONDS = 300  # 签名有效期5分钟

    # 速率限制
    MAX_FAILED_AUTH_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15


# ==================== Token管理 ====================

class TokenManager:
    """Token管理器"""

    @staticmethod
    def generate_token() -> str:
        """生成安全的随机Token"""
        return secrets.token_urlsafe(SecuritySettings.TOKEN_LENGTH)

    @staticmethod
    def hash_token(token: str) -> str:
        """对Token进行安全哈希"""
        # 使用SHA-256 + 盐值
        salt = "distributed-gpu-inference-v1"
        return hashlib.sha256(f"{salt}:{token}".encode()).hexdigest()

    @staticmethod
    def generate_refresh_token() -> str:
        """生成刷新Token"""
        return secrets.token_urlsafe(SecuritySettings.TOKEN_LENGTH * 2)

    @staticmethod
    def verify_token_hash(token: str, token_hash: str) -> bool:
        """验证Token哈希（常量时间比较防止时序攻击）"""
        computed_hash = TokenManager.hash_token(token)
        return hmac.compare_digest(computed_hash, token_hash)


class TokenInfo(BaseModel):
    """Token信息"""
    access_token: str
    refresh_token: str
    expires_at: datetime
    token_type: str = "Bearer"


# ==================== 请求签名 ====================

class RequestSigner:
    """请求签名器 - 防止请求篡改和重放攻击"""

    @staticmethod
    def sign_request(
        method: str,
        path: str,
        body: Optional[str],
        timestamp: int,
        secret: str
    ) -> str:
        """
        生成请求签名

        签名内容: METHOD + PATH + BODY_HASH + TIMESTAMP
        """
        body_hash = hashlib.sha256((body or "").encode()).hexdigest()

        sign_content = f"{method.upper()}:{path}:{body_hash}:{timestamp}"

        signature = hmac.new(
            secret.encode(),
            sign_content.encode(),
            hashlib.sha256
        ).hexdigest()

        return signature

    @staticmethod
    def verify_signature(
        method: str,
        path: str,
        body: Optional[str],
        timestamp: int,
        signature: str,
        secret: str
    ) -> Tuple[bool, str]:
        """
        验证请求签名

        Returns:
            (is_valid, error_message)
        """
        # 检查时间戳是否在有效期内
        current_time = int(datetime.utcnow().timestamp())
        time_diff = abs(current_time - timestamp)

        if time_diff > SecuritySettings.SIGNATURE_VALIDITY_SECONDS:
            return False, "signature_expired"

        # 计算预期签名
        expected_signature = RequestSigner.sign_request(
            method, path, body, timestamp, secret
        )

        # 常量时间比较
        if not hmac.compare_digest(signature, expected_signature):
            return False, "invalid_signature"

        return True, ""


# ==================== 安全服务 ====================

class SecurityService:
    """安全服务"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_worker_credentials(
        self,
        worker: Worker
    ) -> TokenInfo:
        """为Worker创建认证凭据"""
        access_token = TokenManager.generate_token()
        refresh_token = TokenManager.generate_refresh_token()

        expires_at = datetime.utcnow() + timedelta(
            hours=SecuritySettings.TOKEN_VALIDITY_HOURS
        )

        # 存储哈希后的Token
        worker.auth_token_hash = TokenManager.hash_token(access_token)
        worker.refresh_token_hash = TokenManager.hash_token(refresh_token)
        worker.token_expires_at = expires_at
        worker.signing_secret = secrets.token_urlsafe(32)

        await self.db.commit()

        logger.info(f"Created credentials for worker {worker.id}")

        return TokenInfo(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at
        )

    async def refresh_worker_token(
        self,
        worker: Worker,
        refresh_token: str
    ) -> Optional[TokenInfo]:
        """刷新Worker Token"""
        # 验证刷新Token
        if not worker.refresh_token_hash:
            logger.warning(f"Worker {worker.id} has no refresh token")
            return None

        if not TokenManager.verify_token_hash(refresh_token, worker.refresh_token_hash):
            logger.warning(f"Invalid refresh token for worker {worker.id}")
            await self._record_failed_auth(worker)
            return None

        # 生成新Token
        return await self.create_worker_credentials(worker)

    async def verify_worker_auth(
        self,
        worker_id: str,
        token: str,
        check_expiry: bool = True
    ) -> Tuple[Optional[Worker], str]:
        """
        验证Worker认证

        Returns:
            (worker, error_message)
        """
        from uuid import UUID

        result = await self.db.execute(
            select(Worker).where(Worker.id == UUID(worker_id))
        )
        worker = result.scalar_one_or_none()

        if not worker:
            return None, "worker_not_found"

        # 检查是否被锁定
        if worker.locked_until and worker.locked_until > datetime.utcnow():
            return None, "account_locked"

        # 验证Token
        if not TokenManager.verify_token_hash(token, worker.auth_token_hash):
            await self._record_failed_auth(worker)
            return None, "invalid_token"

        # 检查过期
        if check_expiry and worker.token_expires_at:
            if worker.token_expires_at < datetime.utcnow():
                return None, "token_expired"

        # 重置失败计数
        if worker.failed_auth_attempts > 0:
            worker.failed_auth_attempts = 0
            await self.db.commit()

        return worker, ""

    async def verify_request_signature(
        self,
        worker: Worker,
        method: str,
        path: str,
        body: Optional[str],
        timestamp: int,
        signature: str
    ) -> Tuple[bool, str]:
        """验证请求签名"""
        if not worker.signing_secret:
            return False, "no_signing_secret"

        return RequestSigner.verify_signature(
            method, path, body, timestamp, signature, worker.signing_secret
        )

    async def _record_failed_auth(self, worker: Worker):
        """记录失败的认证尝试"""
        worker.failed_auth_attempts = (worker.failed_auth_attempts or 0) + 1
        worker.last_failed_auth = datetime.utcnow()

        # 检查是否需要锁定
        if worker.failed_auth_attempts >= SecuritySettings.MAX_FAILED_AUTH_ATTEMPTS:
            worker.locked_until = datetime.utcnow() + timedelta(
                minutes=SecuritySettings.LOCKOUT_DURATION_MINUTES
            )
            logger.warning(
                f"Worker {worker.id} locked due to {worker.failed_auth_attempts} "
                f"failed auth attempts"
            )

        await self.db.commit()

    def should_refresh_token(self, worker: Worker) -> bool:
        """检查是否应该刷新Token"""
        if not worker.token_expires_at:
            return False

        refresh_threshold = datetime.utcnow() + timedelta(
            hours=SecuritySettings.TOKEN_REFRESH_BEFORE_HOURS
        )

        return worker.token_expires_at < refresh_threshold


# ==================== 审计日志 ====================

class AuditLogger:
    """审计日志记录器"""

    @staticmethod
    def log_auth_event(
        event_type: str,
        worker_id: str,
        ip_address: Optional[str],
        success: bool,
        details: Optional[dict] = None
    ):
        """记录认证事件"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "worker_id": worker_id,
            "ip_address": ip_address,
            "success": success,
            "details": details or {}
        }

        if success:
            logger.info(f"AUTH_EVENT: {json.dumps(log_data)}")
        else:
            logger.warning(f"AUTH_EVENT: {json.dumps(log_data)}")

    @staticmethod
    def log_security_event(
        event_type: str,
        severity: str,
        message: str,
        details: Optional[dict] = None
    ):
        """记录安全事件"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "severity": severity,
            "message": message,
            "details": details or {}
        }

        if severity == "critical":
            logger.critical(f"SECURITY_EVENT: {json.dumps(log_data)}")
        elif severity == "high":
            logger.error(f"SECURITY_EVENT: {json.dumps(log_data)}")
        elif severity == "medium":
            logger.warning(f"SECURITY_EVENT: {json.dumps(log_data)}")
        else:
            logger.info(f"SECURITY_EVENT: {json.dumps(log_data)}")
