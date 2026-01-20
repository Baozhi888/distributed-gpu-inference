"""
企业隐私保护服务
实现：数据匿名化、加密存储、数据保留策略、访问控制、合规审计
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import and_, delete
import hashlib
import hmac
import secrets
import base64
import json
import re
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from app.models.usage import UsageRecord, Enterprise, Bill
from app.models.models import Job

logger = logging.getLogger(__name__)


# ==================== 配置常量 ====================

class PrivacyConfig:
    """隐私保护配置"""
    # 默认数据保留天数
    DEFAULT_RETENTION_DAYS = 30
    MIN_RETENTION_DAYS = 7
    MAX_RETENTION_DAYS = 365

    # 匿名化配置
    ANONYMIZE_IP_MASK = True  # IP地址掩码
    ANONYMIZE_CONTENT = True  # 内容匿名化
    HASH_SALT_LENGTH = 16

    # 加密配置
    ENCRYPTION_KEY_ENV = "PRIVACY_ENCRYPTION_KEY"
    KEY_DERIVATION_ITERATIONS = 100000

    # 敏感字段列表
    SENSITIVE_FIELDS = [
        'prompt', 'messages', 'input', 'output', 'response',
        'api_key', 'token', 'password', 'secret',
        'email', 'phone', 'address', 'name'
    ]

    # PII (个人身份信息) 正则模式
    PII_PATTERNS = {
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'phone_cn': r'1[3-9]\d{9}',
        'phone_intl': r'\+\d{1,3}[-.\s]?\d{1,14}',
        'id_card_cn': r'\d{17}[\dXx]',
        'credit_card': r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',
        'ip_address': r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
    }


# ==================== 数据匿名化服务 ====================

class DataAnonymizer:
    """数据匿名化处理器"""

    def __init__(self, salt: Optional[str] = None):
        self.salt = salt or secrets.token_hex(PrivacyConfig.HASH_SALT_LENGTH)

    def anonymize_string(self, value: str, preserve_format: bool = False) -> str:
        """
        匿名化字符串

        Args:
            value: 原始字符串
            preserve_format: 是否保留格式（如邮箱保留@后的域名）
        """
        if not value:
            return value

        if preserve_format:
            # 检测格式并保留部分信息
            if '@' in value:  # 邮箱
                local, domain = value.rsplit('@', 1)
                hashed_local = self._hash_value(local)[:8]
                return f"{hashed_local}***@{domain}"
            elif re.match(r'^\d+$', value):  # 纯数字
                return self._mask_digits(value)

        # 完全哈希
        return self._hash_value(value)

    def anonymize_ip(self, ip: str) -> str:
        """匿名化IP地址（保留前两段）"""
        if not ip:
            return ip

        parts = ip.split('.')
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.xxx.xxx"

        # IPv6处理
        if ':' in ip:
            parts = ip.split(':')
            if len(parts) >= 4:
                return ':'.join(parts[:2]) + '::xxxx'

        return self._hash_value(ip)[:16]

    def anonymize_content(self, content: str, max_preview: int = 50) -> str:
        """
        匿名化内容（用于日志记录）
        保留长度信息和开头预览
        """
        if not content:
            return content

        length = len(content)
        preview = content[:max_preview] if len(content) > max_preview else content
        # 移除可能的PII
        preview = self._remove_pii(preview)

        return f"[{length} chars] {preview}..."

    def anonymize_dict(self, data: Dict, fields_to_anonymize: List[str] = None) -> Dict:
        """
        匿名化字典中的敏感字段

        Args:
            data: 原始字典
            fields_to_anonymize: 需要匿名化的字段列表
        """
        if not data:
            return data

        fields = fields_to_anonymize or PrivacyConfig.SENSITIVE_FIELDS
        result = {}

        for key, value in data.items():
            if key.lower() in [f.lower() for f in fields]:
                if isinstance(value, str):
                    result[key] = self.anonymize_content(value, max_preview=20)
                elif isinstance(value, list):
                    result[key] = f"[{len(value)} items]"
                elif isinstance(value, dict):
                    result[key] = "{...}"
                else:
                    result[key] = "[REDACTED]"
            elif isinstance(value, dict):
                result[key] = self.anonymize_dict(value, fields)
            elif isinstance(value, list):
                result[key] = [
                    self.anonymize_dict(item, fields) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value

        return result

    def create_pseudonym(self, identifier: str, context: str = "") -> str:
        """
        创建假名（可重复生成相同假名用于关联分析）

        Args:
            identifier: 原始标识符
            context: 上下文（如企业ID），用于隔离不同企业的假名
        """
        combined = f"{context}:{identifier}:{self.salt}"
        return f"pseudo_{hashlib.sha256(combined.encode()).hexdigest()[:16]}"

    def _hash_value(self, value: str) -> str:
        """安全哈希值"""
        combined = f"{self.salt}:{value}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _mask_digits(self, value: str) -> str:
        """掩码数字（保留首尾）"""
        if len(value) <= 4:
            return '*' * len(value)
        return value[:2] + '*' * (len(value) - 4) + value[-2:]

    def _remove_pii(self, text: str) -> str:
        """移除文本中的PII"""
        result = text
        for pattern_name, pattern in PrivacyConfig.PII_PATTERNS.items():
            result = re.sub(pattern, f'[{pattern_name.upper()}]', result)
        return result


# ==================== 敏感数据加密服务 ====================

class DataEncryptor:
    """敏感数据加密器"""

    def __init__(self, encryption_key: Optional[str] = None):
        """
        初始化加密器

        Args:
            encryption_key: 加密密钥（如果不提供则生成新密钥）
        """
        if encryption_key:
            # 从密钥派生Fernet密钥
            self.fernet = self._derive_fernet_key(encryption_key)
        else:
            # 生成新密钥
            self.key = Fernet.generate_key()
            self.fernet = Fernet(self.key)

    def _derive_fernet_key(self, password: str) -> Fernet:
        """从密码派生Fernet密钥"""
        # 使用固定盐值（实际使用中应该存储随机盐）
        salt = b'distributed-gpu-privacy-salt-v1'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=PrivacyConfig.KEY_DERIVATION_ITERATIONS,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return Fernet(key)

    def encrypt(self, plaintext: str) -> str:
        """加密字符串"""
        if not plaintext:
            return plaintext
        return self.fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """解密字符串"""
        if not ciphertext:
            return ciphertext
        try:
            return self.fernet.decrypt(ciphertext.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return "[DECRYPTION_FAILED]"

    def encrypt_dict(self, data: Dict, fields_to_encrypt: List[str]) -> Dict:
        """加密字典中的指定字段"""
        if not data:
            return data

        result = data.copy()
        for field in fields_to_encrypt:
            if field in result and isinstance(result[field], str):
                result[field] = self.encrypt(result[field])
                result[f"_{field}_encrypted"] = True

        return result

    def decrypt_dict(self, data: Dict) -> Dict:
        """解密字典中的加密字段"""
        if not data:
            return data

        result = data.copy()
        encrypted_markers = [k for k in result.keys() if k.endswith('_encrypted')]

        for marker in encrypted_markers:
            field = marker.replace('_encrypted', '').lstrip('_')
            if field in result:
                result[field] = self.decrypt(result[field])
                del result[marker]

        return result


# ==================== 数据保留策略服务 ====================

class DataRetentionService:
    """数据保留策略服务"""

    def __init__(self, db: Session):
        self.db = db

    def cleanup_expired_data(self, enterprise_id: Optional[UUID] = None) -> Dict[str, int]:
        """
        清理过期数据

        Args:
            enterprise_id: 指定企业ID，为None则清理所有企业

        Returns:
            清理统计信息
        """
        stats = {
            'usage_records_deleted': 0,
            'jobs_anonymized': 0,
            'enterprises_processed': 0
        }

        # 获取需要处理的企业
        query = self.db.query(Enterprise).filter(Enterprise.is_active == True)
        if enterprise_id:
            query = query.filter(Enterprise.id == enterprise_id)

        enterprises = query.all()

        for enterprise in enterprises:
            retention_days = enterprise.data_retention_days or PrivacyConfig.DEFAULT_RETENTION_DAYS
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

            # 删除过期使用记录
            deleted = self._delete_expired_usage_records(enterprise.id, cutoff_date)
            stats['usage_records_deleted'] += deleted

            # 匿名化过期任务数据
            anonymized = self._anonymize_expired_jobs(enterprise.id, cutoff_date)
            stats['jobs_anonymized'] += anonymized

            stats['enterprises_processed'] += 1

            logger.info(
                f"Data retention cleanup for enterprise {enterprise.id}: "
                f"deleted {deleted} records, anonymized {anonymized} jobs"
            )

        self.db.commit()
        return stats

    def _delete_expired_usage_records(self, enterprise_id: UUID, cutoff_date: datetime) -> int:
        """删除过期使用记录"""
        result = self.db.execute(
            delete(UsageRecord).where(
                and_(
                    UsageRecord.enterprise_id == enterprise_id,
                    UsageRecord.created_at < cutoff_date
                )
            )
        )
        return result.rowcount

    def _anonymize_expired_jobs(self, enterprise_id: UUID, cutoff_date: datetime) -> int:
        """匿名化过期任务的敏感数据"""
        # 获取关联的任务
        jobs = self.db.query(Job).join(
            UsageRecord, Job.id == UsageRecord.job_id
        ).filter(
            and_(
                UsageRecord.enterprise_id == enterprise_id,
                Job.created_at < cutoff_date,
                Job.params.isnot(None)  # 还没有被匿名化的
            )
        ).all()

        anonymizer = DataAnonymizer()
        count = 0

        for job in jobs:
            if job.params:
                job.params = anonymizer.anonymize_dict(job.params)
            if job.result:
                job.result = anonymizer.anonymize_dict(job.result)
            count += 1

        return count

    def get_retention_status(self, enterprise_id: UUID) -> Dict[str, Any]:
        """获取数据保留状态"""
        enterprise = self.db.query(Enterprise).filter(
            Enterprise.id == enterprise_id
        ).first()

        if not enterprise:
            return {}

        retention_days = enterprise.data_retention_days or PrivacyConfig.DEFAULT_RETENTION_DAYS
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

        # 统计各时间段的数据量
        total_records = self.db.query(UsageRecord).filter(
            UsageRecord.enterprise_id == enterprise_id
        ).count()

        expired_records = self.db.query(UsageRecord).filter(
            and_(
                UsageRecord.enterprise_id == enterprise_id,
                UsageRecord.created_at < cutoff_date
            )
        ).count()

        return {
            'enterprise_id': str(enterprise_id),
            'retention_days': retention_days,
            'cutoff_date': cutoff_date.isoformat(),
            'total_records': total_records,
            'expired_records': expired_records,
            'retention_compliance': expired_records == 0
        }


# ==================== 隐私合规审计服务 ====================

class PrivacyAuditService:
    """隐私合规审计服务"""

    def __init__(self, db: Session):
        self.db = db

    def log_data_access(
        self,
        enterprise_id: UUID,
        accessor_id: str,
        accessor_type: str,
        data_type: str,
        action: str,
        record_count: int = 1,
        details: Optional[Dict] = None
    ):
        """
        记录数据访问日志

        Args:
            enterprise_id: 企业ID
            accessor_id: 访问者ID（用户/API密钥/系统）
            accessor_type: 访问者类型（user/api_key/system）
            data_type: 数据类型（usage_records/jobs/bills）
            action: 操作类型（read/export/delete）
            record_count: 涉及记录数
            details: 额外详情
        """
        audit_log = {
            'timestamp': datetime.utcnow().isoformat(),
            'enterprise_id': str(enterprise_id),
            'accessor_id': accessor_id,
            'accessor_type': accessor_type,
            'data_type': data_type,
            'action': action,
            'record_count': record_count,
            'details': details or {}
        }

        logger.info(f"PRIVACY_AUDIT: {json.dumps(audit_log)}")

    def log_privacy_setting_change(
        self,
        enterprise_id: UUID,
        changed_by: str,
        setting_name: str,
        old_value: Any,
        new_value: Any
    ):
        """记录隐私设置变更"""
        audit_log = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'privacy_setting_change',
            'enterprise_id': str(enterprise_id),
            'changed_by': changed_by,
            'setting_name': setting_name,
            'old_value': str(old_value),
            'new_value': str(new_value)
        }

        logger.warning(f"PRIVACY_SETTING_CHANGE: {json.dumps(audit_log)}")

    def log_data_export(
        self,
        enterprise_id: UUID,
        exporter_id: str,
        export_type: str,
        record_count: int,
        destination: str
    ):
        """记录数据导出"""
        audit_log = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'data_export',
            'enterprise_id': str(enterprise_id),
            'exporter_id': exporter_id,
            'export_type': export_type,
            'record_count': record_count,
            'destination': destination
        }

        logger.warning(f"DATA_EXPORT: {json.dumps(audit_log)}")

    def generate_compliance_report(self, enterprise_id: UUID) -> Dict[str, Any]:
        """生成隐私合规报告"""
        enterprise = self.db.query(Enterprise).filter(
            Enterprise.id == enterprise_id
        ).first()

        if not enterprise:
            return {}

        retention_service = DataRetentionService(self.db)
        retention_status = retention_service.get_retention_status(enterprise_id)

        return {
            'enterprise_id': str(enterprise_id),
            'enterprise_name': enterprise.name,
            'generated_at': datetime.utcnow().isoformat(),
            'privacy_settings': {
                'data_retention_days': enterprise.data_retention_days,
                'allow_logging': enterprise.allow_logging,
                'anonymize_data': enterprise.anonymize_data,
                'private_deployment': enterprise.private_deployment
            },
            'data_retention': retention_status,
            'compliance_status': {
                'retention_policy_configured': enterprise.data_retention_days is not None,
                'retention_policy_enforced': retention_status.get('retention_compliance', False),
                'logging_controlled': True,
                'anonymization_enabled': enterprise.anonymize_data
            },
            'recommendations': self._generate_recommendations(enterprise)
        }

    def _generate_recommendations(self, enterprise: Enterprise) -> List[str]:
        """生成隐私改进建议"""
        recommendations = []

        if enterprise.data_retention_days is None:
            recommendations.append("建议配置数据保留策略以符合合规要求")
        elif enterprise.data_retention_days > 90:
            recommendations.append("数据保留期限较长，建议评估是否需要缩短")

        if enterprise.allow_logging and not enterprise.anonymize_data:
            recommendations.append("启用了日志记录但未开启数据匿名化，建议开启匿名化保护用户隐私")

        if not enterprise.private_deployment:
            recommendations.append("如有高敏感数据需求，可考虑私有化部署方案")

        return recommendations


# ==================== 企业隐私保护主服务 ====================

class EnterprisePrivacyService:
    """企业隐私保护主服务"""

    def __init__(self, db: Session, encryption_key: Optional[str] = None):
        self.db = db
        self.anonymizer = DataAnonymizer()
        self.encryptor = DataEncryptor(encryption_key) if encryption_key else None
        self.retention_service = DataRetentionService(db)
        self.audit_service = PrivacyAuditService(db)

    def get_enterprise_privacy_settings(self, enterprise_id: UUID) -> Dict[str, Any]:
        """获取企业隐私设置"""
        enterprise = self.db.query(Enterprise).filter(
            Enterprise.id == enterprise_id
        ).first()

        if not enterprise:
            return {}

        return {
            'enterprise_id': str(enterprise_id),
            'data_retention_days': enterprise.data_retention_days or PrivacyConfig.DEFAULT_RETENTION_DAYS,
            'allow_logging': enterprise.allow_logging,
            'anonymize_data': enterprise.anonymize_data,
            'private_deployment': enterprise.private_deployment
        }

    def update_privacy_settings(
        self,
        enterprise_id: UUID,
        changed_by: str,
        settings: Dict[str, Any]
    ) -> bool:
        """更新企业隐私设置"""
        enterprise = self.db.query(Enterprise).filter(
            Enterprise.id == enterprise_id
        ).first()

        if not enterprise:
            return False

        # 记录变更并更新
        for key, value in settings.items():
            if hasattr(enterprise, key):
                old_value = getattr(enterprise, key)
                if old_value != value:
                    self.audit_service.log_privacy_setting_change(
                        enterprise_id, changed_by, key, old_value, value
                    )
                    setattr(enterprise, key, value)

        self.db.commit()
        return True

    def process_data_for_storage(
        self,
        enterprise_id: UUID,
        data: Dict,
        data_type: str
    ) -> Dict:
        """
        处理数据以符合企业隐私设置

        Args:
            enterprise_id: 企业ID
            data: 原始数据
            data_type: 数据类型
        """
        settings = self.get_enterprise_privacy_settings(enterprise_id)

        if not settings:
            return data

        processed = data.copy()

        # 如果不允许日志记录，清除敏感内容
        if not settings['allow_logging']:
            processed = self._remove_sensitive_content(processed)

        # 如果需要匿名化
        if settings['anonymize_data']:
            processed = self.anonymizer.anonymize_dict(processed)

        # 如果有加密器，加密敏感字段
        if self.encryptor:
            processed = self.encryptor.encrypt_dict(
                processed,
                PrivacyConfig.SENSITIVE_FIELDS
            )

        return processed

    def process_data_for_retrieval(
        self,
        enterprise_id: UUID,
        data: Dict,
        accessor_id: str
    ) -> Dict:
        """
        处理数据用于检索（解密等）

        Args:
            enterprise_id: 企业ID
            data: 存储的数据
            accessor_id: 访问者ID
        """
        # 记录访问
        self.audit_service.log_data_access(
            enterprise_id, accessor_id, 'api_key', 'usage_data', 'read'
        )

        # 如果有加密器，解密数据
        if self.encryptor:
            return self.encryptor.decrypt_dict(data)

        return data

    def export_enterprise_data(
        self,
        enterprise_id: UUID,
        exporter_id: str,
        export_format: str = 'json',
        include_sensitive: bool = False
    ) -> Tuple[str, Dict]:
        """
        导出企业数据（GDPR数据可携带权）

        Args:
            enterprise_id: 企业ID
            exporter_id: 导出者ID
            export_format: 导出格式
            include_sensitive: 是否包含敏感数据
        """
        enterprise = self.db.query(Enterprise).filter(
            Enterprise.id == enterprise_id
        ).first()

        if not enterprise:
            return '', {}

        # 收集数据
        usage_records = self.db.query(UsageRecord).filter(
            UsageRecord.enterprise_id == enterprise_id
        ).all()

        bills = self.db.query(Bill).filter(
            Bill.enterprise_id == enterprise_id
        ).all()

        export_data = {
            'enterprise': {
                'id': str(enterprise.id),
                'name': enterprise.name,
                'code': enterprise.code,
                'created_at': enterprise.created_at.isoformat() if enterprise.created_at else None
            },
            'usage_records': [{
                'id': str(r.id),
                'job_type': r.job_type,
                'quantity': r.quantity,
                'total_cost': r.total_cost,
                'created_at': r.created_at.isoformat() if r.created_at else None
            } for r in usage_records],
            'bills': [{
                'id': str(b.id),
                'period_start': b.period_start.isoformat() if b.period_start else None,
                'period_end': b.period_end.isoformat() if b.period_end else None,
                'total': b.total,
                'status': b.status
            } for b in bills],
            'export_metadata': {
                'exported_at': datetime.utcnow().isoformat(),
                'exporter_id': exporter_id,
                'include_sensitive': include_sensitive
            }
        }

        # 如果不包含敏感数据，进行匿名化
        if not include_sensitive:
            export_data = self.anonymizer.anonymize_dict(export_data)

        # 记录导出操作
        self.audit_service.log_data_export(
            enterprise_id,
            exporter_id,
            export_format,
            len(usage_records) + len(bills),
            'user_download'
        )

        return json.dumps(export_data, indent=2, ensure_ascii=False), export_data

    def delete_enterprise_data(
        self,
        enterprise_id: UUID,
        requester_id: str,
        confirm: bool = False
    ) -> Dict[str, Any]:
        """
        删除企业数据（GDPR被遗忘权）

        Args:
            enterprise_id: 企业ID
            requester_id: 请求者ID
            confirm: 确认删除
        """
        if not confirm:
            # 预览将删除的数据
            usage_count = self.db.query(UsageRecord).filter(
                UsageRecord.enterprise_id == enterprise_id
            ).count()

            bill_count = self.db.query(Bill).filter(
                Bill.enterprise_id == enterprise_id
            ).count()

            return {
                'status': 'preview',
                'data_to_delete': {
                    'usage_records': usage_count,
                    'bills': bill_count
                },
                'message': '请确认删除操作（设置confirm=True）'
            }

        # 执行删除
        deleted_usage = self.db.query(UsageRecord).filter(
            UsageRecord.enterprise_id == enterprise_id
        ).delete()

        deleted_bills = self.db.query(Bill).filter(
            Bill.enterprise_id == enterprise_id
        ).delete()

        self.db.commit()

        # 记录删除操作
        self.audit_service.log_data_access(
            enterprise_id, requester_id, 'user', 'all', 'delete',
            record_count=deleted_usage + deleted_bills,
            details={'reason': 'right_to_be_forgotten'}
        )

        logger.warning(
            f"Enterprise data deleted: {enterprise_id}, "
            f"usage_records={deleted_usage}, bills={deleted_bills}, "
            f"requested_by={requester_id}"
        )

        return {
            'status': 'completed',
            'deleted': {
                'usage_records': deleted_usage,
                'bills': deleted_bills
            }
        }

    def run_scheduled_cleanup(self) -> Dict[str, Any]:
        """运行定时清理任务"""
        logger.info("Starting scheduled privacy cleanup...")

        stats = self.retention_service.cleanup_expired_data()

        logger.info(f"Privacy cleanup completed: {stats}")

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'cleanup_stats': stats
        }

    def _remove_sensitive_content(self, data: Dict) -> Dict:
        """移除敏感内容（当不允许日志记录时）"""
        result = {}
        for key, value in data.items():
            if key.lower() in [f.lower() for f in PrivacyConfig.SENSITIVE_FIELDS]:
                result[key] = None
            elif isinstance(value, dict):
                result[key] = self._remove_sensitive_content(value)
            else:
                result[key] = value
        return result
