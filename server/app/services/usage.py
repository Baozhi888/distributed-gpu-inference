"""
算力使用量追踪服务
记录、计算和聚合GPU算力消耗
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from app.models.usage import (
    UsageRecord, Enterprise, EnterpriseAPIKey, PricePlan,
    Bill, WorkerUsageSummary, UsageType
)
from app.models.models import Worker, Job


class UsageService:
    """算力使用量服务"""

    @staticmethod
    def record_usage(
        db: Session,
        job: Job,
        worker: Worker,
        enterprise_id: UUID,
        api_key_id: Optional[UUID] = None,
        details: Optional[Dict] = None
    ) -> UsageRecord:
        """
        记录任务使用量

        Args:
            db: 数据库会话
            job: 任务对象
            worker: Worker对象
            enterprise_id: 企业ID
            api_key_id: API密钥ID
            details: 额外详情
        """
        details = details or {}

        # 计算使用量
        usage_data = UsageService._calculate_usage(job, details)

        # 获取价格
        enterprise = db.query(Enterprise).filter(Enterprise.id == enterprise_id).first()
        unit_price, total_cost = UsageService._calculate_cost(
            db, enterprise, usage_data['usage_type'], usage_data['quantity']
        )

        # 创建使用记录
        record = UsageRecord(
            enterprise_id=enterprise_id,
            worker_id=worker.id,
            job_id=job.id,
            api_key_id=api_key_id,
            machine_id=worker.machine_id or str(worker.id)[:32],
            usage_type=usage_data['usage_type'],
            job_type=job.type,
            model_id=details.get('model_id'),
            quantity=usage_data['quantity'],
            unit=usage_data['unit'],
            unit_price=unit_price,
            total_cost=total_cost,
            gpu_seconds=usage_data.get('gpu_seconds', 0),
            gpu_memory_peak_gb=details.get('gpu_memory_peak_gb'),
            started_at=job.started_at,
            completed_at=job.completed_at,
            duration_ms=job.actual_duration_ms,
            request_summary=UsageService._create_request_summary(job, enterprise),
            response_summary=UsageService._create_response_summary(job, enterprise),
            worker_region=worker.region,
            client_ip=job.client_ip,
            client_region=job.client_region
        )

        db.add(record)

        # 更新企业余额
        if enterprise and total_cost > 0:
            enterprise.credit_balance -= total_cost

        db.commit()
        db.refresh(record)

        return record

    @staticmethod
    def _calculate_usage(job: Job, details: Dict) -> Dict[str, Any]:
        """根据任务类型计算使用量"""
        job_type = job.type
        result = job.result or {}
        params = job.params or {}

        # 计算GPU秒数
        gpu_seconds = 0
        if job.actual_duration_ms:
            gpu_seconds = job.actual_duration_ms / 1000.0

        if job_type == 'llm':
            # LLM: 计算Token数
            input_tokens = result.get('usage', {}).get('prompt_tokens', 0)
            output_tokens = result.get('usage', {}).get('completion_tokens', 0)
            total_tokens = input_tokens + output_tokens

            return {
                'usage_type': UsageType.LLM_TOKENS.value,
                'quantity': total_tokens / 1000,  # 千Token
                'unit': 'K tokens',
                'gpu_seconds': gpu_seconds
            }

        elif job_type == 'image_gen':
            # 图像生成: 按张数和像素计算
            width = params.get('width', 1024)
            height = params.get('height', 1024)
            num_images = params.get('num_images', 1)
            total_pixels = width * height * num_images / 1_000_000  # 百万像素

            return {
                'usage_type': UsageType.IMAGE_GEN.value,
                'quantity': num_images,
                'unit': 'images',
                'gpu_seconds': gpu_seconds,
                'pixels': total_pixels
            }

        elif job_type == 'whisper':
            # 语音识别: 按秒数计算
            audio_duration = result.get('duration', details.get('audio_duration', 0))

            return {
                'usage_type': UsageType.WHISPER_SECONDS.value,
                'quantity': audio_duration,
                'unit': 'seconds',
                'gpu_seconds': gpu_seconds
            }

        elif job_type == 'embedding':
            # 嵌入: 按Token数计算
            texts = params.get('texts', [])
            # 粗略估计Token数 (中文约1.5字符/Token，英文约4字符/Token)
            total_chars = sum(len(t) for t in texts)
            estimated_tokens = total_chars / 3

            return {
                'usage_type': UsageType.EMBEDDING_TOKENS.value,
                'quantity': estimated_tokens / 1000,
                'unit': 'K tokens',
                'gpu_seconds': gpu_seconds
            }

        # 默认: 按GPU秒计费
        return {
            'usage_type': UsageType.GPU_SECONDS.value,
            'quantity': gpu_seconds,
            'unit': 'seconds',
            'gpu_seconds': gpu_seconds
        }

    @staticmethod
    def _calculate_cost(
        db: Session,
        enterprise: Optional[Enterprise],
        usage_type: str,
        quantity: float
    ) -> tuple:
        """计算费用"""
        # 获取价格方案
        if enterprise and enterprise.custom_pricing.get(usage_type):
            unit_price = enterprise.custom_pricing[usage_type]
        elif enterprise and enterprise.price_plan_id:
            plan = db.query(PricePlan).filter(PricePlan.id == enterprise.price_plan_id).first()
            unit_price = plan.prices.get(usage_type, 0) if plan else 0
        else:
            # 默认价格
            default_prices = {
                UsageType.LLM_TOKENS.value: 0.002,
                UsageType.LLM_REQUESTS.value: 0.01,
                UsageType.IMAGE_GEN.value: 0.1,
                UsageType.IMAGE_PIXELS.value: 0.00001,
                UsageType.WHISPER_SECONDS.value: 0.006,
                UsageType.EMBEDDING_TOKENS.value: 0.0001,
                UsageType.GPU_SECONDS.value: 0.001,
            }
            unit_price = default_prices.get(usage_type, 0)

        total_cost = unit_price * quantity
        return unit_price, total_cost

    @staticmethod
    def _create_request_summary(job: Job, enterprise: Optional[Enterprise]) -> Optional[Dict]:
        """创建请求摘要（考虑隐私设置）"""
        if enterprise and not enterprise.allow_logging:
            return None

        params = job.params or {}
        summary = {
            'type': job.type,
            'model': params.get('model'),
        }

        # 隐私保护: 不记录完整输入
        if enterprise and enterprise.anonymize_data:
            if 'messages' in params:
                summary['message_count'] = len(params['messages'])
            if 'prompt' in params:
                summary['prompt_length'] = len(params['prompt'])

        return summary

    @staticmethod
    def _create_response_summary(job: Job, enterprise: Optional[Enterprise]) -> Optional[Dict]:
        """创建响应摘要（考虑隐私设置）"""
        if enterprise and not enterprise.allow_logging:
            return None

        result = job.result or {}
        summary = {
            'status': job.status,
            'duration_ms': job.actual_duration_ms
        }

        if 'usage' in result:
            summary['token_usage'] = result['usage']

        return summary

    @staticmethod
    def get_enterprise_usage(
        db: Session,
        enterprise_id: UUID,
        start_time: datetime,
        end_time: datetime,
        group_by: str = 'day'
    ) -> Dict[str, Any]:
        """获取企业使用量统计"""
        records = db.query(UsageRecord).filter(
            and_(
                UsageRecord.enterprise_id == enterprise_id,
                UsageRecord.created_at >= start_time,
                UsageRecord.created_at <= end_time
            )
        ).all()

        # 按类型汇总
        by_type = {}
        total_cost = 0
        total_jobs = len(records)

        for record in records:
            if record.usage_type not in by_type:
                by_type[record.usage_type] = {
                    'quantity': 0,
                    'cost': 0,
                    'count': 0
                }
            by_type[record.usage_type]['quantity'] += record.quantity
            by_type[record.usage_type]['cost'] += record.total_cost or 0
            by_type[record.usage_type]['count'] += 1
            total_cost += record.total_cost or 0

        return {
            'enterprise_id': str(enterprise_id),
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'total_jobs': total_jobs,
            'total_cost': round(total_cost, 4),
            'by_type': by_type
        }

    @staticmethod
    def get_worker_usage(
        db: Session,
        worker_id: UUID,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """获取Worker使用量统计"""
        worker = db.query(Worker).filter(Worker.id == worker_id).first()
        if not worker:
            return {}

        records = db.query(UsageRecord).filter(
            and_(
                UsageRecord.worker_id == worker_id,
                UsageRecord.created_at >= start_time,
                UsageRecord.created_at <= end_time
            )
        ).all()

        total_gpu_seconds = sum(r.gpu_seconds or 0 for r in records)
        total_revenue = sum(r.total_cost or 0 for r in records)

        # 按小时分布
        hourly_dist = {}
        for record in records:
            hour = record.created_at.strftime('%Y-%m-%d %H:00')
            if hour not in hourly_dist:
                hourly_dist[hour] = {'jobs': 0, 'gpu_seconds': 0, 'revenue': 0}
            hourly_dist[hour]['jobs'] += 1
            hourly_dist[hour]['gpu_seconds'] += record.gpu_seconds or 0
            hourly_dist[hour]['revenue'] += record.total_cost or 0

        return {
            'worker_id': str(worker_id),
            'machine_id': worker.machine_id,
            'worker_name': worker.name,
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'total_jobs': len(records),
            'total_gpu_seconds': round(total_gpu_seconds, 2),
            'total_revenue': round(total_revenue, 4),
            'hourly_distribution': hourly_dist
        }

    @staticmethod
    def aggregate_hourly_summary(db: Session, worker_id: UUID, hour: datetime):
        """聚合小时汇总"""
        period_start = hour.replace(minute=0, second=0, microsecond=0)
        period_end = period_start + timedelta(hours=1)

        worker = db.query(Worker).filter(Worker.id == worker_id).first()
        if not worker:
            return

        records = db.query(UsageRecord).filter(
            and_(
                UsageRecord.worker_id == worker_id,
                UsageRecord.created_at >= period_start,
                UsageRecord.created_at < period_end
            )
        ).all()

        if not records:
            return

        summary = WorkerUsageSummary(
            worker_id=worker_id,
            machine_id=worker.machine_id or str(worker_id)[:32],
            period_type='hourly',
            period_start=period_start,
            period_end=period_end,
            total_jobs=len(records),
            completed_jobs=len([r for r in records if r.job_type]),  # 都是完成的
            total_gpu_seconds=sum(r.gpu_seconds or 0 for r in records),
            total_tokens=sum(
                int(r.quantity * 1000) for r in records
                if r.usage_type in [UsageType.LLM_TOKENS.value, UsageType.EMBEDDING_TOKENS.value]
            ),
            total_images=sum(
                int(r.quantity) for r in records
                if r.usage_type == UsageType.IMAGE_GEN.value
            ),
            total_revenue=sum(r.total_cost or 0 for r in records),
            peak_gpu_memory_gb=max(
                (r.gpu_memory_peak_gb or 0 for r in records),
                default=0
            )
        )

        # 检查是否已存在
        existing = db.query(WorkerUsageSummary).filter(
            and_(
                WorkerUsageSummary.worker_id == worker_id,
                WorkerUsageSummary.period_type == 'hourly',
                WorkerUsageSummary.period_start == period_start
            )
        ).first()

        if existing:
            # 更新
            for key, value in summary.__dict__.items():
                if not key.startswith('_') and key != 'id':
                    setattr(existing, key, value)
        else:
            db.add(summary)

        db.commit()

    @staticmethod
    def get_platform_stats(db: Session) -> Dict[str, Any]:
        """获取平台总体统计"""
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # 今日统计
        today_records = db.query(UsageRecord).filter(
            UsageRecord.created_at >= today_start
        ).all()

        # 本月统计
        month_records = db.query(UsageRecord).filter(
            UsageRecord.created_at >= month_start
        ).all()

        # Worker统计
        total_workers = db.query(Worker).count()
        online_workers = db.query(Worker).filter(
            Worker.status.in_(['online', 'busy'])
        ).count()

        # 企业统计
        total_enterprises = db.query(Enterprise).filter(Enterprise.is_active == True).count()

        return {
            'timestamp': now.isoformat(),
            'workers': {
                'total': total_workers,
                'online': online_workers
            },
            'enterprises': {
                'total': total_enterprises
            },
            'today': {
                'jobs': len(today_records),
                'revenue': round(sum(r.total_cost or 0 for r in today_records), 2),
                'gpu_hours': round(sum(r.gpu_seconds or 0 for r in today_records) / 3600, 2)
            },
            'this_month': {
                'jobs': len(month_records),
                'revenue': round(sum(r.total_cost or 0 for r in month_records), 2),
                'gpu_hours': round(sum(r.gpu_seconds or 0 for r in month_records) / 3600, 2)
            }
        }
