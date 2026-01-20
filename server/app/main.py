"""FastAPI主入口"""
import asyncio
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from app.config import settings
from app.db.database import init_db, AsyncSessionLocal
from app.api import jobs, workers
from app.api import admin
from app.services.task_guarantee import TaskGuaranteeBackgroundWorker

logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 后台任务保障服务
task_guarantee_worker = TaskGuaranteeBackgroundWorker(AsyncSessionLocal)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    await init_db()

    # 启动后台任务
    logger.info(f"Starting server in region: {settings.region}")
    background_task = asyncio.create_task(task_guarantee_worker.start())

    yield

    # 关闭后台任务
    task_guarantee_worker.stop()
    background_task.cancel()
    try:
        await background_task
    except asyncio.CancelledError:
        pass
    logger.info("Server shutdown complete")


app = FastAPI(
    title=settings.app_name,
    description="分布式GPU推理服务 - 支持全球多区域部署",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(jobs.router)
app.include_router(workers.router)
app.include_router(admin.router)

# 静态文件服务（管理后台）
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/admin")
@app.get("/admin/{path:path}")
async def admin_spa(path: str = ""):
    """管理后台SPA入口"""
    admin_index = STATIC_DIR / "admin" / "index.html"
    if admin_index.exists():
        return FileResponse(admin_index)
    return {"error": "Admin panel not found"}


@app.get("/")
async def root():
    return {
        "message": "Distributed GPU Inference API",
        "version": "1.0.0",
        "region": settings.region
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "region": settings.region
    }


@app.get("/regions")
async def get_regions():
    """获取所有区域信息"""
    return {
        "current_region": settings.region,
        "available_regions": [
            {"code": "asia-east", "name": "东亚", "description": "中国、日本、韩国"},
            {"code": "asia-south", "name": "东南亚", "description": "新加坡、泰国、越南"},
            {"code": "europe-west", "name": "西欧", "description": "德国、法国、英国"},
            {"code": "europe-east", "name": "东欧", "description": "波兰、俄罗斯"},
            {"code": "america-north", "name": "北美", "description": "美国、加拿大"},
            {"code": "america-south", "name": "南美", "description": "巴西、阿根廷"},
            {"code": "oceania", "name": "大洋洲", "description": "澳大利亚、新西兰"}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
