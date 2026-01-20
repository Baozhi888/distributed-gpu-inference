import os


# 让 server/app 在没有 asyncpg 的环境下也可被导入：
# - 生产默认使用 postgresql+asyncpg
# - 单测/CI 默认改用 sqlite+aiosqlite（无需网络与额外依赖）
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
