from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(
    settings.database_url,
    echo=False,  # Never echo SQL in production; use dedicated SQL logging if needed
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=5,
    pool_recycle=1800,  # Recycle connections every 30 min to avoid stale connections
    pool_timeout=30,
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session
