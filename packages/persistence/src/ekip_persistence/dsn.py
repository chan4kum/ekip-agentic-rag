def to_asyncpg_dsn(url: str) -> str:
    """asyncpg wants plain postgresql://; our settings use the SQLAlchemy-style +asyncpg form."""
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)
