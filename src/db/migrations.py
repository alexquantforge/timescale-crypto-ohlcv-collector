"""
Database initialization and TimescaleDB extension check for the 4 historical databases.
"""
import logging
import asyncpg
from config.settings import settings

logger = logging.getLogger("db_migrations")

ALL_DB_NAMES = [
    settings.db_high_1d,
    settings.db_low_1d,
    settings.db_high_15m,
    settings.db_low_15m,
]


async def ensure_databases_exist() -> None:
    """
    Connects to default postgres database and creates any missing databases from the 4 configured names.
    """
    try:
        conn = await asyncpg.connect(
            host=settings.db_host,
            port=settings.db_port,
            user=settings.db_user,
            password=settings.db_password,
            database="postgres",
        )
        for db_name in ALL_DB_NAMES:
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", db_name
            )
            if not exists:
                logger.info(f"Creating database '{db_name}'...")
                await conn.execute(f'CREATE DATABASE "{db_name}"')
                logger.info(f"✓ Database '{db_name}' created.")

                # Enable TimescaleDB extension if available
                try:
                    db_conn = await asyncpg.connect(
                        host=settings.db_host,
                        port=settings.db_port,
                        user=settings.db_user,
                        password=settings.db_password,
                        database=db_name,
                    )
                    await db_conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
                    await db_conn.close()
                    logger.info(f"✓ TimescaleDB extension enabled for '{db_name}'.")
                except Exception as e:
                    logger.debug(f"TimescaleDB notice for '{db_name}': {e}")
        await conn.close()
    except Exception as e:
        logger.warning(f"Database check notice: {e}")
