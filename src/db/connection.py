"""
Database connection pool manager for the 4 historical databases.
"""
import logging
from typing import Dict, Optional
import asyncpg
from config.settings import settings

logger = logging.getLogger("db_connection")
_db_pools: Dict[str, asyncpg.Pool] = {}


async def get_db_pools(timeframe: str = "1d") -> Dict[str, asyncpg.Pool]:
    """
    Returns connection pools for HIGH and LOW volume databases corresponding to the given timeframe.
    For 1d: (DB_HIGH_1D, DB_LOW_1D)
    For 15m: (DB_HIGH_15M, DB_LOW_15M)
    """
    global _db_pools

    if timeframe == "15m":
        high_db = settings.db_high_15m
        low_db = settings.db_low_15m
    else:
        high_db = settings.db_high_1d
        low_db = settings.db_low_1d

    async def _init_session(conn):
        """
        Hard session timeouts so a stuck query or a lock wait can NEVER hang
        the whole engine (asyncpg has no timeouts by default; a blocked
        DELETE/DROP holds its pool slot forever and every worker freezes).
        """
        await conn.execute(
            "SET statement_timeout = '120000'; "      # 120s per statement
            "SET lock_timeout = '15000'; "            # 15s to acquire a lock
            "SET idle_in_transaction_session_timeout = '300000';"
        )

    for db_name in [high_db, low_db]:
        if db_name not in _db_pools or _db_pools[db_name]._closed:
            logger.info(f"Initializing connection pool for database: {db_name}")
            _db_pools[db_name] = await asyncpg.create_pool(
                host=settings.db_host,
                port=settings.db_port,
                user=settings.db_user,
                password=settings.db_password,
                database=db_name,
                min_size=settings.db_min_pool_size,
                max_size=settings.db_max_pool_size,
                command_timeout=150,
                init=_init_session,
            )

    return {
        "HIGH": _db_pools[high_db],
        "LOW": _db_pools[low_db],
        "high_db_name": high_db,
        "low_db_name": low_db,
    }


async def close_all_db_pools() -> None:
    """Closes all active database pools."""
    global _db_pools
    for db_name, pool in list(_db_pools.items()):
        if pool and not pool._closed:
            await pool.close()
            logger.info(f"Closed pool for database: {db_name}")
    _db_pools.clear()
