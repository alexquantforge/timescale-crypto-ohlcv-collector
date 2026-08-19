"""
Database package: connection pooling, TimescaleDB schema migrations, and repository access.
"""
from src.db.connection import get_db_pools, close_all_db_pools
from src.db.migrations import ensure_databases_exist
from src.db.repository import HistoricalMarketRepository

__all__ = [
    "get_db_pools",
    "close_all_db_pools",
    "ensure_databases_exist",
    "HistoricalMarketRepository",
]
