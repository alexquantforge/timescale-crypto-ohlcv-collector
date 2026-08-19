"""
CCXT Async Exchange Factory and Resource Lifecycle Manager.
"""
import asyncio
import logging
import ccxt.async_support as ccxt_async
from typing import Optional
from config.settings import settings

logger = logging.getLogger("exchange_client")


def create_exchange(ccxt_id: str, proxy: Optional[str] = None) -> ccxt_async.Exchange:
    """
    Instantiates an asynchronous CCXT exchange client with rate-limiting and proxy settings.
    """
    exchange_class = getattr(ccxt_async, ccxt_id)
    config = {
        "enableRateLimit": True,
        "timeout": 40000,
        "options": {},
    }
    proxy_url = proxy or settings.socks5_proxy
    if proxy_url:
        config["socks_proxy"] = proxy_url

    return exchange_class(config)


async def close_exchange_safely(exchange: ccxt_async.Exchange, name: str = "") -> None:
    """
    Safely closes a CCXT exchange instance and releases aiohttp / SOCKS resources
    to prevent unclosed session warnings during garbage collection.
    """
    if exchange is None:
        return

    try:
        await exchange.close()
    except Exception as e:
        logger.debug(f"exchange.close() exception for {name}: {e}")

    try:
        session = getattr(exchange, "session", None)
        if session is not None:
            if getattr(exchange, "own_session", True):
                await session.close()
            exchange.session = None
    except Exception:
        pass

    for attr in ("tcp_connector", "aiohttp_socks_connector"):
        try:
            conn = getattr(exchange, attr, None)
            if conn is not None:
                await conn.close()
                setattr(exchange, attr, None)
        except Exception:
            pass

    try:
        sps = getattr(exchange, "socks_proxy_sessions", None)
        if sps:
            for url in list(sps):
                try:
                    await sps[url].close()
                except Exception:
                    pass
            exchange.socks_proxy_sessions = None
    except Exception:
        pass

    await asyncio.sleep(0.25)
