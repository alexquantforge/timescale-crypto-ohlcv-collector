"""
Hard network timeout helper.

`asyncio.wait_for` is NOT a hard bound on Python 3.11: when the timeout
fires, wait_for cancels the inner task and then awaits its cancellation
(`_cancel_and_wait`) with NO time limit. If the inner task suppresses or
slowly handles CancelledError (aiohttp/ccxt internals occasionally do),
the caller hangs forever — a semaphore slot is held and the engine
silently freezes (which is exactly how the 1D watchdog stalls looked).

`hard_wait_for` bounds the total wait: after `timeout` it cancels the
inner task, waits at most `cancel_grace` more seconds for the
cancellation to land, then ABANDONS the still-running task (it is left
to die on its own socket timeouts) and raises asyncio.TimeoutError so
the worker can move on and release its slot.
"""
import asyncio
import logging
from typing import Any, Awaitable

logger = logging.getLogger("timeouts")


def _silence_zombie(t: asyncio.Task) -> None:
    """Consumes the abandoned task's exception so asyncio never logs
    'Task exception was never retrieved' for a task we deliberately left
    running past its hard timeout."""
    try:
        t.exception()
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


async def hard_wait_for(
    aw: Awaitable[Any],
    timeout: float,
    cancel_grace: float = 5.0,
    label: str = "",
) -> Any:
    """Like asyncio.wait_for, but the total wait is strictly bounded by
    timeout + cancel_grace even if the awaited task ignores cancellation."""
    task = asyncio.ensure_future(aw)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if done:
            return task.result()  # propagates inner exceptions
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=cancel_grace)
        if done:
            try:
                return task.result()
            except asyncio.CancelledError:
                raise asyncio.TimeoutError()
        # Cancellation did not land in time — abandon the zombie task.
        # Mask its cancellation so our caller's TimeoutError is not
        # suppressed by the task's later completion.
        task.uncancel()
        task.add_done_callback(_silence_zombie)
        logger.warning(
            f"hard timeout: inner task ignored cancel for {cancel_grace:.0f}s"
            + (f" ({label})" if label else "")
            + " — abandoning and moving on"
        )
        raise asyncio.TimeoutError()
    except asyncio.CancelledError:
        # The OUTER caller (e.g. engine shutdown) was cancelled — propagate
        # the cancellation into the inner task and honor it.
        task.cancel()
        task.add_done_callback(_silence_zombie)
        raise
