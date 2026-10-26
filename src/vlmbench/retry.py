from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable

from .models import NonRetryableError

log = logging.getLogger(__name__)


def call_with_retries(fn: Callable, *args, max_retries: int = 5) -> tuple[Any, str | None, int]:
    """Call ``fn(*args)``, retrying transient failures with exponential backoff and jitter.

    Returns ``(result, error, attempts)``; ``error`` is None on success. Errors a retry
    cannot fix (``NonRetryableError``) are returned immediately.
    """
    delay = 2.0
    for attempt in range(1, max_retries + 2):
        try:
            return fn(*args), None, attempt
        except NonRetryableError as e:
            return None, f"non_retryable: {e}", attempt
        except Exception as e:  # noqa: BLE001 - network / rate limit / server errors
            if attempt > max_retries:
                return None, f"{type(e).__name__}: {e}", attempt
            wait = min(delay, 120) * (0.5 + random.random())
            log.warning("Attempt %d failed (%s: %s); retrying in %.0fs",
                        attempt, type(e).__name__, str(e)[:200], wait)
            time.sleep(wait)
            delay *= 2
    raise AssertionError("unreachable")
