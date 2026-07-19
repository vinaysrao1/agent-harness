"""Model adapter abstract base class and retry machinery (DESIGN.md §4.1-4.2).

This module defines the contract every provider adapter implements and the
shared retry policy the agent loop uses around ``complete()`` calls. It is
deliberately free of any provider SDK import: concrete adapters (Anthropic,
OpenAI-compatible, ...) live in sibling modules and depend on this one, never
the reverse.
"""

from __future__ import annotations

import abc
import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from harness.types import Capabilities, Message, ModelResponse, ToolSpec

__all__ = [
    "AdapterError",
    "ModelAdapter",
    "retry_with_backoff",
]

T = TypeVar("T")


class AdapterError(Exception):
    """A failure surfaced by a model adapter.

    ``retryable`` distinguishes transient faults (rate limits, 5xx, network
    timeouts) from permanent ones (auth failure, invalid request): the retry
    helper only retries the former.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ModelAdapter(abc.ABC):
    """Abstract base for provider adapters.

    Concrete adapters translate the harness's provider-neutral types
    (:class:`~harness.types.Message` et al.) to and from one provider's SDK.
    Everything above this layer is provider-agnostic (goal G1).
    """

    @property
    @abc.abstractmethod
    def capabilities(self) -> Capabilities:
        """Static capabilities of the underlying model/endpoint.

        The harness negotiates behavior from this (parallel tool dispatch,
        cache breakpoints, context budgeting) instead of assuming a lowest
        common denominator.
        """

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str | None = None,
        **params: Any,
    ) -> ModelResponse:
        """Run one model completion over ``messages``.

        ``tools`` declares what the model may call this turn; ``system`` is
        the system prompt (kept separate because providers treat it
        specially, e.g. for cache control). Extra ``params`` (temperature,
        max_tokens, ...) pass through to the provider. Failures must be
        raised as :class:`AdapterError` with ``retryable`` set accurately —
        the agent loop's retry policy depends on it.
        """

    def count_tokens(self, messages: list[Message]) -> int:
        """Approximate the token count of ``messages``.

        Default implementation: total characters of all textual content
        (message content, tool-call names/arguments, tool-result payloads)
        divided by 4, the standard chars-per-token rule of thumb for English
        and code. Adapters with an exact tokenizer should override this; the
        context manager treats the returned value as ground truth either way,
        so a consistent over- or under-estimate is preferable to a noisy one.
        """
        chars = 0
        for message in messages:
            if message.content:
                chars += len(message.content)
            for call in message.tool_calls:
                chars += len(call.name) + len(repr(call.arguments))
            if message.tool_result is not None:
                chars += len(message.tool_result.content)
        return chars // 4


async def retry_with_backoff(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 5,
    backoff_base: float = 1.0,
    backoff_cap: float = 30.0,
    jitter: Callable[[], float] = random.random,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Call ``fn`` with exponential backoff, retrying only retryable errors.

    Policy (DESIGN.md §4.1 API failure policy):

    - ``fn`` is attempted up to ``max_attempts`` times.
    - Only :class:`AdapterError` with ``retryable=True`` triggers a retry;
      non-retryable adapter errors and all other exceptions propagate
      immediately.
    - Delay before retry *n* (1-indexed) is
      ``min(backoff_base * 2**(n-1), backoff_cap) * (1 + jitter())`` where
      ``jitter()`` returns a float in ``[0, 1)`` — i.e. full exponential
      backoff with up to 2x multiplicative jitter, capped.
    - The final attempt's error is re-raised once attempts are exhausted.

    ``jitter`` and ``sleep`` are injectable so tests can run deterministically
    and without real sleeping.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except AdapterError as exc:
            if not exc.retryable or attempt == max_attempts:
                raise
            delay = min(backoff_base * 2 ** (attempt - 1), backoff_cap)
            await sleep(delay * (1 + jitter()))
    raise AssertionError("unreachable")  # pragma: no cover
