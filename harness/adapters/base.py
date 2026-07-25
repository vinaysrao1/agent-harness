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
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeVar

from harness.types import Capabilities, Message, ModelResponse, ToolSpec

__all__ = [
    "AdapterError",
    "Fault",
    "ModelAdapter",
    "classify_http_fault",
    "retry_with_backoff",
]

T = TypeVar("T")

#: Provider-fault taxonomy: *why* a run died at the provider boundary, as
#: opposed to at the model's own hands. Deliberately provider-neutral and
#: free of any benchmark-harness vocabulary — the mapping onto a specific
#: harness's exception classes lives in that harness's integration module
#: (``harness/integrations/harbor_agent.py`` is the only place allowed to
#: import Harbor). Consumers that do not care may ignore it entirely.
Fault = Literal[
    "auth",
    "quota",
    "rate_limit",
    "server",
    "transport",
    "malformed_response",
]

#: Substrings that mark an HTTP 403 body as *credit/quota exhaustion* rather
#: than one of 403's other meanings. 403 alone is over-broad: gateways also
#: use it for region blocks, moderation refusals, and "your key may not use
#: this model" — all of which are genuine scored failures that must NOT be
#: laundered into an infrastructure fault. Matching is case-insensitive on
#: the provider's own error text; when nothing matches we deliberately
#: classify nothing rather than guess.
_QUOTA_BODY_MARKERS = (
    "insufficient_quota",
    "limit exceeded",
    "credit",
    "quota",
)


def classify_http_fault(status: int | None, body: str = "") -> Fault | None:
    """Classify an HTTP status (plus its error body) into a :data:`Fault`.

    Shared by every HTTP-speaking adapter so the taxonomy cannot drift
    between providers:

    ==============================================  ===============
    Condition                                       ``Fault``
    ==============================================  ===============
    401                                             ``auth``
    402                                             ``quota``
    403 whose body matches a quota marker           ``quota``
    403 otherwise                                   ``None``
    429                                             ``rate_limit``
    5xx                                             ``server``
    anything else (400, 404, ...)                   ``None``
    ==============================================  ===============

    ``None`` means "not a provider fault" — the failure stays an ordinary
    scored failure, which is the right answer for a 400 (a harness bug that
    must remain loudly visible in the numbers) and for the non-quota 403s.
    Statusless transport faults are classified by their callers, which are
    the ones that know a fault had no status at all.
    """
    if status is None:
        return None
    if status == 401:
        return "auth"
    if status == 402:
        return "quota"
    if status == 403:
        lowered = body.lower()
        if any(marker in lowered for marker in _QUOTA_BODY_MARKERS):
            return "quota"
        return None
    if status == 429:
        return "rate_limit"
    if status >= 500:
        return "server"
    return None


class AdapterError(Exception):
    """A failure surfaced by a model adapter.

    ``retryable`` distinguishes transient faults (rate limits, 5xx, network
    timeouts) from permanent ones (auth failure, invalid request): the retry
    helper only retries the former.

    ``fault`` optionally classifies *what kind* of provider fault this was
    (see :data:`Fault`). It is orthogonal to ``retryable``: ``retryable``
    governs this adapter's own bounded retry loop, while ``fault`` describes
    the surviving failure for whoever is above the harness — the agent loop
    carries it out on ``AgentResult.error_kind``. ``None`` (the default,
    which preserves every existing construction site) means "unclassified":
    an ordinary failure that should be scored as one.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        fault: Fault | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.fault = fault


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
    max_elapsed: float | None = None,
    clock: Callable[[], float] = time.monotonic,
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

    Wall-clock ceiling: when ``max_elapsed`` is set, the total time spent in
    this call (including the duration of the ``fn`` attempts themselves, read
    via ``clock``) is bounded — before sleeping for a retry, if the elapsed
    time plus that sleep would reach ``max_elapsed``, the current error is
    re-raised instead. This keeps ``per-attempt-timeout × max_attempts`` from
    silently overrunning an upstream deadline (e.g. a benchmark harness's
    per-agent timeout) on a persistently-hanging provider. ``None`` (default)
    leaves the sequence bounded only by ``max_attempts``.

    ``jitter``, ``sleep``, and ``clock`` are injectable so tests can run
    deterministically and without real sleeping.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    start = clock()
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except AdapterError as exc:
            if not exc.retryable or attempt == max_attempts:
                raise
            delay = min(backoff_base * 2 ** (attempt - 1), backoff_cap) * (
                1 + jitter()
            )
            if max_elapsed is not None and (clock() - start) + delay >= max_elapsed:
                raise
            await sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
