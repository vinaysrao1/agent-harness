"""Which model serves which kind of call (S-106).

A run makes more than one kind of model call, and they have different economics.
The main loop's call needs the run's model. The compaction summarizer is
rendering an evicted transcript span into prose — a job a cheap model does
about as well, several times per long run, on input that is by definition the
part the run has decided it no longer needs verbatim.

Routing that is a config change rather than a code change is the point. The
`Router` resolves a *purpose* to a registry model name, and falls back to the
run's own model for anything unrouted, so the default configuration produces
exactly the calls it produced before.

**Two purposes, not the four the plan named.** `classify` and `lint` are in
the plan; neither is a model call in this harness. `diligence.lint_verification`
is a pure function over a command string. A purpose that nothing can route is a
config surface implying a mechanism that does not exist — the archetype this
project keeps catching — so the enum grows when the call site does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = ["CallPurpose", "Router", "UnroutableModelError"]


class CallPurpose(str, Enum):
    """What a model call is *for*.

    A `str` enum so it lands in a SQLite column and a JSON payload as its
    value without a conversion step at every call site.
    """

    #: The agent loop's own call — the one that chooses tools and writes code.
    MAIN = "main"
    #: Rendering an evicted transcript span into a summary (S-105).
    SUMMARIZE = "summarize"


class UnroutableModelError(ValueError):
    """`[routing]` names a model the registry does not have."""


@dataclass(frozen=True)
class Router:
    """Resolves a :class:`CallPurpose` to a model name.

    ``default`` is the run's own model, and every unrouted purpose gets it, so
    an empty ``overrides`` is exactly today's behaviour.
    """

    default: str
    overrides: dict[str, str] = field(default_factory=dict)

    def model_for(self, purpose: CallPurpose) -> str:
        return self.overrides.get(purpose.value, self.default)

    def is_routed(self, purpose: CallPurpose) -> bool:
        """Whether this purpose goes somewhere other than the run's model.

        The orchestrator asks before building a second adapter: routing to the
        same name should reuse the run's adapter *object*, not construct an
        equivalent one, so the default path is provably unchanged rather than
        merely equivalent.
        """
        routed = self.overrides.get(purpose.value)
        return routed is not None and routed != self.default


def build_router(
    default: str, overrides: dict[str, str], known_models: set[str]
) -> Router:
    """Validate `[routing]` against the model registry and build the router.

    Raises rather than falling back. A silent fallback is the failure where a
    run reports the cheap model in its config and bills the expensive one --
    and the only place that shows up is a bill, weeks later, with no line item
    naming the purpose that did it.

    An unknown *purpose* raises too. `[routing] summarise = "..."` is a
    plausible typo, and an ignored key looks exactly like a working
    configuration.
    """
    valid = {p.value for p in CallPurpose}
    for purpose, model in overrides.items():
        if purpose not in valid:
            raise UnroutableModelError(
                f"unknown call purpose {purpose!r} in [routing]; "
                f"known purposes: {', '.join(sorted(valid))}"
            )
        if model not in known_models:
            available = ", ".join(sorted(known_models)) or "(none)"
            raise UnroutableModelError(
                f"[routing] {purpose} = {model!r} names a model that is not "
                f"in the registry; configured models: {available}"
            )
    return Router(default=default, overrides=dict(overrides))
