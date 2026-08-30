"""Masking known secret values at the tool boundary (S-108).

A secret reaches the transcript by accident, not by design: the agent runs
``env``, or a build script echoes its own configuration, or a stack trace
prints a URL with credentials in it. From there it goes three places at once —
the model's context, the persisted event log, and (for a large result) a spill
file on disk. All three outlive the run.

This masks values the harness already knows: the API keys it resolved to make
the run possible. It cannot mask a secret it was never told about, and it is
not a scanner — there is no entropy heuristic here, because a false positive
corrupts a tool result and a false negative is the status quo.

Two properties do the load-bearing work:

**An empty registry is the identity function.** No secrets registered means
:meth:`SecretRegistry.mask` returns the string it was given, unchanged, by
identity check. That is what makes this Lane A: the benchmark path registers
nothing, so it cannot observe that masking exists.

**Short values are never masked.** A four-character "secret" appears inside
ordinary output constantly, and replacing it would corrupt results in ways far
worse than the leak it prevents. Anything shorter than
:data:`MIN_MASKABLE_LENGTH` is refused at registration, loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["SecretRegistry", "MIN_MASKABLE_LENGTH", "SecretTooShortError"]

#: Below this, a value is too common to replace safely. Real credentials are
#: far longer; a short one is either not a credential or not maskable without
#: mangling unrelated text.
MIN_MASKABLE_LENGTH = 12


#: Environment variables the provider SDKs read directly when config omits an
#: ``api_key``. That fallback is documented and supported, which makes it the
#: likeliest real setup — and it produced an **empty registry** while the live
#: key sat in ``os.environ``, inherited by every ``LocalSandbox`` subprocess.
#: An agent running ``env`` (the opening line of this module's docstring) then
#: leaked it to the model, the log and the spill file with masking a no-op.
#:
#: A fixed list, not a pattern like ``*_API_KEY``: matching on name shape would
#: sweep in variables whose values are short or structural, and a false
#: positive here corrupts tool output everywhere the value appears.
CREDENTIAL_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "MOONSHOT_API_KEY",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "TOGETHER_API_KEY",
    "FIREWORKS_API_KEY",
    "XAI_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


class SecretTooShortError(ValueError):
    """A value too short to mask without corrupting ordinary output."""


@dataclass
class SecretRegistry:
    """Values to redact from anything leaving the tool boundary.

    Masking is **span-based**: every occurrence of every secret is located
    first, overlapping spans are merged, and replacement happens right to
    left. Sequential ``str.replace`` calls cannot do this correctly — with
    ``ABCDEFGHIJKL`` and ``GHIJKLMNOPQR`` both registered, replacing the first
    leaves ``[redacted:a]MNOPQR``, a partially-redacted fragment that looks
    safe and is not. Sorting longest-first fixes *containment* only; overlap
    needs the spans.
    """

    #: ``repr=False`` is the point of the field, not a detail. ``ToolDeps`` is
    #: a dataclass captured in every tool closure, and pytest ``--showlocals``,
    #: rich tracebacks and ``logging.exception`` all render frame locals -- so
    #: a default dataclass repr would print every live credential into exactly
    #: the places this module exists to keep clean. ``ModelConfig.api_key`` is
    #: a ``SecretStr`` for the same reason; unwrapping it into a container that
    #: prints it would undo that.
    _secrets: list[tuple[str, str]] = field(default_factory=list, repr=False)

    def __repr__(self) -> str:
        return f"SecretRegistry({len(self._secrets)} value(s))"

    @property
    def empty(self) -> bool:
        return not self._secrets

    def __len__(self) -> int:
        return len(self._secrets)

    def register(self, label: str, value: str | None) -> bool:
        """Register ``value`` under ``label``. Returns whether it was taken.

        ``None`` and empty values are ignored — an unset API key is the common
        case and is not an error. A value that is present but too short raises:
        silently declining to mask something the caller asked to have masked is
        the failure mode this whole module exists to avoid.
        """
        if not value:
            return False
        if len(value) < MIN_MASKABLE_LENGTH:
            raise SecretTooShortError(
                f"{label!r} is {len(value)} characters; values shorter than "
                f"{MIN_MASKABLE_LENGTH} are not masked because replacing them "
                "would corrupt ordinary tool output"
            )
        if any(existing == value for existing, _ in self._secrets):
            return False
        self._secrets.append((value, label))
        self._secrets.sort(key=lambda item: -len(item[0]))
        return True

    def mask(self, text: str) -> str:
        """Replace every registered secret in ``text`` with its label.

        Identity when nothing is registered — the same object comes back, so
        the benchmark path pays one boolean check and nothing else.
        """
        if not self._secrets or not text:
            return text

        spans: list[tuple[int, int, str]] = []
        for value, label in self._secrets:
            start = text.find(value)
            while start != -1:
                spans.append((start, start + len(value), label))
                start = text.find(value, start + 1)
        if not spans:
            return text

        # Merge overlaps so a fragment of one secret cannot survive inside the
        # replacement of another. The first label wins for a merged span; the
        # secrets are longest-first, so that is the longest match.
        spans.sort(key=lambda span: (span[0], -span[1]))
        merged: list[tuple[int, int, str]] = []
        for begin, end, label in spans:
            if merged and begin < merged[-1][1]:
                previous_begin, previous_end, previous_label = merged[-1]
                merged[-1] = (previous_begin, max(previous_end, end), previous_label)
            else:
                merged.append((begin, end, label))

        for begin, end, label in reversed(merged):
            text = f"{text[:begin]}[redacted:{label}]{text[end:]}"
        return text

    def mask_payload(self, payload):
        """Mask every string inside a JSON-shaped structure.

        Persisted events are dicts of arbitrary depth, and a secret can sit in
        a nested tool argument as easily as in a result. Dict **keys** and
        tuples are walked too: model-supplied ``tool_call.arguments`` is an
        arbitrary object, and a tuple serialises into the log as an array, so
        skipping either leaves a live value in the file.

        Identity when nothing is registered, so this costs one check per event
        on the benchmark path rather than a full structural walk.
        """
        if not self._secrets:
            return payload
        if isinstance(payload, str):
            return self.mask(payload)
        if isinstance(payload, dict):
            return {
                self.mask_payload(key): self.mask_payload(value)
                for key, value in payload.items()
            }
        if isinstance(payload, (list, tuple)):
            masked = [self.mask_payload(item) for item in payload]
            return tuple(masked) if isinstance(payload, tuple) else masked
        return payload
