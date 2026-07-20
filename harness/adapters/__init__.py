"""Model adapter layer (DESIGN.md §4.2).

Public surface:

- :class:`~harness.adapters.base.ModelAdapter` — the abstract adapter
  contract every provider implementation satisfies.
- :class:`~harness.adapters.base.AdapterError` and
  :func:`~harness.adapters.base.retry_with_backoff` — shared error and retry
  machinery.
- :func:`get_adapter` — factory that turns a
  :class:`~harness.config.ModelConfig` registry entry into a live adapter.

Concrete adapters (``anthropic``, ``openai_compat``, ``fake``) are imported
lazily inside :func:`get_adapter` so importing this package never pulls in a
provider SDK you are not using.
"""

from __future__ import annotations

from pathlib import Path

from harness.adapters.base import AdapterError, ModelAdapter, retry_with_backoff
from harness.config import ModelConfig

__all__ = [
    "AdapterError",
    "ModelAdapter",
    "retry_with_backoff",
    "get_adapter",
]

#: Adapter names accepted in ``[models.*].adapter`` config entries.
VALID_ADAPTERS = ("anthropic", "openai", "fake")


def get_adapter(model_config: ModelConfig) -> ModelAdapter:
    """Build the adapter described by one model-registry entry.

    Dispatches on ``model_config.adapter``:

    - ``"anthropic"`` — :class:`~harness.adapters.anthropic.AnthropicAdapter`
      for ``model_config.model``, honoring ``model_config.base_url`` when set
      (proxies/gateways in front of the Messages API).
    - ``"openai"`` — :class:`~harness.adapters.openai_compat.OpenAICompatAdapter`,
      honoring ``model_config.base_url`` for OpenAI-compatible endpoints.
    - ``"fake"`` — :class:`~harness.adapters.fake.FakeAdapter`; when
      ``model_config.model`` is a path to an existing file it is loaded as a
      JSONL script. A ``model`` value that *looks* like a script path (ends
      in ``.jsonl`` or contains a path separator) but does not exist raises
      :class:`FileNotFoundError` instead of silently degrading to an empty
      script; any other plain string is treated as an intentionally empty
      script.

    The API key reference is resolved via
    :meth:`~harness.config.ModelConfig.resolve_api_key` (env var / keychain).
    Raises :class:`ValueError` naming the valid adapters when
    ``model_config.adapter`` is unknown.
    """
    adapter = model_config.adapter
    if adapter == "anthropic":
        from harness.adapters.anthropic import AnthropicAdapter

        return AnthropicAdapter(
            model=model_config.model,
            api_key=model_config.resolve_api_key(),
            base_url=model_config.base_url,
        )
    if adapter == "openai":
        from harness.adapters.openai_compat import OpenAICompatAdapter

        return OpenAICompatAdapter(
            model=model_config.model,
            api_key=model_config.resolve_api_key(),
            base_url=model_config.base_url,
            extra_body=model_config.extra_body,
        )
    if adapter == "fake":
        from harness.adapters.fake import FakeAdapter

        model = model_config.model
        script_path = Path(model).expanduser()
        if script_path.is_file():
            return FakeAdapter(script_path)
        looks_like_path = model.endswith(".jsonl") or "/" in model or "\\" in model
        if looks_like_path:
            raise FileNotFoundError(
                f"fake adapter model {model!r} looks like a script path "
                f"(resolved to {script_path}) but no such file exists; fix "
                "the path or use a plain non-path string for an "
                "intentionally empty script"
            )
        return FakeAdapter([])
    raise ValueError(
        f"unknown adapter {adapter!r} for model {model_config.model!r}; "
        f"valid adapters: {', '.join(VALID_ADAPTERS)}"
    )
