"""Configuration loading for the harness (DESIGN.md §4.2, §4.7, §4.11).

Config lives in a single TOML file parsed with stdlib :mod:`tomllib`. The
search order is: an explicit path passed to :func:`load_config`, then
``$HARNESS_HOME/config.toml`` (``HARNESS_HOME`` defaults to ``~/.harness``).
A missing file is not an error — it yields a fully usable default config.

Secrets are handled as *references*, resolved lazily via
:func:`resolve_secret`:

- ``"env:VAR"``       — read from ``os.environ``
- ``"keychain:name"`` — read from the OS keychain via ``keyring``
- anything else       — treated as the literal value, with a warning

The ``api_key`` field is stored as a pydantic ``SecretStr`` so that neither
the reference nor a literal secret ever appears in ``repr()``/``str()`` of
config objects or in logs.

Expected TOML shape::

    [models.opus]
    adapter  = "anthropic"
    model    = "claude-opus-4-8"
    api_key  = "keychain:anthropic"

    [models.kimi]
    adapter  = "openai"
    base_url = "https://api.moonshot.ai/v1"
    model    = "kimi-k3"
    api_key  = "env:MOONSHOT_API_KEY"

    [sandbox]
    network = "allowlist"          # none | allowlist | open
    image   = "harness-sandbox:latest"

    [permissions]
    default = "gated"              # gated | auto
    allow   = ["mcp.github.create_issue"]   # per-tool/per-pattern overrides
    deny    = ["bash"]                      # (fnmatch globs, DESIGN.md §4.11)
"""

from __future__ import annotations

import os
import tomllib
import warnings
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

__all__ = [
    "ConfigError",
    "SecretResolutionError",
    "NetworkMode",
    "PermissionMode",
    "ModelConfig",
    "SandboxConfig",
    "HarnessConfig",
    "harness_home",
    "resolve_secret",
    "load_config",
]

#: Environment variable that overrides the harness home directory.
HARNESS_HOME_ENV = "HARNESS_HOME"

#: Default harness home directory when ``HARNESS_HOME`` is unset.
DEFAULT_HOME = Path("~/.harness")


class ConfigError(Exception):
    """Raised when the config file exists but cannot be parsed or validated."""


class SecretResolutionError(ConfigError):
    """Raised when a secret reference cannot be resolved to a value."""


class NetworkMode(str, Enum):
    """Sandbox network egress policy (DESIGN.md §4.7)."""

    NONE = "none"
    ALLOWLIST = "allowlist"
    OPEN = "open"


class PermissionMode(str, Enum):
    """Default autonomy mode for the permission engine (DESIGN.md §4.11)."""

    GATED = "gated"
    AUTO = "auto"


def harness_home() -> Path:
    """Return the harness home directory.

    ``$HARNESS_HOME`` if set, else ``~/.harness``; always expanded to an
    absolute path. The directory is not created here — callers that need it
    on disk create it themselves.
    """
    raw = os.environ.get(HARNESS_HOME_ENV)
    return (Path(raw) if raw else DEFAULT_HOME).expanduser()


def resolve_secret(reference: str) -> str:
    """Resolve a secret reference to its actual value.

    ``"env:VAR"`` reads ``os.environ["VAR"]``; ``"keychain:name"`` reads the
    OS keychain entry ``(service="harness", username=name)`` via ``keyring``.
    Any other string is returned as-is under the assumption it is a literal
    secret, with a :class:`UserWarning` recommending a proper reference.

    Raises :class:`SecretResolutionError` if the referenced env var or
    keychain entry is missing, or if ``keyring`` is not installed.
    """
    if reference.startswith("env:"):
        var = reference[len("env:"):]
        try:
            return os.environ[var]
        except KeyError:
            raise SecretResolutionError(
                f"secret reference {reference!r}: environment variable "
                f"{var!r} is not set"
            ) from None
    if reference.startswith("keychain:"):
        name = reference[len("keychain:"):]
        try:
            import keyring
        except ImportError:
            raise SecretResolutionError(
                f"secret reference {reference!r} requires the 'keyring' "
                "package, which is not installed (pip install keyring)"
            ) from None
        value = keyring.get_password("harness", name)
        if value is None:
            raise SecretResolutionError(
                f"secret reference {reference!r}: no keychain entry for "
                f"service 'harness', name {name!r}"
            )
        return value
    warnings.warn(
        "api_key looks like a literal secret; prefer an 'env:VAR' or "
        "'keychain:name' reference so the value stays out of config.toml",
        UserWarning,
        stacklevel=2,
    )
    return reference


class ModelConfig(BaseModel):
    """One entry in the model registry (DESIGN.md §4.2).

    ``adapter`` names the adapter implementation (e.g. ``"anthropic"``,
    ``"openai"``); ``model`` is the provider's model id; ``base_url``
    optionally points an OpenAI-compatible adapter at any compatible
    endpoint. ``api_key`` holds a secret *reference* (or literal) wrapped in
    ``SecretStr`` so it never leaks through ``repr``/``str``; call
    :meth:`resolve_api_key` to obtain the real key.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    adapter: str
    model: str
    base_url: str | None = None
    api_key: SecretStr | None = None

    def resolve_api_key(self) -> str | None:
        """Resolve the stored reference to the actual API key, if any."""
        if self.api_key is None:
            return None
        return resolve_secret(self.api_key.get_secret_value())


class SandboxConfig(BaseModel):
    """Docker sandbox settings (DESIGN.md §4.7)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    network: NetworkMode = NetworkMode.NONE
    image: str = "harness-sandbox:latest"


class HarnessConfig(BaseModel):
    """Top-level harness configuration.

    Built by :func:`load_config`; every field has a sensible default so an
    empty or missing config file still produces a working (if model-less)
    configuration.
    """

    model_config = ConfigDict(frozen=True)

    home: Path = Field(default_factory=harness_home)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    permission_mode: PermissionMode = PermissionMode.GATED
    #: Per-tool/per-pattern policy overrides from ``[permissions]`` (§4.11):
    #: fnmatch globs threaded into every run's :class:`~harness.permissions.Policy`.
    permission_allow: tuple[str, ...] = ()
    permission_deny: tuple[str, ...] = ()


def _describe_validation_error(exc: ValidationError) -> str:
    """Render a pydantic ``ValidationError`` without echoing input values.

    ``str(ValidationError)`` embeds the raw ``input_value`` for each error,
    which for a malformed ``[models.*]`` entry can include a plaintext
    ``api_key``. This renders only the field locations and messages so the
    module's no-secrets-in-strings guarantee holds on the error path too.
    """
    parts = []
    for err in exc.errors(include_url=False, include_input=False):
        loc = ".".join(str(item) for item in err["loc"]) or "(top level)"
        parts.append(f"{loc}: {err['msg']}")
    plural = "s" if exc.error_count() != 1 else ""
    return (
        f"{exc.error_count()} validation error{plural} for {exc.title}: "
        + "; ".join(parts)
    )


def load_config(path: str | Path | None = None) -> HarnessConfig:
    """Load harness configuration from TOML.

    ``path``, if given, must point at the config file (it is an error for an
    explicitly named file to be missing). Otherwise
    ``$HARNESS_HOME/config.toml`` is used, and a missing file silently yields
    defaults. Raises :class:`ConfigError` on malformed TOML or values that
    fail validation.
    """
    home = harness_home()
    if path is not None:
        config_path = Path(path).expanduser()
        if not config_path.is_file():
            raise ConfigError(f"config file not found: {config_path}")
    else:
        config_path = home / "config.toml"
        if not config_path.is_file():
            return HarnessConfig(home=home)

    try:
        with config_path.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc

    for section in ("models", "sandbox", "permissions"):
        value = data.get(section, {})
        if not isinstance(value, dict):
            raise ConfigError(
                f"invalid config in {config_path}: [{section}] must be a "
                f"table, got {type(value).__name__}"
            )
    for name, entry in data.get("models", {}).items():
        if not isinstance(entry, dict):
            raise ConfigError(
                f"invalid config in {config_path}: [models.{name}] must be a "
                f"table, got {type(entry).__name__}"
            )

    permissions = data.get("permissions", {})
    for key in ("allow", "deny"):
        patterns = permissions.get(key, [])
        if not isinstance(patterns, list) or not all(
            isinstance(item, str) for item in patterns
        ):
            raise ConfigError(
                f"invalid config in {config_path}: [permissions] {key!r} "
                "must be a list of strings (fnmatch tool-name patterns)"
            )

    try:
        return HarnessConfig(
            home=home,
            models={
                name: ModelConfig(**entry)
                for name, entry in data.get("models", {}).items()
            },
            sandbox=SandboxConfig(**data.get("sandbox", {})),
            permission_mode=PermissionMode(
                permissions.get("default", PermissionMode.GATED)
            ),
            permission_allow=tuple(permissions.get("allow", [])),
            permission_deny=tuple(permissions.get("deny", [])),
        )
    except ValidationError as exc:
        # ``from None``: chaining the original ValidationError would put its
        # unredacted str() (raw input values, possibly a literal api_key)
        # back into tracebacks and logs.
        raise ConfigError(
            f"invalid config in {config_path}: {_describe_validation_error(exc)}"
        ) from None
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid config in {config_path}: {exc}") from exc
