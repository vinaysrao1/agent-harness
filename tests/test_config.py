"""Unit tests for harness.config."""

import sys
import traceback
from pathlib import Path

import pytest

from harness.config import (
    ConfigError,
    HarnessConfig,
    ModelConfig,
    NetworkMode,
    PermissionMode,
    SandboxConfig,
    SecretResolutionError,
    harness_home,
    load_config,
    resolve_secret,
)

FULL_TOML = """
[models.opus]
adapter = "anthropic"
model   = "claude-opus-4-8"
api_key = "env:ANTHROPIC_API_KEY"

[models.kimi]
adapter  = "openai"
base_url = "https://api.moonshot.ai/v1"
model    = "kimi-k3"
api_key  = "keychain:moonshot"

[sandbox]
network = "allowlist"
image   = "custom-sandbox:v2"

[permissions]
default = "auto"
allow   = ["mcp.github.*", "bash"]
deny    = ["mcp.slack.send_*"]
"""


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Point HARNESS_HOME at a temp dir and return it."""
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path))
    return tmp_path


class TestHarnessHome:
    def test_env_override(self, home):
        assert harness_home() == home

    def test_default_is_dot_harness(self, monkeypatch):
        monkeypatch.delenv("HARNESS_HOME", raising=False)
        assert harness_home() == Path("~/.harness").expanduser()


class TestResolveSecret:
    def test_env_reference(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "sk-secret-123")
        assert resolve_secret("env:MY_KEY") == "sk-secret-123"

    def test_env_reference_missing(self, monkeypatch):
        monkeypatch.delenv("NOPE_VAR", raising=False)
        with pytest.raises(SecretResolutionError, match="NOPE_VAR"):
            resolve_secret("env:NOPE_VAR")

    def test_literal_returns_value_with_warning(self):
        with pytest.warns(UserWarning, match="literal"):
            assert resolve_secret("sk-raw-literal") == "sk-raw-literal"

    def test_keychain_without_keyring_is_clear_error(self, monkeypatch):
        # Force the import inside resolve_secret to fail even if keyring
        # happens to be installed.
        monkeypatch.setitem(sys.modules, "keyring", None)
        with pytest.raises(SecretResolutionError, match="keyring"):
            resolve_secret("keychain:anthropic")

    def test_keychain_with_fake_keyring(self, monkeypatch):
        class FakeKeyring:
            @staticmethod
            def get_password(service, name):
                assert (service, name) == ("harness", "anthropic")
                return "kc-secret"

        monkeypatch.setitem(sys.modules, "keyring", FakeKeyring())
        assert resolve_secret("keychain:anthropic") == "kc-secret"

    def test_keychain_entry_missing(self, monkeypatch):
        class EmptyKeyring:
            @staticmethod
            def get_password(service, name):
                return None

        monkeypatch.setitem(sys.modules, "keyring", EmptyKeyring())
        with pytest.raises(SecretResolutionError, match="no keychain entry"):
            resolve_secret("keychain:missing")


class TestLoadConfig:
    def test_missing_file_yields_defaults(self, home):
        cfg = load_config()
        assert cfg.home == home
        assert cfg.models == {}
        assert cfg.sandbox == SandboxConfig()
        assert cfg.sandbox.network is NetworkMode.NONE
        assert cfg.sandbox.image == "harness-sandbox:latest"
        assert cfg.permission_mode is PermissionMode.GATED

    def test_full_config_parses(self, home):
        (home / "config.toml").write_text(FULL_TOML)
        cfg = load_config()
        assert set(cfg.models) == {"opus", "kimi"}
        opus = cfg.models["opus"]
        assert opus.adapter == "anthropic"
        assert opus.model == "claude-opus-4-8"
        assert opus.base_url is None
        kimi = cfg.models["kimi"]
        assert kimi.base_url == "https://api.moonshot.ai/v1"
        assert cfg.sandbox.network is NetworkMode.ALLOWLIST
        assert cfg.sandbox.image == "custom-sandbox:v2"
        assert cfg.permission_mode is PermissionMode.AUTO
        assert cfg.permission_allow == ("mcp.github.*", "bash")
        assert cfg.permission_deny == ("mcp.slack.send_*",)

    def test_permission_patterns_default_empty(self, home):
        """Regression (§4.11): [permissions] allow/deny are parsed; when
        absent they default to empty tuples."""
        (home / "config.toml").write_text("[permissions]\ndefault = 'gated'\n")
        cfg = load_config()
        assert cfg.permission_allow == ()
        assert cfg.permission_deny == ()

    def test_permission_patterns_must_be_string_lists(self, home):
        (home / "config.toml").write_text("[permissions]\nallow = 'bash'\n")
        with pytest.raises(ConfigError, match="list of strings"):
            load_config()
        (home / "config.toml").write_text("[permissions]\ndeny = [1, 2]\n")
        with pytest.raises(ConfigError, match="list of strings"):
            load_config()

    def test_explicit_path_wins_over_home(self, home, tmp_path_factory):
        (home / "config.toml").write_text("[permissions]\ndefault = 'auto'\n")
        other = tmp_path_factory.mktemp("elsewhere") / "cfg.toml"
        other.write_text("[permissions]\ndefault = 'gated'\n")
        assert load_config(other).permission_mode is PermissionMode.GATED

    def test_explicit_missing_path_is_error(self, home):
        with pytest.raises(ConfigError, match="not found"):
            load_config(home / "does-not-exist.toml")

    def test_malformed_toml_is_config_error(self, home):
        (home / "config.toml").write_text("this is [not toml")
        with pytest.raises(ConfigError, match="invalid TOML"):
            load_config()

    def test_invalid_network_mode_is_config_error(self, home):
        (home / "config.toml").write_text("[sandbox]\nnetwork = 'wide-open'\n")
        with pytest.raises(ConfigError, match="invalid config"):
            load_config()

    def test_invalid_permission_mode_is_config_error(self, home):
        (home / "config.toml").write_text("[permissions]\ndefault = 'yolo'\n")
        with pytest.raises(ConfigError, match="invalid config"):
            load_config()

    def test_model_missing_required_field_is_config_error(self, home):
        (home / "config.toml").write_text("[models.bad]\nadapter = 'openai'\n")
        with pytest.raises(ConfigError, match="invalid config"):
            load_config()


class TestSecretsNeverLeak:
    def make_config(self, home):
        (home / "config.toml").write_text(FULL_TOML)
        return load_config()

    def test_api_key_absent_from_model_repr_and_str(self, home):
        cfg = self.make_config(home)
        for text in (repr(cfg.models["opus"]), str(cfg.models["opus"])):
            assert "ANTHROPIC_API_KEY" not in text
            assert "env:" not in text

    def test_api_key_absent_from_top_level_repr_and_str(self, home):
        cfg = self.make_config(home)
        for text in (repr(cfg), str(cfg)):
            assert "ANTHROPIC_API_KEY" not in text
            assert "keychain:moonshot" not in text
            assert "**********" in text  # SecretStr masking in effect

    def test_literal_key_masked(self):
        mc = ModelConfig(adapter="openai", model="m", api_key="sk-super-secret")
        assert "sk-super-secret" not in repr(mc)
        assert "sk-super-secret" not in str(mc)

    def test_resolve_api_key_via_env(self, monkeypatch):
        monkeypatch.setenv("K_VAR", "the-real-key")
        mc = ModelConfig(adapter="openai", model="m", api_key="env:K_VAR")
        assert mc.resolve_api_key() == "the-real-key"

    def test_resolve_api_key_none(self):
        assert ModelConfig(adapter="openai", model="m").resolve_api_key() is None

    def test_literal_key_absent_from_validation_error(self, home):
        # Regression: pydantic's ValidationError embeds the raw input dict
        # (including a plaintext api_key) in its str(); load_config must not
        # let that reach the ConfigError message or its exception chain.
        (home / "config.toml").write_text(
            "[models.bad]\napi_key = 'sk-ant-xyz123'\n"
        )
        with pytest.raises(ConfigError) as excinfo:
            load_config()
        assert "sk-ant-xyz123" not in str(excinfo.value)
        # The chained/context exceptions would also be printed in tracebacks.
        chain = "".join(
            traceback.format_exception(
                type(excinfo.value), excinfo.value, excinfo.value.__traceback__
            )
        )
        assert "sk-ant-xyz123" not in chain
        # The message should still be actionable.
        assert "adapter" in str(excinfo.value)
        assert "model" in str(excinfo.value)

    def test_literal_key_absent_when_field_has_wrong_type(self, home):
        (home / "config.toml").write_text(
            "[models.bad]\nadapter = 'openai'\nmodel = 7\n"
            "api_key = 'sk-ant-xyz123'\n"
        )
        with pytest.raises(ConfigError) as excinfo:
            load_config()
        assert "sk-ant-xyz123" not in str(excinfo.value)


class TestHarnessConfigDefaults:
    def test_bare_construction_is_usable(self, home):
        cfg = HarnessConfig()
        assert cfg.home == home
        assert cfg.models == {}
        assert cfg.permission_mode is PermissionMode.GATED


class TestMinorFindingRegressions:
    """Regressions for stage-1 review minor findings (fixed inline)."""

    def test_wrong_shaped_section_raises_config_error(self, tmp_path):
        from harness.config import ConfigError, load_config
        import pytest

        for body in ('models = "hi"', 'permissions = "x"', "models = [1, 2]"):
            cfg = tmp_path / f"c{abs(hash(body))}.toml"
            cfg.write_text(body)
            with pytest.raises(ConfigError):
                load_config(cfg)

    def test_wrong_shaped_model_entry_raises_config_error(self, tmp_path):
        from harness.config import ConfigError, load_config
        import pytest

        cfg = tmp_path / "c.toml"
        cfg.write_text('[models]\nopus = "not-a-table"')
        with pytest.raises(ConfigError):
            load_config(cfg)

    def test_unknown_key_rejected(self, tmp_path):
        from harness.config import ConfigError, load_config
        import pytest

        cfg = tmp_path / "c.toml"
        cfg.write_text(
            '[models.m]\nadapter = "openai"\nmodel = "x"\nbaseurl = "typo"\n'
        )
        with pytest.raises(ConfigError):
            load_config(cfg)

    def test_bare_sum_of_usage(self):
        from harness.types import Usage

        total = sum([Usage(input_tokens=1), Usage(input_tokens=2)])
        assert total.input_tokens == 3
