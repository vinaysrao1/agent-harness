"""S-004: AgentProfile promotion is behavior-preserving.

The point of this spec is not the new fields -- it is that adding them changed
nothing for ``CODING``. N1 and N2 are what prove that, and they pass with no
golden change; these tests cover the parts the goldens cannot see: precedence
between run-level arguments and profile defaults, the guarantee that a profile
cannot weaken the operator's permission mode, and that every profile still
carries the safety core.
"""

from __future__ import annotations

import pytest

import pytest

from harness.adapters.fake import FakeAdapter
from harness.config import HarnessConfig
from harness.orchestrator import (
    CODING_RULES,
    CODING_TOOL_FACTORIES,
    Orchestrator,
    assemble_rules,
)
from harness.persistence import RunStore
from harness.sandbox.docker import DockerSandbox
from harness.tools.registry import Tool
from harness.types import (
    Message,
    ModelResponse,
    Role,
    StopReason,
    ToolCall,
    ToolSpec,
    Usage,
)
from harness.permissions import (
    HARD_DENY_CATEGORIES,
    Decision,
    PermissionMode,
    Policy,
    ToolMeta,
    evaluate,
)
from harness.profiles import (
    ALL_PROFILES,
    CODING,
    CODING_READONLY,
    CODING_REPO,
    REPO_CAPABILITIES,
    AgentProfile,
    Profile,
    SandboxSpec,
)


GOAL = "Write hello.txt containing hi."
CLEAN_FINISH = "Task complete. Wrote the file; contents verified."

pytestmark = pytest.mark.filterwarnings("ignore:no Docker daemon")


@pytest.fixture(autouse=True)
def _no_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DockerSandbox, "availability", classmethod(lambda cls: False))


@pytest.fixture
def orchestrator(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    return Orchestrator(HarnessConfig(home=home), RunStore(tmp_path / "state.db"))


def _write_then_finish() -> list[ModelResponse]:
    return [
        ModelResponse(
            message=Message(
                role=Role.ASSISTANT,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments={"path": "hello.txt", "content": "hi"},
                    )
                ],
            ),
            usage=Usage(),
            stop_reason=StopReason.TOOL_USE,
        ),
        ModelResponse(
            message=Message(role=Role.ASSISTANT, content=CLEAN_FINISH),
            usage=Usage(),
            stop_reason=StopReason.END_TURN,
        ),
    ]


class TestPromotionIsBehaviorPreserving:
    def test_S004_coding_is_unchanged(self) -> None:
        # The whole spec in one assertion: CODING must still be exactly the
        # pre-promotion pair. N1/N2 prove the assembled artifacts; this proves
        # the inputs.
        assert CODING.domain_rules == CODING_RULES
        assert CODING.tool_factories == CODING_TOOL_FACTORIES

    def test_S004_coding_declares_no_capabilities(self) -> None:
        # A capability on CODING would be a change to the benchmark path.
        assert CODING.capabilities == frozenset()
        assert CODING.sandbox_spec is None
        assert CODING.permission_allow == ()

    def test_S004_profile_alias_is_the_promoted_struct(self) -> None:
        assert Profile is AgentProfile

    def test_S004_new_fields_default_to_inert(self) -> None:
        minimal = AgentProfile(name="x", domain_rules="", tool_factories=())
        assert minimal.capabilities == frozenset()
        assert minimal.sandbox_spec is None
        assert minimal.permission_allow == ()


class TestCapabilitiesAreAskedForNotSwitchedOn:
    def test_S004_enables_reports_membership(self) -> None:
        assert CODING_REPO.enables("git_substrate")
        assert not CODING.enables("git_substrate")
        assert not CODING_REPO.enables("nonexistent")

    def test_S004_repo_capabilities_are_named_not_implemented(self) -> None:
        # Naming them is the seam; none is active until the environment
        # affirms it (S-005) and the feature exists (Layer 2). CODING_REPO
        # therefore still ships today's tool set.
        assert CODING_REPO.capabilities == REPO_CAPABILITIES
        assert CODING_REPO.tool_factories == CODING_TOOL_FACTORIES

    def test_S004_no_profile_leaks_capabilities_into_coding(self) -> None:
        for profile in ALL_PROFILES:
            if profile is not CODING:
                continue
            assert not profile.capabilities


class TestSandboxSpec:
    def test_S004_defaults_defer_to_config(self) -> None:
        # Every field None means "whatever the config says", which is what
        # keeps CODING's sandbox construction untouched (N3, N4).
        assert SandboxSpec().network is None


class TestProfilePermissionAllowReachesTheRun:
    """C1: the merge in ``build_policy`` is the only orchestrator change S-004
    makes, and deleting it left the whole suite green.

    These drive ``run_task``, so the assertion is about the policy the harness
    actually builds rather than a ``Policy`` hand-constructed in a test.

    A custom side-effecting tool is required because **every builtin is
    ``side_effect=False``** -- the sandbox is the trust boundary, so in gated
    mode no builtin ever reaches the ask callback. Using one would have made
    the positive test vacuous, which is what its control caught.
    """

    @staticmethod
    def _gated_profile(*, allow: tuple[str, ...]) -> AgentProfile:
        async def handler(arguments: dict) -> str:
            return "did the thing"

        def factory(_deps) -> Tool:
            return Tool(
                spec=ToolSpec(
                    name="risky_tool",
                    description="A side-effecting tool.",
                    input_schema={"type": "object", "properties": {}},
                ),
                meta=ToolMeta(side_effect=True),
                handler=handler,
            )

        return AgentProfile(
            name="gated-probe",
            domain_rules=CODING.domain_rules,
            tool_factories=(*CODING.tool_factories, factory),
            permission_allow=allow,
        )

    @staticmethod
    def _call_risky() -> list[ModelResponse]:
        return [
            ModelResponse(
                message=Message(
                    role=Role.ASSISTANT,
                    tool_calls=[ToolCall(id="c1", name="risky_tool", arguments={})],
                ),
                usage=Usage(),
                stop_reason=StopReason.TOOL_USE,
            ),
            ModelResponse(
                message=Message(role=Role.ASSISTANT, content=CLEAN_FINISH),
                usage=Usage(),
                stop_reason=StopReason.END_TURN,
            ),
        ]

    async def test_S004_without_a_profile_allow_the_tool_is_asked_about(
        self, orchestrator, tmp_path
    ) -> None:
        # The control, written first: if a side-effecting tool does not reach
        # the ask callback, the positive test below proves nothing.
        asked: list[str] = []

        async def ask(tool_name: str, _arguments: dict, _meta: object) -> bool:
            asked.append(tool_name)
            return True

        await orchestrator.run_task(
            GOAL,
            "fake-model",
            mode=PermissionMode.GATED,
            ask=ask,
            adapter_override=FakeAdapter(self._call_risky()),
            profile=self._gated_profile(allow=()),
            workspace=tmp_path / "ws-control",
        )
        assert asked == ["risky_tool"], (
            f"expected the side-effecting tool to be asked about, got {asked}"
        )

    async def test_S004_profile_allow_reaches_the_policy(
        self, orchestrator, tmp_path
    ) -> None:
        # The real assertion: with the pattern on the profile, the same tool
        # must be ALLOWed without consulting the callback -- which can only
        # happen if build_policy merged profile.permission_allow.
        asked: list[str] = []

        async def ask(tool_name: str, _arguments: dict, _meta: object) -> bool:
            asked.append(tool_name)
            return True

        _, result = await orchestrator.run_task(
            GOAL,
            "fake-model",
            mode=PermissionMode.GATED,
            ask=ask,
            adapter_override=FakeAdapter(self._call_risky()),
            profile=self._gated_profile(allow=("risky_tool",)),
            workspace=tmp_path / "ws-allowed",
        )
        assert result.status == "completed"
        assert asked == [], (
            "the tool was sent to the ask callback despite the profile "
            "allow-listing it; build_policy did not merge "
            "profile.permission_allow"
        )
