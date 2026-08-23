"""S-002: N3 (zero setup cost) and N4 (filesystem neutrality).

Both need a real ``CODING`` run, so they live apart from the static
invariants. They are still fast: the model is a scripted FakeAdapter and the
sandbox is LocalSandbox on tmp_path.

N3 is the invariant that makes S-005's SetupBudget structurally dead code on
the benchmark path, and N4 generalises the SPILL_DIR lesson -- a run that
writes into the workspace to do its own bookkeeping corrupts the very thing
the grader inspects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from harness.adapters.fake import FakeAdapter
from harness.config import HarnessConfig
from harness.orchestrator import Orchestrator
from harness.persistence import RunStore
from harness.sandbox.base import ExecResult, SandboxPathError
from harness.sandbox.docker import DockerSandbox
from harness.sandbox.local import LocalSandbox
from harness.types import Message, ModelResponse, Role, StopReason, ToolCall, Usage
from tests.conformance.fixture import tree_hash

pytestmark = pytest.mark.filterwarnings("ignore:no Docker daemon")

GOAL = "Do nothing; the workspace is already correct."
CLEAN_FINISH = "Task complete. Nothing needed changing; verified the tree is as required."


@pytest.fixture(autouse=True)
def no_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the LocalSandbox fallback: conformance must not need Docker."""
    monkeypatch.setattr(DockerSandbox, "availability", classmethod(lambda cls: False))


def _write_script() -> list[ModelResponse]:
    """Write one file, then finish -- used to prove the recorder records."""
    return [
        ModelResponse(
            message=Message(
                role=Role.ASSISTANT,
                content=None,
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


def _noop_script() -> list[ModelResponse]:
    """One turn, no tool calls, a finish the diligence gate accepts."""
    return [
        ModelResponse(
            message=Message(role=Role.ASSISTANT, content=CLEAN_FINISH),
            usage=Usage(),
            stop_reason=StopReason.END_TURN,
        )
    ]


class _CountingAdapter(FakeAdapter):
    """Records the sandbox's exec count at the moment of the first model call.

    N3 asks a question about ordering -- did anything execute *before* the
    model was first consulted -- which cannot be answered by counting at the
    end of the run.
    """

    def __init__(self, script: list[ModelResponse], sandbox_holder: dict) -> None:
        super().__init__(script)
        self._holder = sandbox_holder
        self.exec_count_at_first_call: int | None = None

    async def complete(self, *args: Any, **kwargs: Any) -> Any:
        if self.exec_count_at_first_call is None:
            sandbox = self._holder.get("sandbox")
            self.exec_count_at_first_call = (
                getattr(sandbox, "exec_count", 0) if sandbox else 0
            )
        return await super().complete(*args, **kwargs)


def _instrument_local_sandbox(monkeypatch: pytest.MonkeyPatch, holder: dict) -> None:
    """Count execs and record every write path on whatever LocalSandbox runs."""
    real_init = LocalSandbox.__init__
    real_exec = LocalSandbox.exec
    real_write = LocalSandbox.write_file

    def init(self: LocalSandbox, *args: Any, **kwargs: Any) -> None:
        real_init(self, *args, **kwargs)
        self.exec_count = 0
        self.written_paths = []
        holder["sandbox"] = self

    async def exec_(self: LocalSandbox, command: str, timeout: float = 120) -> ExecResult:
        self.exec_count += 1
        return await real_exec(self, command, timeout)

    async def write_file(self: LocalSandbox, path: str, content: str, **kwargs: Any) -> None:
        self.written_paths.append(path)
        return await real_write(self, path, content, **kwargs)

    monkeypatch.setattr(LocalSandbox, "__init__", init)
    monkeypatch.setattr(LocalSandbox, "exec", exec_)
    monkeypatch.setattr(LocalSandbox, "write_file", write_file)


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An orchestrator whose sandbox is instrumented, plus the holder."""
    holder: dict = {}
    _instrument_local_sandbox(monkeypatch, holder)
    home = tmp_path / "home"
    home.mkdir()
    store = RunStore(tmp_path / "state.db")
    return Orchestrator(HarnessConfig(home=home), store), holder, home


class TestN3ZeroSetupCost:
    async def test_S002_n3_no_exec_before_the_first_model_call(self, harness) -> None:
        orchestrator, holder, _ = harness
        adapter = _CountingAdapter(_noop_script(), holder)
        await orchestrator.run_task(GOAL, "fake-model", adapter_override=adapter)
        assert adapter.exec_count_at_first_call == 0, (
            "a CODING run executed in the sandbox before consulting the model; "
            "every setup second is subtracted from work under a wall clock"
        )

    async def test_S002_n3_the_counter_actually_counts(self, harness) -> None:
        # Negative control: if the instrumentation were inert, N3 would pass by
        # not looking. Prove the counter moves when an exec really happens.
        orchestrator, holder, _ = harness
        adapter = _CountingAdapter(_noop_script(), holder)
        await orchestrator.run_task(GOAL, "fake-model", adapter_override=adapter)
        sandbox = holder["sandbox"]
        before = sandbox.exec_count
        await sandbox.exec("true")
        assert sandbox.exec_count == before + 1


class TestN4FilesystemNeutrality:
    """A ``CODING`` run leaves the workspace byte-identical.

    The baseline must be taken on a **pristine** workspace. Taking it after a
    run has already executed bakes that run's residue into the baseline, so a
    second identical run reproduces it and the check passes -- which is how
    the first version of this test managed to accept a run that wrote a
    bookkeeping file on every startup. Residue in the real world is
    deterministic (`.git`, a staged binary, a cache index); a check that only
    catches *non-deterministic* residue catches the case that does not happen.
    """

    async def test_S002_n4_pristine_workspace_unchanged(
        self, harness, tmp_path: Path
    ) -> None:
        orchestrator, holder, _ = harness
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "seed.txt").write_text("seed", encoding="utf-8")
        before = tree_hash(workspace)
        await orchestrator.run_task(
            GOAL,
            "fake-model",
            adapter_override=_CountingAdapter(_noop_script(), holder),
            workspace=workspace,
        )
        assert tree_hash(workspace) == before, (
            "a no-op CODING run modified the workspace tree; the grader "
            "inspects exactly this tree"
        )

    async def test_S002_n4_detects_residue_written_during_the_run(
        self, harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Negative: residue with a FIXED name and FIXED content, written by the
        # run itself. This is the shape every realistic mechanism has (.git, a
        # staged binary, a cache index) and the shape the first version of this
        # test accepted, because it baselined an already-polluted workspace.
        orchestrator, holder, _ = harness
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "seed.txt").write_text("seed", encoding="utf-8")
        before = tree_hash(workspace)

        real_start = LocalSandbox.start

        async def leaky_start(self: LocalSandbox) -> None:
            await real_start(self)
            await LocalSandbox.write_file(self, ".harness-bookkeeping", "state")

        monkeypatch.setattr(LocalSandbox, "start", leaky_start)
        await orchestrator.run_task(
            GOAL,
            "fake-model",
            adapter_override=_CountingAdapter(_noop_script(), holder),
            workspace=workspace,
        )
        assert tree_hash(workspace) != before, (
            "N4 accepted a run that wrote a fixed-name file into the workspace"
        )

    async def test_S002_n4_write_recorder_is_not_vacuous(
        self, harness, tmp_path: Path
    ) -> None:
        # The previous version asserted a predicate over `written_paths` that
        # was empty on every run, so the loop body never executed. Prove the
        # recorder actually records before relying on it.
        orchestrator, holder, _ = harness
        workspace = tmp_path / "ws"
        workspace.mkdir()
        await orchestrator.run_task(
            GOAL,
            "fake-model",
            adapter_override=_CountingAdapter(_write_script(), holder),
            workspace=workspace,
        )
        assert holder["sandbox"].written_paths, (
            "the write recorder saw nothing even though the run wrote a file"
        )
        assert (workspace / "hello.txt").exists()

    async def test_S002_n4_harness_home_gains_only_known_files(
        self, harness, tmp_path: Path
    ) -> None:
        # `home` is the outside-the-workspace location N4's wording mentions.
        # It was previously returned by the fixture and discarded by every
        # consumer -- a check drafted and dropped.
        orchestrator, holder, home = harness
        workspace = tmp_path / "ws"
        workspace.mkdir()
        await orchestrator.run_task(
            GOAL,
            "fake-model",
            adapter_override=_CountingAdapter(_noop_script(), holder),
            workspace=workspace,
        )
        created = {p.name for p in home.rglob("*") if p.is_file()}
        unexpected = {
            name
            for name in created
            if not name.startswith("state.db") and name not in {"INDEX.md"}
        }
        assert not unexpected, (
            f"a CODING run created unexpected files under HARNESS_HOME: "
            f"{sorted(unexpected)}"
        )

    def test_S002_n4_tree_hash_detects_a_change(self, tmp_path: Path) -> None:
        # Control for the hash primitive itself.
        root = tmp_path / "t"
        root.mkdir()
        (root / "a.txt").write_text("a", encoding="utf-8")
        before = tree_hash(root)
        (root / "a.txt").write_text("b", encoding="utf-8")
        assert tree_hash(root) != before
        (root / "a.txt").write_text("a", encoding="utf-8")
        assert tree_hash(root) == before
        (root / "new.txt").write_text("", encoding="utf-8")
        assert tree_hash(root) != before


class TestN4ScopeIsStated:
    """What N4 does and does not enforce, asserted rather than implied.

    N4 pins the workspace tree. It does **not** police writes outside the
    workspace: ``LocalSandbox.write_file`` routes through
    ``resolve_workspace_path``, which raises on any path escaping the root, so
    an assertion that recorded write paths are relative is a tautology
    enforced by the code it claims to check. And the one deliberate
    outside-workspace write -- the spill path -- goes through ``exec``, not
    ``write_file``, so a write recorder cannot see it at all.
    """

    async def test_S002_n4_absolute_write_is_structurally_refused(
        self, harness, tmp_path: Path
    ) -> None:
        orchestrator, holder, _ = harness
        workspace = tmp_path / "ws"
        workspace.mkdir()
        await orchestrator.run_task(
            GOAL,
            "fake-model",
            adapter_override=_CountingAdapter(_noop_script(), holder),
            workspace=workspace,
        )
        with pytest.raises(SandboxPathError):
            await LocalSandbox.write_file(
                holder["sandbox"], "/etc/passwd", "nope"
            )


class TestN3ScopeAtTheBenchmarkEntryPoint:
    """N3's window, stated honestly, and the one exec inside it pinned.

    ``TestN3ZeroSetupCost`` measures execs issued *by the orchestrator*, on
    ``LocalSandbox``, between ``run_task`` and the first model call. A
    Terminal-Bench run enters through
    ``harness.integrations.harbor_agent``, which constructs a
    ``HarborSandbox`` and awaits ``start()`` **before** the orchestrator
    exists -- and ``HarborSandbox.start`` execs ``pwd`` to detect the
    workspace root.

    So the unqualified claim "a CODING run performs zero sandbox execs before
    the first model call" is false on the benchmark path, by exactly one exec.
    Rather than leave the invariant asserting something untrue of the path it
    exists to protect, that exec is pinned here: it is allowed to be one, and
    a second would fail. Growth is what N3 is really guarding against -- a
    probe, an index, a baseline -- and growth is caught either side of the
    boundary.
    """

    async def test_S002_n3_harbor_start_execs_exactly_once(self) -> None:
        from harness.sandbox.harbor_env import HarborSandbox

        calls: list[str] = []

        class _FakeEnvironment:
            async def exec(self, command: str, **kwargs: Any) -> Any:
                calls.append(command)

                class _R:
                    return_code = 0
                    stdout = "/app"
                    stderr = ""

                return _R()

        sandbox = HarborSandbox(_FakeEnvironment())
        await sandbox.start()
        assert len(calls) == 1, (
            f"HarborSandbox.start issued {len(calls)} execs ({calls}); N3's "
            "budget for the benchmark entry point is exactly one (the `pwd` "
            "workspace probe). Any addition is setup cost subtracted from "
            "work under a fixed wall clock."
        )
        assert calls == ["pwd"], f"unexpected startup command: {calls}"

    async def test_S002_n3_harbor_start_is_idempotent(self) -> None:
        # A second start() must not re-probe: the orchestrator and the Harbor
        # adapter both call it on some paths.
        from harness.sandbox.harbor_env import HarborSandbox

        calls: list[str] = []

        class _FakeEnvironment:
            async def exec(self, command: str, **kwargs: Any) -> Any:
                calls.append(command)

                class _R:
                    return_code = 0
                    stdout = "/app"
                    stderr = ""

                return _R()

        sandbox = HarborSandbox(_FakeEnvironment())
        await sandbox.start()
        await sandbox.start()
        assert len(calls) == 1, f"start() re-probed on the second call: {calls}"
