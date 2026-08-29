"""Running graded commands, in the environment they will be graded in (S-401).

Two things live here because validation and trial-grading must share them, and
sharing them is the whole point.

**Grading runs in a sandbox.** The agent's commands run in one; grading used to
run ``subprocess.run(shell=True)`` in the harness process. That undoes the
isolation completely — a ``conftest.py`` the agent added is arbitrary code and
pytest imports it before collection, so choosing Docker to keep model-authored
code off the host and then grading on the host executes exactly the code the
container existed to contain.

**"The command could not run" is not "the tests failed."** Every exec failure
used to collapse to exit code 1. A grading timeout scored as a model failure; a
timeout on the *regression* command scored as ``regressions=1`` — a fabricated
regression, invented by the harness and attributed to the agent. :class:`Graded`
carries ``ran`` so the caller can tell an observation from an absence of one.
"""

from __future__ import annotations

import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

from harness.config import HarnessConfig
from harness.sandbox.base import Sandbox
from harness.sandbox.docker import DockerSandbox
from harness.sandbox.local import LocalSandbox

__all__ = [
    "Graded",
    "Grader",
    "CacheWipeError",
    "sandboxed_runner",
    "wipe_caches",
    "CACHE_DIRECTORIES",
]

#: Removed immediately before every graded command. A ``.pyc`` is validated
#: against its source's **mtime and size**, not its content, so a test file can
#: be gutted, executed to leave a passing cache, and then restored byte-for-byte
#: with its original timestamp — the content hash sees nothing, and the grader
#: runs the cached bytecode. Deleting the caches costs a recompile and closes
#: it.
#:
#: ``PYTHONDONTWRITEBYTECODE`` was here too and is not, because it does not
#: help: it stops Python *writing* bytecode and has no effect on *reading* a
#: cache that already exists, which is the entire attack. Keeping it would have
#: dressed the real defence in a second one that does nothing, and it mutates
#: the command the operator configured.
CACHE_DIRECTORIES = ("__pycache__", ".pytest_cache")


class CacheWipeError(RuntimeError):
    """A compiled-artifact cache could not be removed before grading."""


def wipe_caches(root: Path) -> None:
    """Delete every compiled-artifact cache under ``root``.

    Raises rather than swallowing. This is the *only* thing standing between a
    planted ``.pyc`` and a green grade, so a wipe that silently failed would
    leave the defence looking present and doing nothing — the failure shape
    this project keeps rediscovering.
    """
    failures: list[str] = []
    for name in CACHE_DIRECTORIES:
        for path in root.rglob(name):
            if not path.is_dir():
                continue
            try:
                shutil.rmtree(path)
            except OSError as exc:
                failures.append(f"{path}: {exc}")
    if failures:
        raise CacheWipeError(
            "could not clear compiled-artifact caches before grading, so a "
            "stale cache could decide the result: " + "; ".join(failures)
        )


@dataclass(frozen=True)
class Graded:
    """The outcome of one graded command."""

    exit_code: int
    output: str
    #: Whether the command actually executed to completion. False for a
    #: sandbox failure or a timeout — in which case ``exit_code`` says nothing
    #: about the code under test and must not be read as a verdict.
    ran: bool = True

    @property
    def passed(self) -> bool:
        return self.ran and self.exit_code == 0

    @property
    def failed(self) -> bool:
        """Ran, and the code under test did not satisfy it."""
        return self.ran and self.exit_code != 0


class Grader:
    """Runs graded commands in a sandbox over one directory.

    One sandbox, reused for every command against that directory. A trial uses
    up to three — the regression baseline before the agent runs, then the test
    command and the regression command after — and starting a container per
    command would triple the cost of a suite.
    """

    def __init__(self, directory: Path, config: HarnessConfig) -> None:
        self._directory = directory
        self._config = config
        self._sandbox: Sandbox | None = None

    async def __aenter__(self) -> "Grader":
        if DockerSandbox.availability():
            self._sandbox = DockerSandbox(
                self._directory,
                image=self._config.sandbox.image,
                network=self._config.sandbox.network,
            )
        else:
            warnings.warn(
                "no Docker daemon available; graded commands will run "
                "directly on the host with NO isolation, and a task tree can "
                "contain code the agent wrote",
                UserWarning,
                stacklevel=2,
            )
            self._sandbox = LocalSandbox(self._directory)
        await self._sandbox.start()
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._sandbox is not None:
            await self._sandbox.stop()

    async def run(self, command: str, timeout: float) -> Graded:
        """Run ``command``, reporting whether it ran at all."""
        assert self._sandbox is not None, "use Grader as an async context manager"
        # Before every command, not once per trial: the test command itself
        # writes caches that the regression command would then inherit.
        wipe_caches(self._directory)
        try:
            result = await self._sandbox.exec(command, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            return Graded(1, f"the grading command could not run: {exc}", ran=False)
        output = (result.stdout or "") + (result.stderr or "")
        if getattr(result, "timed_out", False):
            return Graded(
                result.exit_code,
                f"the grading command exceeded its {timeout:g}s timeout\n{output}",
                ran=False,
            )
        return Graded(result.exit_code, output)


def sandboxed_runner(config: HarnessConfig):
    """A :data:`~harness.eval.pr_replay.RunCommand` that sandboxes each call.

    :class:`Grader` binds to one directory, because a trial grades the same
    tree several times and a container per command would triple the cost of a
    suite. Validation is the other shape — two commands in two different
    directories — so it gets a sandbox per call. Two containers per task
    validated, against a step that already runs the whole test suite twice.

    The point of routing validation through this rather than the host is not
    isolation (validation runs before any agent exists): it is that
    "fails at base, passes with the reference answer" has to be established in
    the environment where the verdict will be reached. A suite whose
    ``test_command`` names a host interpreter validated perfectly and then
    failed every trial inside the image, scoring correct solutions as model
    failures.
    """

    async def run(command: str, cwd: Path, timeout: float) -> Graded:
        async with Grader(cwd, config) as grader:
            return await grader.run(command, timeout)

    return run
