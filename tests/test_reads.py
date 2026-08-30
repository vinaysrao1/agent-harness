"""S-102: the file cache and the read ledger.

Two pieces of per-agent state. The cache saves a round-trip; the ledger is the
more valuable half — an `old_string` built from a stale memory of a file is the
commonest way an edit fails, and the failure arrives as "not found" with no
hint that the file moved underneath.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from harness.diligence import looks_unfinished
from harness.reads import MAX_CACHED_FILES, ReadLedger, StaleRead, version_of
from harness.sandbox.local import LocalSandbox
from harness.tools.builtin import (
    bash_tool,
    edit_file_tool,
    multi_edit_tool,
    read_file_tool,
    write_file_tool,
)


@pytest.fixture
def workspace() -> Path:
    directory = Path(tempfile.mkdtemp())
    (directory / "a.py").write_text("value = 1\n")
    (directory / "b.py").write_text("other = 2\n")
    return directory


async def _sandbox(directory: Path) -> LocalSandbox:
    sandbox = LocalSandbox(directory)
    await sandbox.start()
    return sandbox


def _counting(monkeypatch) -> list[str]:
    """Record every file the sandbox is actually asked to read."""
    seen: list[str] = []
    real = LocalSandbox.read_file

    async def spy(self, path):
        seen.append(path)
        return await real(self, path)

    monkeypatch.setattr(LocalSandbox, "read_file", spy)
    return seen


class TestTheCacheSavesTheRoundTrip:
    """Acceptance (1): a repeat read of an unchanged file performs no sandbox
    call. Asserted on the calls, because elapsed time would not distinguish a
    cache hit from a fast filesystem."""

    async def test_S102_a_second_read_does_not_reach_the_sandbox(
        self, workspace, monkeypatch
    ) -> None:
        sandbox = await _sandbox(workspace)
        ledger = ReadLedger()
        tool = read_file_tool(sandbox, ledger)
        seen = _counting(monkeypatch)

        first = await tool.handler({"path": "a.py"})
        second = await tool.handler({"path": "a.py"})

        assert first == second
        assert seen == ["a.py"], seen

    async def test_S102_without_a_ledger_nothing_changes(
        self, workspace, monkeypatch
    ) -> None:
        # Neutrality: the benchmark path passes no ledger today, and must read
        # exactly as often as it did before.
        sandbox = await _sandbox(workspace)
        tool = read_file_tool(sandbox)
        seen = _counting(monkeypatch)
        await tool.handler({"path": "a.py"})
        await tool.handler({"path": "a.py"})
        assert seen == ["a.py", "a.py"]

    async def test_S102_different_files_are_both_read(
        self, workspace, monkeypatch
    ) -> None:
        sandbox = await _sandbox(workspace)
        tool = read_file_tool(sandbox, ReadLedger())
        seen = _counting(monkeypatch)
        await tool.handler({"path": "a.py"})
        await tool.handler({"path": "b.py"})
        assert seen == ["a.py", "b.py"]

    def test_S102_the_cache_is_bounded(self) -> None:
        # A run that reads ten thousand files should not also hold them: this
        # is a latency optimisation, not a store.
        ledger = ReadLedger()
        for index in range(MAX_CACHED_FILES + 20):
            ledger.record_read(f"f{index}.py", f"content {index}")
        assert ledger.cached("f0.py") is None, "the oldest entry survived"
        assert ledger.cached(f"f{MAX_CACHED_FILES + 19}.py") is not None

    def test_S102_reading_refreshes_recency(self) -> None:
        ledger = ReadLedger()
        for index in range(MAX_CACHED_FILES):
            ledger.record_read(f"f{index}.py", "x")
        ledger.cached("f0.py")               # touch the oldest
        ledger.record_read("new.py", "y")    # force one eviction
        assert ledger.cached("f0.py") is not None, "LRU evicted a recent read"


class TestInvalidation:
    """A cache that survives a write hands the model a file that no longer
    exists in that form — worse than the round-trip it saved."""

    async def test_S102_an_edit_invalidates_the_cache(
        self, workspace, monkeypatch
    ) -> None:
        sandbox = await _sandbox(workspace)
        ledger = ReadLedger()
        reader = read_file_tool(sandbox, ledger)
        editor = edit_file_tool(sandbox, reads=ledger)

        await reader.handler({"path": "a.py"})
        await editor.handler({"path": "a.py", "old_string": "1", "new_string": "99"})
        result = await reader.handler({"path": "a.py"})
        assert "99" in result, "the cache served content the edit replaced"

    async def test_S102_a_bash_command_invalidates_everything(
        self, workspace, monkeypatch
    ) -> None:
        # A shell command can rewrite any path without the harness learning
        # which, so every cached file becomes a belief rather than knowledge.
        sandbox = await _sandbox(workspace)
        ledger = ReadLedger()
        reader = read_file_tool(sandbox, ledger)
        shell = bash_tool(sandbox, reads=ledger)

        await reader.handler({"path": "a.py"})
        await shell.handler({"command": "echo 'value = 42' > a.py"})
        result = await reader.handler({"path": "a.py"})
        assert "42" in result, "a bash write was served from a stale cache"

    async def test_S102_a_write_is_known_without_a_reread(
        self, workspace, monkeypatch
    ) -> None:
        sandbox = await _sandbox(workspace)
        ledger = ReadLedger()
        writer = write_file_tool(sandbox, reads=ledger)
        reader = read_file_tool(sandbox, ledger)
        await writer.handler({"path": "c.py", "content": "fresh = 1\n"})
        seen = _counting(monkeypatch)
        result = await reader.handler({"path": "c.py"})
        assert "fresh" in result
        assert seen == [], "the harness re-read content it had just authored"

    async def test_S102_an_append_is_not_claimed_as_knowledge(
        self, workspace, monkeypatch
    ) -> None:
        # After an append the harness knows a change happened but not the
        # resulting whole. Claiming to know it would be worse than admitting
        # it does not.
        sandbox = await _sandbox(workspace)
        ledger = ReadLedger()
        writer = write_file_tool(sandbox, reads=ledger)
        reader = read_file_tool(sandbox, ledger)
        await reader.handler({"path": "a.py"})
        await writer.handler({"path": "a.py", "content": "extra = 3\n", "mode": "append"})
        seen = _counting(monkeypatch)
        result = await reader.handler({"path": "a.py"})
        assert seen == ["a.py"], "an append left a stale cache entry in place"
        assert "extra" in result


class TestStalenessWarnsAndNeverRejects:
    """Acceptance (2): staleness never rejects an edit in any profile."""

    async def test_S102_editing_an_unread_file_warns(self, workspace) -> None:
        sandbox = await _sandbox(workspace)
        ledger = ReadLedger(advise=True)
        result = await edit_file_tool(sandbox, reads=ledger).handler(
            {"path": "a.py", "old_string": "1", "new_string": "2"}
        )
        assert "without being read first" in result
        assert "edited a.py" in result, "the edit did not happen"
        assert (workspace / "a.py").read_text() == "value = 2\n"

    async def test_S102_editing_a_freshly_read_file_is_silent(
        self, workspace
    ) -> None:
        # The control. A warning on ordinary correct behaviour is how an
        # advisory becomes noise and then becomes ignored.
        sandbox = await _sandbox(workspace)
        ledger = ReadLedger(advise=True)
        await read_file_tool(sandbox, ledger).handler({"path": "a.py"})
        result = await edit_file_tool(sandbox, reads=ledger).handler(
            {"path": "a.py", "old_string": "1", "new_string": "2"}
        )
        assert "Note from the harness" not in result, result

    async def test_S102_a_second_edit_in_a_row_is_silent(self, workspace) -> None:
        # The harness authored the new content, so it knows it. Without this
        # every edit after the first on one file would warn.
        sandbox = await _sandbox(workspace)
        ledger = ReadLedger(advise=True)
        reader, editor = read_file_tool(sandbox, ledger), edit_file_tool(sandbox, reads=ledger)
        await reader.handler({"path": "a.py"})
        await editor.handler({"path": "a.py", "old_string": "value", "new_string": "alpha"})
        await reader.handler({"path": "a.py"})
        result = await editor.handler({"path": "a.py", "old_string": "1", "new_string": "2"})
        assert "Note from the harness" not in result, result

    async def test_S102_editing_a_file_the_harness_just_wrote_is_silent(
        self, workspace
    ) -> None:
        # `note_write` records a *version*, not just cache content. Without
        # the version half, writing a file and then editing it warns that the
        # file "was edited without being read first" -- on content the agent
        # authored one call earlier. That is the advisory firing on the most
        # ordinary sequence there is.
        sandbox = await _sandbox(workspace)
        ledger = ReadLedger(advise=True)
        await write_file_tool(sandbox, reads=ledger).handler(
            {"path": "new.py", "content": "value = 1\n"}
        )
        result = await edit_file_tool(sandbox, reads=ledger).handler(
            {"path": "new.py", "old_string": "1", "new_string": "2"}
        )
        assert "Note from the harness" not in result, result

    async def test_S102_a_second_multi_edit_is_silent(self, workspace) -> None:
        sandbox = await _sandbox(workspace)
        ledger = ReadLedger(advise=True)
        tool = multi_edit_tool(sandbox, reads=ledger)
        await tool.handler(
            {"path": "a.py", "edits": [{"old_string": "value", "new_string": "alpha"}]}
        )
        result = await tool.handler(
            {"path": "a.py", "edits": [{"old_string": "1", "new_string": "2"}]}
        )
        assert "Note from the harness" not in result, result

    async def test_S102_an_edit_after_an_external_change_warns(
        self, workspace
    ) -> None:
        sandbox = await _sandbox(workspace)
        ledger = ReadLedger(advise=True)
        await read_file_tool(sandbox, ledger).handler({"path": "a.py"})
        (workspace / "a.py").write_text("value = 7\n")   # changed underneath
        result = await edit_file_tool(sandbox, reads=ledger).handler(
            {"path": "a.py", "old_string": "7", "new_string": "8"}
        )
        assert "changed since it was last read" in result
        assert (workspace / "a.py").read_text() == "value = 8\n"

    async def test_S102_multi_edit_warns_the_same_way(self, workspace) -> None:
        sandbox = await _sandbox(workspace)
        ledger = ReadLedger(advise=True)
        result = await multi_edit_tool(sandbox, reads=ledger).handler(
            {"path": "a.py", "edits": [{"old_string": "1", "new_string": "2"}]}
        )
        assert "without being read first" in result
        assert "applied 1 edit" in result

    async def test_S102_without_a_ledger_no_advisory_appears(
        self, workspace
    ) -> None:
        sandbox = await _sandbox(workspace)
        result = await edit_file_tool(sandbox).handler(
            {"path": "a.py", "old_string": "1", "new_string": "2"}
        )
        assert result == "edited a.py"


class TestTheAdvisoryCannotBeMistakenForUnfinishedWork:
    """Acceptance (3): no promise pattern, never ends in `?`.

    The same constraint `format_syntax_failure` carries. An advisory quoted
    back by the model must not make its own final answer look like a promise
    of future work, or the loop will refuse to accept a finished run.
    """

    @pytest.mark.parametrize(
        "reason",
        [
            "a.py was edited without being read first, so the edit was based "
            "on an assumption about its contents.",
            "a.py changed since it was last read, so the version edited is not "
            "the version seen.",
        ],
    )
    def test_S102_the_advisory_does_not_trip_looks_unfinished(
        self, reason: str
    ) -> None:
        advisory = StaleRead("a.py", reason).advisory()
        unfinished, why = looks_unfinished(advisory, 0)
        assert not unfinished, why

    def test_S102_the_advisory_never_ends_in_a_question(self) -> None:
        advisory = StaleRead("a.py", "a.py changed.").advisory()
        assert not advisory.rstrip().endswith("?")

    def test_S102_the_advisory_names_the_harness_as_author(self) -> None:
        # Text that reads like the model's own tool output would have it
        # debugging a message it never produced.
        assert StaleRead("a.py", "x").advisory().startswith("Note from the harness:")


class TestVersioning:
    def test_S102_identical_content_is_the_same_version(self) -> None:
        assert version_of("abc") == version_of("abc")

    def test_S102_different_content_is_a_different_version(self) -> None:
        assert version_of("abc") != version_of("abd")

    def test_S102_a_ledger_starts_knowing_nothing(self) -> None:
        assert ReadLedger(advise=True).check("a.py", "x") is not None

    def test_S102_advisories_are_off_by_default(self) -> None:
        # The neutrality default. A ledger that warns unless told otherwise
        # would put an advisory on the benchmark path by construction.
        assert ReadLedger().advise is False
        assert ReadLedger().check("a.py", "x") is None


class TestTheBenchmarkPathIsByteIdentical:
    """D2. The first version wired an advising ledger into every profile, so
    the canonical Terminal-Bench sequence — create a file with a `bash`
    heredoc, then edit it — carried a 178-byte advisory saying the file "was
    edited without being read first". That is the modal write path there, and
    it changes tool-result bytes and therefore tokens per turn."""

    async def test_S102_a_heredoc_then_edit_is_unchanged(self, workspace) -> None:
        sandbox = await _sandbox(workspace)
        ledger = ReadLedger()          # advise=False, the CODING default
        await bash_tool(sandbox, reads=ledger).handler(
            {"command": "printf 'x = 1\\n' > solve.py"}
        )
        result = await edit_file_tool(sandbox, reads=ledger).handler(
            {"path": "solve.py", "old_string": "1", "new_string": "2"}
        )
        assert result == "edited solve.py", result

    def test_S102_the_benchmark_profile_does_not_enable_advisories(self) -> None:
        from harness.profiles import CODING, CODING_REPO

        assert not CODING.enables("read_staleness")
        assert CODING_REPO.enables("read_staleness")

    def test_S102_an_unknown_environment_still_affirms_nothing(self) -> None:
        # `read_staleness` is gated on the profile half only. Routing it
        # through `affirms` would have meant returning True unconditionally,
        # and UNKNOWN_ENVIRONMENT affirming something is what S-005's
        # "unknown is not affirmation" rule exists to prevent.
        from harness.environment import UNKNOWN_ENVIRONMENT

        assert not UNKNOWN_ENVIRONMENT.affirms("read_staleness")


class TestOneFileIsOneCacheEntry:
    """D1. Keyed verbatim, `a.py` and `./a.py` were two entries, so writing
    through one spelling left the other serving content no longer on disk —
    reachable from a single agent with no `bash` involved."""

    @pytest.mark.parametrize(
        "written_as", ["./a.py", "a.py", "pkg/../a.py"]
    )
    async def test_S102_a_write_invalidates_every_spelling(
        self, workspace, written_as: str
    ) -> None:
        (workspace / "pkg").mkdir(exist_ok=True)
        sandbox = await _sandbox(workspace)
        ledger = ReadLedger()
        reader = read_file_tool(sandbox, ledger)
        writer = write_file_tool(sandbox, reads=ledger)

        await reader.handler({"path": "a.py"})
        await writer.handler({"path": written_as, "content": "NEW\n"})
        assert "NEW" in await reader.handler({"path": "a.py"}), written_as

    async def test_S102_a_subdirectory_read_is_one_entry(self, workspace) -> None:
        (workspace / "pkg").mkdir(exist_ok=True)
        (workspace / "pkg" / "mod.py").write_text("m = 1\n")
        sandbox = await _sandbox(workspace)
        ledger = ReadLedger()
        reader = read_file_tool(sandbox, ledger)
        await reader.handler({"path": "pkg/mod.py"})
        await edit_file_tool(sandbox, reads=ledger).handler(
            {"path": "pkg/./mod.py", "old_string": "1", "new_string": "7"}
        )
        assert "7" in await reader.handler({"path": "pkg/mod.py"})

    def test_S102_normalisation_collapses_equivalent_spellings(self) -> None:
        from harness.reads import normalise_path

        assert len({normalise_path(p) for p in ("a.py", "./a.py", "pkg/../a.py")}) == 1


class TestTheHarnessKnowsWhatItWrote:
    """D3. `edit_file` invalidated the path but never recorded the result, so
    a second edit warned that the file "changed since it was last read" —
    blaming an external change for the harness's own edit, one call earlier."""

    async def test_S102_two_edits_in_a_row_are_silent(self, workspace) -> None:
        # No re-read in between. The earlier version of this test inserted
        # one, and that re-read was what made it pass.
        sandbox = await _sandbox(workspace)
        ledger = ReadLedger(advise=True)
        await read_file_tool(sandbox, ledger).handler({"path": "a.py"})
        editor = edit_file_tool(sandbox, reads=ledger)
        await editor.handler({"path": "a.py", "old_string": "value", "new_string": "alpha"})
        result = await editor.handler({"path": "a.py", "old_string": "1", "new_string": "9"})
        assert "Note from the harness" not in result, result

    async def test_S102_the_edited_content_is_cached(self, workspace, monkeypatch) -> None:
        sandbox = await _sandbox(workspace)
        ledger = ReadLedger()
        await read_file_tool(sandbox, ledger).handler({"path": "a.py"})
        await edit_file_tool(sandbox, reads=ledger).handler(
            {"path": "a.py", "old_string": "value", "new_string": "alpha"}
        )
        seen = _counting(monkeypatch)
        result = await read_file_tool(sandbox, ledger).handler({"path": "a.py"})
        assert "alpha" in result
        assert seen == [], "re-read content the harness had just produced"


class TestTheCacheIsSharedByAgentsInARun:
    """D4. A subagent runs in the lead's sandbox by default. With a per-agent
    cache, the child could rewrite a file while the parent kept serving the
    pre-spawn bytes until it happened to run a shell command."""

    async def test_S102_one_agents_write_invalidates_anothers_cache(
        self, workspace
    ) -> None:
        from harness.reads import FileCache

        sandbox = await _sandbox(workspace)
        shared = FileCache()
        lead = ReadLedger(cache=shared)
        child = ReadLedger(cache=shared)

        await read_file_tool(sandbox, lead).handler({"path": "a.py"})
        await write_file_tool(sandbox, reads=child).handler(
            {"path": "a.py", "content": "rewritten = 1\n"}
        )
        assert "rewritten" in await read_file_tool(sandbox, lead).handler({"path": "a.py"})

    def test_S102_versions_are_not_shared(self) -> None:
        # Sharing the cache is right; sharing knowledge is not. One agent's
        # read must not silence another's staleness warning.
        from harness.reads import FileCache

        shared = FileCache()
        lead = ReadLedger(cache=shared, advise=True)
        child = ReadLedger(cache=shared, advise=True)
        lead.record_read("a.py", "content")
        assert lead.check("a.py", "content") is None
        assert child.check("a.py", "content") is not None


class TestAConcurrentInvalidationWins:
    """D5. A read suspends at an `await` before it can store what it fetched,
    and tool calls in a turn run concurrently — so a `bash` invalidation could
    land in that window and the read would reinsert pre-command bytes that
    nothing would ever evict again."""

    def test_S102_a_store_from_a_stale_generation_is_dropped(self) -> None:
        from harness.reads import FileCache

        cache = FileCache()
        generation = cache.generation      # a read begins
        cache.drop_all()                   # a concurrent bash lands
        assert cache.put("a.py", "old bytes", generation=generation) is False
        assert cache.get("a.py") is None

    def test_S102_a_store_from_the_current_generation_is_kept(self) -> None:
        from harness.reads import FileCache

        cache = FileCache()
        assert cache.put("a.py", "bytes", generation=cache.generation) is True
        assert cache.get("a.py") == "bytes"


class TestBothInsertionPathsAreBounded:
    """D6. `note_write` was `record_read` minus the eviction loop, so a codegen
    run writing thousands of files held every one of them for the run."""

    def test_S102_writes_are_bounded_too(self) -> None:
        ledger = ReadLedger()
        for index in range(MAX_CACHED_FILES + 50):
            ledger.note_write(f"f{index}.py", f"content {index}")
        assert len(ledger.cache) == MAX_CACHED_FILES

    def test_S102_versions_are_bounded(self) -> None:
        from harness.reads import MAX_TRACKED_VERSIONS

        ledger = ReadLedger(advise=True)
        for index in range(MAX_TRACKED_VERSIONS + 50):
            ledger.note_write(f"f{index}.py", "x")
        assert len(ledger._versions) == MAX_TRACKED_VERSIONS


class TestTheWiringProductionActuallyUses:
    """Every test above builds tools by hand. That is how the first version's
    neutrality break survived: the components behaved correctly and the
    orchestrator wired them the wrong way. These go through
    `Orchestrator._build_registry`, which is what a real run does.
    """

    def _orchestrator(self, tmp_path):
        from harness.config import HarnessConfig
        from harness.orchestrator import Orchestrator
        from harness.persistence import RunStore
        from harness.reads import FileCache

        home = tmp_path / "home"
        home.mkdir(exist_ok=True)
        orchestrator = Orchestrator(HarnessConfig(home=home), RunStore(home / "s.db"))
        orchestrator._file_cache = FileCache()
        return orchestrator

    def _registry(self, orchestrator, tmp_path, factories, *, advise: bool):
        from harness.context import ContextManager
        from harness.memory.store import MemoryStore
        from harness.skills import SkillLibrary

        return orchestrator._build_registry(
            LocalSandbox(tmp_path),
            MemoryStore(tmp_path / "memory"),
            SkillLibrary(tmp_path / "skills"),
            "run",
            "agent",
            ContextManager(
                base_system_prompt="x",
                count_tokens=lambda m: 0,
                max_context=1000,
                summarize=None,
            ),
            factories,
            None,
            None,
            advise,
        )

    async def test_S102_a_coding_registry_produces_no_advisory(
        self, workspace, tmp_path
    ) -> None:
        # The neutrality property, through the real construction path.
        from harness.orchestrator import CODING_TOOL_FACTORIES
        from harness.types import ToolCall

        orchestrator = self._orchestrator(tmp_path)
        registry = self._registry(
            orchestrator, workspace, CODING_TOOL_FACTORIES, advise=False
        )
        await registry.dispatch(ToolCall(
            id="1", name="bash",
            arguments={"command": "printf 'x = 1\\n' > solve.py"},
        ))
        result = await registry.dispatch(ToolCall(
            id="2", name="edit_file",
            arguments={"path": "solve.py", "old_string": "1", "new_string": "2"},
        ))
        assert result.content == "edited solve.py", result.content

    async def test_S102_a_repo_registry_does_produce_one(
        self, workspace, tmp_path
    ) -> None:
        # The control: if this passed too, the flag would be doing nothing.
        from harness.profiles import REPO_TOOL_FACTORIES
        from harness.types import ToolCall

        orchestrator = self._orchestrator(tmp_path)
        registry = self._registry(
            orchestrator, workspace, REPO_TOOL_FACTORIES, advise=True
        )
        result = await registry.dispatch(ToolCall(
            id="1", name="edit_file",
            arguments={"path": "a.py", "old_string": "1", "new_string": "2"},
        ))
        assert "Note from the harness" in result.content

    async def test_S102_two_registries_share_one_cache(
        self, workspace, tmp_path
    ) -> None:
        # A subagent runs in the lead's sandbox. With a per-agent cache the
        # child could rewrite a file while the parent served stale bytes.
        from harness.orchestrator import CODING_TOOL_FACTORIES
        from harness.types import ToolCall

        orchestrator = self._orchestrator(tmp_path)
        lead = self._registry(
            orchestrator, workspace, CODING_TOOL_FACTORIES, advise=False
        )
        child = self._registry(
            orchestrator, workspace, CODING_TOOL_FACTORIES, advise=False
        )
        await lead.dispatch(ToolCall(id="1", name="read_file", arguments={"path": "a.py"}))
        await child.dispatch(ToolCall(
            id="2", name="write_file",
            arguments={"path": "a.py", "content": "rewritten = 1\n"},
        ))
        again = await lead.dispatch(
            ToolCall(id="3", name="read_file", arguments={"path": "a.py"})
        )
        assert "rewritten" in again.content, again.content

    async def test_S102_the_read_only_profile_gets_a_cache(
        self, workspace, tmp_path, monkeypatch
    ) -> None:
        # The one profile where an un-revalidating cache can never be wrong —
        # no bash, no write_file, no edit_file — and the one that was left
        # unwired. A constructed-and-never-read ledger is this project's
        # archetype.
        from harness.profiles import CODING_READONLY
        from harness.types import ToolCall

        orchestrator = self._orchestrator(tmp_path)
        registry = self._registry(
            orchestrator, workspace, CODING_READONLY.tool_factories, advise=False
        )
        seen = _counting(monkeypatch)
        await registry.dispatch(ToolCall(id="1", name="read_file", arguments={"path": "a.py"}))
        await registry.dispatch(ToolCall(id="2", name="read_file", arguments={"path": "a.py"}))
        assert seen == ["a.py"], seen


class TestInvalidationSurvivesAFailedCommand:
    async def test_S102_a_raising_exec_still_invalidates(
        self, workspace, monkeypatch
    ) -> None:
        # A command that raises part way through may already have written.
        # Invalidating only on the success path leaves the whole cache intact
        # over a partially-executed shell command.
        sandbox = await _sandbox(workspace)
        ledger = ReadLedger()
        await read_file_tool(sandbox, ledger).handler({"path": "a.py"})
        assert ledger.cached("a.py") is not None

        async def boom(self, command, **kwargs):
            raise RuntimeError("daemon went away mid-command")

        monkeypatch.setattr(LocalSandbox, "exec", boom)
        with pytest.raises(RuntimeError):
            await bash_tool(sandbox, reads=ledger).handler({"command": "rm -rf x"})
        assert ledger.cached("a.py") is None, "a failed command left a stale cache"


class TestTheProfileDerivationInsideExecute:
    """N3. The tests above pass `advise` to `_build_registry` directly, so
    they cannot see the line in `_execute` that derives it from the profile.
    Setting that line to `True` left every one of them green while every
    Terminal-Bench edit grew an advisory. Only a real `run_task` covers it."""

    @pytest.fixture(autouse=True)
    def _no_docker(self, monkeypatch):
        from harness.sandbox.docker import DockerSandbox

        monkeypatch.setattr(
            DockerSandbox, "availability", classmethod(lambda cls: False)
        )

    async def _edit_result(self, tmp_path, profile) -> str:
        import warnings

        from harness.adapters.fake import FakeAdapter
        from harness.config import HarnessConfig
        from harness.orchestrator import Orchestrator
        from harness.persistence import RunStore
        from harness.types import (
            Message, ModelResponse, Role, StopReason, ToolCall, Usage,
        )

        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "solve.py").write_text("x = 1\n")

        def call(identifier: str, name: str, **arguments):
            return ModelResponse(
                message=Message(role=Role.ASSISTANT, tool_calls=[
                    ToolCall(id=identifier, name=name, arguments=arguments)]),
                usage=Usage(), stop_reason=StopReason.TOOL_USE,
            )

        script = [
            call("e1", "edit_file", path="solve.py", old_string="1", new_string="2"),
            ModelResponse(
                message=Message(role=Role.ASSISTANT,
                                content="Task complete. Edited solve.py and verified it."),
                usage=Usage(), stop_reason=StopReason.END_TURN,
            ),
        ]
        home = tmp_path / "home"
        home.mkdir()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with RunStore(home / "state.db") as store:
                orchestrator = Orchestrator(HarnessConfig(home=home), store)
                run_id, _ = await orchestrator.run_task(
                    "Edit solve.py.", "fake-model",
                    adapter_override=FakeAdapter(script),
                    workspace=workspace, profile=profile,
                )
                agent_id = store.list_agents(run_id)[0].id
                results = [
                    __import__("json").loads(event.payload)["content"]
                    if isinstance(event.payload, str) else event.payload["content"]
                    for event in store.load_events(agent_id)
                    if event.kind == "tool_result"
                ]
        return results[0] if results else ""

    async def test_S102_a_coding_run_produces_no_advisory(self, tmp_path) -> None:
        from harness.profiles import CODING

        assert await self._edit_result(tmp_path, CODING) == "edited solve.py"

    async def test_S102_a_default_profile_run_produces_no_advisory(
        self, tmp_path
    ) -> None:
        # profile=None is what the Harbor bridge passes.
        assert await self._edit_result(tmp_path, None) == "edited solve.py"

    async def test_S102_a_repo_run_does_produce_one(self, tmp_path) -> None:
        # The control. Without this, "no advisory anywhere" would also pass.
        from harness.profiles import CODING_REPO

        result = await self._edit_result(tmp_path, CODING_REPO)
        assert "Note from the harness" in result, result
