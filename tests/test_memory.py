"""Unit tests for harness.memory.store."""

from pathlib import Path

import pytest

from harness.memory.store import (
    EpisodeNotFoundError,
    FactNotFoundError,
    FactType,
    MemoryStore,
    MemoryStoreError,
    _parse_frontmatter,
)


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory")


class TestInitLayout:
    def test_creates_directory_tree(self, tmp_path):
        root = tmp_path / "memory"
        MemoryStore(root)
        assert (root / "facts").is_dir()
        assert (root / "facts" / "archive").is_dir()
        assert (root / "episodes").is_dir()
        assert (root / "INDEX.md").is_file()

    def test_reuses_existing_root_without_clobbering(self, tmp_path):
        root = tmp_path / "memory"
        s1 = MemoryStore(root)
        s1.write_fact("likes-tea", "User likes tea", "user", "Body.")
        s2 = MemoryStore(root)
        assert s2.read_fact("likes-tea").description == "User likes tea"


class TestFrontmatterParser:
    def test_colon_in_value_preserved(self):
        text = "---\nname: x\ndescription: Prefers dark mode: enabled\n---\nBody\n"
        fields, body = _parse_frontmatter(text)
        assert fields["description"] == "Prefers dark mode: enabled"
        assert body == "Body\n"

    def test_missing_opening_fence_raises(self):
        with pytest.raises(ValueError, match="opening"):
            _parse_frontmatter("name: x\n---\nBody\n")

    def test_missing_closing_fence_raises(self):
        with pytest.raises(ValueError, match="closing"):
            _parse_frontmatter("---\nname: x\nBody with no closing fence\n")

    def test_line_without_colon_raises(self):
        with pytest.raises(ValueError, match="no ':'"):
            _parse_frontmatter("---\nnot a valid line\n---\nBody\n")

    def test_blank_lines_between_fences_ignored(self):
        fields, _ = _parse_frontmatter("---\nname: x\n\ndescription: y\n---\nBody\n")
        assert fields == {"name": "x", "description": "y"}


class TestWriteReadFact:
    def test_round_trip(self, store):
        fact = store.write_fact(
            "prefers-dark-mode",
            "User prefers dark mode in all apps",
            "user",
            "Observed across three sessions.",
            sources=["2026-07-19-onboarding.md", "2026-07-20-followup.md"],
        )
        assert fact.type is FactType.USER
        read_back = store.read_fact("prefers-dark-mode")
        assert read_back == fact

    def test_type_accepts_enum_or_string(self, store):
        f1 = store.write_fact("a-fact", "desc", FactType.PROJECT, "body")
        f2 = store.write_fact("b-fact", "desc", "project", "body")
        assert f1.type is FactType.PROJECT
        assert f2.type is FactType.PROJECT

    def test_invalid_type_raises_clear_error(self, store):
        with pytest.raises(MemoryStoreError, match="invalid fact type"):
            store.write_fact("bad-fact", "desc", "nonsense", "body")

    def test_write_twice_overwrites(self, store):
        store.write_fact("likes-tea", "User likes tea", "user", "v1")
        store.write_fact("likes-tea", "User likes green tea specifically", "user", "v2")
        fact = store.read_fact("likes-tea")
        assert fact.description == "User likes green tea specifically"
        assert fact.body == "v2"

    def test_description_with_colon_round_trips(self, store):
        store.write_fact(
            "time-format", "Format: 24-hour clock, no AM/PM", "user", "body"
        )
        fact = store.read_fact("time-format")
        assert fact.description == "Format: 24-hour clock, no AM/PM"

    def test_sources_default_empty(self, store):
        fact = store.write_fact("no-sources", "desc", "reference", "body")
        assert fact.sources == []

    def test_read_missing_fact_raises(self, store):
        with pytest.raises(FactNotFoundError, match="no-such-fact"):
            store.read_fact("no-such-fact")

    def test_non_kebab_name_rejected(self, store):
        with pytest.raises(MemoryStoreError, match="kebab-case"):
            store.write_fact("Not_Kebab", "desc", "user", "body")

    def test_malformed_frontmatter_on_disk_is_clear_error(self, store, tmp_path):
        bad_path = store._facts_dir / "hand-edited.md"
        bad_path.write_text("---\nname: hand-edited\n---\nBody\n", encoding="utf-8")
        with pytest.raises(MemoryStoreError, match="missing required field"):
            store.read_fact("hand-edited")

    def test_missing_description_field_is_clear_error(self, store):
        bad_path = store._facts_dir / "no-desc.md"
        bad_path.write_text(
            "---\nname: no-desc\ntype: user\n---\nBody\n", encoding="utf-8"
        )
        with pytest.raises(MemoryStoreError, match="description"):
            store.read_fact("no-desc")


class TestWriteSideValidation:
    """Values the read-side parser would reject (or misread) must be
    rejected *before* anything lands on disk — a poisoned file would make
    every subsequent list/read/mutation fail (regression tests)."""

    def test_newline_in_description_rejected_and_nothing_written(self, store):
        with pytest.raises(MemoryStoreError, match="single line"):
            store.write_fact("bad", "line1\nline2: sneaky", "user", "body")
        assert not (store._facts_dir / "bad.md").exists()
        assert store.list_facts() == []  # store not bricked

    def test_newline_without_colon_rejected_before_write(self, store):
        with pytest.raises(MemoryStoreError, match="single line"):
            store.write_fact("bad", "line1\nno colon here", "user", "body")
        assert not (store._facts_dir / "bad.md").exists()
        store.rebuild_index()  # still healthy

    def test_empty_description_rejected(self, store):
        for desc in ("", "   "):
            with pytest.raises(MemoryStoreError, match="non-empty"):
                store.write_fact("bad", desc, "user", "body")
        assert not (store._facts_dir / "bad.md").exists()

    def test_sources_items_validated(self, store):
        with pytest.raises(MemoryStoreError, match="comma"):
            store.write_fact("f", "d", "user", "b", sources=["a.md,b.md"])
        with pytest.raises(MemoryStoreError, match="single line"):
            store.write_fact("f", "d", "user", "b", sources=["a\nb.md"])
        with pytest.raises(MemoryStoreError, match="non-empty"):
            store.write_fact("f", "d", "user", "b", sources=[""])
        assert not (store._facts_dir / "f.md").exists()

    def test_episode_title_and_outcome_validated(self, store):
        with pytest.raises(MemoryStoreError, match="single line"):
            store.write_episode("e", "a\nb", "ok", "body", date="2026-01-01")
        with pytest.raises(MemoryStoreError, match="non-empty"):
            store.write_episode("e", "title", "", "body", date="2026-01-01")
        assert list(store._episodes_dir.iterdir()) == []
        assert store.list_episodes() == []

    def test_superseded_by_validated(self, store):
        store.write_fact("old-fact", "d", "user", "b")
        with pytest.raises(MemoryStoreError, match="single line"):
            store.archive_fact("old-fact", superseded_by="a\nb")
        # rejected archive left the fact active and unarchived
        assert store.read_fact("old-fact").description == "d"
        assert not (store._archive_dir / "old-fact.md").exists()

    def test_surrounding_whitespace_normalized_round_trip(self, store):
        fact = store.write_fact("ws", "  desc: with colon  ", "user", "body")
        assert fact.description == "desc: with colon"
        assert store.read_fact("ws") == fact


class TestListFacts:
    def test_lists_sorted_by_name(self, store):
        store.write_fact("zebra-fact", "z", "user", "body")
        store.write_fact("alpha-fact", "a", "user", "body")
        names = [e.name for e in store.list_facts()]
        assert names == ["alpha-fact", "zebra-fact"]

    def test_empty_store_lists_nothing(self, store):
        assert store.list_facts() == []


class TestArchiveFact:
    def test_archive_moves_file_and_removes_from_active(self, store):
        store.write_fact("old-pref", "Old preference", "user", "body")
        store.archive_fact("old-pref")
        with pytest.raises(FactNotFoundError):
            store.read_fact("old-pref")
        archived = store._archive_dir / "old-pref.md"
        assert archived.is_file()
        assert "old-pref" not in {e.name for e in store.list_facts()}

    def test_archive_never_deletes_content(self, store):
        store.write_fact("old-pref", "Old preference", "user", "the body text")
        store.archive_fact("old-pref")
        archived_text = (store._archive_dir / "old-pref.md").read_text()
        assert "the body text" in archived_text
        assert "Old preference" in archived_text

    def test_archive_with_superseded_by_records_pointer(self, store):
        store.write_fact("old-pref", "Old preference", "user", "body")
        store.write_fact("new-pref", "New preference", "user", "body")
        store.archive_fact("old-pref", superseded_by="new-pref")
        archived_text = (store._archive_dir / "old-pref.md").read_text()
        assert "superseded_by: new-pref" in archived_text

    def test_archive_unknown_fact_raises(self, store):
        with pytest.raises(FactNotFoundError):
            store.archive_fact("never-existed")

    def test_archive_rebuilds_index(self, store):
        store.write_fact("old-pref", "Old preference", "user", "body")
        store.archive_fact("old-pref")
        index_text = (store.root / "INDEX.md").read_text()
        assert "old-pref" not in index_text


class TestWriteReadEpisode:
    def test_filename_is_date_prefixed(self, store):
        filename = store.write_episode(
            "fixed-bug", "Fixed the bug", "success", "Details.", date="2026-07-19"
        )
        assert filename == "2026-07-19-fixed-bug.md"

    def test_round_trip(self, store):
        filename = store.write_episode(
            "onboarding",
            "First session",
            "success",
            "Learned the user's timezone.",
            date="2026-07-19",
        )
        episode = store.read_episode(filename)
        assert episode.slug == "onboarding"
        assert episode.title == "First session"
        assert episode.outcome == "success"
        assert episode.date == "2026-07-19"
        assert episode.body == "Learned the user's timezone."

    def test_date_defaults_to_today(self, store):
        import datetime

        filename = store.write_episode("no-date-given", "T", "success", "body")
        assert filename.startswith(datetime.date.today().isoformat())

    def test_invalid_date_raises(self, store):
        with pytest.raises(MemoryStoreError, match="YYYY-MM-DD"):
            store.write_episode("bad-date", "T", "success", "body", date="07-19-2026")

    def test_read_episode_rejects_path_traversal(self, store, tmp_path):
        # a parseable file OUTSIDE the memory root must be unreachable
        secret = tmp_path / "secret.md"
        secret.write_text(
            "---\nslug: s\ndate: 2026-01-01\ntitle: T\noutcome: O\n---\n\nsecret\n",
            encoding="utf-8",
        )
        for name in (
            "../../secret.md",
            "../secret.md",
            "a/b.md",
            "..\\x.md",
            "..",
            ".",
            "",
        ):
            with pytest.raises(MemoryStoreError, match="bare filename"):
                store.read_episode(name)

    def test_read_missing_episode_raises(self, store):
        with pytest.raises(EpisodeNotFoundError):
            store.read_episode("2026-07-19-nonexistent.md")

    def test_non_kebab_slug_rejected(self, store):
        with pytest.raises(MemoryStoreError, match="kebab-case"):
            store.write_episode("Not Kebab", "T", "success", "body")


class TestListEpisodesOrdering:
    def test_most_recent_first(self, store):
        store.write_episode("first", "First", "success", "b", date="2026-07-01")
        store.write_episode("second", "Second", "success", "b", date="2026-07-15")
        store.write_episode("third", "Third", "success", "b", date="2026-07-10")
        slugs = [e.slug for e in store.list_episodes()]
        assert slugs == ["second", "third", "first"]

    def test_empty_store_lists_nothing(self, store):
        assert store.list_episodes() == []


class TestRebuildIndex:
    def test_index_contains_facts_and_episodes(self, store):
        store.write_fact("likes-tea", "User likes tea", "user", "body")
        store.write_episode("onboarding", "First session", "success", "b", date="2026-07-19")
        index_text = (store.root / "INDEX.md").read_text()
        assert "- [likes-tea] User likes tea" in index_text
        assert "2026-07-19-onboarding.md" in index_text
        assert "First session" in index_text

    def test_index_caps_episodes_at_ten(self, store):
        for i in range(12):
            store.write_episode(f"episode-{i:02d}", f"Episode {i}", "success", "b", date="2026-07-01")
        index_text = (store.root / "INDEX.md").read_text()
        assert index_text.count("- [2026-07-01-episode-") == 10

    def test_empty_index_has_placeholders(self, store):
        index_text = (store.root / "INDEX.md").read_text()
        assert "(none yet)" in index_text

    def test_rebuild_index_is_idempotent_and_callable_directly(self, store):
        store.write_fact("a-fact", "desc", "user", "body")
        before = (store.root / "INDEX.md").read_text()
        store.rebuild_index()
        after = (store.root / "INDEX.md").read_text()
        assert before == after


class TestSearch:
    def test_hits_in_fact_and_episode_bodies(self, store):
        store.write_fact("likes-tea", "User likes tea", "user", "Prefers oolong tea in the morning.")
        store.write_episode(
            "tea-chat", "Talked about tea", "success", "We discussed oolong tea varieties.", date="2026-07-19"
        )
        results = store.search("oolong")
        kinds = {(kind, name) for kind, name, _ in results}
        assert ("fact", "likes-tea") in kinds
        assert ("episode", "2026-07-19-tea-chat.md") in kinds

    def test_case_insensitive(self, store):
        store.write_fact("likes-tea", "desc", "user", "Prefers OOLONG tea.")
        results = store.search("oolong")
        assert len(results) == 1
        assert "OOLONG" in results[0][2]

    def test_no_match_returns_empty(self, store):
        store.write_fact("likes-tea", "desc", "user", "Prefers oolong tea.")
        assert store.search("coffee") == []

    def test_empty_query_returns_empty(self, store):
        store.write_fact("likes-tea", "desc", "user", "Prefers oolong tea.")
        assert store.search("   ") == []

    def test_archived_facts_not_searched(self, store):
        store.write_fact("old-pref", "desc", "user", "Prefers oolong tea archived.")
        store.archive_fact("old-pref")
        assert store.search("oolong") == []
