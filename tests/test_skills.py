"""Unit tests for harness.skills."""

import pytest

from harness.skills import Skill, SkillLibrary


def _make_skill(root, dirname: str, name: str, description: str, body: str) -> None:
    skill_dir = root / dirname
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )


class TestDiscovery:
    def test_discovers_skill_directories(self, tmp_path):
        _make_skill(tmp_path, "pr-review", "pr-review", "Review a pull request", "Steps...")
        library = SkillLibrary(tmp_path)
        assert [s.name for s in library.skills()] == ["pr-review"]

    def test_missing_root_yields_empty_library(self, tmp_path):
        library = SkillLibrary(tmp_path / "does-not-exist")
        assert library.skills() == []
        assert library.index_lines() == []

    def test_directory_without_skill_md_is_ignored(self, tmp_path):
        (tmp_path / "not-a-skill").mkdir()
        (tmp_path / "not-a-skill" / "README.md").write_text("hi")
        library = SkillLibrary(tmp_path)
        assert library.skills() == []

    def test_skill_body_captured(self, tmp_path):
        _make_skill(tmp_path, "deploy", "deploy", "Deploy the service", "1. Build\n2. Push\n3. Restart")
        library = SkillLibrary(tmp_path)
        assert library.load("deploy") == "1. Build\n2. Push\n3. Restart"

    def test_multiple_skills_sorted_in_index_lines(self, tmp_path):
        _make_skill(tmp_path, "zebra", "zebra-skill", "Z desc", "body")
        _make_skill(tmp_path, "alpha", "alpha-skill", "A desc", "body")
        library = SkillLibrary(tmp_path)
        lines = library.index_lines()
        assert lines == [
            "- alpha-skill: A desc",
            "- zebra-skill: Z desc",
        ]

    def test_colon_in_description_preserved(self, tmp_path):
        _make_skill(tmp_path, "notes", "notes", "Format: bullet points only", "body")
        library = SkillLibrary(tmp_path)
        assert library.skills()[0].description == "Format: bullet points only"


class TestMalformedSkillsSkipped:
    def test_missing_name_field_skips_with_warning(self, tmp_path):
        skill_dir = tmp_path / "broken"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\ndescription: no name here\n---\nBody\n", encoding="utf-8"
        )
        with pytest.warns(UserWarning, match="broken"):
            library = SkillLibrary(tmp_path)
        assert library.skills() == []

    def test_missing_description_field_skips_with_warning(self, tmp_path):
        skill_dir = tmp_path / "broken2"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: broken2\n---\nBody\n", encoding="utf-8"
        )
        with pytest.warns(UserWarning, match="broken2"):
            library = SkillLibrary(tmp_path)
        assert library.skills() == []

    def test_bad_fences_skip_with_warning(self, tmp_path):
        skill_dir = tmp_path / "broken3"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("no frontmatter at all here", encoding="utf-8")
        with pytest.warns(UserWarning, match="broken3"):
            library = SkillLibrary(tmp_path)
        assert library.skills() == []

    def test_one_malformed_skill_does_not_block_others(self, tmp_path):
        _make_skill(tmp_path, "good", "good-skill", "Works fine", "body")
        skill_dir = tmp_path / "broken"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("garbage", encoding="utf-8")
        with pytest.warns(UserWarning):
            library = SkillLibrary(tmp_path)
        assert [s.name for s in library.skills()] == ["good-skill"]


class TestLoad:
    def test_load_unknown_skill_lists_available(self, tmp_path):
        _make_skill(tmp_path, "pr-review", "pr-review", "Review a PR", "body")
        _make_skill(tmp_path, "deploy", "deploy", "Deploy", "body")
        library = SkillLibrary(tmp_path)
        with pytest.raises(KeyError) as excinfo:
            library.load("nonexistent")
        message = str(excinfo.value)
        assert "nonexistent" in message
        assert "deploy" in message
        assert "pr-review" in message

    def test_load_unknown_skill_empty_library(self, tmp_path):
        library = SkillLibrary(tmp_path)
        with pytest.raises(KeyError, match=r"\(none\)"):
            library.load("anything")


class TestReload:
    def test_reload_picks_up_new_skill(self, tmp_path):
        library = SkillLibrary(tmp_path)
        assert library.skills() == []
        _make_skill(tmp_path, "new-skill", "new-skill", "Just added", "body")
        library.reload()
        assert [s.name for s in library.skills()] == ["new-skill"]

    def test_reload_drops_removed_skill(self, tmp_path):
        _make_skill(tmp_path, "temp", "temp-skill", "Temporary", "body")
        library = SkillLibrary(tmp_path)
        assert [s.name for s in library.skills()] == ["temp-skill"]
        import shutil

        shutil.rmtree(tmp_path / "temp")
        library.reload()
        assert library.skills() == []


class TestSkillDataclass:
    def test_skill_is_frozen_dataclass_with_expected_fields(self, tmp_path):
        _make_skill(tmp_path, "deploy", "deploy", "Deploy the service", "body text")
        library = SkillLibrary(tmp_path)
        skill = library.skills()[0]
        assert isinstance(skill, Skill)
        assert skill.name == "deploy"
        assert skill.description == "Deploy the service"
        assert skill.path.name == "SKILL.md"
        assert skill.body == "body text"
        with pytest.raises(Exception):
            skill.name = "changed"  # frozen dataclass
