"""S-001: the spec process enforces itself.

T1 every spec carries complete, well-formed front-matter
T2 every spec at Accepted or later is named by at least one test
T3 every transcript event kind is grandfathered or owned by a spec

Each rule has a negative test. A traceability check that has never been seen to
reject anything is not known to work -- the same argument S-002 makes for the
neutrality invariants.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.specs import (
    EVENT_KIND_SPECS,
    LANES,
    STATUSES,
    LEGACY_EVENT_KINDS,
    REQUIRED_FIELDS,
    SpecError,
    event_kinds_in_source,
    load_specs,
    parse_spec,
    render_index,
    spec_ids_named_by_tests,
    stale_legacy_kinds,
    unnamed_specs,
    unowned_event_kinds,
    write_index,
)

ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = ROOT / "specs"
TESTS_DIR = ROOT / "tests"
PACKAGE_DIR = ROOT / "harness"

VALID = """---
id: S-042
title: A valid spec
status: Accepted
lane: A
depends: S-001
effort: M
---

body
"""


def _write(tmp_path: Path, text: str, name: str = "S-042-thing.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestT1FrontMatter:
    def test_S001_every_spec_parses(self) -> None:
        specs = load_specs(SPECS_DIR)
        assert specs, "no specs found -- the suite would vacuously pass"
        for spec in specs:
            assert spec.status in STATUSES
            assert spec.lane in LANES
            assert spec.path.name.startswith(spec.id + "-")

    def test_S001_valid_front_matter_round_trips(self, tmp_path: Path) -> None:
        spec = parse_spec(_write(tmp_path, VALID))
        assert spec.id == "S-042"
        assert spec.status == "Accepted"
        assert spec.depends == ("S-001",)

    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    def test_S001_missing_any_required_field_is_rejected(
        self, tmp_path: Path, field: str
    ) -> None:
        text = "\n".join(
            line for line in VALID.splitlines() if not line.startswith(f"{field}:")
        )
        with pytest.raises(SpecError, match="missing front-matter keys"):
            parse_spec(_write(tmp_path, text))

    def test_S001_unknown_field_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SpecError, match="unknown front-matter keys"):
            parse_spec(_write(tmp_path, VALID.replace("effort: M", "effort: M\nowner: x")))

    @pytest.mark.parametrize(
        "bad,match",
        [
            ("id: S42", "not of the form"),
            ("status: Cooking", "status 'Cooking' not one of"),
            ("lane: C", "lane 'C' not one of"),
            ("depends: nonsense", "not a spec id"),
        ],
    )
    def test_S001_malformed_values_are_rejected(
        self, tmp_path: Path, bad: str, match: str
    ) -> None:
        key = bad.split(":")[0]
        text = "\n".join(
            bad if line.startswith(f"{key}:") else line for line in VALID.splitlines()
        )
        with pytest.raises(SpecError, match=match):
            parse_spec(_write(tmp_path, text))

    def test_S001_missing_fence_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SpecError, match="missing opening"):
            parse_spec(_write(tmp_path, "id: S-042\n"))

    def test_S001_unterminated_fence_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SpecError, match="unterminated"):
            parse_spec(_write(tmp_path, "---\nid: S-042\n"))

    def test_S001_filename_must_match_id(self, tmp_path: Path) -> None:
        with pytest.raises(SpecError, match="filename must start with"):
            parse_spec(_write(tmp_path, VALID, name="S-999-wrong.md"))

    def test_S001_dangling_dependency_is_rejected(self, tmp_path: Path) -> None:
        _write(tmp_path, VALID.replace("depends: S-001", "depends: S-777"))
        with pytest.raises(SpecError, match="unknown spec"):
            load_specs(tmp_path)

    def test_S001_duplicate_id_is_rejected(self, tmp_path: Path) -> None:
        _write(tmp_path, VALID.replace("depends: S-001", "depends: -"))
        _write(
            tmp_path,
            VALID.replace("depends: S-001", "depends: -"),
            name="S-042-other.md",
        )
        with pytest.raises(SpecError, match="duplicate spec id"):
            load_specs(tmp_path)


class TestT2SpecsAreNamedByTests:
    def test_S001_accepted_specs_have_a_naming_test(self) -> None:
        named = spec_ids_named_by_tests(TESTS_DIR)
        unnamed = unnamed_specs(load_specs(SPECS_DIR), named)
        assert not unnamed, (
            f"specs at Accepted or later with no test naming them: {unnamed}. "
            f"Add a test called test_{unnamed[0].replace('-', '') if unnamed else 'S000'}_<what>."
        )

    def test_S001_draft_specs_are_exempt(self, tmp_path: Path) -> None:
        # Built from a synthetic spec, not from whatever the repo contains: the
        # repo's only Draft becomes Accepted the moment S-002 lands, at which
        # point a repo-derived version of this test would silently go vacuous.
        draft = parse_spec(_write(tmp_path, VALID.replace("status: Accepted", "status: Draft")))
        assert not draft.needs_test
        assert unnamed_specs([draft], named=set()) == []

    def test_S001_test_name_scan_finds_this_module(self) -> None:
        # Negative control for the scanner: if the regex silently matched
        # nothing, T2 would pass by not looking.
        assert "S-001" in spec_ids_named_by_tests(TESTS_DIR)


class TestT3EventKindsAreOwned:
    def test_S001_every_emitted_event_kind_is_accounted_for(self) -> None:
        emitted = event_kinds_in_source(PACKAGE_DIR)
        assert emitted, "no event kinds found -- the scan is broken"
        unowned = unowned_event_kinds(emitted)
        assert not unowned, (
            f"event kinds with no owning spec: {unowned}. Register each in "
            "harness.specs.EVENT_KIND_SPECS against the spec that introduced it."
        )

    def test_S001_legacy_set_matches_reality(self) -> None:
        # Guards the other direction: a legacy kind deleted from the source
        # should be removed from the frozen set, not left to rot.
        stale = stale_legacy_kinds(event_kinds_in_source(PACKAGE_DIR))
        assert not stale, f"LEGACY_EVENT_KINDS lists kinds no longer emitted: {stale}"

    def test_S001_unresolvable_event_kind_raises(self, tmp_path: Path) -> None:
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "m.py").write_text(
            "KIND = compute()\n"
            "def f(store, agent_id):\n    store.append_event(agent_id, KIND, {})\n",
            encoding="utf-8",
        )
        with pytest.raises(SpecError, match="cannot resolve event kind"):
            event_kinds_in_source(pkg)

    def test_S001_computed_event_kind_raises(self, tmp_path: Path) -> None:
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "m.py").write_text(
            'def f(store, a):\n    store.append_event(a, "x" + "y", {})\n',
            encoding="utf-8",
        )
        with pytest.raises(SpecError, match="computed expression"):
            event_kinds_in_source(pkg)


class TestIndexIsDeterministic:
    def test_S001_index_on_disk_is_current(self) -> None:
        expected = render_index(load_specs(SPECS_DIR))
        actual = (SPECS_DIR / "_index.md").read_text(encoding="utf-8")
        assert actual == expected, "run `python -m harness.specs` to regenerate"

    def test_S001_render_is_independent_of_filesystem_order(
        self, tmp_path: Path
    ) -> None:
        # The real determinism risk is directory iteration order, which
        # comparing render_index(x) to render_index(x) never touches.
        for name, spec_id in (("S-003-c.md", "S-003"), ("S-001-a.md", "S-001")):
            _write(tmp_path, VALID.replace("S-042", spec_id).replace(
                "depends: S-001", "depends: -"), name=name)
        first = render_index(load_specs(tmp_path))
        # Written S-003 first, S-001 second: the index must still order by id,
        # not by whatever order the directory happens to yield.
        assert first.index("S-001") < first.index("S-003")
        assert render_index(load_specs(tmp_path)) == first

    def test_S001_template_is_not_parsed_as_a_spec(self) -> None:
        # The template carries a placeholder id and must never enter the index.
        assert "S-000" not in render_index(load_specs(SPECS_DIR))


class TestRulesRejectViolations:
    """The rules must be shown to FAIL, not merely to pass on a clean repo.

    Every check above asserts the repository is compliant. That is necessary
    but proves nothing about the rule: setting STATUSES_REQUIRING_TESTS to
    empty, or making the scanner return every id, leaves those assertions
    green. These tests drive each rule with a known violation.
    """

    def test_S001_t2_rejects_an_accepted_spec_with_no_test(
        self, tmp_path: Path
    ) -> None:
        spec = parse_spec(_write(tmp_path, VALID.replace("depends: S-001", "depends: -")))
        assert spec.needs_test
        assert unnamed_specs([spec], named=set()) == ["S-042"]
        assert unnamed_specs([spec], named={"S-042"}) == []

    def test_S001_t3_rejects_an_unregistered_event_kind(self, tmp_path: Path) -> None:
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "m.py").write_text(
            'def f(store, a):\n    store.append_event(a, "brand_new_kind", {})\n',
            encoding="utf-8",
        )
        kinds = event_kinds_in_source(pkg)
        assert kinds == {"brand_new_kind"}
        assert unowned_event_kinds(kinds) == ["brand_new_kind"]

    def test_S001_t3_sees_a_kind_passed_by_keyword(self, tmp_path: Path) -> None:
        # Regression: reading args[1] unconditionally skipped keyword calls in
        # silence, so a new event kind could enter the package unaudited.
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "m.py").write_text(
            'def f(store, a):\n    store.append_event(a, kind="kw_kind", payload={})\n',
            encoding="utf-8",
        )
        assert event_kinds_in_source(pkg) == {"kw_kind"}

    def test_S001_t3_sees_a_bare_append_event_call(self, tmp_path: Path) -> None:
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "m.py").write_text(
            'def f(append_event, a):\n    append_event(a, "bare_kind", {})\n',
            encoding="utf-8",
        )
        assert event_kinds_in_source(pkg) == {"bare_kind"}

    @pytest.mark.parametrize(
        "body,match",
        [
            ("def f(s, a, k):\n    s.append_event(*k)\n", r"\*args"),
            ("def f(s, a, k):\n    s.append_event(a, **k)\n", r"\*\*kwargs"),
            ("def f(s, a):\n    s.append_event(a)\n", "no resolvable 'kind'"),
        ],
    )
    def test_S001_t3_raises_on_unauditable_call_shapes(
        self, tmp_path: Path, body: str, match: str
    ) -> None:
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "m.py").write_text(body, encoding="utf-8")
        with pytest.raises(SpecError, match=match):
            event_kinds_in_source(pkg)

    def test_S001_t3_resolves_an_imported_constant(self, tmp_path: Path) -> None:
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "kinds.py").write_text(
            'from typing import Final\nMY_EVENT: Final[str] = "resolved_kind"\n',
            encoding="utf-8",
        )
        (pkg / "m.py").write_text(
            "from pkg.kinds import MY_EVENT\n"
            "def f(s, a):\n    s.append_event(a, MY_EVENT, {})\n",
            encoding="utf-8",
        )
        assert event_kinds_in_source(pkg) == {"resolved_kind"}

    def test_S001_t3_does_not_resolve_an_unimported_lookalike(
        self, tmp_path: Path
    ) -> None:
        # A parameter shadowing an unrelated module's constant must NOT resolve
        # to that constant's value -- a confidently wrong kind is worse than a
        # loud failure, because the real kind then goes unaudited.
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "other.py").write_text('KIND = "value_from_elsewhere"\n', encoding="utf-8")
        (pkg / "m.py").write_text(
            "def f(s, a):\n    s.append_event(a, KIND, {})\n", encoding="utf-8"
        )
        with pytest.raises(SpecError, match="cannot resolve event kind"):
            event_kinds_in_source(pkg)

    def test_S001_t2_ignores_a_spec_id_in_a_comment(self, tmp_path: Path) -> None:
        # A comment is not a test. Counting it would let a spec claim coverage
        # it does not have.
        tests = tmp_path / "t"
        tests.mkdir()
        (tests / "test_fake.py").write_text(
            '# def test_S900_not_real\n'
            '"""def test_S901_also_not_real"""\n'
            'BLAH = "def test_S902_string"\n'
            "def test_S903_real():\n    pass\n",
            encoding="utf-8",
        )
        assert spec_ids_named_by_tests(tests) == {"S-903"}

    def test_S001_dependency_cycle_is_rejected(self, tmp_path: Path) -> None:
        _write(tmp_path, VALID.replace("depends: S-001", "depends: S-043"))
        _write(
            tmp_path,
            VALID.replace("S-042", "S-043").replace("depends: S-001", "depends: S-042"),
            name="S-043-other.md",
        )
        with pytest.raises(SpecError, match="dependency cycle"):
            load_specs(tmp_path)

    def test_S001_self_dependency_is_rejected(self, tmp_path: Path) -> None:
        _write(tmp_path, VALID.replace("depends: S-001", "depends: S-042"))
        with pytest.raises(SpecError, match="dependency cycle"):
            load_specs(tmp_path)

    def test_S001_filename_prefix_respects_a_boundary(self, tmp_path: Path) -> None:
        with pytest.raises(SpecError, match="filename must start with"):
            parse_spec(_write(tmp_path, VALID, name="S-0429-other.md"))

    def test_S001_write_index_refuses_an_empty_directory(self, tmp_path: Path) -> None:
        with pytest.raises(SpecError, match="refusing to write an empty index"):
            write_index(tmp_path)

    def test_S001_write_index_is_idempotent(self, tmp_path: Path) -> None:
        _write(tmp_path, VALID.replace("depends: S-001", "depends: -"))
        first = write_index(tmp_path).read_text(encoding="utf-8")
        assert write_index(tmp_path).read_text(encoding="utf-8") == first

    def test_S001_template_carries_the_seed_sections(self) -> None:
        # AC-3: the template is what a `harness run --spec` would seed into the
        # instruction ledger, so its sections are load-bearing.
        text = (SPECS_DIR / "_template.md").read_text(encoding="utf-8")
        for heading in ("## Contract", "## Invariants", "## Acceptance",
                        "## Telemetry", "## Rollback", "## Neutrality argument"):
            assert heading in text, f"template is missing {heading}"

    def test_S001_stale_legacy_kind_is_reported(self) -> None:
        assert stale_legacy_kinds(set(LEGACY_EVENT_KINDS)) == []
        assert stale_legacy_kinds(set()) == sorted(LEGACY_EVENT_KINDS)

    def test_S001_t3_refuses_a_parameter_shadowing_an_import(
        self, tmp_path: Path
    ) -> None:
        # A parameter shadows the module-level import, so resolving through
        # that import would yield a confidently wrong kind while the real one
        # went unaudited.
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "kinds.py").write_text('K = "from_kinds"\n', encoding="utf-8")
        (pkg / "m.py").write_text(
            "from pkg.kinds import K\n"
            "def f(s, a, K):\n    s.append_event(a, K, {})\n",
            encoding="utf-8",
        )
        with pytest.raises(SpecError, match="parameter of the enclosing function"):
            event_kinds_in_source(pkg)

    def test_S001_t3_keyword_scan_is_order_independent(self, tmp_path: Path) -> None:
        # Rejecting on **kwargs mid-loop made the outcome depend on argument
        # order; an explicit kind must win regardless of where it appears.
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "m.py").write_text(
            "def f(s, a, extra):\n"
            '    s.append_event(a, **extra, kind="after_kwargs")\n',
            encoding="utf-8",
        )
        assert event_kinds_in_source(pkg) == {"after_kwargs"}
