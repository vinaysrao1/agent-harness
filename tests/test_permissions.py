"""Tests for the permission engine (harness/permissions.py, DESIGN.md §4.11)."""

from __future__ import annotations

import fnmatch as fnmatch_module

import pytest

from harness.config import PermissionMode
from harness.permissions import (
    HARD_DENY_CATEGORIES,
    Decision,
    Policy,
    ToolMeta,
    evaluate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def meta(*, side_effect: bool = False, categories: frozenset[str] = frozenset()) -> ToolMeta:
    return ToolMeta(side_effect=side_effect, categories=categories)


def policy(
    mode: PermissionMode,
    *,
    allow: list[str] | None = None,
    deny: list[str] | None = None,
) -> Policy:
    return Policy(mode=mode, allow=allow or [], deny=deny or [])


# ---------------------------------------------------------------------------
# Basic mode behavior (no rules)
# ---------------------------------------------------------------------------


def test_gated_mode_readonly_tool_allowed() -> None:
    assert evaluate("fs.read", meta(side_effect=False), policy(PermissionMode.GATED)) == Decision.ALLOW


def test_gated_mode_side_effect_tool_asks() -> None:
    assert evaluate("gmail.send", meta(side_effect=True), policy(PermissionMode.GATED)) == Decision.ASK


def test_auto_mode_readonly_tool_allowed() -> None:
    assert evaluate("fs.read", meta(side_effect=False), policy(PermissionMode.AUTO)) == Decision.ALLOW


def test_auto_mode_side_effect_tool_allowed() -> None:
    assert evaluate("gmail.send", meta(side_effect=True), policy(PermissionMode.AUTO)) == Decision.ALLOW


# ---------------------------------------------------------------------------
# Deny patterns (highest precedence)
# ---------------------------------------------------------------------------


def test_deny_pattern_exact_match_beats_gated_allow() -> None:
    p = policy(PermissionMode.GATED, deny=["fs.read"])
    assert evaluate("fs.read", meta(side_effect=False), p) == Decision.DENY


def test_deny_pattern_glob_match() -> None:
    p = policy(PermissionMode.AUTO, deny=["mcp.github.*"])
    assert evaluate("mcp.github.delete_repo", meta(side_effect=True), p) == Decision.DENY


def test_deny_pattern_beats_auto_mode() -> None:
    p = policy(PermissionMode.AUTO, deny=["shell.exec"])
    assert evaluate("shell.exec", meta(side_effect=True), p) == Decision.DENY


def test_deny_pattern_beats_conflicting_allow_pattern() -> None:
    # deny is checked first regardless of an overlapping allow entry.
    p = policy(PermissionMode.GATED, allow=["mcp.github.*"], deny=["mcp.github.delete_repo"])
    assert evaluate("mcp.github.delete_repo", meta(side_effect=True), p) == Decision.DENY
    assert evaluate("mcp.github.create_issue", meta(side_effect=True), p) == Decision.ALLOW


def test_deny_pattern_no_match_falls_through() -> None:
    p = policy(PermissionMode.AUTO, deny=["mcp.slack.*"])
    assert evaluate("mcp.github.create_issue", meta(side_effect=True), p) == Decision.ALLOW


# ---------------------------------------------------------------------------
# Hard-deny categories (survive auto mode and allow patterns)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", sorted(HARD_DENY_CATEGORIES))
def test_hard_deny_category_denied_in_gated_mode(category: str) -> None:
    p = policy(PermissionMode.GATED)
    m = meta(side_effect=True, categories=frozenset({category}))
    assert evaluate("some.tool", m, p) == Decision.DENY


@pytest.mark.parametrize("category", sorted(HARD_DENY_CATEGORIES))
def test_hard_deny_category_denied_in_auto_mode(category: str) -> None:
    """DESIGN.md: 'auto mode is not zero policy' -- hard-deny categories still block."""
    p = policy(PermissionMode.AUTO)
    m = meta(side_effect=True, categories=frozenset({category}))
    assert evaluate("some.tool", m, p) == Decision.DENY


@pytest.mark.parametrize("category", sorted(HARD_DENY_CATEGORIES))
def test_hard_deny_category_survives_matching_allow_pattern(category: str) -> None:
    """An allow pattern must NOT override a hard-deny category, in either mode."""
    p_gated = policy(PermissionMode.GATED, allow=["mcp.stripe.*"])
    p_auto = policy(PermissionMode.AUTO, allow=["mcp.stripe.*"])
    m = meta(side_effect=True, categories=frozenset({category}))
    assert evaluate("mcp.stripe.charge", m, p_gated) == Decision.DENY
    assert evaluate("mcp.stripe.charge", m, p_auto) == Decision.DENY


def test_hard_deny_category_survives_exact_name_allow() -> None:
    p = policy(PermissionMode.AUTO, allow=["mcp.stripe.charge"])
    m = meta(side_effect=True, categories=frozenset({"payment"}))
    assert evaluate("mcp.stripe.charge", m, p) == Decision.DENY


def test_multiple_categories_one_hard_deny_still_denies() -> None:
    m = meta(side_effect=True, categories=frozenset({"external_send", "payment"}))
    assert evaluate("some.tool", m, policy(PermissionMode.AUTO)) == Decision.DENY


def test_external_send_alone_is_not_hard_denied() -> None:
    # external_send is a category tag but not in HARD_DENY_CATEGORIES: it is
    # gated via side_effect in gated mode and allowed outright in auto mode.
    m = meta(side_effect=True, categories=frozenset({"external_send"}))
    assert evaluate("gmail.send", m, policy(PermissionMode.AUTO)) == Decision.ALLOW
    assert evaluate("gmail.send", m, policy(PermissionMode.GATED)) == Decision.ASK


def test_non_side_effect_tool_with_hard_deny_category_still_denied() -> None:
    # Even a nominally read-only tool tagged with a hard-deny category (e.g.
    # a "read credential" tool) is denied -- category precedence is absolute.
    m = meta(side_effect=False, categories=frozenset({"credential"}))
    assert evaluate("vault.read_secret", m, policy(PermissionMode.GATED)) == Decision.DENY
    assert evaluate("vault.read_secret", m, policy(PermissionMode.AUTO)) == Decision.DENY


# ---------------------------------------------------------------------------
# Allow patterns
# ---------------------------------------------------------------------------


def test_allow_pattern_upgrades_gated_side_effect_to_allow() -> None:
    p = policy(PermissionMode.GATED, allow=["mcp.github.create_issue"])
    m = meta(side_effect=True)
    assert evaluate("mcp.github.create_issue", m, p) == Decision.ALLOW


def test_allow_pattern_glob_match() -> None:
    p = policy(PermissionMode.GATED, allow=["mcp.github.*"])
    m = meta(side_effect=True)
    assert evaluate("mcp.github.create_issue", m, p) == Decision.ALLOW
    assert evaluate("mcp.github.close_issue", m, p) == Decision.ALLOW


def test_allow_pattern_no_match_falls_through_to_mode_logic() -> None:
    p = policy(PermissionMode.GATED, allow=["mcp.slack.*"])
    m = meta(side_effect=True)
    assert evaluate("mcp.github.create_issue", m, p) == Decision.ASK


def test_allow_pattern_irrelevant_in_auto_mode_without_hard_deny() -> None:
    p = policy(PermissionMode.AUTO, allow=["mcp.github.*"])
    m = meta(side_effect=True)
    assert evaluate("mcp.github.create_issue", m, p) == Decision.ALLOW


# ---------------------------------------------------------------------------
# fnmatch pattern-matching edge cases
# ---------------------------------------------------------------------------


def test_pattern_exact_match_no_trailing_wildcard_does_not_prefix_match() -> None:
    p = policy(PermissionMode.GATED, allow=["mcp.github.create_issue"])
    m = meta(side_effect=True)
    # "mcp.github.create_issue_v2" is NOT matched by the exact pattern.
    assert evaluate("mcp.github.create_issue_v2", m, p) == Decision.ASK


def test_pattern_question_mark_wildcard() -> None:
    p = policy(PermissionMode.GATED, allow=["fs.read?"])
    m = meta(side_effect=True)
    assert evaluate("fs.reads", m, p) == Decision.ALLOW
    assert evaluate("fs.read", m, p) == Decision.ASK  # "?" requires one char


def test_pattern_character_class() -> None:
    p = policy(PermissionMode.GATED, allow=["fs.read[123]"])
    m = meta(side_effect=True)
    assert evaluate("fs.read1", m, p) == Decision.ALLOW
    assert evaluate("fs.read4", m, p) == Decision.ASK


def test_pattern_empty_allow_and_deny_lists_do_not_match_anything() -> None:
    p = policy(PermissionMode.GATED)
    m = meta(side_effect=True)
    assert evaluate("anything.at.all", m, p) == Decision.ASK


def test_pattern_star_matches_dots_fnmatch_is_not_path_aware() -> None:
    # fnmatch's "*" matches any character including ".", unlike glob path
    # semantics -- "mcp.*" matches "mcp.github.create_issue" in full.
    p = policy(PermissionMode.GATED, allow=["mcp.*"])
    m = meta(side_effect=True)
    assert evaluate("mcp.github.create_issue", m, p) == Decision.ALLOW


def test_pattern_case_sensitive_on_posix() -> None:
    # evaluate() uses fnmatch.fnmatchcase (not fnmatch.fnmatch), which is
    # always case-sensitive and never applies os.path.normcase -- so this
    # holds identically on every platform, not just POSIX.
    p = policy(PermissionMode.GATED, allow=["MCP.GITHUB.*"])
    m = meta(side_effect=True)
    assert evaluate("mcp.github.create_issue", m, p) == Decision.ASK
    # The pattern does match a name of the same case.
    assert evaluate("MCP.GITHUB.create_issue", m, p) == Decision.ALLOW


def test_matching_uses_fnmatchcase_not_platform_dependent_fnmatch() -> None:
    # Regression: harness/permissions.py must call fnmatch.fnmatchcase, not
    # fnmatch.fnmatch, in _matches_any. fnmatch.fnmatch applies
    # os.path.normcase to both operands, which is a no-op on POSIX but
    # lowercases on Windows -- making policy evaluation platform-dependent
    # for a security-relevant check. fnmatchcase never normalizes case.
    tool_name, pattern = "mcp.github.create_issue", "MCP.GITHUB.*"
    assert fnmatch_module.fnmatchcase(tool_name, pattern) is False
    p = policy(PermissionMode.GATED, allow=[pattern])
    m = meta(side_effect=True)
    assert evaluate(tool_name, m, p) == Decision.ASK


# ---------------------------------------------------------------------------
# Policy.with_grant immutability
# ---------------------------------------------------------------------------


def test_with_grant_returns_new_policy_object() -> None:
    p1 = policy(PermissionMode.GATED)
    p2 = p1.with_grant("mcp.github.*")
    assert p1 is not p2


def test_with_grant_does_not_mutate_original() -> None:
    p1 = policy(PermissionMode.GATED, allow=["a.b"])
    p2 = p1.with_grant("mcp.github.*")
    assert p1.allow == ("a.b",)
    assert p2.allow == ("a.b", "mcp.github.*")


def test_with_grant_changes_evaluation_only_for_new_policy() -> None:
    p1 = policy(PermissionMode.GATED)
    p2 = p1.with_grant("gmail.send")
    m = meta(side_effect=True)
    assert evaluate("gmail.send", m, p1) == Decision.ASK
    assert evaluate("gmail.send", m, p2) == Decision.ALLOW


def test_with_grant_chaining_accumulates() -> None:
    p = policy(PermissionMode.GATED)
    p = p.with_grant("a.*")
    p = p.with_grant("b.*")
    assert p.allow == ("a.*", "b.*")


def test_policy_is_frozen() -> None:
    p = policy(PermissionMode.GATED)
    with pytest.raises(Exception):
        p.allow = ["x"]  # type: ignore[misc]


def test_with_grant_original_allow_list_object_not_shared_mutation() -> None:
    # Mutating the container returned inside the new policy must not reach
    # back into the original policy's container.
    p1 = policy(PermissionMode.GATED, allow=["a.*"])
    p2 = p1.with_grant("b.*")
    assert p2.allow is not p1.allow
    assert "c.*" not in p1.allow


def test_policy_allow_deny_are_truly_immutable_not_just_shallow_frozen() -> None:
    # Regression: pydantic's frozen=True only blocks attribute reassignment
    # (p.allow = [...]); if allow/deny were plain lists, the list object
    # itself would still be mutable in place (p.allow.append(...)), silently
    # flipping evaluate()'s outcome for every holder of the shared Policy.
    # Storing them as tuples makes that mutation a hard AttributeError.
    p = policy(PermissionMode.GATED)
    assert isinstance(p.allow, tuple)
    assert isinstance(p.deny, tuple)
    with pytest.raises(AttributeError):
        p.allow.append("*")  # type: ignore[attr-defined]


def test_shared_policy_object_evaluation_stable_across_holders() -> None:
    # A Policy handed to multiple agents/subagents must not have its
    # evaluate() outcome change out from under any holder.
    p = policy(PermissionMode.GATED)
    m = meta(side_effect=True)
    before = evaluate("gmail.send", m, p)
    with pytest.raises(AttributeError):
        p.allow.append("*")  # type: ignore[attr-defined]
    assert evaluate("gmail.send", m, p) == before == Decision.ASK


# ---------------------------------------------------------------------------
# Full precedence matrix (deny > hard-deny-category > allow > mode)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "deny", "allow", "categories", "side_effect", "expected"),
    [
        # 1. deny beats everything
        (PermissionMode.AUTO, ["t.*"], ["t.*"], frozenset(), True, Decision.DENY),
        (PermissionMode.GATED, ["t.*"], ["t.*"], frozenset(), False, Decision.DENY),
        # 2. hard-deny category beats allow + auto, when not denied
        (PermissionMode.AUTO, [], ["t.*"], frozenset({"payment"}), True, Decision.DENY),
        (PermissionMode.GATED, [], ["t.*"], frozenset({"credential"}), False, Decision.DENY),
        # 3. allow beats mode logic
        (PermissionMode.GATED, [], ["t.*"], frozenset(), True, Decision.ALLOW),
        # 4. auto mode default allow
        (PermissionMode.AUTO, [], [], frozenset(), True, Decision.ALLOW),
        (PermissionMode.AUTO, [], [], frozenset(), False, Decision.ALLOW),
        # 5. gated mode default: side_effect -> ASK, else ALLOW
        (PermissionMode.GATED, [], [], frozenset(), True, Decision.ASK),
        (PermissionMode.GATED, [], [], frozenset(), False, Decision.ALLOW),
    ],
)
def test_precedence_matrix(
    mode: PermissionMode,
    deny: list[str],
    allow: list[str],
    categories: frozenset[str],
    side_effect: bool,
    expected: Decision,
) -> None:
    p = policy(mode, allow=allow, deny=deny)
    m = meta(side_effect=side_effect, categories=categories)
    assert evaluate("t.tool", m, p) == expected


def test_hard_deny_categories_constant_matches_design_doc() -> None:
    assert HARD_DENY_CATEGORIES == frozenset({"credential", "permanent_delete", "payment"})
