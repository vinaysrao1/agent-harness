"""Build a labeled corpus of self-verification outcomes for judge evaluation.

Every record is a trial where the harness's own self-verification gate
PASSED. The label is what the real grader said afterwards:

    true_pass   - self-verify passed, grader agreed          (108)
    false_pass  - self-verify passed, grader scored 0        ( 33)

The false_pass class is the defect we are trying to detect: the model
authored both the solution and the check that blessed it.

Workspaces are not preserved (containers are deleted after a trial), so
the final workspace is *reconstructed* by replaying the run's file-writing
tool calls, last write wins. That is an approximation of live state - a
file later mutated by a bash command is not captured - and it is the
reason offline results are directional, not conclusive.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3

WRITE_TOOLS = {"write_file", "edit_file"}

#: `cat > path <<'EOF'` / `tee -a path <<EOF`. Agents create most of their
#: files this way rather than through write_file (188 vs 136 across the
#: corpus), so a reconstruction that ignored heredocs would hand the judge a
#: nearly empty workspace and fail for the wrong reason.
_HEREDOC = re.compile(
    r"(?:cat|tee)\s*(?:-a\s*)?>\s*'?\"?(?P<path>[^\s'\"<>|]+)'?\"?\s*"
    r"<<\s*(?P<quote>'?)(?P<delim>[A-Za-z_][A-Za-z0-9_]*)(?P=quote)"
)


def _heredoc_files(command: str) -> dict[str, str]:
    """Extract files a shell command writes via heredoc, best-effort.

    Scans for each heredoc opener and takes everything up to a line equal to
    its delimiter. Anything that does not parse cleanly is skipped rather
    than guessed at - a wrong reconstruction is worse than a missing one.
    """
    files: dict[str, str] = {}
    for m in _HEREDOC.finditer(command):
        path, delim = m.group("path"), m.group("delim")
        rest = command[m.end():]
        lines = rest.split("\n")[1:]  # body starts on the next line
        body: list[str] = []
        for line in lines:
            if line.strip() == delim:
                files[path] = "\n".join(body)
                break
            body.append(line)
    return files


def _events(db: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        for kind, payload in con.execute(
            "SELECT kind, payload FROM transcript_events ORDER BY seq"
        ):
            try:
                out.append((kind, json.loads(payload)))
            except Exception:
                continue
    finally:
        con.close()
    return out


def _reward(doc: dict) -> int:
    vr = doc.get("verifier_result") or {}
    r = (vr.get("rewards") or {}).get("reward", vr.get("reward"))
    return 1 if r else 0


def build_record(result_path: str, run: str) -> dict | None:
    try:
        doc = json.load(open(result_path))
    except Exception:
        return None
    db = os.path.join(os.path.dirname(result_path), "agent", "harness-home", "state.db")
    if not os.path.exists(db):
        return None
    try:
        events = _events(db)
    except Exception:
        return None

    if not any(k == "verification_passed" for k, _ in events):
        return None  # only trials the self-verify gate actually blessed

    instruction, final_answer = "", ""
    workspace: dict[str, str] = {}
    actions: list[dict] = []
    declared, lint_kinds = None, []

    pending: dict[str, dict] = {}
    for kind, p in events:
        if kind == "message":
            role, content = p.get("role"), p.get("content") or ""
            if role == "user" and not instruction:
                instruction = content
            elif role == "assistant" and content:
                final_answer = content
        elif kind == "tool_call":
            name, args = p.get("name"), p.get("arguments") or {}
            cid = p.get("id") or p.get("tool_call_id") or ""
            if name in WRITE_TOOLS:
                path = args.get("path") or args.get("file_path")
                body = args.get("content") or args.get("new_str") or args.get("new_string")
                if path and body is not None:
                    workspace[path] = body  # last write wins
            elif name == "bash":
                cmd = args.get("command") or args.get("cmd") or ""
                workspace.update(_heredoc_files(cmd))  # last write wins
                pending[cid] = {"cmd": cmd}
                actions.append(pending[cid])
        elif kind == "tool_result":
            cid = p.get("tool_call_id") or ""
            if cid in pending:
                pending[cid]["out"] = (p.get("content") or "")[:1200]
        elif kind == "verification_declared":
            declared = p.get("command") or declared
        elif kind == "verification_passed":
            declared = p.get("command") or declared
            lint_kinds = [f.get("kind") for f in (p.get("lint_findings") or [])]

    return {
        "task": doc.get("task_name"),
        "run": run,
        "label": "true_pass" if _reward(doc) else "false_pass",
        "instruction": instruction,
        "declared_command": declared,
        "lint_kinds": lint_kinds,
        "workspace": workspace,
        "actions": actions,
        "final_answer": final_answer,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobdirs", nargs="+")
    ap.add_argument("-o", "--out", default="tools_eval/judge_eval/corpus.jsonl")
    args = ap.parse_args()

    records = []
    for d in args.jobdirs:
        run = os.path.basename(d.rstrip("/"))
        for f in sorted(glob.glob(os.path.join(d, "*", "result.json"))):
            rec = build_record(f, run)
            if rec:
                records.append(rec)

    with open(args.out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    n_false = sum(1 for r in records if r["label"] == "false_pass")
    chars = sum(len(json.dumps(r)) for r in records)
    ws = sum(len(r["workspace"]) for r in records)
    print(f"records      : {len(records)}  (true_pass {len(records)-n_false}, false_pass {n_false})")
    print(f"files rebuilt: {ws}  ({ws/max(1,len(records)):.1f} per trial)")
    print(f"total size   : {chars/1e6:.2f} MB  (~{chars/4/1e6:.2f}M tokens if sent whole)")
    print(f"median record: {sorted(len(json.dumps(r)) for r in records)[len(records)//2]/1024:.0f} KB")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
