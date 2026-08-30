"""Background commands, tracked by the harness (S-104).

A build that outruns the exec cap currently returns nothing: the command is
killed at the timeout and the agent learns only that it took too long. That is
a whole class of Terminal-Bench task — `build-pov-ray` is one — where the right
move is to start the build, do something else, and come back to it.

Built entirely on :meth:`Sandbox.exec`, deliberately. A background primitive on
the sandbox ABC would need a correct implementation in every backend and a new
contract for handles that outlive a call; redirecting to a file and polling it
needs neither, works identically under Docker and Local, and keeps the process
tree inside the sandbox where the existing orphan-kill machinery already
reaches it.

The registry exists because a started job is a *promise*. Nothing else in the
harness knows the process is there, so nothing else can notice that the run
ended with it still going, or that the agent started it and never looked.
"""

from __future__ import annotations

import shlex
import uuid
from dataclasses import dataclass, field

__all__ = [
    "JOBS_DIR",
    "Job",
    "JobRegistry",
    "ABANDONED_EVENT",
    "start_command",
    "poll_command",
    "kill_command",
    "kill_process_command",
    "parse_poll",
]

#: Where job output lands, inside the sandbox. Under ``/tmp`` for the same
#: reason the shadow git store is: it is harness bookkeeping and must not
#: appear in the workspace the agent is being judged on.
JOBS_DIR = "/tmp/.harness-jobs"

#: Emitted when a run ends with a job that was started and never polled.
#: Measurement first: whether the model actually comes back for its build is
#: the question this feature exists to answer, and a mechanism nobody uses
#: should be visible as unused rather than counted as shipped.
ABANDONED_EVENT = "background_job_abandoned"


@dataclass
class Job:
    """One background command."""

    handle: str
    command: str
    pid: str
    #: The process group measured at start time, not assumed from the pid.
    pgid: str = ""
    polled: bool = False
    killed: bool = False
    #: Set once a poll has seen the exit sentinel. ``None`` means "no exit
    #: code observed", which is not the same as "still running" -- see
    #: ``finished``.
    exit_code: str | None = None
    #: The job is known to have exited. Distinct from ``killed``: a job that
    #: ran to completion was never killed, and reporting it as still running
    #: told the model to collect output it had already read -- and made the
    #: reap signal long-dead pids, which in a container recycle fast enough
    #: to hit an unrelated process group. Distinct from ``exit_code is not
    #: None`` too: a job that finishes before its group can even be read is
    #: known to be over while its exit code still sits unread in the sentinel.
    finished: bool = False
    #: Bytes of output already returned, so `since` can resume rather than
    #: re-sending a growing log every poll.
    offset: int = 0
    #: Whether `background_job_abandoned` has already been emitted for this
    #: job. Reaping now happens at the landing turn *and* at teardown, so
    #: without this a run that reached landing emitted the event twice while
    #: one that errored emitted it once -- an uneven double-count of the one
    #: number this feature exists to produce.
    abandonment_reported: bool = False


@dataclass
class JobRegistry:
    """Every background job this agent started."""

    _jobs: dict[str, Job] = field(default_factory=dict)
    _counter: int = 0

    #: Distinguishes this registry's files from every other registry's. Job
    #: output lives at a fixed path derived from the handle, and handles
    #: restart at 1 per registry -- so two agents sharing a sandbox, or two
    #: runs against the same container, both wrote to `job-1.out` and read
    #: each other's exit codes.
    _token: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def next_handle(self) -> str:
        self._counter += 1
        return f"job-{self._token}-{self._counter}"

    def add(self, job: Job) -> None:
        self._jobs[job.handle] = job

    def get(self, handle: str) -> Job | None:
        return self._jobs.get(handle)

    def all(self) -> list[Job]:
        return list(self._jobs.values())

    def live(self) -> list[Job]:
        """Jobs neither killed nor known to have finished.

        The set a run must not end holding. "Not killed" alone was wrong:
        nothing marks natural completion, so a job polled to its exit code
        stayed live forever -- announced at wind-down as still running, and
        signalled at landing on a pid the kernel had long since reused.
        """
        return [
            job for job in self._jobs.values()
            if not job.killed and not job.finished
        ]

    def abandoned(self) -> list[Job]:
        """Started, never polled, not yet reported. A build nobody came back for.

        Excluding the already-reported is not bookkeeping tidiness: the reap
        runs twice on a run that reaches landing and once on a run that
        errors, so counting the raw events would say abandonment is twice as
        common on successful runs.
        """
        return [
            job for job in self._jobs.values()
            if not job.polled and not job.abandonment_reported
        ]


#: Reads a process's group id without `ps`, which the sandbox image does not
#: have. Field 5 of `/proc/<pid>/stat` is the pgrp on Linux; BSD and macOS
#: have no `/proc` but do have `ps`. Either answer is fine; *no* answer is not,
#: because an unverifiable process group is one that may not exist.
_READ_PGID = (
    "awk '{{print $5}}' /proc/{pid}/stat 2>/dev/null "
    "|| ps -o pgid= -p {pid} 2>/dev/null | tr -d ' '"
)


def start_command(handle: str, command: str) -> str:
    """Shell that launches ``command`` detached and prints ``pid pgid``.

    The job must end up in **its own process group**, because a build spawns
    compilers and signalling only the job's shell leaves them running. Getting
    that wrong has now happened twice, both times silently:

    - ``setsid`` is a Linux tool, absent on macOS. Used with a plain-subshell
      fallback, every host-side run took the fallback and the kill reached
      only the shell.
    - ``set -m`` is POSIX job control, and **dash turns it off** when there is
      no controlling terminal — which is every ``docker exec`` without ``-t``,
      i.e. the production path. It prints ``can't access tty; job control
      turned off`` to stderr and carries on, so the job inherits the sandbox
      exec's group and a later ``kill -TERM -<pid>`` is ESRCH.

    So this stops asserting and starts **measuring**: whichever mechanism is
    available is used, and the real group id is read back and returned. The
    caller compares it to the pid and refuses the job if they differ. A
    mechanism that cannot be verified is not a mechanism.

    The closing parenthesis is on its own line because the command's last line
    may be a heredoc terminator or a ``#`` comment, and appending ``)`` to it
    makes the whole thing a syntax error — which presented as a job that
    started, produced nothing, and polled as running forever.

    A subshell rather than a brace group: ``exit 3`` inside braces terminates
    the outer shell, so the exit sentinel is never written.
    """
    out = f"{JOBS_DIR}/{handle}.out"
    rc = f"{JOBS_DIR}/{handle}.rc"
    inner = (
        f"(\n{command}\n) > {shlex.quote(out)} 2>&1\n"
        f"echo $? > {shlex.quote(rc)}\n"
    )
    quoted = shlex.quote(inner)
    read_pgid = _READ_PGID.format(pid="$pid")
    return (
        # `|| exit 1`, not `&&`: a `;` after an `&&` list ends it, so a failed
        # mkdir still launched the job -- with no output file, no sentinel, and
        # a poll that said "still running" forever.
        f"mkdir -p {JOBS_DIR} || exit 1\n"
        f"if command -v setsid >/dev/null 2>&1; then\n"
        f"  setsid sh -c {quoted} </dev/null >/dev/null 2>&1 &\n"
        f"else\n"
        f"  set -m 2>/dev/null\n"
        f"  sh -c {quoted} </dev/null >/dev/null 2>&1 &\n"
        f"fi\n"
        f"pid=$!\n"
        f"pgid=$({read_pgid})\n"
        # A short command can finish before its group can be read, and a
        # process that no longer exists has no group and nothing to kill.
        # Distinguishing "already gone" from "could not be isolated" matters:
        # the first is fine, the second must refuse the job. Conflating them
        # rejected every fast job.
        f'if [ -z "$pgid" ]; then\n'
        f'  if kill -0 "$pid" 2>/dev/null; then pgid=unknown; else pgid=gone; fi\n'
        f"fi\n"
        f'echo "$pid $pgid"\n'
    )


def poll_command(handle: str, offset: int) -> str:
    """Shell that reports new output, whether the job finished, and the offset.

    The new offset comes from ``wc -c`` on the file itself rather than from
    the length of the text the harness received. Deriving it from the decoded
    string was wrong twice over: a non-UTF-8 byte decodes to U+FFFD and
    re-encodes to three, and the harness's own truncation marker was counted
    as job output. Either desynchronised the offset permanently, and the
    symptom was a job that "finished with nothing to show" while the rest of
    its log sat in a file nobody would ever read again.
    """
    out = f"{JOBS_DIR}/{handle}.out"
    rc = f"{JOBS_DIR}/{handle}.rc"
    return (
        f"if [ -f {shlex.quote(rc)} ]; then "
        f"echo \"__HARNESS_EXIT__$(cat {shlex.quote(rc)})\"; "
        f"else echo __HARNESS_RUNNING__; fi\n"
        f"echo \"__HARNESS_OFFSET__$(wc -c < {shlex.quote(out)} 2>/dev/null || echo {offset})\"\n"
        f"tail -c +{offset + 1} {shlex.quote(out)} 2>/dev/null || true\n"
    )


def kill_command(pid: str) -> str:
    """Shell that terminates a job's process group, then the process itself.

    Takes the pid alone, because a registered job's group *is* its pid:
    `_start_background` refuses any job whose measured `pgid` differs, and
    assigns `pgid = pid` on the already-finished path. A `pgid` parameter
    would only ever carry the value it was compared against.

    The negative form targets the group; the plain form is the backstop for
    the process itself. Both are sent because a job that lost its group
    between start and kill should still have its own shell stopped.

    Worth keeping in view: `kill -TERM -<pid>` on a process that is *not* a
    group leader signals whatever group it happens to be in, which on a
    shared group is the sandbox's own shell. That is why the refusal above
    exists rather than a best-effort kill, and why nothing calls this with a
    pid it has not first verified to be a group leader.
    """
    single = shlex.quote(str(pid))
    group = f"-{single}"
    return (
        f"kill -TERM {group} 2>/dev/null; kill -TERM {single} 2>/dev/null\n"
        f"sleep 0.1\n"
        f"kill -KILL {group} 2>/dev/null; kill -KILL {single} 2>/dev/null\n"
        f"true\n"
    )


def kill_process_command(pid: str) -> str:
    """Shell that terminates one process, never a group.

    For the refusal path only: a job whose measured group is not its own pid
    is in *somebody else's* group -- on a `docker exec` that is the sandbox's
    own shell -- so `kill -TERM -<pid>` there would signal the sandbox. The
    started shell still has to be stopped; only it.
    """
    single = shlex.quote(str(pid))
    return (
        f"kill -TERM {single} 2>/dev/null\n"
        f"sleep 0.1\n"
        f"kill -KILL {single} 2>/dev/null\n"
        f"true\n"
    )


def parse_poll(raw: str) -> tuple[str | None, str, int | None]:
    """``(exit_code, output, offset)`` from :func:`poll_command`'s stdout.

    ``None`` for the exit code means still running. Parsed from a sentinel
    rather than from the process table because a finished job leaves no
    process to inspect, and "no such pid" is indistinguishable from "never
    started". The offset is the file's byte length as the *shell* measured it.
    """
    lines = raw.split("\n")
    exit_code: str | None = None
    offset: int | None = None
    body_start = 0
    for index, line in enumerate(lines[:2]):
        if line.startswith("__HARNESS_EXIT__"):
            exit_code = line[len("__HARNESS_EXIT__"):].strip() or "?"
            body_start = index + 1
        elif line.startswith("__HARNESS_RUNNING__"):
            body_start = index + 1
        elif line.startswith("__HARNESS_OFFSET__"):
            raw_offset = line[len("__HARNESS_OFFSET__"):].strip()
            offset = int(raw_offset) if raw_offset.isdigit() else None
            body_start = index + 1
    return exit_code, "\n".join(lines[body_start:]), offset
