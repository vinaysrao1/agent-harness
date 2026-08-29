# The default sandbox image (DESIGN.md §4.7: `harness-sandbox:latest`).
#
# DESIGN.md has named this image since the sandbox was written and nothing
# ever built it, so every Docker-backed `harness run` failed on an image pull
# that could not succeed. The eval (S-401) is the first thing that needed a
# working one.
#
# Deliberately small. It carries what a coding agent and a grader need and
# nothing else: an interpreter, a test runner, git (S-201's substrate will not
# activate without it), and the search tools the agent's prompt assumes exist.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git ripgrep curl ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

# Test runners plus the dependencies real test suites reach for. Network is
# "none" inside a running sandbox by default, so anything a task needs has to
# be baked in here -- a missing one surfaces as a task that "cannot be solved",
# which sends you looking in entirely the wrong place.
RUN pip install --no-cache-dir \
      pytest pytest-timeout pytest-xdist \
      typing_extensions

# Command-line tools test suites shell out to. `less` in particular: click's
# pager tests parametrize over real pagers and fail without it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends less nano vim-tiny \
 && rm -rf /var/lib/apt/lists/*

# git refuses to write a commit without an identity, and S-201's checkpoints
# are commits.
RUN git config --system user.name "harness" \
 && git config --system user.email "harness@localhost" \
 && git config --system --add safe.directory '*'

WORKDIR /workspace
