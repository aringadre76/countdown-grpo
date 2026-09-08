---
name: learn-commit-push
description: >
  Record durable project facts, validate intentional changes, make a concise
  Conventional Commit, push it, and prove the checkout is synchronized.
---

# Learn, Commit, Push

This is the project-specific version of the workflow copied from
`cursor-chat`. Finish with an empty working tree and `main` equal to
`origin/main`.

## 1. Preserve project continuity

Before staging, read `AGENTS.md`, `docs/README.md`, `docs/lessons.md`,
`docs/next-steps.md`, and `CHANGELOG.md` when they exist. Record only durable
project facts in their canonical documentation file:

- measured results and environment probes: README, hardware, or artifacts;
- reproducible commands and protocol changes: reproduction or protocol;
- reusable implementation or workflow lessons: lessons;
- intentional shipped changes: changelog;
- global routing or safety rules: AGENTS.

Do not import raw chat transcripts, credentials, model completions, or private
host paths into project documentation. The adjacent
`agents-memory-routing.md` is retained as the source workflow's historical
reference; this research repository uses its documentation set instead of a
general personal-memory registry.

## 2. Inspect and validate

Inspect `git status --short`, unstaged and staged diffs, and the last three
commits. Run the checks appropriate to the change. For normal project work:

```bash
source .venv/bin/activate
pytest
ruff check .
git diff --check
```

Rerun the bounded model path when it changed, and record actual commands and
outputs. Resolve failures before staging; do not call an unrun check passing.

## 3. Stage and commit

Stage only intentional files. Exclude credentials, downloaded weights, caches,
and unrelated work. Use a concise Conventional Commit subject in imperative
mood, ideally under 50 characters. Add a wrapped body only when the reason is
not clear from the diff. Do not amend, bypass hooks, alter Git configuration,
or add AI attribution.

## 4. Push and prove synchronization

Before pushing, run `git fetch --prune origin` and compare the current branch
with its upstream. If upstream has commits not in the local branch, rebase the
local commits onto it, then revalidate. Never discard work, force-push, or
rewrite `main`.

Push normally. Then fetch again and require:

```bash
git fetch --prune origin
test "$(git rev-parse HEAD)" = "$(git rev-parse @{upstream})"
test -z "$(git status --short)"
```

Report the continuity files updated, checks run, commit SHA and subject, push
result, ref equality, and clean-tree result.
