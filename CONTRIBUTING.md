# How we work in this repo

We follow [GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow). The goal:
`main` always holds the latest **tested** code, and each of us works on our own branch without
stepping on the others.

## The rules

1. **Never commit to `main` directly.** Everything reaches `main` through a pull request (PR).
2. **One branch per piece of work**, named `<your-name>-<topic>`, for example `tanvir-treebench`,
   `bhushan-python-agent`, `tanish-mme-realworld`. Keep branches short-lived: merge, delete, start a new one.
3. **Keep unrelated changes on separate branches.** A dataset port and a CI change are two PRs.
4. **Run the tests before you push:** `pytest -q`. CI runs them again on every PR (`.github/workflows/tests.yml`);
   a PR with failing tests does not get merged.
5. **Every PR gets one review** from a teammate before merging.
6. **Delete the branch after it is merged.** The PR and its commits stay in the history.

## Day to day

```bash
# start new work from the latest main
git switch main && git pull
git switch -c <your-name>-<topic>

# ... edit, then commit small, self-contained changes with clear messages
pytest -q
git add <files> && git commit -m "Add X so that Y"
git push -u origin <your-name>-<topic>      # first push; later just `git push`
```

Then open a PR on GitHub (base: `main`), describe what changed and how you tested it, and ask a
teammate to review.

**When `main` moves while you are working**, bring it into your branch and re-run the tests:

```bash
git fetch origin
git merge origin/main        # resolve conflicts here, on your branch, never on main
pytest -q && git push
```

**Building on a teammate's branch that is not merged yet:** branch from theirs
(`git switch -c tanvir-x origin/bhushan-y`), and say in your PR description that it depends on their PR.
Merge theirs first; then `git merge origin/main` into yours, and your PR shrinks to just your changes.

## What goes in git, and what does not

- **In git:** code, configs, tests, small analysis scripts, and reports or tables you want to share (`docs/`, `analysis/`).
- **Not in git:** `data/`, `runs/`, `logs/` (already in `.gitignore`), API keys (`.env`), large files, and
  copyrighted PDFs (the repo is public). To keep a file out of git on your machine only, add it to
  `.git/info/exclude` instead of the shared `.gitignore`.
- Run results live in `runs/`. To share numbers, commit the report (`vlmbench report ... --out`) or an
  analysis output, not the run folders.

## Changing shared code

`runner.py`, `cli.py`, `datasets/base.py`, `models/` and `agents/` are used by everyone. When a PR changes them:
add or update a test, mention it in the PR description, and check that existing runs still re-score the
same (`vlmbench finalize runs/<run>`). Add new behaviour behind an option or a new version (as the
agent instructions do with `v1`…`v4`) instead of changing what an existing option does, so earlier runs
stay reproducible.

## One-time setup for the repo owner

Turn on branch protection for `main` (Settings → Branches → Add rule, or Settings → Rules):
require a pull request before merging, require 1 approval, and require the `tests` status check to pass.
