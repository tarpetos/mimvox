# Contributing

Bug reports, ideas and pull requests are welcome. Open an issue first for anything larger than a small fix, so the
change can be discussed before you spend time on it.

## Terms for contributions

The project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE), and its author keeps the right to
offer it under other terms, including commercial ones. To keep that possible, every contribution is accepted on these
terms:

- You confirm that you wrote the contribution yourself, or otherwise have the right to submit it under these terms.
- You grant the project's author (tarpetos) a perpetual, worldwide, non-exclusive, royalty-free, irrevocable license
  to use, copy, modify, publish, distribute, sublicense and relicense your contribution, under any terms, including
  commercial ones.
- Your contribution is published in this repository under the project's license.

Opening a pull request means you agree to these terms.

## Ground rules

- Everything in English: code, comments, documentation, UI text and commit messages.
- The app is event-driven: no polling loops, no `while True` with sleeps or timeouts. Background threads post events
  to the UI; the engine runs on one asyncio loop (see [docs/architecture.md](docs/architecture.md)).
- The engine protocol in [contract/engine-protocol.md](contract/engine-protocol.md) is the source of truth: change it
  together with the app and the engine.
- A change that makes setup install something new the app or engine depends on (a Python package, a model, the Applio
  commit) also bumps `tools/setup/runtime_version.txt`: installed apps that update to it then ask the user to run setup
  again.
- The app must stay usable for its purpose only: cloning your own voice, or a voice you have explicit permission for
  (see [DISCLAIMER.md](DISCLAIMER.md)).

## Comments

Code explains itself through names, types and structure. A comment is the exception, written only when it is
fully justified: when a reader would otherwise get something wrong and the code cannot say it.

- **Write one** for a reason the code cannot show: a workaround for a bug or a platform quirk, a constraint from
  outside (the protocol, a thread, `unsafe`, a file format), or where a number comes from when it was measured or
  tuned (one line, with the number).
- **Do not write one** that repeats what the code or a name already says, describes a field or a function whose
  signature is clear, narrates history ("used to", "now", "since 0.3"), or explains the feature to a user.
  Architecture and behaviour belong in `docs/` and `contract/`, not in module headers.
- Keep it short: one or two lines. No essays, no lists of measurements, no cross-references to other files unless the
  link is the point.
- Doc comments (`///`, `//!`, Python docstrings) only on items whose contract is not obvious from the signature.
- When in doubt, leave it out.

## Commit messages

Pull requests are squash-merged: the pull request's title becomes the one commit on `master` (the pull request number
is added to it), and its description becomes the commit body. A check (`PR title`) refuses a title that does not
follow this convention ([Conventional Commits](https://www.conventionalcommits.org/)):

```
type(scope): summary
```

- **type**, one of:
  - `feat`: something new the user can do;
  - `fix`: something that did not work and now does;
  - `perf`: the same, faster or lighter;
  - `refactor`: code changed, behaviour not;
  - `docs`: documentation only;
  - `build`: installers, build scripts, dependencies;
  - `ci`: GitHub workflows;
  - `chore`: the rest (version bumps, housekeeping);
  - `revert`: undoes an earlier commit.
- **scope**, optional, one of:
  - `app`: the Rust desktop app;
  - `engine`: the Python engine;
  - `setup`: `setup.sh` / `setup.cmd` and `tools/setup`;
  - `build`, `ci`, `docs`, `release`, `deps`.
- **summary**: what changes, as seen by a user where there is one. It starts in lower case, has no full stop at the
  end. Keep it short; details belong in the description.
- **Breaking change**: `!` before the colon (`feat(engine)!: ...`), with the details in the description.

Examples:

```
feat(app): update the app from inside it on every platform
fix(setup): find setup.sh in AppImage, .deb and macOS installs
docs: explain the Book mode's chapter controls
chore(release): 0.1.6
```

Direct commits to `master` (maintainers only) follow the same convention.
