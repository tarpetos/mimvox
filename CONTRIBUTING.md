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
- The app must stay usable for its purpose only: cloning your own voice, or a voice you have explicit permission for
  (see [DISCLAIMER.md](DISCLAIMER.md)).
