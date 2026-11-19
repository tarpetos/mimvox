# Notes for coding agents

Follow [CONTRIBUTING.md](../CONTRIBUTING.md), in particular:

- **Comments:** only when fully justified (a workaround, an outside constraint, a measured number), one or two lines.
  Never restate the code, never write module essays or history. When in doubt, leave it out.
- Commit and pull request titles: Conventional Commits, see "Commit messages".
- No attribution lines in commits or pull requests (no "Generated with", no co-author or session trailers).
- Before pushing, run the commands in "Checks"; CI runs the same on Linux, Windows and macOS.
- New non-ML code is Rust. Python stays for the engine's ML work and the code already there.
