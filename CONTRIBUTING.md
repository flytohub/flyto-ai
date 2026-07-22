# Contributing

Use Python 3.10 or newer. Install the development profile with
`python -m pip install -e ".[dev]"` and keep provider/network credentials out of
the repository.

Before implementation, use Flyto2 Indexer to identify affected agent, provider,
tool, memory, safety, channel, and evidence contracts. New capabilities need a
typed contract, guardrail, observable evidence, tests, documentation, and
rollback behavior.

Before opening a pull request, run the commands in
[Operations and verification](docs/OPERATIONS.md#local-verification), regenerate
the source reference, and update project memory/changelog when behavior or
deployment changes. Tests must use fakes unless explicitly marked as live
integration tests.

