# Contributing Guide

Thanks for your interest in improving **arff-csv**! This guide explains how to get started and what we expect from contributions.

## Quick Start

1. Fork and clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -e ".[dev]"`.
4. Run tests to confirm the baseline: `make test`.

## Development Workflow

- **Lint**: `make lint`
- **Type check**: `make typecheck`
- **Tests**: `make test` (or `make test-cov`)
- **Format**: `make format`

Please run lint, typecheck, and tests before opening a PR.

## Submitting Changes

1. Create a descriptive branch name.
2. Keep commits focused and write clear messages.
3. Include tests for new behavior or bug fixes.
4. Open a Pull Request that describes:
   - The problem and the solution.
   - Any breaking changes.
   - How you tested it.

## Reporting Issues

When filing an issue, include:
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Sample input files if relevant (anonymized)

## Security

Please **do not** open public issues for security problems. Follow the process in `SECURITY.md`.

## Code of Conduct

By participating, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).
