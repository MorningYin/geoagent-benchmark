# Contributing to Meta Agents Research Environments
We want to make contributing to this project as easy and transparent as
possible.


## Our Development Process

This repository is the open source version of the Meta Agents Research Environments. Our main development happens on our internal codebase
and is synced to this repository. We will sync changes from the internal codebase to this repository on a regular basis and any Pull Requests opened here will first be reviewed and tested on our internal codebase before being merged.

## Code Style

This project enforces consistent code style through automated tools and CI checks. All contributions must pass these checks before being merged.

### Python Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for Python linting and formatting:

- **Linting**: Ruff checks for code quality issues and enforces Python best practices
- **Formatting**: Ruff format ensures consistent code formatting (similar to Black)
- **Import sorting**: Imports are automatically sorted and organized
- **Type checking**: We use [Pyright](https://github.com/microsoft/pyright) for static type analysis

### TypeScript Code Style

For TypeScript/JavaScript code in the GUI client:

- **Formatting**: [Prettier](https://prettier.io/) ensures consistent code formatting
- **Type checking**: TypeScript compiler (`tsc`) validates type safety
- **GraphQL**: Schema and Relay artifacts must be kept in sync

### Pre-commit Hooks

We use pre-commit hooks to automatically check and fix code style issues before commits:

```bash
# Install pre-commit hooks (one-time setup)
pip install pre-commit
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files
```

The pre-commit configuration includes:
- Basic file checks (syntax, merge conflicts, large files)
- Ruff linting and formatting for Python
- Import sorting
- File header validation
- Trailing whitespace and end-of-file fixes

### Running Style Checks Locally

**Python:**
```bash
# Fix auto-fixable linting issues
uvx ruff check --fix .

# Apply formatting
uvx ruff format .

# Type checking
uv run --extra dev pyright
```

**TypeScript (in are/simulation/gui/client directory):**
```bash
# Type checking
npm run tsc

# Format check
npm run format

# Run tests
npm run test
```

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License
By contributing to Meta Agents Research Environments, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
