# Contributing to HermesKatana

Thank you for your interest in contributing to HermesKatana!

## Getting Started

```bash
git clone https://github.com/claudlos/hermes-katana.git
cd hermes-katana
pip install -e ".[dev]"
pytest tests/ -q  # verify everything passes
```

## Development Workflow

1. Fork the repository and create a feature branch from `master`.
2. Make your changes with tests.
3. Run the test suite: `pytest tests/ -q`
4. Run the linter: `ruff check src/ tests/`
5. Run the formatter: `ruff format src/ tests/`
6. For scanner, policy, routing, or security-threshold changes, run: `scripts/verify_scanner_change.sh`
7. Submit a pull request.

## Code Style

- Python 3.10+ with type annotations
- Line length: 120 characters
- Formatting: `ruff format`
- Linting: `ruff check`
- All public APIs need docstrings

## Testing

- Tests live in `tests/` mirroring the `src/` structure
- Use `pytest` with the fixtures in `tests/conftest.py`
- Aim for high coverage on security-critical paths (scanners, taint, policy)
- Property-based tests use `hypothesis`

## What to Contribute

- New scanner patterns for emerging attack techniques
- Policy engine enhancements
- Documentation improvements
- Bug fixes with regression tests
- Performance improvements with benchmarks

## Pull Request Guidelines

- Keep PRs focused on a single change
- Include tests for new functionality
- Update documentation if behavior changes
- Reference any related issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
