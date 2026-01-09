# Contributing

Contributions are welcome! This document outlines how to contribute to the project.

## Development Setup

```bash
git clone https://github.com/LabStrangeLoop/bitnet.git
cd bitnet
uv sync --group dev
```

## Code Style

This project follows strict code quality standards:

- **Formatter**: ruff (line length 100)
- **Linter**: ruff
- **Type checker**: mypy (strict mode)
- **Tests**: pytest

Run all checks before submitting:

```bash
uv run ruff format .
uv run ruff check .
uv run mypy bitnet experiments analysis tests
uv run pytest tests/ -v
```

## Guidelines

- **File length**: Max 200 lines (hard limit 250)
- **Function length**: Target 10-20 lines, max 30
- **Type hints**: Required on all function signatures
- **Nesting**: Max 2 levels deep; use early returns
- **Tests**: Add tests for new functionality

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Ensure all checks pass
5. Submit a pull request

## Reporting Issues

When reporting bugs, please include:

- Python version
- GPU model and CUDA version
- Steps to reproduce
- Expected vs actual behavior
