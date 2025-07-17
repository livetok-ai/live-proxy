# Development Setup

This project uses modern Python packaging with `pyproject.toml` and `uv` for fast dependency management.

## Setup

### Option 1: Using uv (Recommended - Fast!)

1. Install uv if you haven't already:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# or with brew
brew install uv
```

2. Install the project with development dependencies:
```bash
uv sync --extra dev
```

3. Activate the virtual environment:
```bash
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows
```

### Option 2: Traditional pip

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

2. Install the project with development dependencies:
```bash
pip install -e ".[dev]"
```

## Development Tools

### Using uv (Recommended)

Run commands with uv (automatically manages virtual environment):
```bash
# Code formatting
uv run black *.py

# Linting
uv run ruff check *.py
uv run ruff check --fix *.py

# Type checking
uv run mypy *.py

# Testing
uv run pytest
uv run pytest --cov=.
uv run pytest tests/test_proxy.py::test_basic_import -v
```

### Using activated virtual environment

After activating the virtual environment:
```bash
# Code formatting
black *.py

# Linting
ruff check *.py
ruff check --fix *.py

# Type checking
mypy *.py

# Testing
pytest
pytest --cov=.
pytest tests/test_proxy.py::test_basic_import -v
```

## Project Structure

```
├── pyproject.toml          # Modern Python project configuration
├── .venv/                  # Virtual environment (uv creates this)
├── uv.lock                 # Lock file (created by uv sync)
├── tests/                  # Test directory
│   ├── __init__.py
│   └── test_proxy.py       # Basic tests for proxy module
├── proxy.py                # Main proxy server
├── logger.py               # Logging utilities
├── model*.py               # AI model integrations
└── requirements.txt        # Legacy dependencies (can be removed)
```

## Configuration

All development tool configurations are in `pyproject.toml`:

- **Black**: Line length 120, Python 3.8+ target
- **Ruff**: Modern linting with common rules enabled
- **mypy**: Strict type checking configuration
- **pytest**: Auto async mode, coverage reporting

## Why uv?

`uv` is a fast Python package manager that:
- Installs dependencies 10-100x faster than pip
- Automatically manages virtual environments
- Creates reproducible `uv.lock` files
- Compatible with existing `pyproject.toml` files
- Developed by the creators of Ruff