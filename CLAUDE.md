# AutoClipEval Dev Guide

## Commands
- Build/Run backend: `cd backend && uvicorn app.main:app --reload`
- Run linting: `ruff check .`
- Format code: `ruff format .`
- Install dependencies: `uv add` or `uv add --dev` (run in backend directory)
- Install pre-commit hooks: `pre-commit install`
- Run single test: `pytest path/to/test.py::TestClass::test_function -v`

## Code Style Guidelines
- Python: Use Python 3.12+
- Formatting: Follow ruff formatter conventions
- Imports: Group standard library, third-party, and local imports
- Types: Use type hints for all function parameters and return values
- Naming: snake_case for variables/functions, PascalCase for classes
- Error handling: Use try/except with specific exceptions
- Documentation: Docstrings for all public functions and classes
- Async: Use FastAPI async/await pattern for API endpoints
- Architecture: Keep with FastAPI Full Stack Template conventions