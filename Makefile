PHONY: clean

venv:
	. backend/.venv/bin/activate

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -type f -exec rm -rf {} +
	find . -name "*.ruff_cache" -type d -exec rm -rf {} +

test-atlas: venv
	uv run backend/app/atlas.py

run-agents: venv
	uv run backend/app/agents.py