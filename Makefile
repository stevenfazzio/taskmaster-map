.PHONY: install lint format test clean

install:
	uv sync --extra dev

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format .

test:
	uv run pytest

clean:
	@echo "This will remove all files in data/. Press Ctrl+C to cancel."
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf data/*
