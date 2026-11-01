install:
	uv sync

run:
	uv run python -m src.detector_neumonia.ui

check:
	uv run ruff check .
	uv run pytest
