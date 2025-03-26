.PHONY: test

test:
	uv run pytest tests

.PHONY: format

format:
	uv run black .
	uv run isort .

