PYFILES := $(shell git ls-files "*.py" "**/*.py")
.PHONY: test
test:
	pytest tests

.PHONY: format
format:
	black $(PYFILES)
	isort $(PYFILES)

.PHONY: lint
lint:
	@echo Lint
	pylint $(PYFILES)||:

.PHONY: check
check: lint format test

.DEFAULT_GOAL := check
