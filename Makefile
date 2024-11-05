REPOSITORY := aws-repo
BLACK_ARGS := --line-length 120
ISORT_ARGS := --lines-after-imports 2
LINT_PATHS := energy_explorer/ notebook/

preview-readme:
	@echo "Generating PDF preview of README.md...";
	pandoc README.md -o docs/README.pdf --pdf-engine=xelatex -V geometry:margin=1in -V fontsize=12pt -V colorlinks;
	open docs/README.pdf;

activate:
	poetry shell

install:
	poetry install

lint:
	poetry run black $(BLACK_ARGS) $(LINT_PATHS)
	poetry run isort $(ISORT_ARGS) $(LINT_PATHS)

clean: clean-env clean-build clean-ipython clean-python
	@echo "Overall cleanup completed."

clean-env:
	@echo "Removing Poetry virtual environment..."
	poetry env remove python 2>/dev/null || true

clean-build:
	@echo "Cleaning build artifacts..."
	rm -rf build dist *.egg-info

clean-ipython:
	@echo "Cleaning IPython artifacts..."
	find . -name .ipynb_checkpoints -type d -prune -exec rm -rf '{}' +
	find . -name .virtual_documents -type d -prune -exec rm -rf '{}' +
	find . -name checkpoint.ipynb -exec rm -f '{}'

clean-python:
	@echo "Cleaning Python bytecodes and caches..."
	find . -name "*.pyc" -exec rm -f '{}' +
	find . -name "__pycache__" -exec rm -rf '{}' + 2>/dev/null || true
	find . -name "poetry.lock" -exec rm -rf '{}' + 2>/dev/null || true

reset: clean install

.PHONY: activate install install-lib test publish lint-check lint clean clean-env clean-build clean-ipython clean-python reset
