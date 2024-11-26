#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = UMA-random_forest-SVM
PYTHON_VERSION = 3.12

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	pip install uv
	uv sync
	

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run flake8 random_forest_svm
	uv run isort --check --diff --profile black random_forest_svm/
	uv run black --check --config pyproject.toml random_forest_svm/

## Format source code with black
.PHONY: format
format:
	uv run black --config pyproject.toml random_forest_svm


## Download and preprocess data
.PHONY: prepare_data
prepare_data:
	uv run python random_forest_svm/data/download_data.py
	uv run python random_forest_svm/data/preprocess_data.py


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@uv run python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
