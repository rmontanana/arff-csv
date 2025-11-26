.PHONY: help install install-dev test test-cov lint format typecheck clean build publish-test publish all check

# Default target
help:
	@echo "ARFF-CSV Converter - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install the package"
	@echo "  make install-dev    Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run tests"
	@echo "  make test-cov       Run tests with coverage report"
	@echo "  make test-html      Run tests with HTML coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run linter (ruff check)"
	@echo "  make format         Format code (ruff format)"
	@echo "  make typecheck      Run type checker (mypy)"
	@echo "  make check          Run all quality checks"
	@echo ""
	@echo "Build & Publish:"
	@echo "  make build          Build the package"
	@echo "  make publish-test   Publish to TestPyPI"
	@echo "  make publish        Publish to PyPI"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove build artifacts"
	@echo ""
	@echo "Other:"
	@echo "  make all            Run check, test-cov, and build"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest

test-cov:
	pytest --cov=arff_csv --cov-report=term-missing

test-html:
	pytest --cov=arff_csv --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

# Code quality
lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	MYPYPATH=src mypy -p arff_csv

check: lint typecheck
	@echo "All checks passed!"

# Build and publish
build: clean
	python -m build

publish-test: build
	twine check dist/*
	twine upload --repository testpypi dist/*

publish: build
	twine check dist/*
	twine upload dist/*

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Combined targets
all: check test-cov build
	@echo "All tasks completed successfully!"
