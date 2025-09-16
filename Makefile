# Omega Portfolio Engine Makefile

.PHONY: help install install-dev test lint format clean demo build run stop start

# Default target
help:
	@echo "Omega Portfolio Engine - Available Commands:"
	@echo "============================================="
	@echo "install      - Install the package"
	@echo "install-dev  - Install with development dependencies"
	@echo "start        - Start web application (API + UI)"
	@echo "test         - Run tests"
	@echo "lint         - Run linting"
	@echo "format       - Format code"
	@echo "clean        - Clean build artifacts"
	@echo "demo         - Run complete demo"
	@echo "build        - Build Docker image"
	@echo "run          - Run with Docker Compose"
	@echo "stop         - Stop Docker Compose services"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[api,ui]"
	pre-commit install

# Start web application
start:
	@echo "Starting Omega Portfolio Engine web application..."
	@echo "API will be available at: http://localhost:8000"
	@echo "Web UI will be available at: http://localhost:8501"
	@echo "Press Ctrl+C to stop both services"
	@echo ""
	@echo "Starting API server in background..."
	@python -m api.main &
	@echo "Waiting for API to start..."
	@sleep 3
	@echo "Starting web UI..."
	@streamlit run app/ui.py --server.port 8501 --server.address 0.0.0.0

# Testing
test:
	pytest tests/ -v --cov=engine --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -x

# Code quality
lint:
	ruff check .
	mypy engine/

format:
	ruff format .
	black engine/ api/ app/ tests/

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Demo
demo: install-dev
	@echo "Running Omega Portfolio Engine Demo..."
	@echo "====================================="
	@echo "1. Starting API server..."
	python -m api.main &
	@echo "2. Waiting for API to start..."
	sleep 5
	@echo "3. Starting Streamlit UI..."
	streamlit run app/ui.py --server.port 8501 --server.address 0.0.0.0 &
	@echo "4. Demo is running!"
	@echo "   - API: http://localhost:8000"
	@echo "   - UI: http://localhost:8501"
	@echo "   - Docs: http://localhost:8000/docs"
	@echo "Press Ctrl+C to stop"

# Docker
build:
	docker build -t omega-portfolio-engine .

run:
	docker-compose up -d

stop:
	docker-compose down

logs:
	docker-compose logs -f

# Development
dev-setup: install-dev
	@echo "Setting up development environment..."
	@echo "Creating necessary directories..."
	mkdir -p data/raw data/interim data/processed results logs mlruns
	@echo "Copying environment file..."
	cp env.example .env
	@echo "Development environment ready!"
	@echo "Don't forget to set your API keys in .env"

# Data
download-data:
	@echo "Downloading sample data..."
	python scripts/download_sample_data.py

# Documentation
docs:
	@echo "Generating documentation..."
	sphinx-build -b html docs/ docs/_build/html

# Release
release:
	@echo "Creating release..."
	python -m build
	twine upload dist/*

# CI/CD
ci-test:
	pytest tests/ --cov=engine --cov-report=xml --junitxml=test-results.xml

ci-lint:
	ruff check . --output-format=github
	mypy engine/ --junit-xml=mypy-results.xml
