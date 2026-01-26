.PHONY: setup install test format lint run check

# Default target
all: install

# Setup virtual environment
setup:
	uv venv

# Install dependencies (syncs with uv.lock)
install:
	uv sync

# Run tests
test:
	uv run pytest

# Format code using ruff
format:
	uv run ruff format .
	uv run ruff check --fix .

# Lint code using ruff
lint:
	uv run ruff check .

# Run static type checking
type-check:
	uv run mypy .

# Run validation (lint + type-check + test)
check: lint type-check test

# Run the application (example entry point)
run:
	uv run python -m market_analyst.cli

# --- Docker / Database Operations ---

# Start databases (Postgres, Qdrant, Redis) in background
db-up:
	docker compose -f docker/docker-compose.yml --env-file .env up -d postgres qdrant redis

# Stop databases
db-down:
	docker compose -f docker/docker-compose.yml --env-file .env stop postgres qdrant redis

# View database logs
db-logs:
	docker compose -f docker/docker-compose.yml --env-file .env logs -f postgres qdrant redis

# Build application Docker image
docker-build:
	docker compose -f docker/docker-compose.yml --env-file .env build app

# Start all services (App + DBs)
docker-up:
	docker compose -f docker/docker-compose.yml --env-file .env up -d

# Stop all services (removes containers)
docker-down:
	docker compose -f docker/docker-compose.yml --env-file .env down
