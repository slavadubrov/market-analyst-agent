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
test: install
	uv run pytest

# Format code using ruff
format: install
	uv run ruff format .
	uv run ruff check --fix .

# Lint code using ruff
lint: install
	uv run ruff check .

# Run static type checking
type-check: install
	uv run mypy

# Run validation (lint + type-check + test)
check: lint type-check test

# Run the application (Analysis pipeline)
run-analysis: install db-up
	uv run python -m market_analyst.cli "Analyze NVDA stock"

# Run the trade workflow demo (Guardian policy layer)
run-trade: install db-up
	uv run python -m market_analyst.cli --trade --action buy --ticker NVDA --amount 5000

# Run the combined workflow demo (Full end-to-end)
run: install db-up
	uv run python -m market_analyst.cli --combined "Analyze NVDA stock" --trade-amount 1000

# Run the Gradio Web UI
run-ui: install db-up
	uv run python src/market_analyst/app.py

# Clean up
clean: docker-down
	rm -rf .venv
	rm -rf .cache
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .coverages


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

# --- Part 5: The Habitat -----------------------------------------------------

# Bring up the OTel Collector + Tempo + Loki + Prometheus + Grafana stack.
# Grafana lands on http://localhost:3000 with the "Market Analyst — Overview"
# dashboard preloaded.
observability-up:
	docker compose -f docker/docker-compose.yml --env-file .env --profile observability up -d

observability-down:
	docker compose -f docker/docker-compose.yml --env-file .env --profile observability stop

# Bring up the MCP sidecar (secret-broker pattern: tools behind a proxy).
mcp-up:
	docker compose -f docker/docker-compose.yml --env-file .env --profile mcp up -d mcp-sidecar

mcp-down:
	docker compose -f docker/docker-compose.yml --env-file .env --profile mcp stop mcp-sidecar

# Run the queue worker locally (queue+worker+checkpoint DB shape).
# Producer side: any `market-analyst …` invocation that hits the CLI.
# Consumer side: this loop.
worker: install db-up
	OTEL_SERVICE_NAME=market-analyst-worker uv run python -m market_analyst.runtime.worker

# Push a single test job onto the queue.
queue-push: install
	uv run python -c "from market_analyst.runtime.queue import push_run; print(push_run('Analyze NVDA stock', mode='flash'))"
