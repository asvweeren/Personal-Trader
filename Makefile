.PHONY: help setup up down restart logs backend-shell db-migrate db-upgrade test lint frontend-dev backend-dev

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Initial project setup
	cp -n .env.example .env || true
	docker compose build
	docker compose up -d postgres redis
	sleep 3
	cd backend && uv sync
	cd backend && uv run alembic upgrade head
	cd frontend && npm install

up: ## Start all services
	docker compose up -d

down: ## Stop all services
	docker compose down

restart: ## Restart all services
	docker compose restart

logs: ## Tail all logs
	docker compose logs -f

backend-dev: ## Run backend in development mode
	cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend-dev: ## Run frontend in development mode
	cd frontend && npm run dev

backend-shell: ## Open a shell in the backend container
	docker compose exec backend bash

db-migrate: ## Create a new database migration (usage: make db-migrate msg="description")
	cd backend && uv run alembic revision --autogenerate -m "$(msg)"

db-upgrade: ## Run database migrations
	cd backend && uv run alembic upgrade head

db-downgrade: ## Rollback last migration
	cd backend && uv run alembic downgrade -1

test: ## Run all tests
	cd backend && uv run pytest -v

test-cov: ## Run tests with coverage
	cd backend && uv run pytest --cov=app --cov-report=html -v

lint: ## Run linters
	cd backend && uv run ruff check app/ tests/
	cd backend && uv run ruff format --check app/ tests/

format: ## Format code
	cd backend && uv run ruff check --fix app/ tests/
	cd backend && uv run ruff format app/ tests/
