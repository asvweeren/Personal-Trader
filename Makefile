.PHONY: help setup up down restart logs backend-shell db-migrate db-upgrade test lint frontend-dev backend-dev prod-up prod-down prod-logs prod-restart

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Local Development ──────────────────────────────────────

setup: ## Initial project setup
	cp -n .env.example .env || true
	docker compose build
	docker compose up -d postgres redis
	sleep 3
	cd backend && uv sync
	cd backend && uv run alembic upgrade head
	cd frontend && npm install

up: ## Start all services (dev)
	docker compose up -d

down: ## Stop all services (dev)
	docker compose down

restart: ## Restart all services (dev)
	docker compose restart

logs: ## Tail all logs (dev)
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

# ── Production ─────────────────────────────────────────────

prod-up: ## Start production services
	docker compose -f docker-compose.prod.yml up -d --build

prod-down: ## Stop production services
	docker compose -f docker-compose.prod.yml down

prod-restart: ## Restart production services
	docker compose -f docker-compose.prod.yml restart

prod-logs: ## Tail production logs
	docker compose -f docker-compose.prod.yml logs -f

prod-status: ## Show production service status
	docker compose -f docker-compose.prod.yml ps

prod-deploy: ## Full production deployment
	bash scripts/deploy.sh
