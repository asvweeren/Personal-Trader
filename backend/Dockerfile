FROM python:3.12-slim AS base

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
RUN uv sync --no-dev --no-install-project

COPY alembic.ini .
COPY alembic/ alembic/
COPY app/ app/

# Development target
FROM base AS development
RUN uv sync --no-install-project
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

# Production target
FROM base AS production
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
