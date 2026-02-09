#!/bin/bash
set -e

echo "Starting database setup..."

# Start PostgreSQL if not running
docker compose up -d postgres redis
sleep 3

# Run migrations
cd backend
uv run alembic upgrade head

echo "Database setup complete!"
