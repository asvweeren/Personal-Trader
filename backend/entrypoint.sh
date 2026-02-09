#!/bin/bash
set -e

echo "==> Waiting for PostgreSQL..."
until uv run python -c "
import asyncio, asyncpg, os
async def check():
    conn = await asyncpg.connect(
        host=os.environ.get('POSTGRES_HOST', 'localhost'),
        port=int(os.environ.get('POSTGRES_PORT', 5432)),
        user=os.environ.get('POSTGRES_USER', 'trader'),
        password=os.environ.get('POSTGRES_PASSWORD', 'trader_secret'),
        database=os.environ.get('POSTGRES_DB', 'trader'),
    )
    await conn.close()
asyncio.run(check())
" 2>/dev/null; do
    echo "    PostgreSQL not ready, retrying in 2s..."
    sleep 2
done
echo "==> PostgreSQL is ready"

echo "==> Running database migrations..."
uv run alembic upgrade head
echo "==> Migrations complete"

echo "==> Starting application..."
exec "$@"
