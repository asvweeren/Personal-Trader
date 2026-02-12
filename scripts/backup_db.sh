#!/usr/bin/env bash
# Daily PostgreSQL backup for the AI Trader system.
# Designed to run via crontab: 0 3 * * * /root/trader/scripts/backup_db.sh
#
# - Dumps the trader database via docker compose exec
# - Compresses with gzip
# - Rotates: deletes backups older than 7 days

set -euo pipefail

TRADER_DIR="/root/trader"
BACKUP_DIR="${TRADER_DIR}/backups"
COMPOSE_FILE="${TRADER_DIR}/docker-compose.prod.yml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/trader_${TIMESTAMP}.sql.gz"
RETENTION_DAYS=7

mkdir -p "$BACKUP_DIR"

echo "[$(date -Iseconds)] Starting database backup..."

docker compose -f "$COMPOSE_FILE" exec -T postgres \
  pg_dump -U trader -d trader --no-owner --no-privileges \
  | gzip > "$BACKUP_FILE"

SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "[$(date -Iseconds)] Backup complete: $BACKUP_FILE ($SIZE)"

# Rotate old backups
DELETED=$(find "$BACKUP_DIR" -name "trader_*.sql.gz" -mtime +${RETENTION_DAYS} -print -delete | wc -l)
if [ "$DELETED" -gt 0 ]; then
  echo "[$(date -Iseconds)] Rotated $DELETED backup(s) older than ${RETENTION_DAYS} days"
fi

echo "[$(date -Iseconds)] Done."
