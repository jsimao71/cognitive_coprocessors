#!/usr/bin/env bash
set -euo pipefail

docker compose ps
docker compose exec -T postgres \
  pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"
docker compose exec -T postgres \
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc \
  "SELECT current_setting('transaction_read_only'), extversion FROM pg_extension WHERE extname='vector'"
