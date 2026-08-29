#!/usr/bin/env bash
set -euo pipefail

psql --set=ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  --command "ALTER ROLE \"$POSTGRES_USER\" SET default_transaction_read_only = on"
