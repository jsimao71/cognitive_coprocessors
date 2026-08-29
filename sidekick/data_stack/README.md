# Paper 2.5 WSL data-stack sidekick

This directory is intended for a WSL2 Linux distribution with Docker Engine and
the Docker Compose plugin installed inside WSL. Do not run it through a
Windows-host Docker target. The current Paper 2.5 local-production result does
not depend on this service and does not claim that it has been executed.

## Start and verify

```bash
cd /mnt/c/Users/j.simao/zgit/rd/cognitive_coprocessors/sidekick/data_stack
cp .env.example .env
# Replace the development password before starting the service.
docker compose pull
docker compose up -d postgres
set -a && source .env && set +a
./healthcheck.sh
```

The default host port is `54329`. Keep it bound to the local development host;
do not expose this fixture database publicly. The single fixture role defaults
to read-only after initialization, and the Python adapter also starts every
retrieval transaction with `SET TRANSACTION READ ONLY`.

## Run integration tests

```bash
export CCPU_POSTGRES_DSN="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${CCPU_POSTGRES_PORT}/${POSTGRES_DB}"
python -m pip install -e ".[data,data-services,test]"
python -m pytest -m integration tests/test_paper2_5_services.py
```

Tests skip when `CCPU_POSTGRES_DSN` is absent. They never substitute an in-memory
backend for an unavailable service.

## Stop and clean up

```bash
docker compose down
# Destructive: removes only this Compose project's named fixture volume.
docker compose down --volumes
```

The compose file currently starts one Postgres/pgvector service. Iceberg catalog
and dedicated vector-service profiles will be added only after this boundary is
validated independently.
