# Makefile for agentic-rag-platform.
#
# Wraps every `docker compose` invocation with an `unset` for DB_USER,
# DB_PASSWORD, and DB_NAME. These names collide with envs that some local
# dev tooling exports globally from the shell rc file; without the unset,
# Compose substitutes those values into the postgres service and the API
# fails to connect ("role does not exist").

SHELL := /bin/bash
DC    := unset DB_USER DB_PASSWORD DB_NAME && docker compose

.PHONY: help up down stop clean rebuild build logs status ollama-check ingest psql

help:
	@echo "Common targets:"
	@echo "  make up            - start postgres + api + ui (detached)"
	@echo "  make down          - stop and remove containers + volumes"
	@echo "  make stop          - stop containers, keep volumes"
	@echo "  make rebuild       - down -v + build + up"
	@echo "  make clean         - down -v --rmi local (full reset)"
	@echo "  make logs          - tail logs from all services"
	@echo "  make status        - show container status + API /health"
	@echo "  make ollama-check  - verify agent_api can reach the host's Ollama"
	@echo "  make ingest        - run the ingestion pipeline over documents/"
	@echo "  make psql          - open a psql shell against the vector DB"

up:
	$(DC) up -d

down:
	$(DC) down -v

stop:
	$(DC) stop

build:
	$(DC) build

rebuild:
	$(DC) down -v
	$(DC) up -d --build

clean:
	$(DC) down -v --rmi local

logs:
	$(DC) logs -f --tail=100

status:
	@$(DC) ps
	@echo ""
	@echo -n "API /health: "
	@curl -s http://localhost:8058/health || echo "(API not responding)"
	@echo ""

ollama-check:
	@docker exec agent_api curl -s -m 3 http://host.docker.internal:11434/api/tags > /dev/null 2>&1 \
		&& echo "OK: agent_api can reach Ollama on host.docker.internal:11434" \
		|| echo "FAIL: agent_api cannot reach Ollama. Is 'brew services list' showing ollama as started?"

ingest:
	docker exec -it agent_api python -m ingestion.ingest --documents documents/

psql:
	docker exec -it postgres_pgvector psql -U postgres -d vector_db
