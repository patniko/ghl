.EXPORT_ALL_VARIABLES:

FASTAPI_ENV ?= dev

init:
	poetry install

app-run:
	make docker-up
	poetry run fastapi dev server.py

app-run-prod:
	echo "Starting sql migration"
	poetry run alembic upgrade head
	echo "Starting server"
	poetry run python server.py

app-test:
	make docker-up
	poetry run pytest --capture=no

app-test-ci:
	make migration-run
	poetry run pytest --capture=no

app-test-smoke:
	poetry run pytest --capture=no

docker-up:
	docker-compose up -d
	make migration-run

docker-down:
	docker-compose down

migration-run:
	poetry run alembic upgrade head

migration-new:
	poetry run alembic revision --autogenerate -m $(m)

lint:
	poetry run ruff format
	poetry run ruff check --fix

lint-ci:
	poetry run ruff check

# Task service commands
tasks-run:
	make docker-up
	poetry run python tasks/main.py

tasks-run-prod:
	echo "Starting sql migration"
	poetry run alembic upgrade head
	echo "Starting task service"
	poetry run python tasks/main.py

# Combined commands
run-all:
	@echo "Starting all services (API, Tasks, and UI)"
	@echo "Starting Docker services and running migrations..."
	@make docker-up
	@echo "Starting UI in background..."
	@cd ../ph-ui && npm run dev & UI_PID=$$!
	@echo "Starting Tasks service in background..."
	@poetry run python tasks/main.py & TASKS_PID=$$!
	@echo "Starting API server..."
	@echo "All services started. Press Ctrl+C to stop all services."
	@poetry run fastapi dev server.py
	@kill $$UI_PID $$TASKS_PID 2>/dev/null || true

# Run API with integrated tasks
app-run-with-tasks:
	@echo "Starting API server with integrated tasks"
	@make docker-up
	@poetry run python server.py --enable-tasks

# Run all services with integrated tasks
run-all-integrated:
	@echo "Starting all services with integrated tasks (API+Tasks, and UI)"
	@echo "Starting Docker services and running migrations..."
	@make docker-up
	@echo "Starting UI in background..."
	@cd ../ph-ui && npm run dev & UI_PID=$$!
	@echo "Starting API server with integrated tasks..."
	@echo "All services started. Press Ctrl+C to stop all services."
	@poetry run python server.py --enable-tasks
	@kill $$UI_PID 2>/dev/null || true
