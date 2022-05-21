# Makefile
.PHONY: help
help:
	@echo "Commands:"
	@echo "run_api                 : Launch FastAPI api."


.PHONY: run_api
run_api:
	uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

.PHONY: docker_build
docker_build:
	docker build -t vorphus/eda-cv:1.0-slim .

.PHONY: docker_run
docker_run:
	docker run -it --rm --name eda-cv -p 8080:8080 vorphus/eda-cv:1.0-slim

.PHONY: install-dev
install-dev:
	python -m pip install -e ".[dev]" --no-cache-dir
	pre-commit install
	pre-commit autoupdate
