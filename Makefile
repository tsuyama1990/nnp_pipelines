.PHONY: build up clean

build:
	docker-compose build

up:
	docker-compose up -d

clean:
	docker-compose down

test:
	uv run pytest -m "not integration and not docker"

test-integration:
	uv run pytest -m "integration"

test-all:
	uv run pytest
