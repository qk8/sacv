.PHONY: test test-unit test-integration test-e2e build-sandbox lint type-check

test-unit:
	pytest tests/unit -v -m unit

test-integration:
	pytest tests/integration -v -m integration

test-e2e:
	pytest tests/e2e -v -m e2e

test:
	pytest tests/ -v --ignore=tests/e2e

build-sandbox:
	docker build -f Dockerfile.sandbox -t sacv-sandbox:latest .

lint:
	ruff check src/ tests/

type-check:
	mypy src/

install:
	pip install -e ".[dev]"
	playwright install chromium
