.PHONY: install dev lint test run-api run-ui ingest eval clean docker-up docker-down

# ── Setup ──────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

dev: install
	pip install -e ".[dev]"
	pre-commit install

# ── Quality ────────────────────────────────────────────────────────────────────
lint:
	ruff check src/ cli/ tests/ config/
	black --check src/ cli/ tests/ config/
	isort --check-only src/ cli/ tests/ config/

format:
	black src/ cli/ tests/ config/
	isort src/ cli/ tests/ config/

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/test_arxiv.py tests/test_rag.py -v --tb=short -x

# ── Run ────────────────────────────────────────────────────────────────────────
run-api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run frontend/app.py --server.port 8501

# ── Data ops ───────────────────────────────────────────────────────────────────
ingest:
	python cli/main.py ingest --query "$(QUERY)" --max $(MAX_PAPERS)

# Default benchmark run
eval:
	python cli/main.py evaluate \
		--benchmark data/benchmarks/benchmark.json \
		--name scirag_uq_v1 \
		--output results/

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
	rm -rf chroma_db/ logs/

clean-data:
	rm -rf data/raw/ data/processed/
	mkdir -p data/raw data/processed
