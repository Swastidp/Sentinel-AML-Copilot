# Sentinel AML Copilot - task runner.
#
# On Windows, GNU Make is installed as `mingw32-make` (C:\MinGW\bin).
# Use `mingw32-make <target>`; on Linux/macOS and in CI, plain `make <target>`.
# Activate the virtualenv first: .venv\Scripts\activate  (Windows)  /  source .venv/bin/activate

PY  := python
PKG := sentinel

.PHONY: help install doctor bootstrap pipeline api ui test test-integration eval \
        loadtest report metrics check fmt lint types cov clean

help:
	@echo "Sentinel AML Copilot"
	@echo ""
	@echo "  install           install the package with dev extras"
	@echo "  doctor            ping every store and verify the LLM model slugs"
	@echo "  bootstrap         create Postgres schema, Neo4j constraints, Qdrant collections"
	@echo "  pipeline          full batch: generate -> validate -> load all stores"
	@echo "  api               run the FastAPI service (reload)"
	@echo "  ui                run the Streamlit console"
	@echo "  test              unit tests only - no network, no tokens"
	@echo "  test-integration  add tests that hit live cloud stores"
	@echo "  eval              agent eval harness - COSTS LLM TOKENS"
	@echo "  loadtest          measure /v1/score latency (p50/p95/p99)"
	@echo "  report            metrics + Excel MIS pack"
	@echo "  check             ruff + mypy --strict + unit tests (what CI runs)"
	@echo "  fmt               auto-fix lint and format"
	@echo "  cov               unit tests with a coverage report"
	@echo "  clean             remove caches and generated artefacts"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

doctor:
	$(PY) scripts/check_connectivity.py

bootstrap:
	$(PY) scripts/bootstrap_stores.py

pipeline:
	$(PY) -m $(PKG).pipeline.run

api:
	$(PY) -m uvicorn $(PKG).api.main:app --reload --port 8000

ui:
	$(PY) -m streamlit run src/$(PKG)/ui/app.py

test:
	$(PY) -m pytest tests/unit -m "not integration and not eval"

test-integration:
	$(PY) -m pytest tests -m "not eval"

eval:
	$(PY) -m pytest tests/eval -m eval -s

loadtest:
	$(PY) scripts/loadtest_score.py

metrics:
	$(PY) -m $(PKG).reporting.metrics

report: metrics
	$(PY) -m $(PKG).reporting.excel

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

types:
	$(PY) -m mypy src scripts tests/unit/test_port_conformance.py

check: lint types test

fmt:
	$(PY) -m ruff check --fix src tests scripts
	$(PY) -m ruff format src tests scripts

cov:
	$(PY) -m pytest tests/unit --cov=$(PKG) --cov-report=term-missing

clean:
	$(PY) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	$(PY) -c "import shutil; [shutil.rmtree(d, ignore_errors=True) for d in ('.pytest_cache','.mypy_cache','.ruff_cache','htmlcov')]"
