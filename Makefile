# Sentinel AML Copilot - task runner.
#
# On Windows, GNU Make is installed as `mingw32-make` (C:\MinGW\bin).
# Use `mingw32-make <target>`; on Linux/macOS, plain `make <target>`.
# Activate the virtualenv first: .venv\Scripts\activate  (Windows)  /  source .venv/bin/activate

PY  := python
PKG := sentinel

.PHONY: help install doctor bootstrap migrate generate quality features load pipeline api ui test \
        test-integration eval loadtest report metrics check fmt lint types cov clean

help:
	@echo "Sentinel AML Copilot"
	@echo ""
	@echo "  install           install the package with dev extras"
	@echo "  doctor            ping every store and verify the LLM model slugs"
	@echo "  bootstrap         create Postgres schema, Neo4j constraints, Qdrant collections"
	@echo "  migrate           apply the Postgres migrations - additive, idempotent, safe to repeat"
	@echo "  load              push the curated zone to Neon and verify counts against DuckDB"
	@echo "  generate          synthesise the population and land it in the raw zone"
	@echo "  quality           run the data-quality gate - exits 1 if a BLOCKING check fails"
	@echo "  detect            run rules R001-R007 + Isolation Forest, upsert alerts to Postgres"
	@echo "  features          gate, promote raw -> curated, then compute the 7D/30D features"
	@echo "  pipeline          full batch: generate -> validate -> load all stores"
	@echo "  api               run the FastAPI service (reload)"
	@echo "  ui                run the Streamlit console"
	@echo "  test              unit tests only - no network, no tokens"
	@echo "  test-integration  add tests that hit live cloud stores"
	@echo "  eval              agent eval harness - COSTS LLM TOKENS"
	@echo "  loadtest          measure /v1/score latency (p50/p95/p99)"
	@echo "  metrics           precision/recall/AUC against the ground-truth labels"
	@echo "  report            metrics + Excel MIS pack"
	@echo "  lint              ruff check + format --check"
	@echo "  types             mypy --strict"
	@echo "  check             ruff + mypy --strict + unit tests (run this before pushing)"
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

# Additive only, and idempotent: running it twice prints identical output and changes nothing.
migrate:
	$(PY) -m $(PKG).infrastructure.postgres.migrate

generate:
	$(PY) -m $(PKG).pipeline.generate

# Exits non-zero when a BLOCKING check fails, so `make quality` failing IS the gate working.
# The report is still written to data/quality/dq_report.json either way.
quality:
	$(PY) -m $(PKG).pipeline.quality

# Runs the gate itself first, so a BLOCKING failure exits non-zero here too and nothing is promoted.
# Idempotent: re-running replaces the curated partitions and the feature file rather than appending.
features:
	$(PY) -m $(PKG).pipeline.features

# Migrates first (a no-op when the schema is already there), then loads and verifies. Idempotent:
# re-running leaves identical row counts, which is what the ON CONFLICT DO UPDATE writes buy.
load:
	$(PY) -m $(PKG).pipeline.load

# Reads the curated feature file, fits and persists the Isolation Forest, and upserts one alert per
# finding. Idempotent: `alert_id` is derived from (account, rule, window_end), and the upsert leaves
# `status` alone so a re-run cannot reopen an alert an analyst has already moved.
detect:
	$(PY) -m $(PKG).pipeline.detect

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

# The third line is the T7.1 label-leakage rule: it fails if anything outside the metrics path
# reads `typology_labels`. A leak there would make every precision figure unfalsifiable, and it
# would be invisible in the output - the numbers would simply get better.
lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts
	$(PY) scripts/check_label_leakage.py

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
