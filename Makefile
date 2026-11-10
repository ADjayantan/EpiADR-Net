.PHONY: help audit baselines zeroshot results test lint clean

PY ?= python3

help:
	@echo "make audit      - reproduce the pipeline validity audit  -> results/audit_findings.json"
	@echo "make baselines  - prevalence / fingerprint / tissue-ablation suite -> results/baselines.json"
	@echo "make zeroshot   - leave-one-organ-out generalisation      -> results/zeroshot.json"
	@echo "make results    - all of the above, in order"
	@echo "make test       - pytest suite"
	@echo "make lint       - ruff check"

audit:
	$(PY) scripts/audit_pipeline.py

baselines:
	$(PY) scripts/run_baselines.py

zeroshot:
	$(PY) scripts/run_zeroshot.py

results: audit baselines zeroshot

test:
	$(PY) -m pytest -q tests/

lint:
	ruff check . --ignore E501

clean:
	rm -rf results/*.json __pycache__ */__pycache__ .pytest_cache
