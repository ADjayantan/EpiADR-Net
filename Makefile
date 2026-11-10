.PHONY: help audit baselines zeroshot targets expression mechanism ablation results figures test lint clean

PY ?= python3

help:
	@echo "make audit       - pipeline validity audit        -> results/audit_findings.json"
	@echo "make baselines   - baselines + tissue ablation    -> results/baselines.json"
	@echo "make zeroshot    - S1/S2/S3 generalisation        -> results/zeroshot.json"
	@echo "make mechanism   - drug-target x expression (M0-M9) -> results/mechanism.json"
	@echo "make ablation    - permutation test of organ->expression -> results/ablation_test.json"
	@echo "make results     - all of the above, in order"
	@echo "make figures     - render results/figures/*.png"
	@echo ""
	@echo "  external data (cached + committed; only needed to refetch):"
	@echo "make targets     - PubChem drug-target activity   -> data/drug_target_matrix.csv"
	@echo "make expression  - HPA tissue nTPM for the panel  -> data/gene_tissue_expression.csv"
	@echo ""
	@echo "make test        - pytest"
	@echo "make lint        - ruff check"

audit:
	$(PY) scripts/audit_pipeline.py

baselines:
	$(PY) scripts/run_baselines.py

zeroshot:
	$(PY) scripts/run_zeroshot.py

targets:
	$(PY) scripts/fetch_targets.py

expression:
	$(PY) scripts/build_expression.py

mechanism:
	$(PY) scripts/run_mechanism.py

ablation:
	$(PY) scripts/run_ablation_test.py --permutations 12

results: audit baselines zeroshot mechanism ablation

figures:
	$(PY) scripts/make_figures.py

test:
	$(PY) -m pytest -q tests/test_honest_pipeline.py tests/test_mechanism.py

lint:
	ruff check .

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
