# AuditNeSy / MIMIC-IV convenience targets.
# Set MIMIC_DIR to your MIMIC-IV hosp folder before running data steps.

MIMIC_DIR ?= ./mimiciv/hosp
AUDITNESY_OUT ?= ./auditnesy_mimic_out
export MIMIC_DIR
export AUDITNESY_OUT

.PHONY: install check cohort run test all clean

install:
	pip install -r requirements.txt

check:            ## STEP 0: verify lab itemids against d_labitems
	python scripts/00_check_itemids.py

cohort:           ## STEP 1: build + cache the AKI cohort (gitignored output)
	python scripts/01_build_cohort.py

run:              ## STEP 2: run the AuditNeSy suite (20 seeds + negative controls)
	python scripts/02_run_experiment.py

test:             ## run unit tests (no MIMIC access needed)
	python -m pytest tests/ -q

all: check cohort run

clean:
	rm -rf $(AUDITNESY_OUT) **/__pycache__ .pytest_cache
