.PHONY: install smoke test uniform adalora grad hwaware hwaware05 all sweep metrics clean

PY ?= python
SEED ?= 42

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest src/tests -q

uniform:
	bash experiments/run_uniform_lora.sh $(SEED)

adalora:
	bash experiments/run_adalora.sh $(SEED)

grad:
	bash experiments/run_gradient_adaptive_lora.sh $(SEED)

hwaware:
	bash experiments/run_hardware_aware_lora.sh $(SEED)

hwaware05:
	bash experiments/run_hardware_aware_lora_alpha0_5.sh $(SEED)

all: uniform adalora grad hwaware

# Full Phase 6.8 sweep: 5 method-configs × 3 seeds + auto-aggregate.
# Override via env: `SEEDS="42 43" METHODS="uniform adalora" make sweep`
sweep:
	bash experiments/run_sweep.sh

# Re-run the aggregator without re-training (when results/raw_logs is populated).
metrics:
	$(PY) -m src.metrics \
		--logs-dir results/raw_logs \
		--summaries-dir results/summaries \
		--figures-dir results/figures

smoke:
	$(PY) -m src.train --config configs/uniform_lora.yaml --seed 42 --smoke

clean:
	rm -rf results/raw_logs/* results/summaries/* results/figures/*
