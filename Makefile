.PHONY: install smoke test uniform adalora grad hwaware all clean

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

all: uniform adalora grad hwaware

smoke:
	$(PY) -m src.train --config configs/uniform_lora.yaml --seed 42 --smoke

clean:
	rm -rf results/raw_logs/* results/summaries/* results/figures/*
