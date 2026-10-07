# Pipeline of the execution-RL repository (Hafsi & Vittori 2026 and the
# learning configuration of DESIGN.md Section 7).
#
# Usage:
#   make install            create .venv with uv and install the three packages
#   make test               full test-suite
#   make test-fast          skip tests marked "slow"
#   make check-market       market validity on every seed/scenario (D13), run before results are used
#   make train              Fig. 4: one DQN run per learning rate of CONFIG (seed 10) plus the
#                           extra seeds of the selected rate, in parallel
#   make evaluate-baselines TWAP / Passive / Random on every scenario of CONFIG
#   make evaluate-rl        RL policy from CHECKPOINT on every scenario
#   make price-paths        Fig. 3 sample paths (4 h and 30 min windows)
#   make figures            all figures and tables from results/
#   make all                check-market, train, evaluate, price-paths, figures
#   make clean              remove caches and build artefacts (keeps results/)
#
# CONFIG selects the configuration (default: the learning configuration
# experiments/configs/learn.yaml; the paper's MDP is experiments/configs/paper.yaml).
# Learning rates, seeds, budget and output directory are read from it.
#   make all CONFIG=experiments/configs/paper.yaml
#   make evaluate-rl CHECKPOINT=results/learn/training/lr_0.0002_seed_10/checkpoint
#   make train TOTAL_TIMESTEPS=20000

SHELL := /bin/bash
.ONESHELL:

VENV            ?= .venv
PYTHON          ?= $(VENV)/bin/python
UV              ?= uv
PYTHON_VERSION  ?= 3.10

CONFIG          ?= experiments/configs/learn.yaml
# read from the config's name / training section (override on the command line if needed)
# learning rates are written like the run directories: 1e-3, 5e-4 (mantissa e exponent)
cfgval          = $(shell $(PYTHON) -c "import yaml,math; c=yaml.safe_load(open('$(CONFIG)')); f=lambda x: '%ge%d' % (float(x)/10**math.floor(math.log10(float(x))), math.floor(math.log10(float(x)))); print($(1))")
NAME            ?= $(call cfgval,c.get('name'))
SEED            ?= $(call cfgval,c['training'].get('seed'))
TOTAL_TIMESTEPS ?= $(call cfgval,c['training'].get('total_timesteps'))
LEARNING_RATES  ?= $(call cfgval,' '.join(f(x) for x in c['training']['learning_rates']))
SELECTED_LR     ?= $(call cfgval,f(c['training']['selected_learning_rate']))
EXTRA_SEEDS     ?= $(call cfgval,' '.join(str(s) for s in c['training'].get('extra_seeds', [])))
RESULTS         ?= results/$(NAME)
CHECKPOINT      ?= $(RESULTS)/training/lr_$(SELECTED_LR)_seed_$(SEED)/checkpoint

# One thread per process: the training runs share the machine and ABIDES is
# single-threaded anyway.
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1

.PHONY: help install test test-fast train evaluate-baselines evaluate-rl \
        price-paths figures all clean

help:
	@sed -n '1,21p' Makefile | sed 's/^# \{0,1\}//'

install:
	$(UV) venv --python $(PYTHON_VERSION) $(VENV)
	$(UV) pip install --python $(PYTHON) -r requirements.txt \
		-e ./abides-core -e ./abides-markets -e ./abides-gym

test:
	$(PYTHON) -m pytest tests -q

test-fast:
	$(PYTHON) -m pytest tests -q -m 'not slow'

# Market validity (D13): on every evaluation seed and scenario of $(CONFIG),
# both market makers quoting, no empty side of the book, no discarded market
# order, TWAP executing exactly X0. Exit status 1 otherwise.
check-market:
	$(PYTHON) -m experiments.check_market --config $(CONFIG) \
		--out $(RESULTS)/check_market.csv

# Figure 4: one DQN run per initial learning rate (seed $(SEED)) plus the extra
# seeds of the selected rate (Fig. 4b), all started in the background and
# awaited. Each run writes progress.csv, episodes.csv and a final checkpoint
# under $(RESULTS)/training/lr_<lr>_seed_<seed>/. The other training settings
# (decay, exploration, discount, network) come from the config.
train:
	mkdir -p $(RESULTS)/training
	for lr in $(LEARNING_RATES); do
		$(PYTHON) -m experiments.train_dqn --config $(CONFIG) \
			--lr $$lr --seed $(SEED) --total-timesteps $(TOTAL_TIMESTEPS) \
			--out $(RESULTS)/training/lr_$${lr}_seed_$(SEED) &
	done
	for seed in $(EXTRA_SEEDS); do
		$(PYTHON) -m experiments.train_dqn --config $(CONFIG) \
			--lr $(SELECTED_LR) --seed $$seed --total-timesteps $(TOTAL_TIMESTEPS) \
			--out $(RESULTS)/training/lr_$(SELECTED_LR)_seed_$$seed &
	done
	wait

# Hand-crafted baselines (process pool) on every scenario listed under
# evaluation.scenarios in $(CONFIG).
evaluate-baselines:
	$(PYTHON) -m experiments.evaluate --config $(CONFIG) \
		--policies TWAP Passive Random --out $(RESULTS)

# Greedy policy of the trained DQN checkpoint (main process, needs RLlib).
evaluate-rl:
	@test -d "$(CHECKPOINT)" || { echo "checkpoint $(CHECKPOINT) not found; run 'make train' first or set CHECKPOINT=..."; exit 1; }
	$(PYTHON) -m experiments.evaluate --config $(CONFIG) \
		--policies RL --checkpoint $(CHECKPOINT) --out $(RESULTS)

# Figure 3: undisturbed best-ask paths for several seeds (4 h window, about
# 14 000 one-second steps as in the paper) plus a 30 min window matching the
# execution horizon.
price-paths:
	$(PYTHON) -m experiments.sample_price_paths --config $(CONFIG) \
		--seeds 5 --window 04:00:00 --out $(RESULTS)/price_paths_4h.parquet
	$(PYTHON) -m experiments.sample_price_paths --config $(CONFIG) \
		--seeds 5 --window 00:30:00 --out $(RESULTS)/price_paths_30min.parquet

figures:
	$(PYTHON) -m experiments.make_figures --config $(CONFIG)

all: check-market train evaluate-baselines evaluate-rl price-paths figures

clean:
	find . -type d -name __pycache__ -not -path './$(VENV)/*' -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache ray_results
	rm -rf abides-core/build abides-markets/build abides-gym/build
	find . -type d -name '*.egg-info' -not -path './$(VENV)/*' -prune -exec rm -rf {} +
