# PACELC Swarm CRDT Simulation Artifact

This repository contains the discrete-event simulation artifact for the paper
"Measured-Divergence Coordination for Mobile Swarms: A PACELC Analysis of
CRDT Replication versus Linearizable Agreement."

The artifact compares three intent-replication semantics behind a common
swarm coordination interface:

- Raft-style leader-and-quorum intent replication
- overwrite-by-arrival epidemic gossip
- CRDT intent replication using set-union merge and deterministic resolution

## Contents

- `src/`: simulation model, network model, protocol implementations, experiment
  definitions, and plotting code.
- `run_smoke.py`: quick sanity run.
- `run_experiments.py`: executes the experiment suite.
- `run_plots.py`: regenerates figures from saved results.
- `requirements.txt`: Python package requirements.
- `results/raw/summary.csv`: aggregated experiment outputs used for summary
  tables.
- `results/raw/time_series.csv`: time-series outputs used for convergence and
  healing plots.
- `results/figures/`: generated PDF and PNG figures.

## Requirements

Python 3.10 or later is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
