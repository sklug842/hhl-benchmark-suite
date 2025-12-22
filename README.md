# HHL Benchmark Suite

This repository contains code to generate benchmark instances for the HHL (Harrow–Hassidim–Lloyd) linear-systems algorithm and to evaluate solution-quality metrics across different execution backends:
- PennyLane simulators (ideal; optional GPU via `lightning.gpu`)
- Qiskit Aer simulators (ideal and noisy via backend-derived noise models)
- IBM Quantum Runtime / hardware execution (via `SamplerV2`)

The suite supports repeated runs per instance and records summary statistics (mean/std) for effective shots and distance/error metrics.

---

## Repository Structure

- `Code/`
  - `help_classes/`
    - `wrapper_hhl.py` — main HHL wrapper implementation (PennyLane + Qiskit paths)
  - `instance_generation/`
    - `Data_For_Experiments.py` — generates and stores benchmark instances / records (`.npy`)
  - `test_HHL/`
    - `HHL_Test.py` — runs experiments on stored records (ideal/noisy/hardware depending on flags)
- `Data/`
  - Stores generated `.npy` record files (not tracked by default if large)
- `Paper/` (optional)
  - Paper artifacts / experiment notes (if included)

---

## Requirements

Tested with Python 3.10+.

Key dependencies:
- `numpy`, `scipy`, `matplotlib`
- `pennylane` (and optional `pennylane-lightning[gpu]` for GPU)
- `qiskit`, `qiskit-aer`
- `qiskit-ibm-runtime`

Recommended: create a virtual environment.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
