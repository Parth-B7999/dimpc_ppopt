# Distributed multi-parametric Model Predictive Control (DiMPC)

This repository contains an implementation of **Distributed multi-parametric Model Predictive Control (DiMPC)** algorithms. It extends multi-parametric quadratic programming (mpQP) techniques from the `ppopt` package into distributed frameworks.

## Features

- **FACET-DiMPC**: An iteration-free distributed MPC method utilizing advanced critical region facet detection.
- **Fast Facet Detection**: Includes an optimized LI-mpDiMPC V2 inspired hyperplane comparison approach to bypass time-consuming linear programming steps during offline calculation.
- **I-mpDiMPC / IF-mpDiMPC**: Alternative distributed multi-parametric controllers with convergence properties.
- **Randomized Plant Generation**: A built-in plant generator to simulate complex multi-agent system bidding and distributed constrained dynamics.
- **Offline / Online Phasing**: Pre-computes and caches control solutions to optimize online trajectory calculations.

## Project Structure

- `dimpc/`: Core library for distributed MPC solvers.
  - `dimpc_solver.py`: Base iterative DiMPC solver.
  - `facet_dimpc_solver.py`: FACET-DiMPC solver implementation.
  - `if_mpdimpc_solver.py`: Iteration-free IF-mpDiMPC logic.
  - `facet_finder.py`: Implements both geometric hyperplane-matching and rigorous LP-based algorithms to detect neighbors between critical regions.
  - `cr_store.py`: Abstractions for explicitly storing and querying critical regions efficiently.
  - `mp_solver.py`: Interfaces with the `ppopt` multi-parametric QP algorithm.
  - `plant.py` & `plant_gen.py`: Plant and subsystem factory logic.
- `tests/`: Extensive test suite validating iterations, bounds, pre-computed matrices, and fallback rates. Contains the core case study generator (`test_case_study.py`).

## Installation

Ensure you have Python 3.10+ installed.

1. Clone the repository
2. Install dependencies (e.g., `numpy`, `scipy`, `ppopt`)
   
## Running the Case Study

To run the benchmarking case study and compare the computational speed of the various solvers (DiMPC, I-mpDiMPC, IF-mpDiMPC, FACET-DiMPC):

```bash
python tests/test_case_study.py
```

This will run offline multi-parametric analysis (using the fast hyperplane detection method by default) followed by an online trajectory simulation comparison. Output statistics will be collected and verified against convergence expectations.

## References

Implementation is based on:
- Saini, Brahmbhatt et al. (C&ChE 2025)
- Brahmbhatt et al. (ACC 2026)
