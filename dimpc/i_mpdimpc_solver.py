"""
i_mpdimpc_solver.py — Iterative explicit DiMPC (I-mpDiMPC).

Implements Algorithm 1 from:
  Saini, Brahmbhatt et al. (C&ChE 2025), Section 2.4, Eq. (16)

Identical to DiMPC (same Wegstein loop, same warm start, same convergence
check) with ONE change at the inner loop:

    DiMPC:       U_i = solve_local_qp(θ_i, mat_i)        ← scipy SLSQP
    I-mpDiMPC:   U_i = mp_sol[i].evaluate(θ_i)           ← affine CR lookup
                      else solve_local_qp(θ_i, mat_i)    ← fallback if outside

The CR lookup is  O(n_CR × n_ineq)  per controller — much cheaper than
solving a QP — giving speedup proportional to how many active constraints
the QP solver needs to check.

The outer Wegstein loop structure is shared via dimpc_solver._run_algorithm1.
"""

from __future__ import annotations

import numpy as np

from .plant import Plant
from .cr_store import MPSolutions
from .dimpc_solver import (
    LocalQPMatrices,
    SimResult,
    precompute_qp_matrices,
    solve_local_qp,
    _run_algorithm1,
)


# ─────────────────────────────────────────────────────────────────────────────
#  CR lookup with QP fallback
# ─────────────────────────────────────────────────────────────────────────────

class _CRSolveFn:
    """
    Callable that tries CR lookup first, falls back to QP if θ is outside
    all critical regions.  Tracks fallback count for diagnostics.
    """

    def __init__(self, ctrl_sol, mat: LocalQPMatrices):
        self._ctrl_sol = ctrl_sol
        self._mat      = mat
        self.n_calls   = 0
        self.n_fallback = 0

    def __call__(self, theta_i: np.ndarray) -> np.ndarray:
        self.n_calls += 1
        U = self._ctrl_sol.evaluate(theta_i)   # full-horizon U_i or None
        if U is not None:
            return U
        # θ is outside all CRs — should be rare for well-formulated problems
        self.n_fallback += 1
        return solve_local_qp(theta_i, self._mat)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_i_mpdimpc(
    plant: Plant,
    x0: np.ndarray,
    T: int,
    mp_sol: MPSolutions,
    p_max: int = 100,
    eps: float = 1e-8,
    w_min: float = -5.0,
    w_max: float = 0.0,
    Q_list: list[np.ndarray] | None = None,
    R_list: list[np.ndarray] | None = None,
    P_list: list[np.ndarray] | None = None,
    rho_list: list[float] | None = None,
    qp_mats: list[LocalQPMatrices] | None = None,
    verbose: bool = True,
) -> SimResult:
    """
    Iterative explicit DiMPC (I-mpDiMPC) closed-loop simulation.

    Parameters
    ----------
    plant   : Plant
    x0      : (nx,) initial state
    T       : number of simulation time steps
    mp_sol  : MPSolutions — precomputed explicit mp solution for all controllers
              (from mp_solver.solve_all_mp)
    p_max   : max intermediate iterations per time step  (paper: 100)
    eps     : convergence tolerance  (paper: 1e-8)
    w_min, w_max : Wegstein weight bounds  (paper default: [-5, 0])
    Q_list, R_list, P_list, rho_list : cost weights (must match mp_sol weights)
    qp_mats : precomputed QP matrices for fallback  (if None, computed here)
    verbose : print per-step summary + fallback statistics

    Returns
    -------
    SimResult  (identical structure to DiMPC result — directly comparable)
    """
    if qp_mats is None:
        if verbose:
            print("[I-mpDiMPC] Precomputing QP matrices for fallback...")
        qp_mats = precompute_qp_matrices(plant, Q_list, R_list, P_list, rho_list)

    # Build one solve function per controller
    solve_fns = [
        _CRSolveFn(mp_sol[i], qp_mats[i])
        for i in range(plant.M)
    ]

    result = _run_algorithm1(
        plant, x0, T, p_max, eps, w_min, w_max,
        solve_fns, label="I-mpDiMPC", verbose=verbose,
    )

    if verbose:
        total_calls    = sum(f.n_calls    for f in solve_fns)
        total_fallback = sum(f.n_fallback for f in solve_fns)
        pct = 100.0 * total_fallback / max(total_calls, 1)
        print(f"\n[I-mpDiMPC] CR lookups: {total_calls - total_fallback}  "
              f"QP fallbacks: {total_fallback}  ({pct:.1f}%)")

    return result