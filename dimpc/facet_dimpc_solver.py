"""
facet_dimpc_solver.py — FACET-DiMPC (FAcet-based Critical region Exploration).

Paper: Brahmbhatt et al. (ACC 2026), Section III

FACET-DiMPC extends IF-mpDiMPC by restricting the online CR combination
search to facet-adjacent neighbors of the previous time step's valid CRs.

At time k, for each controller i:
    S_i = { v*_i(k-1) }  ∪  { facet_neighbors of v*_i(k-1) }

Instead of searching all  n_CR,0 × n_CR,1 × ... × n_CR,M  combinations
(IF-mpDiMPC), FACET-DiMPC only searches  |S_0| × |S_1| × ... × |S_M|
combinations — typically 2-10× smaller.

Fallback chain (paper: "reverts to iterative I-mpDiMPC"):
  1. Restricted search  (FACET-DiMPC subset)    — fastest
  2. Full search        (IF-mpDiMPC all combos)  — slower but complete
  3. One-shot I-mpDiMPC (p=1, CR lookup)         — rare safety net

Prerequisite: run facet_finder.find_all_facet_neighbors(mp_sol) offline
              to populate cr.facet_neighbors before calling run_facet_dimpc.
"""

from __future__ import annotations
import itertools
import time

import numpy as np

from .plant import Plant
from .cr_store import MPSolutions
from .dimpc_solver import (
    LocalQPMatrices, SimResult,
    precompute_qp_matrices,
    assemble_warm_start, saturate_inputs,
)
from .if_mpdimpc_solver import (
    _u_offsets, _solve_combination, _fallback_one_step,
)


def run_facet_dimpc(
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
    FACET-DiMPC closed-loop simulation.

    Requires mp_sol to have cr.facet_neighbors pre-populated by
    facet_finder.find_all_facet_neighbors(mp_sol) before calling.

    Parameters
    ----------
    plant   : Plant
    x0      : (nx,) initial state
    T       : number of simulation time steps
    mp_sol  : MPSolutions with facet_neighbors filled in
    qp_mats : precomputed QP matrices for fallback
    verbose : print per-step summary

    Returns
    -------
    SimResult
      iter_counts[k] = combinations tried in RESTRICTED search at step k
                       (1 = found immediately with prev-combo; never counts
                        full-search fallback combos)
      converged[k]   = True if valid combo found in RESTRICTED search
    """
    if qp_mats is None:
        if verbose:
            print("[FACET-DiMPC] Precomputing QP matrices for fallback...")
        qp_mats = precompute_qp_matrices(plant, Q_list, R_list, P_list, rho_list)

    # Warn if facet_neighbors is empty for all CRs (finder not run)
    total_nb = sum(len(cr.facet_neighbors)
                   for s in mp_sol.solutions for cr in s.regions)
    if total_nb == 0:
        print("[FACET-DiMPC] WARNING: facet_neighbors is empty for all CRs. "
              "Run facet_finder.find_all_facet_neighbors(mp_sol) first.")

    M  = plant.M
    nx = plant.nx
    sizes, offsets = _u_offsets(plant)

    # Full CR index lists (for fallback to IF-mpDiMPC)
    all_cr_indices = [list(range(mp_sol[i].n_cr)) for i in range(M)]

    x_traj      = np.zeros((T + 1, nx))
    u_traj      = {i: np.zeros((T, plant.subsystems[i].nu)) for i in range(M)}
    iter_counts = np.zeros(T, dtype=int)
    converged   = np.zeros(T, dtype=bool)
    solve_times = np.zeros(T)

    x_traj[0]  = x0.copy()
    U_opt_prev: dict[int, np.ndarray] | None = None

    n_restricted_fallback = 0
    n_full_fallback       = 0

    for k in range(T):
        t_start = time.perf_counter()
        x_k = x_traj[k]

        U_warm = assemble_warm_start(U_opt_prev, plant)
        U_warm = saturate_inputs(U_warm, plant)

        # ── build restricted search sets ──────────────────────────────────────
        search_sets = []
        warm_combo  = []
        for i in range(M):
            # 1) Build parameter vector theta_i from warm start
            theta_i = []
            for j in range(M):
                if j == i: continue
                theta_i.append(U_warm[j].flatten())
            theta_i_vec = np.concatenate([x_k] + theta_i)

            # 2) Find which CR contains this theta (PointLocation in MATLAB)
            v_warm = None
            for v, cr in enumerate(mp_sol[i].regions):
                if cr.contains(theta_i_vec):
                    v_warm = v
                    break

            # 3) Build S_i: the warm CR and its facet neighbors
            if v_warm is not None:
                warm_combo.append(v_warm)
                S_i = sorted(set([v_warm] + cr.facet_neighbors))
                search_sets.append(S_i)
            else:
                warm_combo.append(None)
                search_sets.append(all_cr_indices[i])

        # ── restricted search ────────────────────────────────────────────────
        U_bar        = None
        combo_tried  = 0

        # Try the warm_combo first if it's fully valid
        restricted_iter = None
        if all(v is not None for v in warm_combo):
            warm_tup = tuple(warm_combo)
            rest = (c for c in itertools.product(*search_sets) if c != warm_tup)
            restricted_iter = itertools.chain([warm_tup], rest)
        else:
            restricted_iter = itertools.product(*search_sets)

        for combo in restricted_iter:
            combo_tried += 1
            cr_combo = [mp_sol[i][v] for i, v in enumerate(combo)]
            U_sol = _solve_combination(cr_combo, x_k, plant, sizes, offsets)
            if U_sol is not None:
                U_bar      = U_sol
                converged[k] = True
                break

        iter_counts[k] = combo_tried

        # ── fallback 1: full combination search (IF-mpDiMPC) ─────────────────
        if U_bar is None:
            n_restricted_fallback += 1
            for combo in itertools.product(*all_cr_indices):
                cr_combo = [mp_sol[i][v] for i, v in enumerate(combo)]
                U_sol = _solve_combination(cr_combo, x_k, plant, sizes, offsets)
                if U_sol is not None:
                    U_bar      = U_sol
                    break

        # ── fallback 2: one-shot I-mpDiMPC ────────────────────────────────────
        if U_bar is None:
            n_full_fallback += 1
            U_bar      = _fallback_one_step(x_k, U_warm, mp_sol, qp_mats, plant)

        U_bar = saturate_inputs(U_bar, plant)

        u_k = {i: U_bar[i][:plant.subsystems[i].nu] for i in range(M)}
        for i in range(M):
            u_traj[i][k] = u_k[i]

        x_traj[k + 1] = plant.step(x_k, u_k)
        U_opt_prev     = {i: U_bar[i].copy() for i in range(M)}
        solve_times[k] = time.perf_counter() - t_start

        if verbose and (k % 10 == 0 or k == T - 1):
            tag = ('FACET' if converged[k]
                   else ('IF-fb' if n_restricted_fallback > (k - sum(converged[:k]))
                         else 'I-fb'))
            print(f"  [FACET-DiMPC] k={k:3d}  "
                  f"restricted={combo_tried:3d}  "
                  f"{'VALID' if converged[k] else 'FALLBACK'}  "
                  f"||x||={np.linalg.norm(x_traj[k+1]):.4f}  "
                  f"t={solve_times[k]*1000:.2f}ms")

    if verbose:
        n_combo_restricted = sum(len(s) for s in [
            sorted(set([0] + mp_sol[i][0].facet_neighbors)) for i in range(M)
        ])  # rough estimate
        print(f"\n[FACET-DiMPC] Restricted fallbacks: {n_restricted_fallback}/{T}  "
              f"Full fallbacks: {n_full_fallback}/{T}  "
              f"Avg restricted combos tried: {iter_counts.mean():.1f}")

    return SimResult(
        x_traj=x_traj,
        u_traj=u_traj,
        iter_counts=iter_counts,
        converged=converged,
        solve_times=solve_times,
    )
