"""
if_mpdimpc_solver.py — Iteration-Free explicit DiMPC (IF-mpDiMPC).

Implements Algorithm 2 from:
  Saini, Brahmbhatt et al. (C&ChE 2025), Section 3.1, Eq. (19)-(20)

Key idea: instead of iterating (Wegstein loop), simultaneously solve ALL
M affine CR functions as one linear system at each sample time step.

At time k, for a candidate CR combination (v_0, v_1, ..., v_{M-1}):
  Each CR gives an affine equation:
      U_i = A^{v_i}_x  x(k)  +  sum_{j≠i} A^{v_i}_{ij} U_j  +  b^{v_i}_i

  Rearranging into a block linear system (Eq. 20):
      L  U  =  R x(k) + d
  where:
      L ∈ R^{n_u_total × n_u_total}  — I on diagonal, -A_{ij} off-diagonal
      R ∈ R^{n_u_total × nx}         — stack of A^{v_i}_x
      d ∈ R^{n_u_total}              — stack of b^{v_i}_i

  Solve U* = L⁻¹ (R x(k) + d).
  Validate: for each i, check  Φ^{v_i} θ_i ≤ φ^{v_i}  with θ_i = [x(k); U*_{j≠i}].
  If valid → U*(k) found.  Try next combination if not.

Communication load: ONE exchange per sample time (just x(k)) — no inter-
controller iteration.  Worst case: try all n_CR,0 × n_CR,1 × … combinations.
Fallback to I-mpDiMPC if no valid combination found (numerical edge case).
"""

from __future__ import annotations
import itertools
import time
from typing import Sequence

import numpy as np

from .plant import Plant
from .cr_store import CriticalRegion, MPSolutions
from .dimpc_solver import (
    LocalQPMatrices, SimResult,
    precompute_qp_matrices, assemble_theta,
    assemble_warm_start, saturate_inputs,
    solve_local_qp, _run_algorithm1,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Linear system assembly helpers
# ─────────────────────────────────────────────────────────────────────────────

def _u_offsets(plant: Plant) -> tuple[list[int], list[int]]:
    """
    Compute start offsets and sizes for each controller's block in U_total.

    Returns
    -------
    sizes   : [Np*nu_0, Np*nu_1, ..., Np*nu_{M-1}]
    offsets : [0, sizes[0], sizes[0]+sizes[1], ...]
    """
    sizes   = [plant.Np * s.nu for s in plant.subsystems]
    offsets = [sum(sizes[:i]) for i in range(plant.M)]
    return sizes, offsets


def _assemble_linear_system(
    cr_combo: Sequence[CriticalRegion],
    plant: Plant,
    sizes: list[int],
    offsets: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build  L, R, d  for the simultaneous linear system  L U = R x(k) + d.

    For each controller i, cr_combo[i].A has columns:
        [0 : nx]          → A^{v_i}_x   (w.r.t. x(k))
        [nx : nx+n_uj1]   → A^{v_i}_{i,j1}   (w.r.t. first other ctrl)
        ...in the same order as assemble_theta (j < i, then j > i)

    The rearranged equation for controller i:
        U_i - sum_{j≠i} A^{v_i}_{ij} U_j = A^{v_i}_x x(k) + b^{v_i}_i
    """
    M  = plant.M
    nx = plant.nx
    n_u_total = sum(sizes)

    L = np.eye(n_u_total)            # diagonal = I blocks
    R = np.zeros((n_u_total, nx))
    d = np.zeros(n_u_total)

    for i, cr in enumerate(cr_combo):
        rs, re = offsets[i], offsets[i] + sizes[i]

        # Coefficient of x(k)
        R[rs:re, :] = cr.A[:, :nx]
        d[rs:re]    = cr.b

        # Off-diagonal blocks: -A^{v_i}_{i,j} for each j ≠ i
        others = [j for j in range(M) if j != i]
        col = nx
        for j in others:
            cs, ce = offsets[j], offsets[j] + sizes[j]
            L[rs:re, cs:ce] = -cr.A[:, col:col + sizes[j]]
            col += sizes[j]

    return L, R, d


def _solve_combination(
    cr_combo: Sequence[CriticalRegion],
    x_k: np.ndarray,
    plant: Plant,
    sizes: list[int],
    offsets: list[int],
    tol: float = 1e-6,
) -> dict[int, np.ndarray] | None:
    """
    Solve L U = R x(k) + d for this CR combination and validate.

    Returns dict {i: U_i} if the solution is valid in all chosen CRs,
    else None.
    """
    L, R, d = _assemble_linear_system(cr_combo, plant, sizes, offsets)

    # Skip singular systems
    if abs(np.linalg.det(L)) < 1e-10:
        return None

    U_all = np.linalg.solve(L, R @ x_k + d)   # (n_u_total,)

    # Split into per-controller dict
    U_dict = {i: U_all[offsets[i]:offsets[i] + sizes[i]]
              for i in range(plant.M)}

    # Validate: for each controller i, θ_i = [x(k); U_{j≠i}^solved] must be
    # inside CR_i^{v_i}  (paper: Φ^{v_i} θ_i ≤ φ^{v_i})
    for i, cr in enumerate(cr_combo):
        theta_i = assemble_theta(x_k, U_dict, i, plant)
        if not cr.contains(theta_i, tol=tol):
            return None

    return U_dict


# ─────────────────────────────────────────────────────────────────────────────
#  IF-mpDiMPC fallback: one-shot I-mpDiMPC step (no Wegstein)
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_one_step(
    x_k: np.ndarray,
    U_warm: dict[int, np.ndarray],
    mp_sol: MPSolutions,
    qp_mats: list[LocalQPMatrices],
    plant: Plant,
) -> dict[int, np.ndarray]:
    """
    One-shot I-mpDiMPC: CR lookup for each controller using warm-start values
    of other controllers.  Equivalent to I-mpDiMPC with p=1 (no Wegstein).
    Used as fallback when no valid CR combination is found.
    """
    U_bar = {}
    for i in range(plant.M):
        theta_i = assemble_theta(x_k, U_warm, i, plant)
        U = mp_sol[i].evaluate(theta_i)
        U_bar[i] = U if U is not None else solve_local_qp(theta_i, qp_mats[i])
    return U_bar


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_if_mpdimpc(
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
    Iteration-Free mpDiMPC closed-loop simulation  (Algorithm 2).

    At each time step:
      1. Search all CR combinations for a valid simultaneous solution.
      2. ONE communication exchange — just x(k), no inter-controller iteration.
      3. Fallback to one-shot I-mpDiMPC if no valid combination found.

    Parameters
    ----------
    plant   : Plant
    x0      : (nx,) initial state
    T       : number of simulation time steps
    mp_sol  : MPSolutions — precomputed explicit mp solution for all controllers
    p_max, eps, w_min, w_max : passed to I-mpDiMPC fallback (rarely needed)
    qp_mats : precomputed QP matrices for fallback (if None, computed here)
    verbose : print per-step summary

    Returns
    -------
    SimResult
      iter_counts[k] = number of CR combinations tried at step k
                       (1 = found immediately, n_CR_0*n_CR_1*... = exhaustive)
      converged[k]   = True if valid combination found without fallback
    """
    if qp_mats is None:
        if verbose:
            print("[IF-mpDiMPC] Precomputing QP matrices for fallback...")
        qp_mats = precompute_qp_matrices(plant, Q_list, R_list, P_list, rho_list)

    M  = plant.M
    nx = plant.nx
    sizes, offsets = _u_offsets(plant)
    n_u_total     = sum(sizes)

    # All CR index lists — one per controller
    cr_indices = [list(range(mp_sol[i].n_cr)) for i in range(M)]

    # ── storage ──────────────────────────────────────────────────────────────
    x_traj      = np.zeros((T + 1, nx))
    u_traj      = {i: np.zeros((T, plant.subsystems[i].nu)) for i in range(M)}
    iter_counts = np.zeros(T, dtype=int)     # combinations tried
    converged   = np.zeros(T, dtype=bool)    # valid combo found (no fallback)
    solve_times = np.zeros(T)

    x_traj[0]  = x0.copy()
    U_opt_prev: dict[int, np.ndarray] | None = None
    prev_combo: tuple[int, ...] | None = None  # CR combination from k-1 (warm hint)

    fallback_count = 0

    for k in range(T):
        t_start = time.perf_counter()
        x_k = x_traj[k]

        U_warm = assemble_warm_start(U_opt_prev, plant)
        U_warm = saturate_inputs(U_warm, plant)

        # ── combination search (Algorithm 2 nested loops) ─────────────────────
        U_bar  = None
        combo_tried = 0

        # Build iteration order: try previous combo first (practical speedup)
        if prev_combo is not None:
            search_order = itertools.chain(
                [prev_combo],
                (c for c in itertools.product(*cr_indices) if c != prev_combo),
            )
        else:
            search_order = itertools.product(*cr_indices)

        for combo in search_order:
            combo_tried += 1
            cr_combo = [mp_sol[i][v] for i, v in enumerate(combo)]
            U_sol = _solve_combination(cr_combo, x_k, plant, sizes, offsets)
            if U_sol is not None:
                U_bar      = U_sol
                prev_combo = combo
                converged[k] = True
                break

        iter_counts[k] = combo_tried

        # ── fallback: one-shot I-mpDiMPC (p=1) ───────────────────────────────
        if U_bar is None:
            fallback_count += 1
            U_bar      = _fallback_one_step(x_k, U_warm, mp_sol, qp_mats, plant)
            prev_combo = None   # reset combo hint after fallback

        # Saturate
        U_bar = saturate_inputs(U_bar, plant)

        # ── apply first control action ────────────────────────────────────────
        u_k = {i: U_bar[i][:plant.subsystems[i].nu] for i in range(M)}
        for i in range(M):
            u_traj[i][k] = u_k[i]

        x_traj[k + 1] = plant.step(x_k, u_k)
        U_opt_prev     = {i: U_bar[i].copy() for i in range(M)}
        solve_times[k] = time.perf_counter() - t_start

        if verbose and (k % 10 == 0 or k == T - 1):
            print(f"  [IF-mpDiMPC] k={k:3d}  combos={combo_tried:4d}  "
                  f"{'VALID' if converged[k] else 'FALLBACK'}  "
                  f"||x||={np.linalg.norm(x_traj[k+1]):.4f}  "
                  f"t={solve_times[k]*1000:.2f}ms")

    if verbose:
        n_comb_total = 1
        for i in range(M):
            n_comb_total *= mp_sol[i].n_cr
        print(f"\n[IF-mpDiMPC] Total combinations: {n_comb_total}  "
              f"Fallbacks: {fallback_count}/{T}  "
              f"Avg combos tried: {iter_counts.mean():.1f}")

    return SimResult(
        x_traj=x_traj,
        u_traj=u_traj,
        iter_counts=iter_counts,
        converged=converged,
        solve_times=solve_times,
    )