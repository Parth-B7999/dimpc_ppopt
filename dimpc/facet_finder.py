"""
facet_finder.py — Offline LP-based and Hyperplane-based facet neighbor detection.

Paper: Brahmbhatt et al. (ACC 2026), Section III, Eq. (14)

For each pair of CRs (v, w) within the SAME controller's solution,
determines whether they share a proper common facet (not merely a
shared hyperplane or a point).

Three geometric cases (Fig. 1 of ACC 2026):
  a) Non-intersecting : regions share a hyperplane equation but no points
  b) Point intersection: regions touch at exactly one point on the hyperplane
  c) Common facet     : regions share a (d-1)-dimensional face  ← this is a neighbor

Methods:
  - "lp": solves the LP from Eq. (14) to robustly verify shared facets.
  - "hyperplane": uses simple normal vector opposite direction comparison
    (similar to LI-mpDiMPC V2) to quickly identify separating boundaries.

Fills cr.facet_neighbors in-place for every CriticalRegion in a
ControllerSolution, then returns the enriched ControllerSolution.
"""

from __future__ import annotations
import time
import numpy as np
from scipy.optimize import linprog

from .cr_store import CriticalRegion, ControllerSolution, MPSolutions


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: detect axis-aligned (box) constraints
# ─────────────────────────────────────────────────────────────────────────────

def _is_box_constraint(e: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Return True if e is a pure axis-aligned unit vector (±e_k), i.e.
    exactly one non-zero entry equal to ±1.

    The parameter-space box constraints   A_t θ ≤ b_t   come from:
        [+I; -I] θ ≤ [θ_max; -θ_min]
    so every CR produced by PPOPT inherits 2*n_theta ±e_k rows.
    These rows appear identically (or with the exact opposite sign) in every
    CR, so they always trigger the opposite-normal check and produce a false
    positive "facet neighbor" for every single pair of CRs.

    Skipping them in the hyperplane method eliminates the false positives
    without discarding genuine facets, because genuine facets arise from the
    active constraints of the QP (G_i U_i ≤ b_i + F_i θ_i), not from the
    parameter-space bounding box.
    """
    nz = np.count_nonzero(np.abs(e) > tol)
    if nz != 1:
        return False
    val = np.abs(e[np.abs(e) > tol][0])
    return abs(val - 1.0) < tol


# ─────────────────────────────────────────────────────────────────────────────
#  Pre-check: does hyperplane H = {x : e^T x = f} appear in CR_k?
# ─────────────────────────────────────────────────────────────────────────────

def _hyperplane_in_cr(
    e: np.ndarray,
    f: float,
    cr_k: CriticalRegion,
    tol: float = 1e-6,
    check_same_direction: bool = True,
) -> bool:
    """
    Quick check: is the hyperplane {x : e^T x = f} a shared boundary with CR_k?

    Normalise both sides and compare. Two adjacent CRs share the exact same
    constraint row at the common facet. If check_same_direction is False,
    only opposite signs (pointing outwards) are matched, which acts as a fast
    facet test without LP.

    NOTE: axis-aligned (box) constraint rows are always skipped — they come
    from the parameter-space bounding box A_t θ ≤ b_t and appear identically
    in every CR, so they would cause every pair of CRs to be flagged as
    neighbors.
    """
    e_norm = e / (np.linalg.norm(e) + 1e-14)
    f_s    = f  / (np.linalg.norm(e) + 1e-14)

    for l in range(cr_k.n_ineq):
        ek = cr_k.E[l]
        # ── skip box constraints on both sides ────────────────────────────────
        if _is_box_constraint(e) or _is_box_constraint(ek):
            continue

        nrm = np.linalg.norm(ek) + 1e-14
        ek_norm = ek / nrm
        fk_s    = cr_k.f[l] / nrm

        # Same direction: e ~ e_k, f ~ f_k
        if check_same_direction and np.linalg.norm(e_norm - ek_norm) < tol and abs(f_s - fk_s) < tol:
            return True
        # Opposite direction: e ~ -e_k, f ~ -f_k  (boundary between two regions)
        if np.linalg.norm(e_norm + ek_norm) < tol and abs(f_s + fk_s) < tol:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  LP-based facet test  (Eq. 14)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_facet_lp(
    cr_i: CriticalRegion,
    cr_k: CriticalRegion,
    j: int,
) -> float | None:
    """
    Solve Eq. (14): max t s.t. x ∈ CR_i, x ∈ CR_k, t ≤ d(x, H^l_i) for l ≠ j.

    Variables: z = [x; t] ∈ R^{n_theta + 1}

    LP in scipy standard form  (min c^T z  s.t. A_ub z ≤ b_ub):
        c       = [0,...,0, -1]        ← maximise t
        [E_i|0] z ≤ f_i               ← x ∈ CR_i
        [E_k|0] z ≤ f_k               ← x ∈ CR_k
        [E_i[l]|1] z ≤ f_i[l]  l≠j   ← t ≤ slack to non-j facets

    Returns t* (optimal), or None if the LP is infeasible (no intersection).
    """
    n_theta = cr_i.E.shape[1]
    n_var   = n_theta + 1          # [x (n_theta); t (1)]
    zero1   = np.zeros((cr_i.n_ineq, 1))
    zerok   = np.zeros((cr_k.n_ineq, 1))

    # Objective: min -t
    c = np.zeros(n_var)
    c[-1] = -1.0

    # x ∈ CR_i
    A_ri = np.hstack([cr_i.E, zero1])
    b_ri = cr_i.f

    # x ∈ CR_k
    A_rk = np.hstack([cr_k.E, zerok])
    b_rk = cr_k.f

    # t ≤ f_i[l] - E_i[l,:] x  for l ≠ j  →  [E_i[l,:] | 1] z ≤ f_i[l]
    not_j = [l for l in range(cr_i.n_ineq) if l != j]
    if not_j:
        ones = np.ones((len(not_j), 1))
        A_t  = np.hstack([cr_i.E[not_j], ones])
        b_t  = cr_i.f[not_j]
    else:
        A_t = np.zeros((0, n_var))
        b_t = np.zeros(0)

    A_ub = np.vstack([A_ri, A_rk, A_t])
    b_ub = np.concatenate([b_ri, b_rk, b_t])
    bounds = [(-1e6, 1e6)] * n_var

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                  options={'disp': False})

    if res.status == 0:          # optimal
        return float(res.x[-1])
    if res.status == 3:          # infeasible → no intersection
        return None
    return None                  # other failure (treat as infeasible)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API: find neighbors for one controller
# ─────────────────────────────────────────────────────────────────────────────

def find_facet_neighbors(
    ctrl_sol: ControllerSolution,
    method: str = "hyperplane",
    tol_t: float = 1e-6,
    tol_pre: float = 1e-6,
    verbose: bool = True,
) -> ControllerSolution:
    """
    Compute and store facet neighbors for all CRs in one ControllerSolution.

    Fills cr.facet_neighbors in-place for every CR in ctrl_sol.regions.
    Clears any existing neighbor lists first (idempotent).

    Parameters
    ----------
    ctrl_sol : ControllerSolution  (from mp_solver)
    method   : "lp" (robust equation 14) or "hyperplane" (fast normal comparison)
    tol_t    : LP threshold — t* > tol_t means proper shared facet
    tol_pre  : hyperplane-match tolerance for pre-check
    verbose  : print summary

    Returns
    -------
    ctrl_sol  (modified in-place, returned for chaining)
    """
    t0   = time.perf_counter()
    n_cr = ctrl_sol.n_cr

    if method not in ["lp", "hyperplane"]:
        raise ValueError(f"Unknown facet finding method: {method}")

    # Clear existing neighbor lists
    for cr in ctrl_sol.regions:
        cr.facet_neighbors.clear()

    n_shared = 0
    n_lps    = 0
    n_pairs  = n_cr * (n_cr - 1) // 2

    for v in range(n_cr):
        for w in range(v + 1, n_cr):
            cr_v, cr_w = ctrl_sol[v], ctrl_sol[w]

            found = False
            if method == "lp":
                # Check all facets of CR_v against CR_w
                for j in range(cr_v.n_ineq):
                    if not _hyperplane_in_cr(cr_v.E[j], cr_v.f[j], cr_w, tol=tol_pre, check_same_direction=True):
                        continue
                    n_lps += 1
                    t_opt = _solve_facet_lp(cr_v, cr_w, j)
                    if t_opt is not None and t_opt > tol_t:
                        found = True
                        break
            elif method == "hyperplane":
                # Simple normal comparison: match MATLAB's A./b == A./b
                # which covers BOTH same and opposite constraint directions.
                for j in range(cr_v.n_ineq):
                    if _hyperplane_in_cr(cr_v.E[j], cr_v.f[j], cr_w, tol=tol_pre, check_same_direction=True):
                        found = True
                        break

            if found:
                cr_v.facet_neighbors.append(w)
                cr_w.facet_neighbors.append(v)
                n_shared += 1

    elapsed = time.perf_counter() - t0
    avg_nb  = sum(len(cr.facet_neighbors) for cr in ctrl_sol.regions) / max(n_cr, 1)

    if verbose:
        lp_str = f"{n_lps} LPs solved, " if method == "lp" else "0 LPs solved, "
        print(f"  ctrl {ctrl_sol.controller_index} ({method}): "
              f"{n_cr} CRs, {n_pairs} pairs checked, "
              f"{lp_str}{n_shared} facet-neighbor pairs, "
              f"avg {avg_nb:.1f} neighbors/CR  ({elapsed:.2f}s)")

    return ctrl_sol


# ─────────────────────────────────────────────────────────────────────────────
#  Public API: find neighbors for all controllers
# ─────────────────────────────────────────────────────────────────────────────

def find_all_facet_neighbors(
    mp_sol: MPSolutions,
    method: str = "hyperplane",
    tol_t: float = 1e-6,
    tol_pre: float = 1e-6,
    verbose: bool = True,
) -> MPSolutions:
    """
    Run find_facet_neighbors for all M controllers in mp_sol.

    Modifies mp_sol in-place (fills facet_neighbors for every CR).
    Returns mp_sol for chaining / re-assignment.

    Methods:
    - 'lp': Rigorous LP solve to ensure proper facets (robust but slow).
    - 'hyperplane': Compares opposite direction normal vectors (fast).

    Typical runtime:
    - lp: 10-60 minutes for M=3
    - hyperplane: 1-2 minutes for M=3
    """
    t0 = time.perf_counter()
    if verbose:
        print(f"[facet_finder] Finding facet neighbors for {mp_sol.M} controllers using '{method}' method...")

    for i in range(mp_sol.M):
        find_facet_neighbors(mp_sol[i], method=method, tol_t=tol_t, tol_pre=tol_pre, verbose=verbose)

    if verbose:
        print(f"[facet_finder] Done in {time.perf_counter()-t0:.2f}s  "
              f"(total neighbor pairs: "
              f"{sum(len(cr.facet_neighbors) for s in mp_sol.solutions for cr in s.regions) // 2})")

    return mp_sol
