"""
dimpc_solver.py — Iterative cooperative DiMPC (QP-based).

Implements Algorithm 1 from:
  Saini, Brahmbhatt et al. (C&ChE 2025), Section 2.3
  with Wegstein acceleration (Wegstein, 1958).

This is the baseline "online" solver — it solves a QP for each local
controller at every intermediate iteration at each sample time step.
I-mpDiMPC (next file) replaces the QP call with a CR lookup from the
offline mp solution, but the outer iteration loop is identical.

Pipeline:
  Offline:  precompute_qp_matrices  — build H_qp, H_par, G, b, F for each i
  Online:   run_dimpc               — Algorithm 1 loop over k = 0..T-1
"""

from __future__ import annotations
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .plant import Plant
from .mp_solver import (
    build_prediction_matrices,
    build_Q_full,
    build_cost_matrices,
    build_constraint_matrices,
    default_weights,
    _build_block_diag_cost,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Precomputed QP matrices (same structure used by I-mpDiMPC)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LocalQPMatrices:
    """
    Precomputed matrices for the local QP of controller i.

    The QP solved at each intermediate iteration p:
        min_{U_i}  1/2 U_i^T H_qp U_i  +  (H_par @ θ_i)^T U_i
        s.t.       G U_i  ≤  b  +  F θ_i

    where  θ_i = [x(k); U_j for j≠i]  is assembled online.
    """
    H_qp:  np.ndarray   # (n_u, n_u)        Hessian — same for all θ
    H_par: np.ndarray   # (n_u, n_theta)    parametric cost
    G:     np.ndarray   # (n_c, n_u)        constraint LHS
    b:     np.ndarray   # (n_c,)            constraint RHS constant
    F:     np.ndarray   # (n_c, n_theta)    constraint RHS parametric
    nu_i:  int
    n_u:   int          # = Np * nu_i
    n_theta: int


def precompute_qp_matrices(
    plant: Plant,
    Q_list: list[np.ndarray] | None = None,
    R_list: list[np.ndarray] | None = None,
    P_list: list[np.ndarray] | None = None,
    rho_list: list[float] | None = None,
) -> list[LocalQPMatrices]:
    """
    Precompute local QP matrices for all M controllers (offline step).

    These matrices are the same for both DiMPC (QP solve) and
    I-mpDiMPC (CR lookup) — the difference is only in how they're used online.

    Returns
    -------
    list of LocalQPMatrices, one per controller (index 0..M-1)
    """
    if Q_list is None or R_list is None or P_list is None or rho_list is None:
        Q_list, R_list, P_list, rho_list = default_weights(plant)

    M  = plant.M
    Np = plant.Np
    nx = plant.nx

    B_list = [plant.B_j(j) for j in range(M)]
    Phi_x, Gamma_list = build_prediction_matrices(plant.A, B_list, Np)

    Q_stage    = _build_block_diag_cost(plant, Q_list,  rho_list)
    Q_terminal = _build_block_diag_cost(plant, P_list, rho_list)
    Q_full     = build_Q_full(nx, Q_stage, Q_terminal, Np)

    x_lb = np.concatenate([s.x_lb for s in plant.subsystems])
    x_ub = np.concatenate([s.x_ub for s in plant.subsystems])

    mats = []
    for i in range(M):
        si    = plant.subsystems[i]
        nu_i  = si.nu
        others = [j for j in range(M) if j != i]

        R_bar_i = rho_list[i] * np.kron(np.eye(Np), R_list[i])
        H_qp, H_par, M_theta = build_cost_matrices(
            Phi_x, Gamma_list[i], [Gamma_list[j] for j in others], Q_full, R_bar_i
        )
        G, b_vec, F = build_constraint_matrices(
            Gamma_list[i], M_theta, x_lb, x_ub, si.u_lb, si.u_ub, Np
        )
        mats.append(LocalQPMatrices(
            H_qp=H_qp, H_par=H_par, G=G, b=b_vec, F=F,
            nu_i=nu_i, n_u=Np * nu_i, n_theta=M_theta.shape[1],
        ))
    return mats


# ─────────────────────────────────────────────────────────────────────────────
#  Online helpers
# ─────────────────────────────────────────────────────────────────────────────

def assemble_theta(
    x_k: np.ndarray,
    U_bar: dict[int, np.ndarray],
    i: int,
    plant: Plant,
) -> np.ndarray:
    """
    Assemble the parametric vector for controller i  (paper Eq. 17):
        θ_i = [x(k);  U_1;  U_2; ... U_{i-1};  U_{i+1}; ... U_M]

    x_k   : (nx,)  current state
    U_bar : {j: (Np*nu_j,)}  current broadcast iterate for each controller
    """
    others = [j for j in range(plant.M) if j != i]
    return np.concatenate([x_k] + [U_bar[j] for j in others])


def assemble_warm_start(
    U_opt_prev: dict[int, np.ndarray] | None,
    plant: Plant,
) -> dict[int, np.ndarray]:
    """
    Build the warm start U^(0)(k) from U*(k-1)  (paper Eq. 13):
        U_i^(0)(k) = [u_i(1|k-1), ..., u_i(Np-1|k-1), 0]

    Shift the previous optimal trajectory by one step and pad with zero.
    At k=0 (no previous solution), returns zeros clipped to input bounds.
    """
    Np = plant.Np
    U_warm = {}
    for i, s in enumerate(plant.subsystems):
        nu_i = s.nu
        if U_opt_prev is None:
            # k=0: start from zero (feasible since bounds span zero)
            U_warm[i] = np.zeros(Np * nu_i)
        else:
            U_prev = U_opt_prev[i]          # (Np*nu_i,)
            # Shift by nu_i (one time step), append zeros for the last step
            U_warm[i] = np.concatenate([U_prev[nu_i:], np.zeros(nu_i)])
    return U_warm


def saturate_inputs(
    U_bar: dict[int, np.ndarray],
    plant: Plant,
) -> dict[int, np.ndarray]:
    """Clip each U_i to the per-step input bounds [u_lb_i, u_ub_i]."""
    Np = plant.Np
    result = {}
    for i, s in enumerate(plant.subsystems):
        lb = np.tile(s.u_lb, Np)
        ub = np.tile(s.u_ub, Np)
        result[i] = np.clip(U_bar[i], lb, ub)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Local QP solver  (scipy SLSQP — no warm-starting, cold start every call)
# ─────────────────────────────────────────────────────────────────────────────

def solve_local_qp(
    theta_i: np.ndarray,
    mat: LocalQPMatrices,
) -> np.ndarray:
    """
    Solve the local QP for controller i given θ_i  (paper Eq. 15).

        min  1/2 U^T H_qp U  +  (H_par @ θ_i)^T U
        s.t. G U  ≤  b  +  F θ_i

    Uses scipy SLSQP — cold start at every call (no warm-starting).
    This is intentional: DiMPC is the baseline and should not benefit
    from any solver tricks that would hide its true iteration cost.

    Returns U_i* of shape (n_u,).
    """
    q     = mat.H_par @ theta_i          # (n_u,) linear cost for this θ
    b_eff = mat.b + mat.F @ theta_i      # (n_c,) effective RHS

    def obj(u):
        return 0.5 * u @ mat.H_qp @ u + q @ u

    def jac(u):
        return mat.H_qp @ u + q

    constraints = {'type': 'ineq',
                   'fun':  lambda u: b_eff - mat.G @ u,   # G u ≤ b_eff
                   'jac':  lambda u: -mat.G}

    res = minimize(
        obj, np.zeros(mat.n_u),
        jac=jac,
        method='SLSQP',
        constraints=constraints,
        options={'ftol': 1e-10, 'maxiter': 500, 'disp': False},
    )
    return res.x


# ─────────────────────────────────────────────────────────────────────────────
#  Simulation result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    """
    Closed-loop simulation result.

    x_traj[k]    = x(k)  for k = 0, 1, ..., T      shape (T+1, nx)
    u_traj[i][k] = u_i(k) applied at step k        shape (T, nu_i) per controller
    iter_counts[k] = intermediate iterations at step k
    converged[k]   = True if Wegstein loop converged before p_max
    solve_times[k] = wall-clock time (seconds) for step k
    """
    x_traj:      np.ndarray
    u_traj:      dict[int, np.ndarray]
    iter_counts: np.ndarray
    converged:   np.ndarray
    solve_times: np.ndarray

    @property
    def T(self) -> int:
        return len(self.iter_counts)

    def summary(self) -> str:
        return (f"SimResult: T={self.T}, "
                f"iter avg={self.iter_counts.mean():.1f}, "
                f"iter max={self.iter_counts.max()}, "
                f"converged={self.converged.mean()*100:.0f}%, "
                f"total time={self.solve_times.sum():.3f}s")


# ─────────────────────────────────────────────────────────────────────────────
#  Algorithm 1 — shared core (used by DiMPC and I-mpDiMPC)
# ─────────────────────────────────────────────────────────────────────────────

from typing import Callable

def _run_algorithm1(
    plant: Plant,
    x0: np.ndarray,
    T: int,
    p_max: int,
    eps: float,
    w_min: float,
    w_max: float,
    solve_fns: list[Callable[[np.ndarray], np.ndarray]],
    label: str = "DiMPC",
    verbose: bool = True,
) -> SimResult:
    """
    Algorithm 1 core — Wegstein-accelerated iterative DiMPC loop.

    Parameters
    ----------
    solve_fns : list of callables, one per controller.
        solve_fns[i](theta_i) → np.ndarray, shape (Np*nu_i,)
        For DiMPC     : wraps solve_local_qp (scipy SLSQP)
        For I-mpDiMPC : wraps ControllerSolution.evaluate (CR lookup + fallback)
    label : printed in verbose output to distinguish DiMPC / I-mpDiMPC
    """
    M  = plant.M
    nx = plant.nx

    x_traj      = np.zeros((T + 1, nx))
    u_traj      = {i: np.zeros((T, plant.subsystems[i].nu)) for i in range(M)}
    iter_counts = np.zeros(T, dtype=int)
    converged   = np.zeros(T, dtype=bool)
    solve_times = np.zeros(T)

    x_traj[0] = x0.copy()
    U_opt_prev: dict[int, np.ndarray] | None = None

    for k in range(T):
        t_start = time.perf_counter()
        x_k = x_traj[k]

        # ── warm start (paper Eq. 13) ─────────────────────────────────────────
        U_bar = assemble_warm_start(U_opt_prev, plant)
        U_bar = saturate_inputs(U_bar, plant)

        # ── Wegstein state ────────────────────────────────────────────────────
        r          = 1      # 1 = first pass, 2 = Wegstein active
        U_bar_prev = None
        U_raw_prev = None
        
        # U_sat is the saturated broadcast vector (used for theta and convergence)
        U_sat      = {i: U_bar[i].copy() for i in range(M)}
        U_sat_prev = None

        # ── intermediate iteration loop ───────────────────────────────────────
        p = 0
        for p in range(1, p_max + 1):

            # ── solve each controller ─────────────────────────────────────────
            U_raw_new = {}
            for i in range(M):
                theta_i       = assemble_theta(x_k, U_sat, i, plant)
                U_raw_new[i]  = solve_fns[i](theta_i)

            # ── Wegstein update ───────────────────────────────────────────────
            if r == 1:
                U_bar_prev = {i: U_bar[i].copy() for i in range(M)}
                U_bar      = {i: U_raw_new[i].copy() for i in range(M)}
                r = 2
            else:
                U_bar_new = {}
                for i in range(M):
                    denom = U_bar[i] - U_bar_prev[i]
                    numer = U_raw_new[i] - U_raw_prev[i]
                    safe  = np.abs(denom) > 1e-12
                    a     = np.where(safe, numer / np.where(safe, denom, 1.0), 0.0)
                    w     = np.where(np.abs(a - 1) > 1e-12, a / (a - 1), w_max)
                    w     = np.clip(w, w_min, w_max)
                    U_bar_new[i] = w * U_bar[i] + (1.0 - w) * U_raw_new[i]
                U_bar_prev = {i: U_bar[i].copy() for i in range(M)}
                U_bar      = {i: U_bar_new[i].copy() for i in range(M)}

            U_raw_prev = {i: U_raw_new[i].copy() for i in range(M)}
            
            U_sat_prev = {i: U_sat[i].copy() for i in range(M)}
            U_sat      = saturate_inputs(U_bar, plant)

            # ── convergence check ─────────────────────────────────────────────
            if U_sat_prev is not None:
                delta = np.concatenate([np.abs(U_sat[i] - U_sat_prev[i])
                                        for i in range(M)])
                if np.all(delta < eps):
                    converged[k] = True
                    break

        iter_counts[k] = p
        solve_times[k] = time.perf_counter() - t_start

        u_k = {i: U_sat[i][:plant.subsystems[i].nu] for i in range(M)}
        for i in range(M):
            u_traj[i][k] = u_k[i]

        x_traj[k + 1] = plant.step(x_k, u_k)
        U_opt_prev     = {i: U_sat[i].copy() for i in range(M)}

        if verbose and (k % 10 == 0 or k == T - 1):
            print(f"  [{label}] k={k:3d}  iters={iter_counts[k]:3d}  "
                  f"{'CONV' if converged[k] else 'MAX '}  "
                  f"||x||={np.linalg.norm(x_traj[k+1]):.4f}  "
                  f"t={solve_times[k]*1000:.1f}ms")

    return SimResult(
        x_traj=x_traj,
        u_traj=u_traj,
        iter_counts=iter_counts,
        converged=converged,
        solve_times=solve_times,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Public API: run_dimpc
# ─────────────────────────────────────────────────────────────────────────────

def run_dimpc(
    plant: Plant,
    x0: np.ndarray,
    T: int,
    p_max: int = 100,
    eps: float = 1e-8,
    w_min: float = -5.0,
    w_max: float = 5.0,
    Q_list: list[np.ndarray] | None = None,
    R_list: list[np.ndarray] | None = None,
    P_list: list[np.ndarray] | None = None,
    rho_list: list[float] | None = None,
    qp_mats: list[LocalQPMatrices] | None = None,
    verbose: bool = True,
) -> SimResult:
    """
    Iterative DiMPC closed-loop simulation  (Algorithm 1, QP-based).

    Each intermediate iteration solves a scipy SLSQP QP per controller.
    This is the computational baseline — I-mpDiMPC replaces the QP call
    with a precomputed CR affine lookup.
    """
    if qp_mats is None:
        if verbose:
            print("[DiMPC] Precomputing QP matrices...")
        qp_mats = precompute_qp_matrices(plant, Q_list, R_list, P_list, rho_list)

    solve_fns = [
        (lambda i: lambda theta: solve_local_qp(theta, qp_mats[i]))(i)
        for i in range(plant.M)
    ]
    return _run_algorithm1(plant, x0, T, p_max, eps, w_min, w_max,
                           solve_fns, label="DiMPC", verbose=verbose)