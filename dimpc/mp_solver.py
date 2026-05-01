"""
mp_solver.py — Offline mp programming solver for local DiMPC controllers.

Takes a Plant and builds + solves the mpQP for each local controller i,
returning ControllerSolution (list of CriticalRegions) via PPOPT.

Paper: Saini, Brahmbhatt et al. (C&ChE 2025), Sec. 2.4, Eq. (14)-(16)

Offline computation pipeline (per controller i):
  1. build_prediction_matrices  — Φ_x, Γ_j  (state rollout)
  2. build_cost_matrices        — H_qp, H_par  (quadratic + parametric cost)
  3. build_constraint_matrices  — G_i, b_i, F_i  (state + input bounds)
  4. build_parameter_space      — A_t, b_t  (box on θ_i)
  5. solve_local_mp             — call PPOPT → ControllerSolution
  6. solve_all_mp               — loop i=0..M-1 → MPSolutions

mpQP standard form passed to PPOPT (from mp_gne_solver.py convention):
  min_{U_i}  1/2 U_i^T Q U_i  +  (H @ θ_i)^T U_i   +  c^T U_i
  s.t.       G_i U_i  ≤  b_i  +  F_i θ_i
             A_t θ_i  ≤  b_t

where θ_i = [x(k); U_1; ...; U_{i-1}; U_{i+1}; ...; U_M]  (Eq. 17)
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import solve_discrete_are, LinAlgError

from ppopt.mpqp_program import MPQP_Program
from ppopt.mp_solvers.solve_mpqp import solve_mpqp, mpqp_algorithm

from .plant import Plant
from .cr_store import ControllerSolution, MPSolutions, from_ppopt_solution


# ─────────────────────────────────────────────────────────────────────────────
#  *** ALGORITHM SELECTION — change this line to switch PPOPT solver ***
#
#  combinatorial            — Gupta et al. 2011, deterministic, no output
#  combinatorial_parallel   — same but multi-core, prints depth + timing ← verbose
#  geometric                — facet-walking, good for larger problems
#  geometric_parallel       — facet-walking multi-core, prints active sets ← verbose
#  graph                    — graph-based, fast for small n_theta
#  graph_parallel           — graph-based multi-core
#
#  Rule of thumb:
#    small problem  (n_u ≤ 5, n_theta ≤ 10):  combinatorial  (fastest, deterministic)
#    medium problem (n_u ≤ 15):               geometric_parallel  (good balance + verbose)
#    large problem  (n_u > 15):               geometric_parallel_exp  (uses pruning)
#
#  For verbose CR-generation output during development use either
#  combinatorial_parallel or geometric_parallel — these print timing per depth/pass.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ALGORITHM = mpqp_algorithm.combinatorial_parallel   # ← change here


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1: Prediction matrices
# ─────────────────────────────────────────────────────────────────────────────

def build_prediction_matrices(
    A: np.ndarray,
    B_list: list[np.ndarray],
    Np: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Compute state rollout matrices for the full coupled system.

    Full system:  x(k+1) = A x(k) + sum_j B_j u_j(k)

    Predicted states stacked as X = [x(1|k); ...; x(Np|k)]:
        X = Phi_x x(k) + sum_j Gamma_j U_j

    Parameters
    ----------
    A : (nx, nx)
    B_list : list of (nx, nu_j)  — one per controller j
    Np : prediction horizon

    Returns
    -------
    Phi_x : (Np*nx, nx)
        Phi_x = [A; A²; ...; A^{Np}]
    Gamma_list : list of (Np*nx, Np*nu_j)
        Gamma_j[l,q] block = A^{l-q} B_j  for q≤l,  else 0
        (lower block-Toeplitz, eq. to paper Eq. 14)
    """
    nx = A.shape[0]

    # Precompute A^0, A^1, ..., A^{Np}
    A_pows = [np.eye(nx)]
    for _ in range(Np):
        A_pows.append(A_pows[-1] @ A)

    # Phi_x = [A^1; A^2; ...; A^{Np}]
    Phi_x = np.vstack([A_pows[l + 1] for l in range(Np)])   # (Np*nx, nx)

    # Gamma_j: lower block-Toeplitz
    Gamma_list = []
    for B_j in B_list:
        nu_j = B_j.shape[1]
        G = np.zeros((Np * nx, Np * nu_j))
        for l in range(Np):          # row block: prediction time l+1
            for q in range(l + 1):   # col block: input applied at step q
                rs, cs = l * nx, q * nu_j
                G[rs:rs + nx, cs:cs + nu_j] = A_pows[l - q] @ B_j
        Gamma_list.append(G)

    return Phi_x, Gamma_list


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2: Cost matrices
# ─────────────────────────────────────────────────────────────────────────────

def build_Q_full(
    nx: int,
    Q_stage: np.ndarray,
    Q_terminal: np.ndarray,
    Np: int,
) -> np.ndarray:
    """
    Block-diagonal state cost matrix over the prediction horizon.

        Q_full = blkdiag(Q_stage, ..., Q_stage, Q_terminal)  ← Np blocks

    Shape: (Np*nx, Np*nx).

    Covers predicted states X = [x(1|k); ...; x(Np|k)].
    The current state x(0|k) = x(k) is a constant; its cost is ignored.
    """
    blocks = [Q_stage] * (Np - 1) + [Q_terminal]
    return np.block([[b if i == j else np.zeros((nx, nx))
                      for j, b in enumerate(blocks)]
                     for i, b in enumerate(blocks)])


def build_cost_matrices(
    Phi_x: np.ndarray,
    Gamma_i: np.ndarray,
    Gamma_others: list[np.ndarray],
    Q_full: np.ndarray,
    R_bar_i: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cost matrices for controller i's mpQP  (paper Eq. 15).

    Full cost: J = 1/2 X^T Q_full X + 1/2 U_i^T R_bar_i U_i  + ...

    with X = M_theta θ_i + Gamma_i U_i,
         M_theta = [Phi_x | Gamma_{j1} | Gamma_{j2} | ...]  (Np*nx × n_theta)

    Expanding and collecting U_i terms:
        H_qp  = Gamma_i^T Q_full Gamma_i + R_bar_i   (n_u × n_u)  ← PPOPT's Q
        H_par = Gamma_i^T Q_full M_theta              (n_u × n_theta) ← PPOPT's H
        c     = 0_{n_u}                               (no constant linear term)

    Returns
    -------
    H_qp  : (n_u, n_u)    Hessian (must be PD for unique solution)
    H_par : (n_u, n_theta) Parametric cost  — PPOPT cost = (H_par @ θ)^T U
    M_theta : (Np*nx, n_theta)
    """
    M_theta = np.hstack([Phi_x] + Gamma_others)      # (Np*nx, n_theta)
    H_qp  = Gamma_i.T @ Q_full @ Gamma_i + R_bar_i   # (n_u, n_u)
    H_par = Gamma_i.T @ Q_full @ M_theta              # (n_u, n_theta)
    return H_qp, H_par, M_theta


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3: Constraint matrices
# ─────────────────────────────────────────────────────────────────────────────

def build_constraint_matrices(
    Gamma_i: np.ndarray,
    M_theta: np.ndarray,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    u_lb_i: np.ndarray,
    u_ub_i: np.ndarray,
    Np: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    G_i U_i ≤ b_i + F_i θ_i   (paper Eq. 15 constraints)

    State constraints (for all subsystems, all predicted steps l=1..Np):
        x_lb ≤ X_l ≤ x_ub  →  ±Gamma_i U_i ≤ ±(x_{ub/lb}_rep - M_theta θ_i)

    Input constraints (for controller i at all steps):
        u_lb_i ≤ u_i(l) ≤ u_ub_i  →  ±I U_i ≤ ±u_{ub/lb}_rep

    Returns
    -------
    G_i : (n_c, n_u)
    b_i : (n_c,)        constant part of RHS
    F_i : (n_c, n_theta) parametric part of RHS
    """
    nx    = x_lb.shape[0]
    nu_i  = u_lb_i.shape[0]
    n_u   = Np * nu_i
    n_theta = M_theta.shape[1]

    # ── state bounds on X = [x(1|k); ...; x(Np|k)] ──────────────────────────
    x_ub_rep = np.tile(x_ub, Np)        # (Np*nx,)
    x_lb_rep = np.tile(x_lb, Np)

    G_xu =  Gamma_i                     # (Np*nx, n_u)
    G_xl = -Gamma_i
    b_xu =  x_ub_rep                    # constant part
    b_xl = -x_lb_rep
    F_xu = -M_theta                     # parametric part: move M_theta θ to RHS
    F_xl =  M_theta

    # ── input bounds on U_i ──────────────────────────────────────────────────
    u_ub_rep = np.tile(u_ub_i, Np)      # (Np*nu_i,)
    u_lb_rep = np.tile(u_lb_i, Np)

    I_u = np.eye(n_u)
    G_uu =  I_u
    G_ul = -I_u
    b_uu =  u_ub_rep
    b_ul = -u_lb_rep
    F_zero = np.zeros((n_u, n_theta))

    # ── stack ────────────────────────────────────────────────────────────────
    G_i = np.vstack([G_xu, G_xl, G_uu, G_ul])
    b_i = np.concatenate([b_xu, b_xl, b_uu, b_ul])
    F_i = np.vstack([F_xu, F_xl, F_zero, F_zero])

    return G_i, b_i, F_i


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4: Parameter space
# ─────────────────────────────────────────────────────────────────────────────

def build_parameter_space(
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    u_lbs_others: list[np.ndarray],
    u_ubs_others: list[np.ndarray],
    Np: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Box constraint on θ_i = [x(k); U_{j1}; U_{j2}; ...]  (eq. PPOPT A_t, b_t).

    Written as:  [I; -I] θ_i ≤ [θ_max; -θ_min]

    Parameters
    ----------
    x_lb, x_ub : global state bounds  (nx,)
    u_lbs_others, u_ubs_others : input bounds for controllers j≠i (each nu_j,)
    Np : prediction horizon

    Returns
    -------
    A_t : (2*n_theta, n_theta)
    b_t : (2*n_theta, 1)   ← PPOPT expects column vector
    """
    theta_min = np.concatenate(
        [x_lb] + [np.tile(lb, Np) for lb in u_lbs_others]
    )
    theta_max = np.concatenate(
        [x_ub] + [np.tile(ub, Np) for ub in u_ubs_others]
    )
    n_theta = len(theta_min)
    A_t = np.vstack([np.eye(n_theta), -np.eye(n_theta)])
    b_t = np.concatenate([theta_max, -theta_min]).reshape(-1, 1)
    return A_t, b_t


# ─────────────────────────────────────────────────────────────────────────────
#  Default weight matrices
# ─────────────────────────────────────────────────────────────────────────────

def default_weights(
    plant: Plant,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float]]:
    """
    Sensible defaults: Q_i = I, R_i = I, P_i from DARE (or I on failure).

    Returns Q_list, R_list, P_list, rho_list  (one entry per subsystem).
    rho_i = 1/M so that sum(rho_i) = 1.
    """
    M   = plant.M
    rho = [1.0 / M] * M
    Q_list, R_list, P_list = [], [], []

    for s in plant.subsystems:
        Q_i = np.eye(s.nx)
        R_i = np.eye(s.nu)
        # DARE for terminal cost using local dynamics A_i, B_{i,i}, Q_i, R_i
        try:
            P_i = solve_discrete_are(s.A, s.B[s.index], Q_i, R_i)
        except (LinAlgError, ValueError):
            P_i = Q_i
        Q_list.append(Q_i)
        R_list.append(R_i)
        P_list.append(P_i)

    return Q_list, R_list, P_list, rho


# ─────────────────────────────────────────────────────────────────────────────
#  Step 5: Solve one local controller
# ─────────────────────────────────────────────────────────────────────────────

def solve_local_mp(
    plant: Plant,
    i: int,
    Q_list: list[np.ndarray] | None = None,
    R_list: list[np.ndarray] | None = None,
    P_list: list[np.ndarray] | None = None,
    rho_list: list[float] | None = None,
    algorithm: mpqp_algorithm = DEFAULT_ALGORITHM,
    verbose: bool = True,
) -> ControllerSolution:
    """
    Solve the mpQP for local controller i and return its ControllerSolution.

    Parameters
    ----------
    plant : Plant
    i : int   (0-based controller index)
    Q_list, R_list, P_list : weight matrices per subsystem (default: identity / DARE)
    rho_list : plantwide cost weights per subsystem (default: 1/M each)
    algorithm : PPOPT mpqp_algorithm enum
    verbose : print progress

    Returns
    -------
    ControllerSolution  with all critical regions for controller i
    """
    if Q_list is None or R_list is None or P_list is None or rho_list is None:
        Q_list, R_list, P_list, rho_list = default_weights(plant)

    M  = plant.M
    Np = plant.Np
    si = plant.subsystems[i]
    nx = plant.nx
    nu_i = si.nu

    # ── 1. Prediction matrices ────────────────────────────────────────────────
    B_list = [plant.B_j(j) for j in range(M)]           # global B_j matrices
    Phi_x, Gamma_list = build_prediction_matrices(plant.A, B_list, Np)
    Gamma_i = Gamma_list[i]                              # (Np*nx, Np*nu_i)

    # Others' Gamma in the order they appear in θ_i
    others = [j for j in range(M) if j != i]
    Gamma_others = [Gamma_list[j] for j in others]

    # ── 2. Cost matrices ─────────────────────────────────────────────────────
    Q_stage    = _build_block_diag_cost(plant, Q_list, rho_list)
    Q_terminal = _build_block_diag_cost(plant, P_list, rho_list)
    Q_full     = build_Q_full(nx, Q_stage, Q_terminal, Np)   # (Np*nx, Np*nx)

    R_bar_i = rho_list[i] * np.kron(np.eye(Np), R_list[i])   # (Np*nu_i, Np*nu_i)

    H_qp, H_par, M_theta = build_cost_matrices(
        Phi_x, Gamma_i, Gamma_others, Q_full, R_bar_i
    )

    # ── 3. Constraint matrices ────────────────────────────────────────────────
    x_lb = np.concatenate([s.x_lb for s in plant.subsystems])
    x_ub = np.concatenate([s.x_ub for s in plant.subsystems])
    G_i, b_i_vec, F_i = build_constraint_matrices(
        Gamma_i, M_theta, x_lb, x_ub, si.u_lb, si.u_ub, Np
    )

    # ── 4. Parameter space ────────────────────────────────────────────────────
    u_lbs_others = [plant.subsystems[j].u_lb for j in others]
    u_ubs_others = [plant.subsystems[j].u_ub for j in others]
    A_t, b_t = build_parameter_space(x_lb, x_ub, u_lbs_others, u_ubs_others, Np)

    n_u     = Np * nu_i
    n_theta = M_theta.shape[1]
    n_c     = G_i.shape[0]
    c       = np.zeros((n_u, 1))
    b_i_col = b_i_vec.reshape(-1, 1)
    F_i_col = F_i   # (n_c, n_theta)

    if verbose:
        print(f"\n[mp_solver] Controller {i}:")
        print(f"  n_u={n_u}, n_theta={n_theta}, n_c={n_c}")
        print(f"  lambda_min(H_qp)={np.linalg.eigvalsh(H_qp).min():.4f}  "
              f"(must be > 0)")

    # ── 5. Solve mpQP with PPOPT ──────────────────────────────────────────────
    # MPQP_Program(A, b, c, H, Q, A_t, b_t, F)
    #   A, b, F  ↔  G_i, b_i, F_i  (constraints)
    #   c        ↔  0  (no constant linear cost)
    #   H        ↔  H_par  (parametric cost, n_u × n_theta)
    #   Q        ↔  H_qp   (Hessian, n_u × n_u)
    #   A_t, b_t ↔  parameter space box
    problem = MPQP_Program(
        G_i,       # A
        b_i_col,   # b
        c,         # c
        H_par,     # H  — parametric cost
        H_qp,      # Q  — Hessian
        A_t,       # A_t
        b_t,       # b_t
        F_i_col,   # F  — parametric constraint
    )

    solution = solve_mpqp(problem, algorithm=algorithm)

    if verbose:
        print(f"  → {len(solution.critical_regions)} critical regions")

    return from_ppopt_solution(solution, controller_index=i, nu_i=nu_i, Np=Np)


def _build_block_diag_cost(
    plant: Plant,
    weight_list: list[np.ndarray],
    rho_list: list[float],
) -> np.ndarray:
    """Block-diagonal global cost matrix: blkdiag(rho_i * W_i for i in 0..M-1)."""
    blocks = [rho_list[i] * weight_list[i] for i in range(plant.M)]
    nx_total = sum(b.shape[0] for b in blocks)
    result = np.zeros((nx_total, nx_total))
    offset = 0
    for b in blocks:
        n = b.shape[0]
        result[offset:offset + n, offset:offset + n] = b
        offset += n
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Step 6: Solve all controllers
# ─────────────────────────────────────────────────────────────────────────────

def solve_all_mp(
    plant: Plant,
    Q_list: list[np.ndarray] | None = None,
    R_list: list[np.ndarray] | None = None,
    P_list: list[np.ndarray] | None = None,
    rho_list: list[float] | None = None,
    algorithm: mpqp_algorithm = DEFAULT_ALGORITHM,
    verbose: bool = True,
) -> MPSolutions:
    """
    Solve mpQP for all M local controllers and return MPSolutions.

    This is the full offline precomputation step — equivalent to running
    PAROC/MPT3 offline in MATLAB.

    Returns
    -------
    MPSolutions  with M ControllerSolution objects
    """
    if Q_list is None or R_list is None or P_list is None or rho_list is None:
        Q_list, R_list, P_list, rho_list = default_weights(plant)

    solutions = []
    for i in range(plant.M):
        ctrl_sol = solve_local_mp(
            plant, i,
            Q_list=Q_list, R_list=R_list, P_list=P_list, rho_list=rho_list,
            algorithm=algorithm, verbose=verbose,
        )
        solutions.append(ctrl_sol)

    mp_sol = MPSolutions(solutions=solutions)
    if verbose:
        print(f"\n[mp_solver] Done. {mp_sol.summary()}")
    return mp_sol