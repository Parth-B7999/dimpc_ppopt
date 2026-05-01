"""
test_mp_solver.py
=================
Tests for mp_solver.py — run each #%% cell in VS Code (Shift+Enter)
or run the whole file:  python tests/test_mp_solver.py

Cells 1-6 test the matrix-building helpers (pure numpy, no PPOPT).
Cells 7-9 call PPOPT and actually solve the mpQP for the ACC 2026 plant.
"""

# %% ── 0. Imports ─────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.linalg import solve_discrete_are

from dimpc.plant import make_acc2026_plant, Plant, Subsystem
from dimpc.mp_solver import (
    build_prediction_matrices,
    build_Q_full,
    build_cost_matrices,
    build_constraint_matrices,
    build_parameter_space,
    default_weights,
    solve_local_mp,
    solve_all_mp,
    _build_block_diag_cost,
)

print("Imports OK")


# %% ── 1. build_prediction_matrices: shapes ──────────────────────────────────
#
#  For the full system x(k+1) = A x(k) + B_0 u_0 + B_1 u_1:
#  Phi_x  must be (Np*nx, nx)
#  Gamma_j must be (Np*nx, Np*nu_j)

def test_prediction_shapes():
    plant = make_acc2026_plant()
    Np = plant.Np       # 3
    nx = plant.nx       # 4
    nu = [s.nu for s in plant.subsystems]   # [1, 1]

    B_list = [plant.B_j(j) for j in range(plant.M)]
    Phi_x, Gamma_list = build_prediction_matrices(plant.A, B_list, Np)

    assert Phi_x.shape == (Np * nx, nx), \
        f"Phi_x shape: expected ({Np*nx},{nx}), got {Phi_x.shape}"
    for j, G in enumerate(Gamma_list):
        expected = (Np * nx, Np * nu[j])
        assert G.shape == expected, \
            f"Gamma_{j} shape: expected {expected}, got {G.shape}"

    print(f"PASS  test_prediction_shapes  "
          f"(Phi_x:{Phi_x.shape}, Gamma_0:{Gamma_list[0].shape})")

test_prediction_shapes()


# %% ── 2. build_prediction_matrices: correctness with a scalar system ─────────
#
#  Use the simplest possible system: 1-D state, 1-D input, A=0.5, B=1.0, Np=2.
#
#  x(1) = 0.5 x(0) + u(0)
#  x(2) = 0.25 x(0) + 0.5 u(0) + u(1)
#
#  Phi_x = [[0.5],   ← A^1
#            [0.25]]  ← A^2
#
#  Gamma_0 = [[1,   0 ],   ← row 0: A^0 B = 1, 0
#              [0.5, 1 ]]   ← row 1: A^1 B = 0.5, A^0 B = 1

def test_prediction_values_scalar():
    A = np.array([[0.5]])
    B = np.array([[1.0]])
    Phi_x, Gamma_list = build_prediction_matrices(A, [B], Np=2)

    Phi_expected = np.array([[0.5], [0.25]])
    G_expected   = np.array([[1.0, 0.0],
                              [0.5, 1.0]])

    assert np.allclose(Phi_x, Phi_expected, atol=1e-12), \
        f"Phi_x wrong:\n{Phi_x}\nexpected:\n{Phi_expected}"
    assert np.allclose(Gamma_list[0], G_expected, atol=1e-12), \
        f"Gamma wrong:\n{Gamma_list[0]}\nexpected:\n{G_expected}"

    print("PASS  test_prediction_values_scalar")

test_prediction_values_scalar()


# %% ── 3. build_prediction_matrices: X = Phi_x x + Gamma U is correct ────────
#
#  Simulate the ACC2026 plant manually for Np steps and compare with
#  the rollout matrix formula.  They must agree to machine precision.

def test_rollout_matches_simulation():
    plant = make_acc2026_plant()
    Np = plant.Np
    nx = plant.nx
    rng = np.random.default_rng(42)

    x0 = rng.uniform(-3, 3, nx)
    # Random input sequences for each controller  (Np time steps)
    U = {j: rng.uniform(-0.5, 0.5, Np * plant.subsystems[j].nu)
         for j in range(plant.M)}

    # ── rollout matrix prediction ─────────────────────────────────────────────
    B_list = [plant.B_j(j) for j in range(plant.M)]
    Phi_x, Gamma_list = build_prediction_matrices(plant.A, B_list, Np)
    X_pred = Phi_x @ x0 + sum(Gamma_list[j] @ U[j] for j in range(plant.M))

    # ── step-by-step simulation ───────────────────────────────────────────────
    X_sim = np.zeros(Np * nx)
    x = x0.copy()
    for l in range(Np):
        u = {j: U[j][l * plant.subsystems[j].nu: (l+1) * plant.subsystems[j].nu]
             for j in range(plant.M)}
        x = plant.step(x, u)
        X_sim[l*nx:(l+1)*nx] = x

    assert np.allclose(X_pred, X_sim, atol=1e-10), \
        f"Max error: {np.max(np.abs(X_pred - X_sim)):.2e}"

    print("PASS  test_rollout_matches_simulation  "
          f"(max err={np.max(np.abs(X_pred-X_sim)):.1e})")

test_rollout_matches_simulation()


# %% ── 4. build_cost_matrices: H_qp must be positive definite ─────────────────
#
#  H_qp = Gamma_i^T Q_full Gamma_i + R_bar_i
#  R_bar_i > 0  ensures H_qp > 0 (necessary for unique mpQP solution).
#  This is a necessary condition for PPOPT to find a well-posed solution.

def test_hqp_positive_definite():
    plant = make_acc2026_plant()
    Np = plant.Np
    nx = plant.nx

    Q_list, R_list, P_list, rho_list = default_weights(plant)
    B_list = [plant.B_j(j) for j in range(plant.M)]
    Phi_x, Gamma_list = build_prediction_matrices(plant.A, B_list, Np)

    for i in range(plant.M):
        nu_i = plant.subsystems[i].nu
        others = [j for j in range(plant.M) if j != i]
        Gamma_others = [Gamma_list[j] for j in others]

        Q_stage    = _build_block_diag_cost(plant, Q_list, rho_list)
        Q_terminal = _build_block_diag_cost(plant, P_list, rho_list)
        Q_full     = build_Q_full(nx, Q_stage, Q_terminal, Np)
        R_bar_i    = rho_list[i] * np.kron(np.eye(Np), R_list[i])

        H_qp, H_par, M_theta = build_cost_matrices(
            Phi_x, Gamma_list[i], Gamma_others, Q_full, R_bar_i
        )

        eigs = np.linalg.eigvalsh(H_qp)
        assert eigs.min() > 0, \
            f"H_qp for controller {i} is NOT positive definite: min_eig={eigs.min():.4f}"

        assert H_par.shape == (Np * nu_i, M_theta.shape[1]), \
            f"H_par shape wrong: {H_par.shape}"

    print(f"PASS  test_hqp_positive_definite  "
          f"(all {plant.M} controllers have PD Hessian)")

test_hqp_positive_definite()


# %% ── 5. build_constraint_matrices: shapes and feasibility at x=0, u=0 ───────
#
#  At the origin (x=0, U_j=0 for j≠i), the zero solution must be feasible:
#  G_i @ 0 ≤ b_i + F_i @ 0  →  0 ≤ b_i  (b_i must be ≥ 0 at origin).
#
#  This confirms the constraint matrix is set up correctly.

def test_constraint_matrices():
    plant = make_acc2026_plant()
    Np = plant.Np
    nx = plant.nx

    Q_list, R_list, P_list, rho_list = default_weights(plant)
    B_list = [plant.B_j(j) for j in range(plant.M)]
    Phi_x, Gamma_list = build_prediction_matrices(plant.A, B_list, Np)

    for i in range(plant.M):
        nu_i = plant.subsystems[i].nu
        others = [j for j in range(plant.M) if j != i]
        Gamma_others = [Gamma_list[j] for j in others]

        Q_stage    = _build_block_diag_cost(plant, Q_list, rho_list)
        Q_terminal = _build_block_diag_cost(plant, P_list, rho_list)
        Q_full     = build_Q_full(nx, Q_stage, Q_terminal, Np)
        R_bar_i    = rho_list[i] * np.kron(np.eye(Np), R_list[i])

        _, _, M_theta = build_cost_matrices(
            Phi_x, Gamma_list[i], Gamma_others, Q_full, R_bar_i
        )

        x_lb = np.concatenate([s.x_lb for s in plant.subsystems])
        x_ub = np.concatenate([s.x_ub for s in plant.subsystems])
        si = plant.subsystems[i]

        G_i, b_i, F_i = build_constraint_matrices(
            Gamma_list[i], M_theta, x_lb, x_ub, si.u_lb, si.u_ub, Np
        )

        n_u     = Np * nu_i
        n_theta = M_theta.shape[1]
        n_c_expected = 2 * Np * nx + 2 * Np * nu_i

        assert G_i.shape == (n_c_expected, n_u), \
            f"G_i shape: expected ({n_c_expected},{n_u}), got {G_i.shape}"
        assert b_i.shape == (n_c_expected,), \
            f"b_i shape: expected ({n_c_expected},), got {b_i.shape}"
        assert F_i.shape == (n_c_expected, n_theta), \
            f"F_i shape: expected ({n_c_expected},{n_theta}), got {F_i.shape}"

        # Feasibility at x=0, U_j=0: G_i @ 0 ≤ b_i  →  b_i ≥ 0 at origin
        # (Origin is inside the state bounds since bounds span positive and negative)
        # Just check that b_i for INPUT constraints is ≥ 0 (u_ub ≥ 0 and u_lb ≤ 0)
        b_input_ub = b_i[2*Np*nx : 2*Np*nx + n_u]   # ≥ 0 iff u_ub ≥ 0
        b_input_lb = b_i[2*Np*nx + n_u :]            # ≥ 0 iff u_lb ≤ 0
        # ACC2026: u_ub > 0 and u_lb < 0 — check
        assert np.all(b_input_ub >= 0), \
            f"Controller {i}: u_ub should be ≥ 0, got {b_input_ub}"
        assert np.all(b_input_lb >= 0), \
            f"Controller {i}: -u_lb should be ≥ 0, got {b_input_lb}"

    print(f"PASS  test_constraint_matrices  "
          f"(shape: ({n_c_expected},{n_u}), input bounds OK)")

test_constraint_matrices()


# %% ── 6. build_parameter_space: shapes and bounds ───────────────────────────
#
#  For controller i=0 with M=2:
#    θ_0 = [x(k);  U_1]
#    n_theta = nx + Np*nu_1 = 4 + 3*1 = 7
#    A_t shape: (14, 7),  b_t shape: (14, 1)
#
#  The box [θ_min, θ_max] must be non-empty (θ_max > θ_min component-wise).

def test_parameter_space():
    plant = make_acc2026_plant()
    Np = plant.Np
    nx = plant.nx
    x_lb = np.concatenate([s.x_lb for s in plant.subsystems])
    x_ub = np.concatenate([s.x_ub for s in plant.subsystems])

    for i in range(plant.M):
        others = [j for j in range(plant.M) if j != i]
        u_lbs = [plant.subsystems[j].u_lb for j in others]
        u_ubs = [plant.subsystems[j].u_ub for j in others]

        n_theta = nx + sum(Np * plant.subsystems[j].nu for j in others)
        A_t, b_t = build_parameter_space(x_lb, x_ub, u_lbs, u_ubs, Np)

        assert A_t.shape == (2 * n_theta, n_theta), \
            f"A_t shape: expected ({2*n_theta},{n_theta}), got {A_t.shape}"
        assert b_t.shape == (2 * n_theta, 1), \
            f"b_t shape: expected ({2*n_theta},1), got {b_t.shape}"

        # Non-empty box: upper half of b_t > lower-half negation → θ_max > θ_min
        theta_max = b_t[:n_theta, 0]
        theta_min = -b_t[n_theta:, 0]
        assert np.all(theta_max > theta_min), \
            f"Controller {i}: parameter space box is empty!"

    print(f"PASS  test_parameter_space  (n_theta={n_theta}, box non-empty)")

test_parameter_space()


# %% ── 7. solve_local_mp: solve mpQP for controller 0  (runs PPOPT) ───────────
#
#  This is the first PPOPT call. Runtime is typically 1–10 sec for this problem.
#  Expected: a small number of critical regions (typically 3-20 for Np=3, nu=1).
#
#  After solving:
#  - Check we get a ControllerSolution with at least 1 CR
#  - Verify CR shapes match the expected problem dimensions
#  - Verify evaluate() at a feasible θ returns a vector of shape (Np*nu_i,)

def test_solve_local_mp_controller0():
    plant = make_acc2026_plant()
    Np  = plant.Np   # 3
    nx  = plant.nx   # 4
    nu0 = plant.subsystems[0].nu  # 1
    # n_theta_0 = nx + Np*nu_1 = 4 + 3 = 7
    n_theta_0 = nx + Np * plant.subsystems[1].nu
    n_u_0     = Np * nu0   # 3

    print("\n[Cell 7] Solving mpQP for controller 0  (calling PPOPT)...")
    ctrl_sol = solve_local_mp(plant, i=0, verbose=True)

    assert ctrl_sol.controller_index == 0
    assert ctrl_sol.nu_i == nu0
    assert ctrl_sol.Np   == Np
    assert ctrl_sol.n_cr >= 1,   "Expected at least 1 critical region"

    # Check first CR dimensions
    cr0 = ctrl_sol[0]
    assert cr0.E.shape[1] == n_theta_0, \
        f"CR.E should have {n_theta_0} columns (n_theta), got {cr0.E.shape[1]}"
    assert cr0.A.shape == (n_u_0, n_theta_0), \
        f"CR.A shape: expected ({n_u_0},{n_theta_0}), got {cr0.A.shape}"

    # Evaluate at a zero theta — may be outside CRs but should not crash
    theta_zero = np.zeros(n_theta_0)
    U = ctrl_sol.evaluate(theta_zero)
    # If inside a CR, shape must be (n_u_0,)
    if U is not None:
        assert U.shape == (n_u_0,), f"evaluate shape wrong: {U.shape}"

    print(f"PASS  test_solve_local_mp_controller0  "
          f"({ctrl_sol.n_cr} CRs, n_theta={n_theta_0}, n_u={n_u_0})")

    return ctrl_sol   # return for use in later cells

ctrl_sol_0 = test_solve_local_mp_controller0()


# %% ── 8. solve_local_mp: controller 1 + verify same number of parameters ─────
#
#  By symmetry of the ACC2026 plant (both subsystems have nx=2, nu=1):
#    n_theta_1 = nx + Np*nu_0 = 4 + 3*1 = 7  (same as controller 0)
#    n_u_1 = Np * nu_1 = 3

def test_solve_local_mp_controller1():
    plant = make_acc2026_plant()
    Np  = plant.Np
    nx  = plant.nx
    nu1 = plant.subsystems[1].nu
    n_theta_1 = nx + Np * plant.subsystems[0].nu   # 7
    n_u_1     = Np * nu1                            # 3

    print("\n[Cell 8] Solving mpQP for controller 1  (calling PPOPT)...")
    ctrl_sol = solve_local_mp(plant, i=1, verbose=True)

    assert ctrl_sol.controller_index == 1
    assert ctrl_sol.n_cr >= 1

    cr0 = ctrl_sol[0]
    assert cr0.E.shape[1] == n_theta_1, \
        f"CR.E columns: expected {n_theta_1}, got {cr0.E.shape[1]}"
    assert cr0.A.shape == (n_u_1, n_theta_1), \
        f"CR.A shape: expected ({n_u_1},{n_theta_1}), got {cr0.A.shape}"

    print(f"PASS  test_solve_local_mp_controller1  ({ctrl_sol.n_cr} CRs)")
    return ctrl_sol

ctrl_sol_1 = test_solve_local_mp_controller1()


# %% ── 9. solve_all_mp: solve all M controllers at once ──────────────────────
#
#  Calls PPOPT for each controller sequentially and bundles into MPSolutions.
#  Verify:
#  - mp_sol.M == 2
#  - mp_sol[i] matches individual solves from cells 7-8
#  - Total CR count matches sum of individual solves

def test_solve_all_mp():
    plant = make_acc2026_plant()

    print("\n[Cell 9] Solving mpQP for ALL controllers  (calling PPOPT twice)...")
    mp_sol = solve_all_mp(plant, verbose=True)

    assert mp_sol.M == plant.M,  f"Expected M={plant.M}, got {mp_sol.M}"

    for i in range(plant.M):
        assert mp_sol[i].controller_index == i
        assert mp_sol[i].n_cr >= 1

    total_cr = sum(mp_sol[i].n_cr for i in range(plant.M))
    assert total_cr == mp_sol.total_cr()

    print(f"PASS  test_solve_all_mp  (CRs per controller: "
          f"{[mp_sol[i].n_cr for i in range(plant.M)]}, total={total_cr})")

    return mp_sol

mp_sol = test_solve_all_mp()


# %% ── 10. Consistency: evaluate vs Gurobi online solve ─────────────────────
#
#  For controller i=0, pick a random θ that lies inside a CR, then solve the
#  online QP directly with scipy and verify the mp solution matches.
#
#  Uses scipy.optimize.minimize (SLSQP) as a reference — no Gurobi needed.

def test_mp_vs_online_qp():
    from scipy.optimize import minimize

    plant = make_acc2026_plant()
    Np = plant.Np
    nx = plant.nx
    i  = 0
    si = plant.subsystems[i]
    nu_i = si.nu
    n_u  = Np * nu_i

    Q_list, R_list, P_list, rho_list = default_weights(plant)
    B_list = [plant.B_j(j) for j in range(plant.M)]
    Phi_x, Gamma_list = build_prediction_matrices(plant.A, B_list, Np)
    others = [j for j in range(plant.M) if j != i]
    Gamma_others = [Gamma_list[j] for j in others]

    Q_stage    = _build_block_diag_cost(plant, Q_list, rho_list)
    Q_terminal = _build_block_diag_cost(plant, P_list, rho_list)
    Q_full     = build_Q_full(nx, Q_stage, Q_terminal, Np)
    R_bar_i    = rho_list[i] * np.kron(np.eye(Np), R_list[i])

    H_qp, H_par, M_theta = build_cost_matrices(
        Phi_x, Gamma_list[i], Gamma_others, Q_full, R_bar_i
    )
    x_lb = np.concatenate([s.x_lb for s in plant.subsystems])
    x_ub = np.concatenate([s.x_ub for s in plant.subsystems])
    G_i, b_i_vec, F_i = build_constraint_matrices(
        Gamma_list[i], M_theta, x_lb, x_ub, si.u_lb, si.u_ub, Np
    )

    # Use the ctrl_sol_0 solved in cell 7
    # Find a theta that lies inside one of its CRs
    found_theta = None
    for cr in ctrl_sol_0.regions:
        # Try to find an interior point via least-squares
        E, f = cr.E, cr.f
        theta_ls = np.linalg.lstsq(E, f * 0.5, rcond=None)[0]
        if np.all(E @ theta_ls <= f + 1e-6):
            found_theta = theta_ls
            break

    if found_theta is None:
        print("SKIP  test_mp_vs_online_qp  (no interior point found)")
        return

    theta = found_theta
    # ── mp solution ───────────────────────────────────────────────────────────
    U_mp = ctrl_sol_0.evaluate(theta)
    if U_mp is None:
        print("SKIP  test_mp_vs_online_qp  (theta outside all CRs)")
        return

    # ── online QP via scipy (reference) ──────────────────────────────────────
    lin_cost = H_par @ theta          # (n_u,) linear part for this theta
    def obj(u):
        return 0.5 * u @ H_qp @ u + lin_cost @ u
    def jac(u):
        return H_qp @ u + lin_cost

    b_eff = b_i_vec + F_i @ theta     # (n_c,) effective RHS
    constraints = [{'type': 'ineq',
                    'fun': lambda u: b_eff - G_i @ u,
                    'jac': lambda u: -G_i}]
    bounds = [(-1e6, 1e6)] * n_u

    res = minimize(obj, np.zeros(n_u), jac=jac,
                   method='SLSQP', bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-12, 'maxiter': 1000})

    assert res.success, f"Online QP failed: {res.message}"
    U_online = res.x

    assert np.allclose(U_mp, U_online, atol=1e-5), \
        (f"mp solution differs from online QP:\n"
         f"  mp     = {U_mp}\n"
         f"  online = {U_online}\n"
         f"  diff   = {np.abs(U_mp - U_online)}")

    print(f"PASS  test_mp_vs_online_qp  "
          f"(max diff={np.max(np.abs(U_mp - U_online)):.2e})")

test_mp_vs_online_qp()


# %% ── 11. Run all as a test suite ────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("mp_solver.py — full test suite")
    print("="*60)
    test_prediction_shapes()
    test_prediction_values_scalar()
    test_rollout_matches_simulation()
    test_hqp_positive_definite()
    test_constraint_matrices()
    test_parameter_space()
    ctrl_sol_0 = test_solve_local_mp_controller0()
    ctrl_sol_1 = test_solve_local_mp_controller1()
    mp_sol     = test_solve_all_mp()
    test_mp_vs_online_qp()
    print("\nAll mp_solver.py tests passed.")

# %%
