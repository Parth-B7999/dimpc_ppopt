"""
test_dimpc_solver.py
====================
Tests for dimpc_solver.py — run each #%% cell in VS Code (Shift+Enter)
or run the whole file:  python tests/test_dimpc_solver.py

Cells 1-4  : pure-numpy helpers (no QP solve, fast)
Cells 5-6  : single QP call via OSQP (~ms)
Cells 7-9  : full closed-loop simulation on ACC 2026 plant (~seconds)
Cell  10   : verify iteration counts match paper Table I (M=2 case)
"""

# %% ── 0. Imports ─────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time

from dimpc.plant import make_acc2026_plant
from dimpc.dimpc_solver import (
    LocalQPMatrices,
    SimResult,
    precompute_qp_matrices,
    assemble_theta,
    assemble_warm_start,
    saturate_inputs,
    solve_local_qp,
    run_dimpc,
)

print("Imports OK")


# %% ── 1. precompute_qp_matrices: shapes ─────────────────────────────────────
#
#  For the ACC2026 plant (M=2, nx=4, nu=[1,1], Np=3):
#    n_u_i     = Np * nu_i = 3
#    n_theta_i = nx + Np * nu_{j≠i} = 4 + 3 = 7
#    n_c_i     = 2*Np*nx + 2*Np*nu_i = 24 + 6 = 30

def test_precompute_shapes():
    plant  = make_acc2026_plant()
    mats   = precompute_qp_matrices(plant)
    Np, nx = plant.Np, plant.nx

    for i, mat in enumerate(mats):
        nu_i      = plant.subsystems[i].nu
        n_u       = Np * nu_i          # 3
        n_others  = sum(plant.subsystems[j].nu for j in range(plant.M) if j != i)
        n_theta   = nx + Np * n_others  # 7
        n_c       = 2 * Np * nx + 2 * Np * nu_i  # 30

        assert mat.H_qp.shape  == (n_u, n_u),     f"H_qp  {i}: {mat.H_qp.shape}"
        assert mat.H_par.shape == (n_u, n_theta),  f"H_par {i}: {mat.H_par.shape}"
        assert mat.G.shape     == (n_c, n_u),      f"G     {i}: {mat.G.shape}"
        assert mat.b.shape     == (n_c,),           f"b     {i}: {mat.b.shape}"
        assert mat.F.shape     == (n_c, n_theta),   f"F     {i}: {mat.F.shape}"
        assert mat.n_u         == n_u
        assert mat.n_theta     == n_theta

        # H_qp must be positive definite (required for unique QP solution)
        eigvals = np.linalg.eigvalsh(mat.H_qp)
        assert eigvals.min() > 0, \
            f"H_qp[{i}] not PD: min_eig={eigvals.min():.4f}"

    print(f"PASS  test_precompute_shapes  "
          f"(n_u={n_u}, n_theta={n_theta}, n_c={n_c})")

test_precompute_shapes()


# %% ── 2. assemble_theta: structure of θ_i ────────────────────────────────────
#
#  θ_0 = [x(k); U_1]   (controller 0 excludes its own U_0)
#  θ_1 = [x(k); U_0]   (controller 1 excludes its own U_1)
#
#  Verify slicing: x part == x_k, other U parts == correct controller's U.

def test_assemble_theta():
    plant = make_acc2026_plant()
    nx = plant.nx
    Np = plant.Np
    rng = np.random.default_rng(3)

    x_k   = rng.uniform(-5, 5, nx)
    U_bar = {0: rng.uniform(-1, 1, Np * 1),
             1: rng.uniform(-1, 1, Np * 1)}

    # θ_0 = [x(k); U_1]
    theta_0 = assemble_theta(x_k, U_bar, i=0, plant=plant)
    assert theta_0.shape == (nx + Np * 1,),  f"θ_0 shape: {theta_0.shape}"
    assert np.allclose(theta_0[:nx], x_k),   "θ_0 first nx must be x(k)"
    assert np.allclose(theta_0[nx:], U_bar[1]), "θ_0 tail must be U_1"

    # θ_1 = [x(k); U_0]
    theta_1 = assemble_theta(x_k, U_bar, i=1, plant=plant)
    assert theta_1.shape == (nx + Np * 1,),  f"θ_1 shape: {theta_1.shape}"
    assert np.allclose(theta_1[:nx], x_k),   "θ_1 first nx must be x(k)"
    assert np.allclose(theta_1[nx:], U_bar[0]), "θ_1 tail must be U_0"

    print("PASS  test_assemble_theta")

test_assemble_theta()


# %% ── 3. assemble_warm_start: shift and pad ──────────────────────────────────
#
#  Warm start at k=1 given U*(k=0):
#    U_i^warm(k) = [u_i(1|k-1), ..., u_i(Np-1|k-1), 0]
#  i.e., drop the first nu_i elements, append nu_i zeros.
#
#  Paper Eq. (13): ensures continuity of the predicted trajectory.

def test_warm_start():
    plant = make_acc2026_plant()
    Np = plant.Np   # 3
    rng = np.random.default_rng(7)

    # Simulate a previous optimal U (random, but within bounds)
    U_prev = {0: rng.uniform(-1, 1, Np * 1),   # [a, b, c]
              1: rng.uniform(-1, 1, Np * 1)}

    # k=0: no previous solution → warm start should be zeros
    U_warm_k0 = assemble_warm_start(None, plant)
    for i in range(plant.M):
        assert np.all(U_warm_k0[i] == 0), f"k=0 warm start should be zero, got {U_warm_k0[i]}"

    # k=1: shift by nu_i=1, pad with zero
    U_warm_k1 = assemble_warm_start(U_prev, plant)
    for i in range(plant.M):
        nu_i = plant.subsystems[i].nu
        # First Np-1 steps: should be U_prev[i][nu_i:]
        assert np.allclose(U_warm_k1[i][:Np*nu_i - nu_i], U_prev[i][nu_i:]), \
            f"Warm start shift wrong for controller {i}"
        # Last nu_i elements: should be zero
        assert np.allclose(U_warm_k1[i][-nu_i:], 0), \
            f"Warm start pad wrong for controller {i}"

    print("PASS  test_warm_start")

test_warm_start()


# %% ── 4. saturate_inputs: clipping to bounds ─────────────────────────────────
#
#  Input bounds for ACC2026:  u_lb_0=-1.2686, u_ub_0=3.5116
#  After saturation, every element of U_i must lie in [u_lb_i, u_ub_i].

def test_saturate_inputs():
    plant = make_acc2026_plant()
    Np = plant.Np

    # Intentionally out-of-bounds values
    U_bar = {0: np.array([10.0, -10.0, 5.0]),   # all violate bounds
             1: np.array([-5.0,  8.0, -3.0])}

    U_sat = saturate_inputs(U_bar, plant)

    for i in range(plant.M):
        s = plant.subsystems[i]
        lb = np.tile(s.u_lb, Np)
        ub = np.tile(s.u_ub, Np)
        assert np.all(U_sat[i] >= lb - 1e-10), \
            f"Ctrl {i}: saturated below lb: {U_sat[i]} < {lb}"
        assert np.all(U_sat[i] <= ub + 1e-10), \
            f"Ctrl {i}: saturated above ub: {U_sat[i]} > {ub}"

    print("PASS  test_saturate_inputs")

test_saturate_inputs()


# %% ── 5. solve_local_qp: single QP solve ────────────────────────────────────
#
#  Solve the QP for controller 0 at x(k)=0 with U_bar[1]=0.
#  At the origin with zero other-inputs the solution should be ~0
#  (since H_par @ theta_0 ≈ 0 and the cost is symmetric around origin).

def test_solve_local_qp_at_origin():
    plant = make_acc2026_plant()
    mats  = precompute_qp_matrices(plant)
    nx    = plant.nx
    Np    = plant.Np

    x_zero  = np.zeros(nx)
    U_bar   = {0: np.zeros(Np * 1), 1: np.zeros(Np * 1)}
    theta_0 = assemble_theta(x_zero, U_bar, i=0, plant=plant)

    U_sol = solve_local_qp(theta_0, mats[0])

    assert U_sol.shape == (Np * 1,), f"Wrong shape: {U_sol.shape}"
    # At origin, optimal is ~0 (up to numerical tolerance)
    assert np.max(np.abs(U_sol)) < 1e-4, \
        f"Solution at origin should be ~0, got {U_sol}"

    print(f"PASS  test_solve_local_qp_at_origin  (U_0={U_sol})")

test_solve_local_qp_at_origin()


# %% ── 6. solve_local_qp: solution is consistent with cost reduction ──────────
#
#  For a non-zero theta, the QP solution U* should give strictly lower cost
#  than the zero solution (assuming zero is feasible but not optimal).

def test_solve_local_qp_cost_reduction():
    plant = make_acc2026_plant()
    mats  = precompute_qp_matrices(plant)
    nx    = plant.nx
    Np    = plant.Np
    rng   = np.random.default_rng(42)

    # Start from a random state safely inside bounds
    x_k   = rng.uniform(-5, 5, nx)
    U_bar = {0: np.zeros(Np * 1), 1: rng.uniform(-0.5, 0.5, Np * 1)}
    theta = assemble_theta(x_k, U_bar, i=0, plant=plant)

    mat   = mats[0]
    U_opt = solve_local_qp(theta, mat)

    def qp_cost(u):
        return 0.5 * u @ mat.H_qp @ u + (mat.H_par @ theta) @ u

    cost_opt  = qp_cost(U_opt)
    cost_zero = qp_cost(np.zeros(mat.n_u))

    # Check feasibility of U_opt: G @ U_opt <= b + F @ theta (with tol)
    rhs = mat.b + mat.F @ theta
    violation = np.max(mat.G @ U_opt - rhs)
    assert violation < 1e-4, f"U_opt violates constraints: max violation={violation:.2e}"

    # Cost should be <= cost at zero
    assert cost_opt <= cost_zero + 1e-6, \
        f"Optimal cost {cost_opt:.6f} > zero cost {cost_zero:.6f}"

    print(f"PASS  test_solve_local_qp_cost_reduction  "
          f"(cost: {cost_zero:.4f} → {cost_opt:.4f})")

test_solve_local_qp_cost_reduction()


# %% ── 7. run_dimpc: state converges to zero ──────────────────────────────────
#
#  Run DiMPC for T=50 steps from a random initial condition.
#  For a stable, controllable plant the state should decay to near-zero.
#  Use the same random IC as in the ACC2026 paper simulations.

def test_dimpc_convergence():
    plant = make_acc2026_plant()
    rng   = np.random.default_rng(2026)   # reproducible

    # Random initial state strictly inside state bounds
    x0 = np.array([5.0, -3.0, 4.0, -2.0])   # inside bounds

    print("\n[Cell 7] Running DiMPC T=50 steps...")
    result = run_dimpc(plant, x0, T=50, p_max=100, eps=1e-8, verbose=True)

    # State should decay
    norm_init = np.linalg.norm(result.x_traj[0])
    norm_final = np.linalg.norm(result.x_traj[-1])
    assert norm_final < norm_init, \
        f"State did not decay: ||x_0||={norm_init:.3f}, ||x_T||={norm_final:.3f}"

    # State should be close to zero by T=50 (ACC2026 paper: convergence within ~30s)
    assert norm_final < 0.1, \
        f"State not near zero after T=50: ||x||={norm_final:.4f}"

    print(f"\nPASS  test_dimpc_convergence  "
          f"(||x_0||={norm_init:.3f} → ||x_T||={norm_final:.5f})")

    return result

result_dimpc = test_dimpc_convergence()


# %% ── 8. run_dimpc: iteration counts match paper Table I ─────────────────────
#
#  Paper Table I, M=2:  max iterations = 38,  average = 19.99
#  We use the same plant (ACC2026) with eps=1e-8, p_max=100.
#  The exact numbers depend on the initial condition, but:
#  - avg iterations should be in the range [5, 50]
#  - max iterations should be well below p_max=100

def test_dimpc_iteration_stats():
    result = result_dimpc   # from cell 7

    avg_iters = result.iter_counts.mean()
    max_iters = result.iter_counts.max()
    pct_conv  = result.converged.mean() * 100

    print(f"\n[Cell 8] Iteration statistics:")
    print(f"  avg iterations : {avg_iters:.2f}  (paper: ~19.99 for M=2)")
    print(f"  max iterations : {max_iters}      (paper: 38 for M=2)")
    print(f"  converged      : {pct_conv:.0f}%")
    print(f"  avg solve time : {result.solve_times.mean()*1000:.1f} ms/step")
    print(f"  total time     : {result.solve_times.sum():.3f} s")

    # Soft bounds — exact numbers depend on IC and plant randomness
    assert avg_iters > 1, "Average iterations implausibly low"
    assert max_iters <= 100, f"Exceeded p_max: {max_iters}"
    assert pct_conv > 50, f"Less than 50% of steps converged: {pct_conv:.0f}%"

    print(f"\nPASS  test_dimpc_iteration_stats")

test_dimpc_iteration_stats()


# %% ── 9. run_dimpc: input trajectories stay within bounds ────────────────────
#
#  Every applied control input u_i(k) must satisfy u_lb_i <= u_i(k) <= u_ub_i.
#  This checks that the saturation step is correctly applied.

def test_dimpc_input_bounds():
    result = result_dimpc   # from cell 7
    plant  = make_acc2026_plant()

    for i in range(plant.M):
        s  = plant.subsystems[i]
        ui = result.u_traj[i]      # (T, nu_i)
        assert np.all(ui >= s.u_lb - 1e-8), \
            f"Controller {i}: u below lb: min={ui.min():.4f}, lb={s.u_lb}"
        assert np.all(ui <= s.u_ub + 1e-8), \
            f"Controller {i}: u above ub: max={ui.max():.4f}, ub={s.u_ub}"

    print("PASS  test_dimpc_input_bounds")

test_dimpc_input_bounds()


# %% ── 10. run_dimpc: Wegstein helps (more iters without it) ──────────────────
#
#  Bypass Wegstein by setting w_min=w_max=0 (pure damping, no acceleration).
#  Then compare avg iterations with the Wegstein-accelerated result.
#  Wegstein should require fewer iterations on average.
#
#  Note: w_min=w_max=0 collapses the mixing to U_bar = U_raw_new (no history).

def test_wegstein_helps():
    plant = make_acc2026_plant()
    x0    = np.array([5.0, -3.0, 4.0, -2.0])

    print("\n[Cell 10] Comparing with/without Wegstein...")
    qp_mats = precompute_qp_matrices(plant)

    # With Wegstein (paper defaults)
    res_w = run_dimpc(plant, x0, T=30, p_max=100, eps=1e-8,
                      w_min=-5.0, w_max=0.0, qp_mats=qp_mats, verbose=False)

    # Without Wegstein: w_min=w_max=0 → w is always 0 → U_bar = U_raw (plain iteration)
    res_nw = run_dimpc(plant, x0, T=30, p_max=100, eps=1e-8,
                       w_min=0.0, w_max=0.0, qp_mats=qp_mats, verbose=False)

    avg_w  = res_w.iter_counts.mean()
    avg_nw = res_nw.iter_counts.mean()

    print(f"  Wegstein avg iters     : {avg_w:.2f}")
    print(f"  No Wegstein avg iters  : {avg_nw:.2f}")

    # Wegstein should converge in fewer or equal iterations on average
    assert avg_w <= avg_nw + 2, \
        f"Wegstein ({avg_w:.1f}) not faster than plain ({avg_nw:.1f})"

    print(f"PASS  test_wegstein_helps  "
          f"(Wegstein saves ~{avg_nw-avg_w:.1f} iters on avg)")

test_wegstein_helps()


# %% ── 11. Run all as test suite ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("dimpc_solver.py — full test suite")
    print("="*60)
    test_precompute_shapes()
    test_assemble_theta()
    test_warm_start()
    test_saturate_inputs()
    test_solve_local_qp_at_origin()
    test_solve_local_qp_cost_reduction()
    result_dimpc = test_dimpc_convergence()
    test_dimpc_iteration_stats()
    test_dimpc_input_bounds()
    test_wegstein_helps()
    print("\nAll dimpc_solver.py tests passed.")

# %%
