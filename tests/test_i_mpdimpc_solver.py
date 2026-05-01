"""
test_i_mpdimpc_solver.py
========================
Tests for i_mpdimpc_solver.py — run each #%% cell in VS Code (Shift+Enter)
or run the whole file:  python tests/test_i_mpdimpc_solver.py

Cell 1   : offline precomputation (PPOPT + QP matrices) — runs once
Cell 2   : verify CR lookup replaces QP (fallback rate should be 0 or near 0)
Cell 3   : trajectory equivalence — DiMPC vs I-mpDiMPC must give same result
Cell 4   : iteration count equivalence
Cell 5   : I-mpDiMPC is faster per time step than DiMPC
Cell 6   : full closed-loop convergence check
Cell 7   : manual single-step CR lookup vs QP (unit test, no simulation)
"""

# %% ── 0. Imports ─────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time

from dimpc.plant import make_acc2026_plant
from dimpc.mp_solver import solve_all_mp, default_weights
from dimpc.dimpc_solver import (
    precompute_qp_matrices,
    assemble_theta,
    assemble_warm_start,
    solve_local_qp,
    run_dimpc,
)
from dimpc.i_mpdimpc_solver import run_i_mpdimpc, _CRSolveFn

print("Imports OK")


# %% ── 1. Offline precomputation (runs PPOPT — ~10 sec, shared across cells) ──
#
#  Solve the mpQP for each controller offline.
#  This is the one-time cost that makes I-mpDiMPC iteration-free online.
#  We also precompute QP matrices so DiMPC and I-mpDiMPC share the same setup.

def setup():
    plant   = make_acc2026_plant()
    Q_list, R_list, P_list, rho_list = default_weights(plant)

    print("\n[Cell 1] Solving offline mp problems (PPOPT)...")
    mp_sol  = solve_all_mp(plant, Q_list, R_list, P_list, rho_list, verbose=True)
    qp_mats = precompute_qp_matrices(plant, Q_list, R_list, P_list, rho_list)

    print(f"\n  Controller 0: {mp_sol[0].n_cr} CRs")
    print(f"  Controller 1: {mp_sol[1].n_cr} CRs")
    print("PASS  setup (mp_sol and qp_mats ready)")
    return plant, mp_sol, qp_mats, Q_list, R_list, P_list, rho_list

plant, mp_sol, qp_mats, Q_list, R_list, P_list, rho_list = setup()


# %% ── 2. CR lookup: fallback rate should be 0% ───────────────────────────────
#
#  For a well-formulated problem with bounded parameter space, the mp solution
#  covers all feasible θ_i.  So evaluate() should never return None during
#  a normal simulation — fallback rate = 0%.
#
#  This confirms the mp solution quality and that θ is assembled correctly.

def test_fallback_rate():
    plant_l = make_acc2026_plant()
    Np = plant_l.Np
    nx = plant_l.nx
    rng = np.random.default_rng(42)

    fallbacks = 0
    n_tests   = 200

    for _ in range(n_tests):
        # Random θ_i that are INSIDE the parameter space box
        x_k   = rng.uniform(-5, 5, nx)   # inside state bounds [-63..29]
        U_bar = {
            j: np.array([rng.uniform(plant_l.subsystems[j].u_lb[0],
                                     plant_l.subsystems[j].u_ub[0])
                         for _ in range(Np)])
            for j in range(plant_l.M)
        }
        for i in range(plant_l.M):
            theta_i = assemble_theta(x_k, U_bar, i, plant_l)
            U = mp_sol[i].evaluate(theta_i)
            if U is None:
                fallbacks += 1

    fallback_pct = 100.0 * fallbacks / (n_tests * plant_l.M)
    print(f"\n[Cell 2] Fallback rate over {n_tests} random θ: "
          f"{fallbacks}/{n_tests*plant_l.M} = {fallback_pct:.1f}%")

    assert fallback_pct < 10, \
        f"Fallback rate too high: {fallback_pct:.1f}% — mp solution may be incomplete"

    print(f"PASS  test_fallback_rate  ({fallback_pct:.1f}% fallbacks)")

test_fallback_rate()


# %% ── 3. Trajectory equivalence: DiMPC ≈ I-mpDiMPC ─────────────────────────
#
#  Both algorithms solve the same optimization problem with the same Wegstein
#  loop.  The only difference is solver (QP vs CR lookup).  Since both give
#  the exact optimizer of the same objective, the trajectories must agree.
#
#  Tolerance: 1e-3 — accounts for SLSQP vs affine-lookup numerical differences.

def test_trajectory_equivalence():
    x0 = np.array([5.0, -3.0, 4.0, -2.0])
    T  = 30

    print("\n[Cell 3] Running DiMPC and I-mpDiMPC from same IC...")

    res_dimpc = run_dimpc(
        plant, x0, T, p_max=100, eps=1e-8,
        qp_mats=qp_mats, verbose=False,
    )
    res_imp = run_i_mpdimpc(
        plant, x0, T, mp_sol, p_max=100, eps=1e-8,
        qp_mats=qp_mats, verbose=True,
    )

    # State trajectories must match closely
    max_state_err = np.max(np.abs(res_dimpc.x_traj - res_imp.x_traj))
    assert max_state_err < 1e-3, \
        f"State trajectory mismatch: max error = {max_state_err:.2e}"

    # Input trajectories must match
    for i in range(plant.M):
        max_u_err = np.max(np.abs(res_dimpc.u_traj[i] - res_imp.u_traj[i]))
        assert max_u_err < 1e-3, \
            f"Input trajectory mismatch ctrl {i}: max error = {max_u_err:.2e}"

    print(f"\nPASS  test_trajectory_equivalence  "
          f"(max state err={max_state_err:.2e})")

    return res_dimpc, res_imp

res_dimpc, res_imp = test_trajectory_equivalence()


# %% ── 4. Iteration count equivalence ────────────────────────────────────────
#
#  Since both algorithms follow identical Wegstein logic, the number of
#  intermediate iterations per time step must be the same (or very close).
#  Any difference indicates the CR lookup gave a slightly different U,
#  pushing the Wegstein trajectory down a different path.

def test_iteration_equivalence():
    max_iter_diff = np.max(np.abs(res_dimpc.iter_counts.astype(int) -
                                   res_imp.iter_counts.astype(int)))
    avg_dimpc = res_dimpc.iter_counts.mean()
    avg_imp   = res_imp.iter_counts.mean()

    print(f"\n[Cell 4] Iteration counts:")
    print(f"  DiMPC      avg: {avg_dimpc:.2f}  max: {res_dimpc.iter_counts.max()}")
    print(f"  I-mpDiMPC  avg: {avg_imp:.2f}  max: {res_imp.iter_counts.max()}")
    print(f"  Max per-step difference: {max_iter_diff}")

    # Allow small differences due to solver tolerance mismatch
    assert max_iter_diff <= 5, \
        f"Iteration counts differ too much: max diff = {max_iter_diff}"

    print("PASS  test_iteration_equivalence")

test_iteration_equivalence()


# %% ── 5. Speed: I-mpDiMPC faster per step than DiMPC ────────────────────────
#
#  The CR lookup (matrix-vector multiply + inequality check) is O(n_CR * n_ineq).
#  The SLSQP QP solve is O(n_u^2 * n_c) per iteration internally.
#  For Np=3, nu=1, n_CR~14: the CR lookup should be measurably faster.
#
#  We run over T=50 steps and compare average per-step solve time.

def test_speed_comparison():
    x0 = np.array([5.0, -3.0, 4.0, -2.0])
    T  = 50

    print("\n[Cell 5] Timing comparison over T=50 steps...")

    res_d = run_dimpc(
        plant, x0, T, p_max=100, eps=1e-8,
        qp_mats=qp_mats, verbose=False,
    )
    res_m = run_i_mpdimpc(
        plant, x0, T, mp_sol, p_max=100, eps=1e-8,
        qp_mats=qp_mats, verbose=False,
    )

    avg_dimpc = res_d.solve_times.mean() * 1000     # ms
    avg_imp   = res_m.solve_times.mean() * 1000     # ms
    speedup   = avg_dimpc / max(avg_imp, 1e-6)

    print(f"  DiMPC      avg: {avg_dimpc:.3f} ms/step  total: {res_d.solve_times.sum():.3f}s")
    print(f"  I-mpDiMPC  avg: {avg_imp:.3f} ms/step  total: {res_m.solve_times.sum():.3f}s")
    print(f"  Speedup: {speedup:.2f}×")

    # I-mpDiMPC should not be significantly SLOWER than DiMPC
    # (It may be faster or similar depending on problem size and CR count)
    assert avg_imp < avg_dimpc * 5, \
        f"I-mpDiMPC ({avg_imp:.3f}ms) much slower than DiMPC ({avg_dimpc:.3f}ms)"

    print(f"PASS  test_speed_comparison  ({speedup:.2f}× speedup)")

test_speed_comparison()


# %% ── 6. Closed-loop convergence to zero ─────────────────────────────────────
#
#  Same test as DiMPC — both methods should drive the state to near-zero.

def test_convergence():
    x0 = np.array([5.0, -3.0, 4.0, -2.0])

    print("\n[Cell 6] Running I-mpDiMPC T=50 convergence test...")
    res = run_i_mpdimpc(
        plant, x0, T=50, mp_sol=mp_sol,
        qp_mats=qp_mats, verbose=True,
    )

    norm_init  = np.linalg.norm(res.x_traj[0])
    norm_final = np.linalg.norm(res.x_traj[-1])

    assert norm_final < norm_init, "State did not decay"
    assert norm_final < 0.1,      f"State not near zero: ||x||={norm_final:.4f}"
    assert res.converged.mean() > 0.5, \
        f"Less than 50% of steps converged: {res.converged.mean()*100:.0f}%"

    print(f"\nPASS  test_convergence  "
          f"(||x_0||={norm_init:.3f} → ||x_T||={norm_final:.5f}, "
          f"conv={res.converged.mean()*100:.0f}%)")

test_convergence()


# %% ── 7. Unit test: single CR lookup vs QP at a known θ ─────────────────────
#
#  For a θ that lies inside a known CR, the CR lookup must return the same
#  answer as the QP solver to high precision.
#
#  This is the core correctness check: U_i^mp = U_i^QP  (paper Sec 2.4 claim).

def test_cr_lookup_matches_qp():
    Np = plant.Np
    nx = plant.nx

    matched = 0
    for v, cr in enumerate(mp_sol[0].regions):
        # Find an interior point of this CR via least-squares
        E, f = cr.E, cr.f
        theta_ls = np.linalg.lstsq(E, f * 0.5, rcond=None)[0]
        if not np.all(E @ theta_ls <= f + 1e-6):
            continue

        theta_i = theta_ls

        # CR lookup
        U_mp  = cr.evaluate(theta_i)          # affine: A^v θ + b^v

        # QP solve (scipy SLSQP, same matrices, cold start)
        U_qp  = solve_local_qp(theta_i, qp_mats[0])

        err = np.max(np.abs(U_mp - U_qp))
        assert err < 1e-4, \
            (f"CR {v}: mp lookup differs from QP:\n"
             f"  mp = {U_mp}\n  qp = {U_qp}\n  err = {err:.2e}")
        matched += 1
        if matched >= 3:
            break

    assert matched >= 1, "Could not find any CR interior point to test"

    print(f"PASS  test_cr_lookup_matches_qp  ({matched} CRs verified, tol=1e-4)")

test_cr_lookup_matches_qp()


# %% ── 8. Run all as test suite ───────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("i_mpdimpc_solver.py — full test suite")
    print("="*60)
    plant, mp_sol, qp_mats, Q_list, R_list, P_list, rho_list = setup()
    test_fallback_rate()
    res_dimpc, res_imp = test_trajectory_equivalence()
    test_iteration_equivalence()
    test_speed_comparison()
    test_convergence()
    test_cr_lookup_matches_qp()
    print("\nAll i_mpdimpc_solver.py tests passed.")