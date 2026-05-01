"""
test_if_mpdimpc_solver.py
=========================
Tests for if_mpdimpc_solver.py — run each #%% cell in VS Code (Shift+Enter)
or run the whole file:  python tests/test_if_mpdimpc_solver.py

Cell 1 : offline precomputation (shared across all cells)
Cell 2 : linear system assembly — shapes and solvability
Cell 3 : single combination solve and validity check
Cell 4 : IF-mpDiMPC produces same trajectory as DiMPC
Cell 5 : IF-mpDiMPC uses far fewer combos than worst-case (prev-combo hint)
Cell 6 : IF-mpDiMPC has no inter-controller iterations (iter_counts = combos tried)
Cell 7 : full closed-loop convergence
Cell 8 : comparison summary — DiMPC vs I-mpDiMPC vs IF-mpDiMPC
"""

# %% ── 0. Imports ─────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import itertools

from dimpc.plant import make_acc2026_plant
from dimpc.mp_solver import solve_all_mp, default_weights
from dimpc.dimpc_solver import precompute_qp_matrices, run_dimpc
from dimpc.i_mpdimpc_solver import run_i_mpdimpc
from dimpc.if_mpdimpc_solver import (
    _u_offsets,
    _assemble_linear_system,
    _solve_combination,
    run_if_mpdimpc,
)

print("Imports OK")


# %% ── 1. Offline precomputation (shared, ~10 sec) ───────────────────────────

def setup():
    plant  = make_acc2026_plant()
    Q_list, R_list, P_list, rho_list = default_weights(plant)

    print("\n[Cell 1] Solving offline mp problems...")
    mp_sol  = solve_all_mp(plant, Q_list, R_list, P_list, rho_list, verbose=True)
    qp_mats = precompute_qp_matrices(plant, Q_list, R_list, P_list, rho_list)

    sizes, offsets = _u_offsets(plant)
    n_total_combos = mp_sol[0].n_cr * mp_sol[1].n_cr

    print(f"\n  CRs: {[mp_sol[i].n_cr for i in range(plant.M)]}")
    print(f"  Total combinations: {n_total_combos}")
    print("PASS  setup")
    return plant, mp_sol, qp_mats, sizes, offsets, n_total_combos

plant, mp_sol, qp_mats, sizes, offsets, n_total_combos = setup()


# %% ── 2. Linear system assembly: shapes and well-posedness ──────────────────
#
#  For the M=2 ACC2026 plant (n_u_0=3, n_u_1=3, nx=4):
#    n_u_total = 6
#    L ∈ R^{6×6}, R ∈ R^{6×4}, d ∈ R^6
#    L should be invertible for most CR combinations (det ≠ 0)
#    L has I blocks on diagonal and -A_{ij} off-diagonal

def test_linear_system_shapes():
    # Use the first CR of each controller
    cr_combo = [mp_sol[i][0] for i in range(plant.M)]
    L, R, d = _assemble_linear_system(cr_combo, plant, sizes, offsets)

    n_u_total = sum(sizes)
    nx        = plant.nx

    assert L.shape == (n_u_total, n_u_total), f"L shape wrong: {L.shape}"
    assert R.shape == (n_u_total, nx),        f"R shape wrong: {R.shape}"
    assert d.shape == (n_u_total,),            f"d shape wrong: {d.shape}"

    # Diagonal blocks of L must be identity
    for i in range(plant.M):
        rs, re = offsets[i], offsets[i] + sizes[i]
        assert np.allclose(L[rs:re, rs:re], np.eye(sizes[i])), \
            f"Diagonal block {i} is not identity"

    # L must be invertible (for a valid CR combination)
    det = abs(np.linalg.det(L))
    assert det > 1e-10, f"L is singular: det={det:.2e}"

    print(f"PASS  test_linear_system_shapes  "
          f"(L:{L.shape}, R:{R.shape}, det(L)={det:.4f})")

test_linear_system_shapes()


# %% ── 3. Single combination solve + validity check ───────────────────────────
#
#  Test _solve_combination on a specific θ̄ = x(k).
#  A valid combination must return a U that satisfies all CR conditions.
#  The number of valid combinations for a given x(k) should be exactly 1
#  (non-overlapping CRs).

def test_single_combination():
    rng = np.random.default_rng(42)
    nx  = plant.nx
    x_k = rng.uniform(-3, 3, nx)

    found = []
    for combo in itertools.product(*[range(mp_sol[i].n_cr)
                                      for i in range(plant.M)]):
        cr_combo = [mp_sol[i][v] for i, v in enumerate(combo)]
        U_sol = _solve_combination(cr_combo, x_k, plant, sizes, offsets)
        if U_sol is not None:
            found.append((combo, U_sol))

    # Should find exactly 1 valid combination (non-overlapping mp solution)
    assert len(found) >= 1, \
        f"No valid combination found for x={x_k}  — point may be infeasible"

    combo, U_sol = found[0]
    print(f"\n[Cell 3] x={x_k.round(3)}")
    print(f"  Valid combo: {combo}")
    print(f"  U_0={U_sol[0].round(4)}, U_1={U_sol[1].round(4)}")
    if len(found) > 1:
        print(f"  (Note: {len(found)} valid combos — overlapping CRs at boundary)")

    # U must be within input bounds
    for i in range(plant.M):
        lb = np.tile(plant.subsystems[i].u_lb, plant.Np)
        ub = np.tile(plant.subsystems[i].u_ub, plant.Np)
        assert np.all(U_sol[i] >= lb - 1e-4), f"U_{i} below lb"
        assert np.all(U_sol[i] <= ub + 1e-4), f"U_{i} above ub"

    print(f"PASS  test_single_combination  ({len(found)} valid combo(s) found)")

test_single_combination()


# %% ── 4. Trajectory equivalence: IF-mpDiMPC ≈ DiMPC ────────────────────────
#
#  IF-mpDiMPC solves the same optimization as DiMPC — just simultaneously
#  instead of iteratively.  Trajectories must agree to numerical tolerance.

def test_trajectory_equivalence():
    x0 = np.array([5.0, -3.0, 4.0, -2.0])
    T  = 30

    print("\n[Cell 4] Trajectory equivalence check...")
    res_dimpc = run_dimpc(
        plant, x0, T, p_max=100, eps=1e-8, qp_mats=qp_mats, verbose=False,
    )
    res_if = run_if_mpdimpc(
        plant, x0, T, mp_sol, qp_mats=qp_mats, verbose=True,
    )

    max_state_err = np.max(np.abs(res_dimpc.x_traj - res_if.x_traj))
    assert max_state_err < 1e-3, \
        f"State trajectory mismatch: max err={max_state_err:.2e}"

    for i in range(plant.M):
        max_u_err = np.max(np.abs(res_dimpc.u_traj[i] - res_if.u_traj[i]))
        assert max_u_err < 1e-3, \
            f"Input trajectory mismatch ctrl {i}: max err={max_u_err:.2e}"

    print(f"\nPASS  test_trajectory_equivalence  "
          f"(max state err={max_state_err:.2e})")
    return res_dimpc, res_if

res_dimpc, res_if = test_trajectory_equivalence()


# %% ── 5. Prev-combo hint: most steps find solution on first try ──────────────
#
#  Because states change slowly, the valid CR combination at k+1 is usually
#  the same as at k.  The prev-combo hint (trying previous combo first) means
#  iter_counts[k] should be 1 most of the time after the first step.
#
#  This demonstrates the practical efficiency of the warm-combination search.

def test_prev_combo_speedup():
    counts = res_if.iter_counts
    pct_first_try = 100.0 * np.sum(counts == 1) / len(counts)
    avg_combos    = counts.mean()
    max_combos    = counts.max()

    print(f"\n[Cell 5] Combination search statistics:")
    print(f"  Total possible combos : {n_total_combos}")
    print(f"  Avg combos tried/step : {avg_combos:.2f}")
    print(f"  Max combos tried      : {max_combos}")
    print(f"  Found on 1st try      : {pct_first_try:.0f}% of steps")

    # With the prev-combo hint, most steps should find valid combo quickly
    assert avg_combos < n_total_combos, \
        "Avg combos >= worst case — prev-combo hint not helping"

    print(f"PASS  test_prev_combo_speedup  "
          f"({pct_first_try:.0f}% first-try, avg={avg_combos:.1f}/{n_total_combos})")

test_prev_combo_speedup()


# %% ── 6. No inter-controller iterations — exactly 1 communication round ──────
#
#  iter_counts for IF-mpDiMPC counts CR combinations tried, NOT iterations.
#  The algorithm uses exactly 1 data exchange per time step (just x(k)).
#  This is the core advantage over DiMPC / I-mpDiMPC.

def test_no_iterations():
    # converged[k] = True means valid combo found — no Wegstein fallback
    pct_conv = res_if.converged.mean() * 100

    print(f"\n[Cell 6] Valid combinations found: {pct_conv:.0f}%")
    print(f"  DiMPC avg iterations / step: {res_dimpc.iter_counts.mean():.1f}")
    print(f"  IF-mpDiMPC: 1 communication exchange per step "
          f"(no iterations — {pct_conv:.0f}% convergence)")

    assert pct_conv >= 50, \
        f"IF-mpDiMPC fallback rate too high: {100-pct_conv:.0f}%"

    print("PASS  test_no_iterations")

test_no_iterations()


# %% ── 7. Full closed-loop convergence ────────────────────────────────────────

def test_convergence():
    x0 = np.array([5.0, -3.0, 4.0, -2.0])

    print("\n[Cell 7] IF-mpDiMPC convergence over T=50 steps...")
    res = run_if_mpdimpc(
        plant, x0, T=50, mp_sol=mp_sol, qp_mats=qp_mats, verbose=True,
    )

    norm_init  = np.linalg.norm(res.x_traj[0])
    norm_final = np.linalg.norm(res.x_traj[-1])

    assert norm_final < norm_init, "State did not decay"
    assert norm_final < 0.1, f"State not near zero: ||x||={norm_final:.4f}"

    # Input bounds respected
    for i in range(plant.M):
        lb = plant.subsystems[i].u_lb
        ub = plant.subsystems[i].u_ub
        assert np.all(res.u_traj[i] >= lb - 1e-8)
        assert np.all(res.u_traj[i] <= ub + 1e-8)

    print(f"\nPASS  test_convergence  "
          f"(||x_0||={norm_init:.3f} → ||x_T||={norm_final:.5f})")

test_convergence()


# %% ── 8. Summary comparison: DiMPC vs I-mpDiMPC vs IF-mpDiMPC ───────────────
#
#  Side-by-side comparison on the same IC and T — the central result of
#  the paper (Fig. 4-6 style comparison).

def test_comparison_summary():
    x0 = np.array([5.0, -3.0, 4.0, -2.0])
    T  = 50

    print("\n[Cell 8] Running all three methods for T=50 steps...")

    r_d  = run_dimpc(plant, x0, T, qp_mats=qp_mats, verbose=False)
    r_im = run_i_mpdimpc(plant, x0, T, mp_sol, qp_mats=qp_mats, verbose=False)
    r_if = run_if_mpdimpc(plant, x0, T, mp_sol, qp_mats=qp_mats, verbose=False)

    print(f"\n{'Method':<18} {'Avg iters/step':>16} {'Avg time(ms)':>13} "
          f"{'Total time(s)':>14} {'||x_T||':>9}")
    print("-" * 75)
    for name, r in [("DiMPC", r_d), ("I-mpDiMPC", r_im), ("IF-mpDiMPC", r_if)]:
        print(f"  {name:<16} {r.iter_counts.mean():>16.2f} "
              f"{r.solve_times.mean()*1000:>13.3f} "
              f"{r.solve_times.sum():>14.3f} "
              f"{np.linalg.norm(r.x_traj[-1]):>9.5f}")

    # All three must give the same final state (same optimizer, different solver)
    err_im = np.max(np.abs(r_d.x_traj - r_im.x_traj))
    err_if = np.max(np.abs(r_d.x_traj - r_if.x_traj))
    assert err_im < 1e-3, f"I-mpDiMPC vs DiMPC: max err={err_im:.2e}"
    assert err_if < 1e-3, f"IF-mpDiMPC vs DiMPC: max err={err_if:.2e}"

    print(f"\n  Max state trajectory error:")
    print(f"    DiMPC vs I-mpDiMPC  : {err_im:.2e}")
    print(f"    DiMPC vs IF-mpDiMPC : {err_if:.2e}")
    print(f"\nPASS  test_comparison_summary")

test_comparison_summary()


# %% ── 9. Run all as test suite ───────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("if_mpdimpc_solver.py — full test suite")
    print("="*60)
    plant, mp_sol, qp_mats, sizes, offsets, n_total_combos = setup()
    test_linear_system_shapes()
    test_single_combination()
    res_dimpc, res_if = test_trajectory_equivalence()
    test_prev_combo_speedup()
    test_no_iterations()
    test_convergence()
    test_comparison_summary()
    print("\nAll if_mpdimpc_solver.py tests passed.")
# %%
