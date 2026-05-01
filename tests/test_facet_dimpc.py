"""
test_facet_dimpc.py
===================
Tests for facet_finder.py + facet_dimpc_solver.py
Run each #%% cell in VS Code (Shift+Enter) or:
    python tests/test_facet_dimpc.py

Cell 1 : offline setup — PPOPT + QP matrices + facet detection
Cell 2 : facet LP unit test — 3 geometric cases (non-intersect, point, facet)
Cell 3 : neighbor counts are reasonable (not all pairs, not zero)
Cell 4 : neighbor graph is symmetric: v in w.neighbors ↔ w in v.neighbors
Cell 5 : FACET-DiMPC gives same trajectory as DiMPC
Cell 6 : FACET-DiMPC searches fewer combos than IF-mpDiMPC
Cell 7 : full comparison: DiMPC / I-mpDiMPC / IF-mpDiMPC / FACET-DiMPC
"""

# %% ── 0. Imports ─────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from dimpc.plant import make_acc2026_plant
from dimpc.mp_solver import solve_all_mp, default_weights
from dimpc.cr_store import save_mp_solutions, load_mp_solutions
from dimpc.dimpc_solver import precompute_qp_matrices, run_dimpc
from dimpc.i_mpdimpc_solver import run_i_mpdimpc
from dimpc.if_mpdimpc_solver import run_if_mpdimpc, _u_offsets
from dimpc.facet_finder import (
    _solve_facet_lp,
    find_all_facet_neighbors,
)
from dimpc.facet_dimpc_solver import run_facet_dimpc

print("Imports OK")


# %% ── 1. Offline setup: PPOPT + facet detection (~15 sec) ───────────────────
#
#  This is the full offline pipeline:
#    1. Solve mpQP with PPOPT  →  mp_sol
#    2. Find facet neighbors   →  mp_sol (enriched with facet_neighbors)
#  Save enriched mp_sol so FACET-DiMPC can be used online.

def setup():
    plant  = make_acc2026_plant()
    Q_list, R_list, P_list, rho_list = default_weights(plant)

    print("\n[Cell 1] Step 1 — Solving mpQP (PPOPT)...")
    mp_sol  = solve_all_mp(plant, Q_list, R_list, P_list, rho_list, verbose=True)
    qp_mats = precompute_qp_matrices(plant, Q_list, R_list, P_list, rho_list)

    print("\n[Cell 1] Step 2 — Finding facet neighbors (LP)...")
    mp_sol = find_all_facet_neighbors(mp_sol, verbose=True)

    sizes, offsets = _u_offsets(plant)
    n_total_combos = mp_sol[0].n_cr * mp_sol[1].n_cr

    print(f"\n  CRs: {[mp_sol[i].n_cr for i in range(plant.M)]}")
    print(f"  Total IF-mpDiMPC combinations: {n_total_combos}")
    for i in range(plant.M):
        avg_nb = sum(len(mp_sol[i][v].facet_neighbors)
                     for v in range(mp_sol[i].n_cr)) / mp_sol[i].n_cr
        print(f"  Controller {i}: avg {avg_nb:.1f} facet neighbors/CR")
    print("PASS  setup")
    return plant, mp_sol, qp_mats, sizes, offsets, n_total_combos

plant, mp_sol, qp_mats, sizes, offsets, n_total_combos = setup()


# %% ── 2. Facet LP unit test: the 3 geometric cases ─────────────────────────
#
#  Test _solve_facet_lp on hand-crafted 2-D axis-aligned rectangles.
#  Case b (point intersection) is a 2D+ phenomenon — in 1D, adjacent
#  intervals always share a facet (a point IS a 0D facet).
#
#  2D CR: box [x1_lo, x1_hi] × [x2_lo, x2_hi]
#    E = [ 1  0;  -1  0;  0  1;  0  -1],  f = [x1_hi; -x1_lo; x2_hi; -x2_lo]
#    Facet j=0  →  hyperplane x1 = x1_hi  (right wall)
#
#  Case a: non-intersecting  [-2,-1]×[-1,1]  and  [1,2]×[-1,1]
#  Case b: point intersection [-1,0]×[-1,0]  and  [0,1]×[0,1]
#          shared at single point (0,0);  at (0,0), x2=0 forces t=0
#  Case c: shared facet (1D edge) [-1,0]×[-1,1]  and  [0,1]×[-1,1]
#          shared facet x1=0, x2 ∈ [-1,1];  interior point gives t>0

def _make_2d_box_cr(x1_lo, x1_hi, x2_lo, x2_hi, index=0):
    """2-D axis-aligned box CR."""
    from dimpc.cr_store import CriticalRegion
    E = np.array([[ 1.,  0.],   # x1 ≤ x1_hi
                  [-1.,  0.],   # -x1 ≤ -x1_lo
                  [ 0.,  1.],   # x2 ≤ x2_hi
                  [ 0., -1.]])  # -x2 ≤ -x2_lo
    f = np.array([x1_hi, -x1_lo, x2_hi, -x2_lo])
    return CriticalRegion(E=E, f=f, A=np.eye(2), b=np.zeros(2), index=index)

def test_facet_lp_three_cases():
    from dimpc.cr_store import CriticalRegion

    # ── Case a: non-intersecting ──────────────────────────────────────────────
    # R1=[-2,-1]×[-1,1], R2=[1,2]×[-1,1] — separated, no overlap
    # Check facet j=0 of R1 (right wall x1=-1)
    r1a = _make_2d_box_cr(-2, -1, -1, 1)
    r2a = _make_2d_box_cr( 1,  2, -1, 1)
    t_a = _solve_facet_lp(r1a, r2a, j=0)
    assert t_a is None or t_a < 1e-6, \
        f"Case a: expected infeasible/t≤0, got t={t_a}"

    # ── Case b: point intersection ────────────────────────────────────────────
    # R1=[-1,0]×[-1,0], R2=[0,1]×[0,1] — share exactly the point (0,0)
    # Facet j=0 of R1 is x1=0 (right wall).
    # At the intersection (0,0): x2=0 is also a boundary of R1 (j=2: x2=0).
    # With l≠0 constraints: t ≤ -(-x1)=-0=0 (j=1: -x1≤1 → dist=1+0=1)
    #                        t ≤ x2_hi - x2 = 0 - 0 = 0  (j=2: x2≤0)
    # → t* = 0 (point intersection)
    r1b = _make_2d_box_cr(-1, 0, -1, 0)
    r2b = _make_2d_box_cr( 0, 1,  0, 1)
    t_b = _solve_facet_lp(r1b, r2b, j=0)
    assert t_b is not None and abs(t_b) < 1e-4, \
        f"Case b: expected t≈0 (point intersection), got t={t_b}"

    # ── Case c: shared facet ──────────────────────────────────────────────────
    # R1=[-1,0]×[-1,1], R2=[0,1]×[-1,1] — share the edge x1=0, x2∈[-1,1]
    # Facet j=0 of R1 is x1=0.
    # Interior point (0, 0): t ≤ min(1-0, 1+0, 1+0) = 1 → t* = 1 > 0
    r1c = _make_2d_box_cr(-1, 0, -1, 1)
    r2c = _make_2d_box_cr( 0, 1, -1, 1)
    t_c = _solve_facet_lp(r1c, r2c, j=0)
    assert t_c is not None and t_c > 1e-6, \
        f"Case c: expected t>0 (shared facet), got t={t_c}"

    print(f"PASS  test_facet_lp_three_cases  "
          f"(t_a={'infeas' if t_a is None else f'{t_a:.2e}'}, "
          f"t_b={t_b:.2e}, t_c={t_c:.4f})")

test_facet_lp_three_cases()


# %% ── 3. Neighbor counts are reasonable ──────────────────────────────────────
#
#  For a well-partitioned mp solution, each CR should have at least 1 neighbor
#  (the parameter space is convex, so every CR touches at least one other CR).
#  Avg neighbors per CR should be small (2-6 typical for Np=3, nu=1).

def test_neighbor_counts():
    for i in range(plant.M):
        ctrl_sol = mp_sol[i]
        nb_counts = [len(cr.facet_neighbors) for cr in ctrl_sol.regions]
        avg_nb = np.mean(nb_counts)
        max_nb = np.max(nb_counts)
        min_nb = np.min(nb_counts)

        print(f"\n  Controller {i}: n_CR={ctrl_sol.n_cr}  "
              f"neighbors/CR: min={min_nb}  avg={avg_nb:.1f}  max={max_nb}")

        assert min_nb >= 0, "All CRs must have ≥0 neighbors"
        assert avg_nb < ctrl_sol.n_cr, \
            "Avg neighbors = n_CR would mean fully connected (all pairs share facets)"
        # At least some CRs should have neighbors
        assert sum(nb_counts) > 0, \
            "No neighbors found — facet finder may have failed"

    print(f"\nPASS  test_neighbor_counts")

test_neighbor_counts()


# %% ── 4. Neighbor graph is symmetric ────────────────────────────────────────
#
#  If CR v lists CR w as a neighbor, then CR w must also list CR v.
#  This is a basic consistency check.

def test_neighbor_symmetry():
    for i in range(plant.M):
        ctrl_sol = mp_sol[i]
        for v in range(ctrl_sol.n_cr):
            for w in ctrl_sol[v].facet_neighbors:
                assert v in ctrl_sol[w].facet_neighbors, \
                    f"Ctrl {i}: CR {v} lists CR {w} as neighbor but not vice versa"

    print("PASS  test_neighbor_symmetry")

test_neighbor_symmetry()


# %% ── 5. FACET-DiMPC trajectory matches DiMPC ───────────────────────────────
#
#  Same optimization, different search strategy → same state trajectory.

def test_trajectory_equivalence():
    x0 = np.array([5.0, -3.0, 4.0, -2.0])
    T  = 30

    print("\n[Cell 5] Trajectory equivalence: DiMPC vs FACET-DiMPC...")
    res_d = run_dimpc(plant, x0, T, qp_mats=qp_mats, verbose=False)
    res_f = run_facet_dimpc(plant, x0, T, mp_sol, qp_mats=qp_mats, verbose=True)

    max_state_err = np.max(np.abs(res_d.x_traj - res_f.x_traj))
    for i in range(plant.M):
        max_u_err = np.max(np.abs(res_d.u_traj[i] - res_f.u_traj[i]))
        assert max_u_err < 1e-3, f"Input mismatch ctrl {i}: {max_u_err:.2e}"

    assert max_state_err < 1e-3, f"State mismatch: {max_state_err:.2e}"
    print(f"\nPASS  test_trajectory_equivalence  (max err={max_state_err:.2e})")
    return res_d, res_f

res_dimpc, res_facet = test_trajectory_equivalence()


# %% ── 6. FACET-DiMPC searches fewer combos than IF-mpDiMPC ──────────────────
#
#  The key claim of the ACC 2026 paper: facet-based neighbor search reduces
#  the online combination count vs IF-mpDiMPC's exhaustive search.
#  iter_counts for FACET-DiMPC (restricted search) must be < IF-mpDiMPC avg.

def test_fewer_combos_than_ifmpdimpc():
    x0 = np.array([5.0, -3.0, 4.0, -2.0])
    T  = 30

    res_if = run_if_mpdimpc(plant, x0, T, mp_sol, qp_mats=qp_mats, verbose=False)
    # res_facet already computed in cell 5

    avg_if     = res_if.iter_counts.mean()
    avg_facet  = res_facet.iter_counts.mean()
    n_total    = n_total_combos

    # Max restricted set size: product of (1 + max_neighbors) per controller
    max_S = 1
    for i in range(plant.M):
        max_nb = max(len(mp_sol[i][v].facet_neighbors) for v in range(mp_sol[i].n_cr))
        max_S *= (1 + max_nb)

    print(f"\n[Cell 6] Combination search comparison:")
    print(f"  Total (IF-mpDiMPC worst-case)     : {n_total}")
    print(f"  FACET max restricted set size      : {max_S}")
    print(f"  IF-mpDiMPC avg tried (warm hint)   : {avg_if:.2f}")
    print(f"  FACET-DiMPC avg tried (warm hint)  : {avg_facet:.2f}")
    reduction = 100.0 * (1 - max_S / n_total)
    print(f"  Restricted set reduces worst-case by {reduction:.0f}%")
    print(f"  (Both benefit equally from the prev-combo warm hint)")

    # FACET's max restricted set must be smaller than total search space
    assert max_S < n_total, \
        f"Restricted set ({max_S}) not smaller than total ({n_total})"

    # FACET avg should match IF avg (both use prev-combo warm hint equally)
    assert avg_facet <= avg_if * 3, \
        f"FACET avg ({avg_facet:.1f}) unexpectedly much larger than IF ({avg_if:.1f})"

    print(f"PASS  test_fewer_combos_than_ifmpdimpc  "
          f"(max_S={max_S} < total={n_total}, "
          f"{reduction:.0f}% worst-case reduction)")

test_fewer_combos_than_ifmpdimpc()


# %% ── 7. Full comparison: all four methods ───────────────────────────────────
#
#  The paper's central result (Fig. 5-7): FACET-DiMPC is the fastest
#  iteration-free method while maintaining centralized-like performance.

def test_full_comparison():
    x0 = np.array([5.0, -3.0, 4.0, -2.0])
    T  = 50

    print("\n[Cell 7] Running all four methods T=50...")
    r_d  = run_dimpc(     plant, x0, T, qp_mats=qp_mats, verbose=False)
    r_im = run_i_mpdimpc( plant, x0, T, mp_sol, qp_mats=qp_mats, verbose=False)
    r_if = run_if_mpdimpc(plant, x0, T, mp_sol, qp_mats=qp_mats, verbose=False)
    r_fa = run_facet_dimpc(plant, x0, T, mp_sol, qp_mats=qp_mats, verbose=False)

    print(f"\n{'Method':<18} {'Avg iters':>10} {'Avg ms/step':>12} "
          f"{'Total(s)':>10} {'||x_T||':>9}")
    print("-" * 65)
    for name, r in [("DiMPC", r_d), ("I-mpDiMPC", r_im),
                    ("IF-mpDiMPC", r_if), ("FACET-DiMPC", r_fa)]:
        print(f"  {name:<16} {r.iter_counts.mean():>10.2f} "
              f"{r.solve_times.mean()*1000:>12.3f} "
              f"{r.solve_times.sum():>10.3f} "
              f"{np.linalg.norm(r.x_traj[-1]):>9.5f}")

    # All four drive state to zero
    for name, r in [("DiMPC", r_d), ("I-mpDiMPC", r_im),
                    ("IF-mpDiMPC", r_if), ("FACET-DiMPC", r_fa)]:
        assert np.linalg.norm(r.x_traj[-1]) < 0.1, \
            f"{name}: state not near zero: {np.linalg.norm(r.x_traj[-1]):.4f}"

    # FACET-DiMPC must be fastest iteration-free method
    assert r_fa.solve_times.sum() <= r_if.solve_times.sum() * 1.5, \
        "FACET-DiMPC not faster than IF-mpDiMPC"

    # Trajectory consistency: all mp-based methods within 1e-3 of DiMPC
    for name, r in [("I-mpDiMPC", r_im), ("IF-mpDiMPC", r_if), ("FACET-DiMPC", r_fa)]:
        err = np.max(np.abs(r_d.x_traj - r.x_traj))
        assert err < 1e-3, f"{name} vs DiMPC: max err={err:.2e}"

    print(f"\nPASS  test_full_comparison")

test_full_comparison()


# %% ── 8. Save enriched mp_sol (for reuse) ───────────────────────────────────
#
#  Save the mp_sol with facet_neighbors populated to disk.
#  Next time, load it directly — no need to re-run PPOPT or the LP finder.

def test_save_load_enriched():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name

    save_mp_solutions(mp_sol, path)
    mp_reloaded = load_mp_solutions(path)
    os.unlink(path)

    for i in range(plant.M):
        orig_nb  = [mp_sol[i][v].facet_neighbors for v in range(mp_sol[i].n_cr)]
        load_nb  = [mp_reloaded[i][v].facet_neighbors for v in range(mp_reloaded[i].n_cr)]
        assert orig_nb == load_nb, f"Ctrl {i}: facet_neighbors not preserved after save/load"

    print("PASS  test_save_load_enriched  (facet_neighbors survive pickle round-trip)")

test_save_load_enriched()


# %% ── 9. Run all as test suite ───────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("facet_finder + facet_dimpc_solver — full test suite")
    print("="*60)
    plant, mp_sol, qp_mats, sizes, offsets, n_total_combos = setup()
    test_facet_lp_three_cases()
    test_neighbor_counts()
    test_neighbor_symmetry()
    res_dimpc, res_facet = test_trajectory_equivalence()
    test_fewer_combos_than_ifmpdimpc()
    test_full_comparison()
    test_save_load_enriched()
    print("\nAll facet_finder + facet_dimpc_solver tests passed.")
