"""
test_cr_store.py
================
Tests for cr_store.py — run each #%% cell in VS Code (Shift+Enter)
or run the whole file:  python tests/test_cr_store.py

Each cell is independent and prints PASS/FAIL.
"""

# %% ── 0. Imports ─────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pickle
import tempfile

from dimpc.cr_store import (
    CriticalRegion,
    ControllerSolution,
    MPSolutions,
    from_ppopt_solution,
    save_mp_solutions,
    load_mp_solutions,
)

print("Imports OK")


# %% ── 1. CriticalRegion: construction and shape normalisation ───────────────
#
#  PPOPT stores f as (n_ineq, 1) column vector and b as (n_u, 1).
#  Our CriticalRegion.__post_init__ must flatten these to 1-D.
#  This mirrors the MATLAB struct where phi was always a column vec.

def test_cr_construction():
    # Build a simple 2-D parameter space, 1-D input
    # CR: theta in [-5,5]^2  →  E = [I; -I], f = [5,5,5,5]
    E = np.vstack([np.eye(2), -np.eye(2)])   # (4, 2)
    f_col = np.array([[5.], [5.], [5.], [5.]])  # (4,1)  ← PPOPT column format
    A = np.array([[1.0, 0.5]])               # (1, 2)  U = A θ + b
    b_col = np.array([[0.1]])                # (1,1)   ← PPOPT column format

    cr = CriticalRegion(E=E, f=f_col, A=A, b=b_col, index=0)

    assert cr.E.shape == (4, 2),  f"E shape wrong: {cr.E.shape}"
    assert cr.f.shape == (4,),    f"f must be 1-D after __post_init__, got {cr.f.shape}"
    assert cr.A.shape == (1, 2),  f"A shape wrong: {cr.A.shape}"
    assert cr.b.shape == (1,),    f"b must be 1-D after __post_init__, got {cr.b.shape}"
    assert cr.n_theta  == 2
    assert cr.n_u_full == 1
    assert cr.n_ineq   == 4
    print("PASS  test_cr_construction")

test_cr_construction()


# %% ── 2. CriticalRegion.contains ─────────────────────────────────────────────
#
#  The CR is the box [-5,5]^2.
#  Points strictly inside must return True; points outside must return False.
#  Points on the boundary (within tol) must return True.
#
#  Paper context: at online time k, we check Φ^v θ ≤ φ^v  (Eq. 16).

def test_cr_contains():
    E = np.vstack([np.eye(2), -np.eye(2)])
    f = np.array([5., 5., 5., 5.])
    cr = CriticalRegion(E=E, f=f, A=np.eye(2), b=np.zeros(2), index=0)

    # strictly inside
    assert cr.contains(np.array([0.0, 0.0])),   "origin must be inside"
    assert cr.contains(np.array([4.9, -4.9])),  "[4.9,-4.9] must be inside"

    # on the boundary — within default tol
    assert cr.contains(np.array([5.0, 0.0])),   "boundary point must be inside (tol)"

    # outside
    assert not cr.contains(np.array([5.1, 0.0])), "[5.1,0] must be outside"
    assert not cr.contains(np.array([0.0, -6.0])), "[0,-6] must be outside"

    print("PASS  test_cr_contains")

test_cr_contains()


# %% ── 3. CriticalRegion.evaluate  ────────────────────────────────────────────
#
#  U* = A θ + b  (paper Eq. 16 affine solution).
#  evaluate_first returns only the first nu_i elements (receding horizon).

def test_cr_evaluate():
    # U = [2*theta_0 + theta_1,  theta_0 - theta_1]  → 2 outputs
    A = np.array([[2.0, 1.0],
                  [1.0, -1.0]])
    b = np.array([0.5, -0.5])
    E = np.vstack([np.eye(2), -np.eye(2)])
    f = np.ones(4) * 10.0
    cr = CriticalRegion(E=E, f=f, A=A, b=b, index=0)

    theta = np.array([3.0, 1.0])
    U_full = cr.evaluate(theta)
    expected = np.array([2*3 + 1 + 0.5, 3 - 1 - 0.5])   # [7.5, 1.5]
    assert np.allclose(U_full, expected), f"evaluate: got {U_full}, want {expected}"

    # evaluate_first with nu_i=1 returns only U_full[0]
    u_first = cr.evaluate_first(theta, nu_i=1)
    assert u_first.shape == (1,),         f"evaluate_first shape wrong: {u_first.shape}"
    assert np.isclose(u_first[0], 7.5),   f"evaluate_first value wrong: {u_first}"

    print("PASS  test_cr_evaluate")

test_cr_evaluate()


# %% ── 4. ControllerSolution: MATLAB-style indexing  ─────────────────────────
#
#  MATLAB:  CRs{i}{v}  →  Python:  solutions[i][v]
#  This cell verifies that the ControllerSolution indexing works identically.

def _make_box_cr(v: int, lo: float, hi: float) -> CriticalRegion:
    """Helper: 1-D CR  lo ≤ θ ≤ hi,  U = θ."""
    E = np.array([[1.0], [-1.0]])
    f = np.array([hi, -lo])
    A = np.array([[1.0]])
    b = np.array([0.0])
    return CriticalRegion(E=E, f=f, A=A, b=b, index=v)

def test_controller_solution_indexing():
    # Two non-overlapping CRs: CR0 = [-5, 0], CR1 = [0, 5]
    cr0 = _make_box_cr(0, lo=-5.0, hi=0.0)
    cr1 = _make_box_cr(1, lo=0.0,  hi=5.0)
    ctrl_sol = ControllerSolution(controller_index=0, nu_i=1, Np=3,
                                  regions=[cr0, cr1])

    # MATLAB CRs{0}{0} == Python solutions[0][0]
    assert ctrl_sol[0] is cr0, "ctrl_sol[0] should return cr0"
    assert ctrl_sol[1] is cr1, "ctrl_sol[1] should return cr1"
    assert len(ctrl_sol) == 2
    assert ctrl_sol.n_cr == 2
    assert ctrl_sol.n_u_full == 3 * 1   # Np * nu_i

    print("PASS  test_controller_solution_indexing")

test_controller_solution_indexing()


# %% ── 5. ControllerSolution.locate (point location)  ────────────────────────
#
#  Online at time k: given θ_i, find active CR v*  (paper Eq. 16).
#  This is the "critical region search" step that replaces solving a QP
#  in I-mpDiMPC and IF-mpDiMPC.

def test_point_location():
    cr0 = _make_box_cr(0, lo=-5.0, hi=0.0)
    cr1 = _make_box_cr(1, lo=0.0,  hi=5.0)
    ctrl_sol = ControllerSolution(controller_index=0, nu_i=1, Np=3,
                                  regions=[cr0, cr1])

    # θ = -3.0 should be in CR0
    v = ctrl_sol.locate(np.array([-3.0]))
    assert v == 0, f"θ=-3 should be in CR0, got CR{v}"

    # θ = 2.5 should be in CR1
    v = ctrl_sol.locate(np.array([2.5]))
    assert v == 1, f"θ=2.5 should be in CR1, got CR{v}"

    # θ = 10.0 is outside all CRs → None
    v = ctrl_sol.locate(np.array([10.0]))
    assert v is None, f"θ=10 should return None, got {v}"

    # boundary θ = 0.0 should land in CR0 (first match wins, boundary within tol)
    v = ctrl_sol.locate(np.array([0.0]))
    assert v in (0, 1), f"θ=0 at boundary should be in 0 or 1, got {v}"

    print("PASS  test_point_location")

test_point_location()


# %% ── 6. ControllerSolution.evaluate  ───────────────────────────────────────
#
#  Combines locate + CR.evaluate in one call.
#  This is the full mpDiMPC online step for one controller at one iteration.

def test_controller_evaluate():
    # CR0: θ in [-5, 0],  U = 2*θ + 1
    E0 = np.array([[1.0], [-1.0]])
    cr0 = CriticalRegion(E=E0, f=np.array([0.0, 5.0]),
                         A=np.array([[2.0]]), b=np.array([1.0]), index=0)

    # CR1: θ in [0, 5],   U = -θ + 3
    cr1 = CriticalRegion(E=E0, f=np.array([5.0, 0.0]),
                         A=np.array([[-1.0]]), b=np.array([3.0]), index=1)

    ctrl_sol = ControllerSolution(controller_index=1, nu_i=1, Np=3,
                                  regions=[cr0, cr1])

    # θ = -2 → CR0 → U = 2*(-2) + 1 = -3
    U = ctrl_sol.evaluate(np.array([-2.0]))
    assert U is not None and np.isclose(U[0], -3.0), f"expected -3.0, got {U}"

    # θ = 4 → CR1 → U = -4 + 3 = -1
    U = ctrl_sol.evaluate(np.array([4.0]))
    assert U is not None and np.isclose(U[0], -1.0), f"expected -1.0, got {U}"

    # θ outside → None
    U = ctrl_sol.evaluate(np.array([99.0]))
    assert U is None, "out-of-range θ should return None"

    print("PASS  test_controller_evaluate")

test_controller_evaluate()


# %% ── 7. MPSolutions container  ──────────────────────────────────────────────
#
#  MATLAB: CRs{i}{v}  →  Python: mp_sol[i][v]
#  This is the top-level object saved/loaded from disk.

def test_mp_solutions_container():
    cr0 = _make_box_cr(0, lo=-5.0, hi=0.0)
    cr1 = _make_box_cr(1, lo=0.0,  hi=5.0)
    ctrl0 = ControllerSolution(controller_index=0, nu_i=1, Np=3, regions=[cr0, cr1])

    cr2 = _make_box_cr(0, lo=-10.0, hi=5.0)
    ctrl1 = ControllerSolution(controller_index=1, nu_i=1, Np=3, regions=[cr2])

    mp = MPSolutions(solutions=[ctrl0, ctrl1])

    assert mp.M == 2
    assert mp.n_cr(0) == 2
    assert mp.n_cr(1) == 1
    assert mp.total_cr() == 3

    # MATLAB-style 2-level indexing: mp[i][v]
    assert mp[0][0] is cr0,  "mp[0][0] should be cr0"
    assert mp[0][1] is cr1,  "mp[0][1] should be cr1"
    assert mp[1][0] is cr2,  "mp[1][0] should be cr2"

    print(mp.summary())
    print("PASS  test_mp_solutions_container")

test_mp_solutions_container()


# %% ── 8. Facet neighbor storage  ─────────────────────────────────────────────
#
#  FACET-DiMPC stores which CRs share a proper facet (offline precomputation).
#  MATLAB: neighbors{i}{v} = [list of CR indices sharing a facet with CR v]
#  Python: cr.facet_neighbors = [list of int indices]
#
#  This cell verifies the storage works — facet_finder.py will fill these.

def test_facet_neighbor_storage():
    cr0 = _make_box_cr(0, lo=-5.0, hi=0.0)
    cr1 = _make_box_cr(1, lo=0.0,  hi=5.0)
    cr2 = _make_box_cr(2, lo=5.0,  hi=10.0)

    # Suppose facet_finder found: CR0 ↔ CR1, CR1 ↔ CR2
    cr0.facet_neighbors = [1]
    cr1.facet_neighbors = [0, 2]
    cr2.facet_neighbors = [1]

    ctrl_sol = ControllerSolution(controller_index=0, nu_i=1, Np=3,
                                  regions=[cr0, cr1, cr2])

    # At time k, current CR is 1; FACET-DiMPC only searches {0, 1, 2}
    current_cr_idx = 1
    search_set = set([current_cr_idx] + ctrl_sol[current_cr_idx].facet_neighbors)
    assert search_set == {0, 1, 2}, f"search set wrong: {search_set}"

    # At time k, current CR is 0; only search {0, 1}
    current_cr_idx = 0
    search_set = set([current_cr_idx] + ctrl_sol[current_cr_idx].facet_neighbors)
    assert search_set == {0, 1}, f"search set for CR0 wrong: {search_set}"

    print("PASS  test_facet_neighbor_storage")

test_facet_neighbor_storage()


# %% ── 9. Save and load (pickle round-trip)  ──────────────────────────────────
#
#  Equivalent to MATLAB's save('solutions.mat', 'CRs', 'neighbors').
#  We use pickle exactly like the existing mp_gne_solution.pkl workflow.

def test_save_load():
    cr0 = _make_box_cr(0, lo=-5.0, hi=0.0)
    cr0.facet_neighbors = [1]
    cr1 = _make_box_cr(1, lo=0.0, hi=5.0)
    cr1.facet_neighbors = [0]
    ctrl = ControllerSolution(controller_index=0, nu_i=1, Np=3, regions=[cr0, cr1])
    mp_orig = MPSolutions(solutions=[ctrl])

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        path = tmp.name

    save_mp_solutions(mp_orig, path)
    mp_loaded = load_mp_solutions(path)
    os.unlink(path)

    assert mp_loaded.M == 1
    assert mp_loaded.total_cr() == 2
    assert np.allclose(mp_loaded[0][0].E, cr0.E), "E matrix not preserved"
    assert np.allclose(mp_loaded[0][0].f, cr0.f), "f vector not preserved"
    assert np.allclose(mp_loaded[0][1].A, cr1.A), "A matrix not preserved"
    assert mp_loaded[0][0].facet_neighbors == [1],  "neighbors not preserved"
    assert mp_loaded[0][1].facet_neighbors == [0],  "neighbors not preserved"

    print("PASS  test_save_load")

test_save_load()


# %% ── 10. from_ppopt_solution: real PPOPT conversion  ────────────────────────
#
#  Convert the existing mp_gne_solution.pkl (from your GNE_PPOPT project)
#  into a ControllerSolution to verify the PPOPT → cr_store bridge.
#  This is a live test using your actual PPOPT data.
#
#  GNE case: theta = x (4-D state), n_u_full = M*Np*nu = 2*3*2 = 12
#  For DiMPC case later: theta_i = [x, U_{j≠i}], n_u_full = Np * nu_i

def test_from_ppopt_solution():
    pkl_path = os.path.join(
        os.path.dirname(__file__),
        "../../GNE_PPOPT/mp_gne_solution.pkl"
    )
    if not os.path.exists(pkl_path):
        print("SKIP  test_from_ppopt_solution  (mp_gne_solution.pkl not found)")
        return

    with open(pkl_path, "rb") as fh:
        ppopt_sol = pickle.load(fh)

    n_cr_ppopt = len(ppopt_sol.critical_regions)
    pcr0 = ppopt_sol.critical_regions[0]
    n_theta = pcr0.E.shape[1]
    n_u_full = pcr0.A.shape[0]
    nu_i  = 2     # 2 inputs per agent (GNE case)
    Np    = 3     # prediction horizon

    ctrl_sol = from_ppopt_solution(ppopt_sol,
                                   controller_index=0,
                                   nu_i=nu_i,
                                   Np=Np)

    # Structural checks
    assert ctrl_sol.n_cr == n_cr_ppopt, \
        f"CR count mismatch: {ctrl_sol.n_cr} vs {n_cr_ppopt}"
    assert ctrl_sol[0].n_theta == n_theta, \
        f"n_theta mismatch: {ctrl_sol[0].n_theta} vs {n_theta}"
    assert ctrl_sol[0].n_u_full == n_u_full, \
        f"n_u_full mismatch: {ctrl_sol[0].n_u_full} vs {n_u_full}"

    # f and b must be 1-D (not (n,1) PPOPT column format)
    assert ctrl_sol[0].f.ndim == 1, \
        f"f should be 1-D, got shape {ctrl_sol[0].f.shape}"
    assert ctrl_sol[0].b.ndim == 1, \
        f"b should be 1-D, got shape {ctrl_sol[0].b.shape}"

    # Evaluate at a zero parameter vector — should not crash
    theta_test = np.zeros(n_theta)
    U = ctrl_sol.evaluate(theta_test)
    # U could be None if zero is outside all CRs; just check type
    assert U is None or U.shape == (n_u_full,), \
        f"evaluate shape wrong: {U.shape if U is not None else 'None'}"

    print(f"PASS  test_from_ppopt_solution "
          f"({n_cr_ppopt} CRs, n_theta={n_theta}, n_u_full={n_u_full})")

test_from_ppopt_solution()


# %% ── 11. Consistency: our evaluate vs PPOPT solution.evaluate  ──────────────
#
#  For the GNE solution, our ControllerSolution.evaluate(theta) must give the
#  same answer as ppopt_solution.evaluate(theta) for any theta inside the CRs.

def test_ppopt_evaluate_consistency():
    pkl_path = os.path.join(
        os.path.dirname(__file__),
        "../../GNE_PPOPT/mp_gne_solution.pkl"
    )
    if not os.path.exists(pkl_path):
        print("SKIP  test_ppopt_evaluate_consistency  (pkl not found)")
        return

    with open(pkl_path, "rb") as fh:
        ppopt_sol = pickle.load(fh)

    ctrl_sol = from_ppopt_solution(ppopt_sol, controller_index=0, nu_i=2, Np=3)

    # PPOPT's cr.evaluate(theta) requires theta as a column vector (n_theta, 1).
    # Passing a flat (n_theta,) array causes a shape-broadcast bug in PPOPT.
    # Our CriticalRegion.evaluate(theta) works correctly with flat arrays.
    # We compare at the individual CR level: pcr.A @ theta + pcr.b  vs  ours.

    matched = 0
    for v, pcr in enumerate(ppopt_sol.critical_regions):
        # Pick a point strictly inside this CR by using the Chebyshev centre
        # approximation: take mean of a few constraint rows' RHS projections.
        E = pcr.E                    # (n_ineq, n_theta)
        f = pcr.f.ravel()            # (n_ineq,)
        # Use a simple interior point: scale down the centroid direction
        theta = np.linalg.lstsq(E, f * 0.5, rcond=None)[0]

        # Verify it is actually inside this CR
        if not np.all(E @ theta <= f + 1e-6):
            continue  # couldn't find interior point, skip this CR

        # PPOPT primal: cr.A @ theta + cr.b  (direct matrix formula)
        U_ppopt_direct = pcr.A @ theta + pcr.b.ravel()

        # Our evaluate
        U_ours = ctrl_sol[v].evaluate(theta)

        assert np.allclose(U_ppopt_direct, U_ours, atol=1e-10), \
            (f"CR {v}: mismatch\n"
             f"  PPOPT direct = {U_ppopt_direct[:4]}\n"
             f"  ours         = {U_ours[:4]}")
        matched += 1

    assert matched >= 1, "No CRs could be tested"
    print(f"PASS  test_ppopt_evaluate_consistency  ({matched}/{len(ppopt_sol.critical_regions)} CRs verified)")

test_ppopt_evaluate_consistency()


# %% ── 12. Run all as a test suite  ───────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("cr_store.py — full test suite")
    print("="*60)
    test_cr_construction()
    test_cr_contains()
    test_cr_evaluate()
    test_controller_solution_indexing()
    test_point_location()
    test_controller_evaluate()
    test_mp_solutions_container()
    test_facet_neighbor_storage()
    test_save_load()
    test_from_ppopt_solution()
    test_ppopt_evaluate_consistency()
    print("\nAll cr_store.py tests passed.")
# %%
