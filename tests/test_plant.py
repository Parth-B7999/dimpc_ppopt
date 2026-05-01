"""
Tests for plant.py.

Run with:  python -m pytest tests/test_plant.py -v
       or: python tests/test_plant.py   (no pytest needed)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dimpc.plant import Subsystem, Plant, make_acc2026_plant


# ── helpers ─────────────────────────────────────────────────────────────────

def assert_shape(arr, expected, name="array"):
    assert arr.shape == expected, f"{name}: expected shape {expected}, got {arr.shape}"


# ── Test 1: Subsystem basic properties ──────────────────────────────────────

def test_subsystem_dimensions():
    plant = make_acc2026_plant()
    s0, s1 = plant.subsystems

    assert s0.nx == 2,  f"sub0 nx should be 2, got {s0.nx}"
    assert s0.nu == 1,  f"sub0 nu should be 1, got {s0.nu}"
    assert s1.nx == 2,  f"sub1 nx should be 2, got {s1.nx}"
    assert s1.nu == 1,  f"sub1 nu should be 1, got {s1.nu}"
    print("PASS  test_subsystem_dimensions")


# ── Test 2: Plant global matrices ───────────────────────────────────────────

def test_plant_global_matrices():
    plant = make_acc2026_plant()

    # Global state matrix should be block-diagonal 4x4
    A = plant.A
    assert_shape(A, (4, 4), "plant.A")

    # Off-diagonal blocks must be zero (coupling only through inputs, not states)
    s0, s1 = plant.subsystems
    assert np.allclose(A[:2, 2:], 0), "A top-right block should be 0"
    assert np.allclose(A[2:, :2], 0), "A bottom-left block should be 0"

    # Diagonal blocks must match A_i
    assert np.allclose(A[:2, :2], s0.A), "A[:2,:2] should equal A1"
    assert np.allclose(A[2:, 2:], s1.A), "A[2:,2:] should equal A2"

    # B_j(0) should be [B_{1,0}; B_{2,0}] = [B11; B21], shape (4,1)
    B0 = plant.B_j(0)
    assert_shape(B0, (4, 1), "B_j(0)")
    assert np.allclose(B0[:2], s0.B[0]), "B_j(0) top half should be B11"
    assert np.allclose(B0[2:], s1.B[0]), "B_j(0) bottom half should be B21"

    print("PASS  test_plant_global_matrices")


# ── Test 3: One-step simulation consistency ──────────────────────────────────

def test_simulation_consistency():
    """
    Global plant.step must match manually assembling x_next from subsystems.
    """
    plant = make_acc2026_plant()
    rng = np.random.default_rng(42)

    x1 = rng.uniform(-5, 5, 2)
    x2 = rng.uniform(-5, 5, 2)
    u = {0: np.array([0.5]), 1: np.array([-0.3])}

    x_global = np.concatenate([x1, x2])
    x_next_global = plant.step(x_global, u)

    # Manually compute from subsystem dynamics
    x1_next = plant.subsystems[0].step(x1, u)
    x2_next = plant.subsystems[1].step(x2, u)
    x_next_manual = np.concatenate([x1_next, x2_next])

    assert np.allclose(x_next_global, x_next_manual, atol=1e-12), (
        f"Mismatch:\n  global: {x_next_global}\n  manual: {x_next_manual}")
    print("PASS  test_simulation_consistency")


# ── Test 4: state_slice utility ──────────────────────────────────────────────

def test_state_slice():
    plant = make_acc2026_plant()
    rng = np.random.default_rng(7)
    x = rng.uniform(-10, 10, plant.nx)

    for i, s in enumerate(plant.subsystems):
        xi_extracted = x[plant.state_slice(i)]
        assert len(xi_extracted) == s.nx, f"Slice for sub {i} has wrong length"

    # Slices must cover all states without overlap
    all_indices = []
    for i in range(plant.M):
        sl = plant.state_slice(i)
        all_indices.extend(range(sl.start, sl.stop))
    assert sorted(all_indices) == list(range(plant.nx)), "Slices don't cover all states"
    print("PASS  test_state_slice")


# ── Test 5: Plant totals ──────────────────────────────────────────────────────

def test_plant_totals():
    plant = make_acc2026_plant()
    assert plant.M == 2
    assert plant.nx == 4   # 2 + 2
    assert plant.nu == 2   # 1 + 1
    assert plant.Np == 3
    print("PASS  test_plant_totals")


# ── run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_subsystem_dimensions()
    test_plant_global_matrices()
    test_simulation_consistency()
    test_state_slice()
    test_plant_totals()
    print("\nAll plant.py tests passed.")
