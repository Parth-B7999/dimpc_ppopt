"""
plant_gen.py — Random stable controllable plant generator.

Matches the case study setup in:
  Saini, Brahmbhatt et al. (C&ChE 2025), Section 4
  Brahmbhatt et al. (ACC 2026), Section IV-A

Per-subsystem parameters (fixed):
  nx_i = 2 states,  nu_i = 1 input
  A_i  elements ~ Uniform[-1, 1],  scaled so spectral radius ≤ rho_max
  B_{i,j} elements ~ Uniform[-1, 1]  (all j, full coupling)
  State bounds: lb ∈ Uniform[-100, -10]^nx,  ub ∈ Uniform[10, 100]^nx
  Input bounds: lb ∈ Uniform[-5,  -1],        ub ∈ Uniform[1, 5]
"""

from __future__ import annotations
import numpy as np
from .plant import Subsystem, Plant


# ─────────────────────────────────────────────────────────────────────────────
#  Single random plant
# ─────────────────────────────────────────────────────────────────────────────

def make_random_plant(
    M: int,
    Np: int = 3,
    nx_i: int = 2,
    nu_i: int = 1,
    rho_max: float = 0.95,
    rng: np.random.Generator | None = None,
) -> Plant:
    """
    Generate one random stable plant with M coupled subsystems.

    A_i is rescaled so spectral radius < rho_max.  B_{i,j} is fully random
    (all subsystems coupled to all inputs), matching the paper setup.

    Parameters
    ----------
    M       : number of subsystems
    Np      : prediction horizon
    nx_i    : states per subsystem (paper: 2)
    nu_i    : inputs per subsystem (paper: 1)
    rho_max : stability threshold for each A_i
    rng     : numpy Generator  (created internally if None)
    """
    if rng is None:
        rng = np.random.default_rng()

    subsystems = []
    for i in range(M):

        # ── A_i: random, rescaled for stability ──────────────────────────────
        A_i  = rng.uniform(-1.0, 1.0, (nx_i, nx_i))
        rho  = np.max(np.abs(np.linalg.eigvals(A_i)))
        if rho >= rho_max:
            A_i *= (rho_max * 0.9) / rho     # scale to 0.9*rho_max

        # ── B_{i,j}: fully random coupling ───────────────────────────────────
        B = {j: rng.uniform(-1.0, 1.0, (nx_i, nu_i)) for j in range(M)}

        # ── Bounds (paper ranges) ─────────────────────────────────────────────
        x_lb = rng.uniform(-100.0, -10.0, nx_i)
        x_ub = rng.uniform( 10.0, 100.0, nx_i)
        u_lb = rng.uniform( -5.0,  -1.0, nu_i)
        u_ub = rng.uniform(  1.0,   5.0, nu_i)

        subsystems.append(Subsystem(
            index=i, A=A_i, B=B,
            x_lb=x_lb, x_ub=x_ub,
            u_lb=u_lb, u_ub=u_ub,
        ))

    return Plant(subsystems=subsystems, Np=Np)


# ─────────────────────────────────────────────────────────────────────────────
#  Batch generator
# ─────────────────────────────────────────────────────────────────────────────

def make_random_plants(
    M: int,
    N: int,
    Np: int = 3,
    nx_i: int = 2,
    nu_i: int = 1,
    rho_max: float = 0.95,
    seed: int = 2025,
) -> list[Plant]:
    """
    Generate N random plants for the case study.

    Parameters
    ----------
    M    : number of subsystems
    N    : number of plants  (paper: 100)
    seed : fixed seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    return [make_random_plant(M, Np, nx_i, nu_i, rho_max, rng) for _ in range(N)]


# ─────────────────────────────────────────────────────────────────────────────
#  Random initial condition (inside bounds, near origin)
# ─────────────────────────────────────────────────────────────────────────────

def make_ic(plant: Plant, scale: float = 0.1, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample a random initial state inside the state bounds.

    Uses x0 = Uniform[scale * x_lb, scale * x_ub] per subsystem so the IC
    is well inside the feasible region.  The same IC is used for all methods.
    """
    if rng is None:
        rng = np.random.default_rng()

    parts = []
    for s in plant.subsystems:
        # scale down from bounds: lb*scale..ub*scale
        lo = np.minimum(s.x_lb * scale, s.x_ub * scale)
        hi = np.maximum(s.x_lb * scale, s.x_ub * scale)
        parts.append(rng.uniform(lo, hi))
    return np.concatenate(parts)