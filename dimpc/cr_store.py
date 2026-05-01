"""
cr_store.py — Critical region storage and lookup for explicit mpDiMPC.

Paper notation → PPOPT attribute → our attribute
─────────────────────────────────────────────────
Φ^v  (CR matrix)   cr.E         CriticalRegion.E     shape (n_ineq, n_theta)
φ^v  (CR vector)   cr.f         CriticalRegion.f     shape (n_ineq,)
f^v_A (affine coef)  cr.A      CriticalRegion.A     shape (n_u_full, n_theta)
f^v_b (affine offset) cr.b     CriticalRegion.b     shape (n_u_full,)

CR definition  (paper Eq. 16):   Φ^v θ_i ≤ φ^v
Affine solution (paper Eq. 16):  U_i = f^v(θ_i) = A^v θ_i + b^v

MATLAB equivalent
─────────────────
MATLAB cell array    →    Python
CRs{i}{v}           →    solutions[i][v]          (ControllerSolution.__getitem__)
CRs{i}{v}.Phi       →    solutions[i][v].E
CRs{i}{v}.phi       →    solutions[i][v].f
CRs{i}{v}.fA        →    solutions[i][v].A
CRs{i}{v}.fb        →    solutions[i][v].b
neighbors{i}{v}     →    solutions[i][v].facet_neighbors

Persistence: pickle (same as existing mp_gne_solution.pkl workflow)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import pickle
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Core data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CriticalRegion:
    """
    One critical region for a single local controller.

    The region is the polyhedron  { θ : E θ ≤ f }.
    Inside this region the optimal control sequence is  U* = A θ + b.

    Attributes
    ----------
    E : ndarray, shape (n_ineq, n_theta)
        Half-space normals.  Equivalent to Φ^v in the paper.
    f : ndarray, shape (n_ineq,)
        Half-space offsets.  Equivalent to φ^v in the paper.
        Stored as 1-D to avoid (n,1) vs (n,) bugs.
    A : ndarray, shape (n_u_full, n_theta)
        Affine solution coefficient.  U_full* = A θ + b
        n_u_full = Np * nu_i  (full horizon sequence).
    b : ndarray, shape (n_u_full,)
        Affine solution offset.
    index : int
        Position of this CR in its parent ControllerSolution (0-based).
    facet_neighbors : list[int]
        Indices of other CRs (same controller) that share a proper facet
        with this region.  Filled in by facet_finder.py offline.
    """

    E: np.ndarray
    f: np.ndarray
    A: np.ndarray
    b: np.ndarray
    index: int = 0
    facet_neighbors: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.E = np.atleast_2d(np.asarray(self.E, dtype=float))
        self.f = np.asarray(self.f, dtype=float).ravel()
        self.A = np.atleast_2d(np.asarray(self.A, dtype=float))
        self.b = np.asarray(self.b, dtype=float).ravel()

    # ── geometry ─────────────────────────────────────────────────────────────

    def contains(self, theta: np.ndarray, tol: float = 1e-8) -> bool:
        """Return True if θ ∈ CR:  E θ ≤ f + tol."""
        theta = np.asarray(theta, dtype=float).ravel()
        return bool(np.all(self.E @ theta <= self.f + tol))

    # ── solution ─────────────────────────────────────────────────────────────

    def evaluate(self, theta: np.ndarray) -> np.ndarray:
        """Return full optimal sequence  U_full* = A θ + b,  shape (n_u_full,)."""
        theta = np.asarray(theta, dtype=float).ravel()
        return self.A @ theta + self.b

    def evaluate_first(self, theta: np.ndarray, nu_i: int) -> np.ndarray:
        """Return only the first control action u_i*(k) = first nu_i elements."""
        return self.evaluate(theta)[:nu_i]

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def n_theta(self) -> int:
        return self.E.shape[1]

    @property
    def n_u_full(self) -> int:
        return self.A.shape[0]

    @property
    def n_ineq(self) -> int:
        return self.E.shape[0]


@dataclass
class ControllerSolution:
    """
    Explicit mp solution for one local controller i.

    Equivalent to the MATLAB cell CRs{i} — a list of CriticalRegion objects.

    Attributes
    ----------
    controller_index : int
        Zero-based index i of this controller.
    nu_i : int
        Dimension of u_i (single time-step input, not full horizon).
    Np : int
        Prediction horizon.
    regions : list[CriticalRegion]
        All critical regions, indexed 0..nCR-1.
        MATLAB: CRs{i}{v}  →  Python: solution[v]  or  solution.regions[v]
    """

    controller_index: int
    nu_i: int
    Np: int
    regions: list[CriticalRegion] = field(default_factory=list)

    # ── MATLAB-style indexing: solutions[i][v] ────────────────────────────────

    def __getitem__(self, v: int) -> CriticalRegion:
        return self.regions[v]

    def __len__(self) -> int:
        return len(self.regions)

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def n_cr(self) -> int:
        return len(self.regions)

    @property
    def n_u_full(self) -> int:
        return self.Np * self.nu_i

    # ── point location ───────────────────────────────────────────────────────

    def locate(self, theta: np.ndarray, tol: float = 1e-8) -> int | None:
        """
        Find which critical region contains θ.

        Returns the index v (0-based) of the first matching CR, or None
        if θ is outside all regions (infeasible or numerical edge).
        Equivalent to MATLAB's mpt3 region search.
        """
        theta = np.asarray(theta, dtype=float).ravel()
        for cr in self.regions:
            if cr.contains(theta, tol=tol):
                return cr.index
        return None

    def evaluate(self, theta: np.ndarray, tol: float = 1e-8) -> np.ndarray | None:
        """
        Locate CR and return U_full* = A^v θ + b^v.

        Returns None if θ is outside all critical regions.
        """
        v = self.locate(theta, tol=tol)
        if v is None:
            return None
        return self.regions[v].evaluate(theta)

    def evaluate_first(self, theta: np.ndarray, tol: float = 1e-8) -> np.ndarray | None:
        """Locate CR and return only the first control action u_i*(k)."""
        v = self.locate(theta, tol=tol)
        if v is None:
            return None
        return self.regions[v].evaluate_first(theta, self.nu_i)


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-controller container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MPSolutions:
    """
    Explicit mp solutions for all M local controllers.

    MATLAB equivalent:  CRs{i}{v}  →  Python: mp.solutions[i][v]

    Attributes
    ----------
    solutions : list[ControllerSolution]
        Length M.  solutions[i] holds all CRs for controller i.
    """

    solutions: list[ControllerSolution] = field(default_factory=list)

    # ── MATLAB-style indexing: mp_sol[i][v] ──────────────────────────────────

    def __getitem__(self, i: int) -> ControllerSolution:
        return self.solutions[i]

    def __len__(self) -> int:
        return len(self.solutions)

    @property
    def M(self) -> int:
        return len(self.solutions)

    def n_cr(self, i: int) -> int:
        return self.solutions[i].n_cr

    def total_cr(self) -> int:
        return sum(s.n_cr for s in self.solutions)

    def summary(self) -> str:
        lines = [f"MPSolutions: M={self.M} controllers"]
        for s in self.solutions:
            lines.append(f"  controller {s.controller_index}: {s.n_cr} CRs, "
                         f"nu_i={s.nu_i}, Np={s.Np}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  PPOPT → our format converter
# ─────────────────────────────────────────────────────────────────────────────

def from_ppopt_solution(ppopt_solution, controller_index: int,
                        nu_i: int, Np: int) -> ControllerSolution:
    """
    Convert a PPOPT Solution object to a ControllerSolution.

    PPOPT CriticalRegion fields used:
        cr.E  — shape (n_ineq, n_theta)  — CR half-space matrix
        cr.f  — shape (n_ineq, 1)        — CR half-space RHS
        cr.A  — shape (n_u_full, n_theta) — affine coeff  (U* = A θ + b)
        cr.b  — shape (n_u_full, 1)       — affine offset

    Parameters
    ----------
    ppopt_solution : ppopt.solution.Solution
        Returned by solve_mpqp().
    controller_index : int
        Zero-based index of this controller.
    nu_i : int
        Single time-step input dimension for controller i.
    Np : int
        Prediction horizon.

    Returns
    -------
    ControllerSolution
    """
    regions = []
    for v, pcr in enumerate(ppopt_solution.critical_regions):
        cr = CriticalRegion(
            E=pcr.E,
            f=pcr.f.ravel(),          # (n_ineq,1) → (n_ineq,)
            A=pcr.A,
            b=pcr.b.ravel(),          # (n_u_full,1) → (n_u_full,)
            index=v,
        )
        regions.append(cr)
    return ControllerSolution(
        controller_index=controller_index,
        nu_i=nu_i,
        Np=Np,
        regions=regions,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Persistence  (pickle — same as mp_gne_solution.pkl)
# ─────────────────────────────────────────────────────────────────────────────

def save_mp_solutions(mp_solutions: MPSolutions, path: str) -> None:
    """Pickle MPSolutions to disk."""
    with open(path, "wb") as fh:
        pickle.dump(mp_solutions, fh)
    print(f"[save] {path}  ({mp_solutions.total_cr()} total CRs across "
          f"{mp_solutions.M} controllers)")


def load_mp_solutions(path: str) -> MPSolutions:
    """Load pickled MPSolutions from disk."""
    with open(path, "rb") as fh:
        mp_solutions = pickle.load(fh)
    print(f"[load] {path}  ({mp_solutions.total_cr()} total CRs across "
          f"{mp_solutions.M} controllers)")
    return mp_solutions