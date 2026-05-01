"""
test_case_study.py
==================
Reproduces the case study from:
  Saini, Brahmbhatt et al. (C&ChE 2025), Sections 4-5
  Brahmbhatt et al. (ACC 2026), Section IV

Compares DiMPC / I-mpDiMPC / IF-mpDiMPC / FACET-DiMPC across
M ∈ {2, 3, 4} subsystems on N_PLANTS random plants each.

Run in VS Code cell-by-cell (#%%) or:
    python tests/test_case_study.py

OFFLINE COMPUTATION NOTES
--------------------------
  M=2 : ~1-30 min offline per plant (depends on CR count)
  M=3 : ~10-60 min offline per plant (facet detection dominates)
  M=4 : ~several hours offline per plant

Checkpoint files are saved after every plant so long runs survive
interruptions.  Re-running the script loads existing checkpoints.

USER CONFIG — edit the block below, then run.
"""

# %% ── 0. Config ──────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import pickle
import traceback
import numpy as np
from ppopt.mp_solvers.solve_mpqp import mpqp_algorithm

# ═══════════════════════════════════════════════════════════════════════════════
N_PLANTS  = 5          # paper uses 100  →  increase for full reproduction
T_SIM     = 30         # paper uses 100  →  increase for full paper sim
M_LIST    = [2, 3, 4]  # subsystem counts to run
ALGO      = mpqp_algorithm.geometric_parallel_exp   # as requested

# p_max per M (paper uses 100 for all, but M≥3 often needs more)
P_MAX = {2: 100, 3: 200, 4: 300}

# facet finding method: "hyperplane" (fast, LI-mpDiMPC V2 style) or "lp" (rigorous, Eq 14)
FACET_METHOD = "hyperplane"

SEED      = 2025       # reproducibility
CKPT_DIR  = os.path.join(os.path.dirname(__file__), "checkpoints")
# ═══════════════════════════════════════════════════════════════════════════════

os.makedirs(CKPT_DIR, exist_ok=True)

from dimpc.plant_gen       import make_random_plants, make_ic
from dimpc.mp_solver        import solve_all_mp, default_weights
from dimpc.dimpc_solver     import precompute_qp_matrices, run_dimpc
from dimpc.i_mpdimpc_solver import run_i_mpdimpc
from dimpc.if_mpdimpc_solver import run_if_mpdimpc
from dimpc.facet_finder     import find_all_facet_neighbors
from dimpc.facet_dimpc_solver import run_facet_dimpc

print(f"Config: N_PLANTS={N_PLANTS}, T_SIM={T_SIM}, M_LIST={M_LIST}")
print(f"Algorithm: {ALGO}")
print(f"Checkpoints → {CKPT_DIR}\n")


# %% ── 1. Per-plant result dataclass ─────────────────────────────────────────

def _ckpt_path(M, plant_idx):
    return os.path.join(CKPT_DIR, f"M{M}_plant{plant_idx:03d}.pkl")


def _save_plant_result(M, plant_idx, data: dict):
    path = _ckpt_path(M, plant_idx)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _load_plant_result(M, plant_idx) -> dict | None:
    path = _ckpt_path(M, plant_idx)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# %% ── 2. Run one plant through all methods ───────────────────────────────────

def run_one_plant(plant, x0, M, plant_idx, p_max):
    """
    Run offline solve + all 4 online methods on one plant.
    Returns a results dict.  Wraps each step in try/except for robustness.
    """
    result = {
        "M": M, "plant_idx": plant_idx,
        "nx": plant.nx, "nu": plant.nu, "Np": plant.Np,
        "n_crs": None, "n_combos": None,
        "offline_time": None,
        "methods": {},
        "error": None,
    }

    # ── offline ──────────────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        Q_list, R_list, P_list, rho_list = default_weights(plant)

        print(f"    [offline] mp solve ({ALGO})...", flush=True)
        mp_sol = solve_all_mp(
            plant, Q_list, R_list, P_list, rho_list,
            algorithm=ALGO, verbose=False,
        )
        crs = [mp_sol[i].n_cr for i in range(M)]
        n_combos = 1
        for c in crs:
            n_combos *= c

        result["n_crs"]    = crs
        result["n_combos"] = n_combos
        print(f"           CRs={crs}  total_combos={n_combos}")

        print(f"    [offline] facet detection (method={FACET_METHOD})...", flush=True)
        mp_sol  = find_all_facet_neighbors(mp_sol, method=FACET_METHOD, verbose=True)
        qp_mats = precompute_qp_matrices(plant, Q_list, R_list, P_list, rho_list)

        result["offline_time"] = time.perf_counter() - t0
        print(f"    [offline] done  ({result['offline_time']:.1f}s total)")

    except Exception as e:
        result["error"] = f"offline: {e}"
        print(f"    [offline] ERROR: {e}")
        traceback.print_exc()
        return result

    # ── online methods ────────────────────────────────────────────────────────
    methods_cfg = {
        "DiMPC":       lambda: run_dimpc(
                            plant, x0, T_SIM,
                            p_max=p_max, eps=1e-8,
                            qp_mats=qp_mats, verbose=False),
        "I-mpDiMPC":   lambda: run_i_mpdimpc(
                            plant, x0, T_SIM, mp_sol,
                            p_max=p_max, eps=1e-8,
                            qp_mats=qp_mats, verbose=False),
        "IF-mpDiMPC":  lambda: run_if_mpdimpc(
                            plant, x0, T_SIM, mp_sol,
                            qp_mats=qp_mats, verbose=False),
        "FACET-DiMPC": lambda: run_facet_dimpc(
                            plant, x0, T_SIM, mp_sol,
                            qp_mats=qp_mats, verbose=False),
    }

    for name, fn in methods_cfg.items():
        try:
            r = fn()
            result["methods"][name] = {
                "avg_iters":   float(r.iter_counts.mean()),
                "max_iters":   int(r.iter_counts.max()),
                "avg_time_ms": float(r.solve_times.mean()) * 1000,
                "total_time":  float(r.solve_times.sum()),
                "conv_pct":    float(r.converged.mean()) * 100,
                "final_norm":  float(np.linalg.norm(r.x_traj[-1])),
            }
        except Exception as e:
            result["methods"][name] = {"error": str(e)}
            print(f"    [{name}] ERROR: {e}")
            traceback.print_exc()

    return result


# %% ── 3. Case study for one M ────────────────────────────────────────────────

def run_case_study_M(M: int) -> list[dict]:
    """
    Run N_PLANTS random plants for this M, loading checkpoints where available.
    Returns list of per-plant result dicts.
    """
    print(f"\n{'='*62}")
    print(f"  M = {M} subsystems  |  N = {N_PLANTS}  |  T = {T_SIM}  |  p_max={P_MAX[M]}")
    print(f"{'='*62}")

    plants = make_random_plants(M, N_PLANTS, seed=SEED)
    rng_ic = np.random.default_rng(SEED + 1000 * M)
    all_results = []

    for idx, plant in enumerate(plants):
        # ── try checkpoint first ─────────────────────────────────────────────
        ckpt = _load_plant_result(M, idx)
        if ckpt is not None:
            print(f"\n  Plant {idx+1}/{N_PLANTS} — loaded from checkpoint")
            _print_plant_row(ckpt)
            all_results.append(ckpt)
            continue

        # ── run fresh ────────────────────────────────────────────────────────
        x0 = make_ic(plant, scale=0.1, rng=rng_ic)
        print(f"\n  ── Plant {idx+1}/{N_PLANTS}  "
              f"(nx={plant.nx}, nu={plant.nu}, Np={plant.Np}) ──")

        res = run_one_plant(plant, x0, M, idx, p_max=P_MAX[M])
        _save_plant_result(M, idx, res)
        _print_plant_row(res)
        all_results.append(res)

    return all_results


def _print_plant_row(res: dict):
    if res.get("error"):
        print(f"    ERROR: {res['error']}")
        return
    print(f"    CRs={res['n_crs']}  combos={res['n_combos']}  "
          f"offline={res['offline_time']:.1f}s")
    print(f"    {'Method':<14} {'AvgIter':>8} {'MaxIter':>8} "
          f"{'AvgMs':>8} {'Conv%':>7} {'||xT||':>9}")
    print(f"    {'-'*58}")
    for name in ["DiMPC", "I-mpDiMPC", "IF-mpDiMPC", "FACET-DiMPC"]:
        m = res["methods"].get(name, {})
        if "error" in m:
            print(f"    {name:<14}  ERROR: {m['error']}")
        else:
            print(f"    {name:<14} {m['avg_iters']:>8.1f} {m['max_iters']:>8d} "
                  f"{m['avg_time_ms']:>8.3f} {m['conv_pct']:>6.0f}% "
                  f"{m['final_norm']:>9.5f}")


# %% ── 4. Run all M ───────────────────────────────────────────────────────────

all_M_results: dict[int, list[dict]] = {}

for M in M_LIST:
    all_M_results[M] = run_case_study_M(M)


# %% ── 5. Aggregate summary (paper-style) ────────────────────────────────────

def aggregate(results: list[dict], method: str, key: str) -> list[float]:
    """Collect 'key' from all plants for 'method', skip errors."""
    vals = []
    for r in results:
        m = r.get("methods", {}).get(method, {})
        if key in m:
            vals.append(m[key])
    return vals


def print_summary(all_M_results: dict[int, list[dict]]):
    print(f"\n\n{'='*72}")
    print(f"  AGGREGATE SUMMARY  —  N={N_PLANTS} plants/M, T={T_SIM}")
    print(f"{'='*72}")

    METHOD_NAMES = ["DiMPC", "I-mpDiMPC", "IF-mpDiMPC", "FACET-DiMPC"]

    # ── Table I: avg / max iterations for DiMPC ─────────────────────────────
    print(f"\n  DiMPC iterations  (paper Table I):")
    print(f"  {'M':>3}  {'Max iters':>10}  {'Avg iters':>10}  "
          f"{'Conv%':>7}  (paper: M2 max=38 avg=19.99 | M3 max=62 avg=33.97)")
    print(f"  {'-'*52}")
    for M, results in all_M_results.items():
        max_v = aggregate(results, "DiMPC", "max_iters")
        avg_v = aggregate(results, "DiMPC", "avg_iters")
        cnv_v = aggregate(results, "DiMPC", "conv_pct")
        if max_v:
            print(f"  {M:>3}  {max(max_v):>10.0f}  {np.mean(avg_v):>10.2f}  "
                  f"{np.mean(cnv_v):>6.0f}%")
        else:
            print(f"  {M:>3}  {'N/A':>10}  {'N/A':>10}")

    # ── Computation time ─────────────────────────────────────────────────────
    print(f"\n  Average per-step computation time (ms)  (paper Fig. 5-6):")
    print(f"  {'M':>3}  {'DiMPC':>10}  {'I-mpDiMPC':>11}  "
          f"{'IF-mpDiMPC':>12}  {'FACET-DiMPC':>13}")
    print(f"  {'-'*55}")
    for M, results in all_M_results.items():
        row = [f"  {M:>3}"]
        for name in METHOD_NAMES:
            v = aggregate(results, name, "avg_time_ms")
            row.append(f"{np.mean(v):>10.3f}" if v else f"{'N/A':>10}")
        print("  ".join(row))

    # ── Speedup vs DiMPC ─────────────────────────────────────────────────────
    print(f"\n  Speedup vs DiMPC (paper: FACET ~98% faster than DiMPC):")
    print(f"  {'M':>3}  {'I-mpDiMPC':>11}  {'IF-mpDiMPC':>12}  {'FACET-DiMPC':>13}")
    print(f"  {'-'*42}")
    for M, results in all_M_results.items():
        base = aggregate(results, "DiMPC", "avg_time_ms")
        if not base:
            continue
        base_ms = np.mean(base)
        row = [f"  {M:>3}"]
        for name in ["I-mpDiMPC", "IF-mpDiMPC", "FACET-DiMPC"]:
            v = aggregate(results, name, "avg_time_ms")
            if v:
                sp = base_ms / np.mean(v)
                row.append(f"{sp:>10.1f}×")
            else:
                row.append(f"{'N/A':>10}")
        print("  ".join(row))

    # ── Communication (Fig. 4) ────────────────────────────────────────────────
    print(f"\n  Communication load — avg data exchanges per step:")
    print(f"  {'M':>3}  {'DiMPC':>10}  {'I-mpDiMPC':>11}  "
          f"{'IF/FACET':>10}  (iterative vs 1 exchange)")
    print(f"  {'-'*46}")
    for M, results in all_M_results.items():
        d  = aggregate(results, "DiMPC",     "avg_iters")
        im = aggregate(results, "I-mpDiMPC", "avg_iters")
        d_s  = f"{np.mean(d):>10.1f}"  if d  else f"{'N/A':>10}"
        im_s = f"{np.mean(im):>11.1f}" if im else f"{'N/A':>11}"
        print(f"  {M:>3}  {d_s}  {im_s}  {'~1':>10}")

    # ── Control quality ───────────────────────────────────────────────────────
    print(f"\n  Control quality — avg ||x_T|| (all methods should ≈ 0):")
    print(f"  {'M':>3}  {'DiMPC':>10}  {'I-mpDiMPC':>11}  "
          f"{'IF-mpDiMPC':>12}  {'FACET-DiMPC':>13}")
    print(f"  {'-'*55}")
    for M, results in all_M_results.items():
        row = [f"  {M:>3}"]
        for name in METHOD_NAMES:
            v = aggregate(results, name, "final_norm")
            row.append(f"{np.mean(v):>10.5f}" if v else f"{'N/A':>10}")
        print("  ".join(row))

    # ── Offline time ─────────────────────────────────────────────────────────
    print(f"\n  Offline solve + facet detection (per plant, averaged):")
    print(f"  {'M':>3}  {'Avg CRs/ctrl':>14}  {'Avg combos':>12}  "
          f"{'Avg offline(s)':>15}")
    print(f"  {'-'*50}")
    for M, results in all_M_results.items():
        valid = [r for r in results if r.get("n_crs") is not None]
        if not valid:
            continue
        avg_crs     = np.mean([np.mean(r["n_crs"]) for r in valid])
        avg_combos  = np.mean([r["n_combos"] for r in valid])
        avg_offline = np.mean([r["offline_time"] for r in valid
                               if r["offline_time"] is not None])
        print(f"  {M:>3}  {avg_crs:>14.1f}  {avg_combos:>12.0f}  "
              f"{avg_offline:>15.1f}")

    print(f"\n{'='*72}")


print_summary(all_M_results)


# %% ── 6. Assertions ──────────────────────────────────────────────────────────

def run_assertions(all_M_results: dict[int, list[dict]]):
    print("\nRunning assertions...")
    errors = []

    for M, results in all_M_results.items():
        valid = [r for r in results if not r.get("error")]
        if not valid:
            print(f"  M={M}: no valid results, skipping assertions")
            continue

        # All iteration-free methods achieve good control (||x_T|| < 5)
        for name in ["IF-mpDiMPC", "FACET-DiMPC"]:
            norms = aggregate(valid, name, "final_norm")
            if norms:
                avg_n = np.mean(norms)
                if avg_n >= 5.0:
                    errors.append(f"M={M} {name}: avg ||x_T||={avg_n:.3f} ≥ 5.0")

        # DiMPC and I-mpDiMPC produce same avg final norm (same optimizer)
        d_n  = aggregate(valid, "DiMPC",     "final_norm")
        im_n = aggregate(valid, "I-mpDiMPC", "final_norm")
        if d_n and im_n:
            diff = abs(np.mean(d_n) - np.mean(im_n))
            if diff > 1.0:
                errors.append(
                    f"M={M}: DiMPC vs I-mpDiMPC final norm diverge: {diff:.4f}")

        # IF-mpDiMPC and FACET-DiMPC give same final norm (same solution)
        if_n = aggregate(valid, "IF-mpDiMPC",  "final_norm")
        fa_n = aggregate(valid, "FACET-DiMPC", "final_norm")
        if if_n and fa_n:
            diff = abs(np.mean(if_n) - np.mean(fa_n))
            if diff > 0.1:
                errors.append(
                    f"M={M}: IF vs FACET final norm diverge: {diff:.4f}")

        # IF/FACET-DiMPC should achieve ≥50% convergence
        for name in ["IF-mpDiMPC", "FACET-DiMPC"]:
            conv = aggregate(valid, name, "conv_pct")
            if conv and np.mean(conv) < 50:
                errors.append(
                    f"M={M} {name}: convergence {np.mean(conv):.0f}% < 50%")

    if errors:
        print("  FAILED assertions:")
        for e in errors:
            print(f"    ✗ {e}")
    else:
        print("  All assertions passed.")
    return len(errors) == 0


run_assertions(all_M_results)


# %% ── 7. Entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nNote: N_PLANTS={N_PLANTS}, T_SIM={T_SIM}.")
    print(f"For full paper reproduction: set N_PLANTS=100, T_SIM=100.")
    print(f"Checkpoints saved in: {CKPT_DIR}")
    print(f"Re-run the script to resume from checkpoints after interruption.")
# %%
