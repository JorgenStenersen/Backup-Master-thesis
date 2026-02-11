#!/usr/bin/env python3

import statistics

from model import run_model       # stochastic model
from benchmark import run_deterministic_benchmark   # deterministic benchmark


# ---------------------------------------------------------------------------
# FUNCTION: run_vrmss_experiment
# ---------------------------------------------------------------------------

def run_vrmss_experiment(
    time_str: str,
    n: int,
    num_runs: int = 6,
    base_seed: int | None = None,
    verbose_runs: bool = False,
    **run_model_kwargs,
):
    """
    Runs the stochastic model and deterministic benchmark multiple times
    using identical seeds per run. Computes Z_sto, Z_det and VRMSS for each run.
    """

    # --- 1) Storage lists ---
    Z_sto_list = []
    Z_det_list = []
    VRMSS_list = []

    print("\n=== START VRMSS EXPERIMENT ===")
    print(f"time_str = {time_str}, n = {n}, num_runs = {num_runs}")
    print(f"Base seed = {base_seed}\n")

    # ------------------------------------------------------------------
    # Main loop over Monte-Carlo runs
    # ------------------------------------------------------------------
    for k in range(num_runs):

        seed = None
        if base_seed is not None:
            seed = base_seed + k

        # --- 1) Deterministic model with seed 
        Z_det_scalar = run_deterministic_benchmark(
            time_str=time_str,
            n=n,
            seed=seed,                 
        )

        if Z_det_scalar == 0:
            raise ValueError("Deterministic objective value is zero; VRMSS undefined.")

        # --- 2) Run stochastic model with seed ---
        res_sto = run_model(
            time_str=time_str,
            n=n,
            seed=seed,
            verbose=verbose_runs,
            **run_model_kwargs,
        )

        model_sto = res_sto["model"]
        Z_sto = model_sto.objVal

       # Save results
        Z_sto_list.append(Z_sto)
        Z_det_list.append(Z_det_scalar)

        # --- 3) Compute VRMSS ---
        vrmss = (Z_sto - Z_det_scalar) / Z_det_scalar
        VRMSS_list.append(vrmss)

        print(f"[RUN {k}] seed={seed}")
        print(f"  Z_sto  = {Z_sto:.6f}")
        print(f"  Z_det  = {Z_det_scalar:.6f}")
        print(f"  VRMSS  = {vrmss:.6f}\n")

    # ------------------------------------------------------------------
    # 5) Averages
    # ------------------------------------------------------------------
    avg_Z_sto = statistics.fmean(Z_sto_list)
    avg_Z_det = statistics.fmean(Z_det_list)
    avg_VRMSS = statistics.fmean(VRMSS_list)

    # ------------------------------------------------------------------
    # 6) Print summary
    # ------------------------------------------------------------------
    print("=== VRMSS SUMMARY ===")
    print(f"Runs          : {num_runs}")
    print(f"Scenarios n   : {n}")
    print(f"Avg Z^sto     : {avg_Z_sto:.6f}")
    print(f"Avg Z^det     : {avg_Z_det:.6f}")
    print(f"Avg VRMSS     : {avg_VRMSS:.6f}")
    print("======================\n")

    # Return results dictionary
    results = {
        "num_runs": num_runs,
        "n_scenarios": n,
        "Z_sto_all": Z_sto_list,
        "Z_det_all": Z_det_list,
        "VRMSS_all": VRMSS_list,
        "avg_Z_sto": avg_Z_sto,
        "avg_Z_det": avg_Z_det,
        "avg_VRMSS": avg_VRMSS,
    }

    return results


# ---------------------------------------------------------------------------
# SIMPLE ENTRY POINT (EDIT THESE VALUES AND RUN THE SCRIPT)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 🔧 EDIT THESE PARAMETERS AS YOU LIKE:
    time_str = "2025-10-09 13:00:00+00:00"   # e.g. your dataset key / timestamp
    n = 6                                   # number of scenarios
    num_runs = 4                            # number of Monte-Carlo repetitions
    base_seed = 16                           # or None for no seeding
    verbose_runs = False                     # True if you want detailed run_model output

    results = run_vrmss_experiment(
        time_str=time_str,
        n=n,
        num_runs=num_runs,
        base_seed=base_seed,
        verbose_runs=verbose_runs,
    )

    print("Average Z^sto :", results["avg_Z_sto"])
    print("Average Z^det :", results["avg_Z_det"])
    print("Average VRMSS :", results["avg_VRMSS"])