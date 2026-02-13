
from src.model import run_model
import statistics



def run_robustness_experiment(
    time_str: str,
    n: int,
    num_runs: int = 20,
    base_seed: int | None = None,
    verbose_runs: bool = False,
    **run_model_kwargs,
):
    """
    Runs `run_model` num_runs times and returns averages and variances
    of objective values and runtimes.
    If base_seed is not None, seeds will be base_seed, base_seed+1, ...
    """

    objectives = []
    runtimes = []

    for k in range(num_runs):
        # Optional: different seeds per run
        seed = None
        if base_seed is not None:
            seed = base_seed + k

        res = run_model(
            time_str=time_str,
            n=n,
            verbose=verbose_runs,
            **run_model_kwargs,
            seed=seed,          # assumes run_model has a seed parameter
        )

        obj = res["objective"]
        rt = res["runtime"]

        objectives.append(obj)
        runtimes.append(rt)

        print(f"[RUN {k}] seed={seed}, obj={obj:.4f}, runtime={rt:.4f}s")

    # Use population statistics or sample statistics as you prefer:
    avg_obj = statistics.fmean(objectives)
    std_obj = statistics.pstdev(objectives)   # population std dev

    avg_rt = statistics.fmean(runtimes)
    std_rt = statistics.pstdev(runtimes)

    results = {
        "num_runs": num_runs,
        "n_scenarios": n,
        "avg_objective": avg_obj,
        "var_objective": std_obj,
        "avg_runtime": avg_rt,
        "var_runtime": std_rt,
        "objectives": objectives,
        "runtimes": runtimes,
    }

    print("\n=== ROBUSTNESS SUMMARY ===")
    print(f"Runs         : {num_runs}")
    print(f"Scenarios n  : {n}")
    print(f"Objective avg: {avg_obj:.4f}")
    print(f"Objective std: {std_obj:.4f}")
    print(f"Runtime avg  : {avg_rt:.4f} s")
    print(f"Runtime std  : {std_rt:.6f} s")

    return results

