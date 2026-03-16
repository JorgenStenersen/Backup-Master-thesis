import argparse
import csv
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import src.read as read
import src.tree as tree
from src.model import get_market_products
from src.solvers.progressive_hedging import (
    adapt_alpha,
    compute_consensus,
    compute_convergence_gap,
    compute_dual_residual,
    initialize_shadow_costs,
    print_final_consensus,
    print_iteration_header,
    print_iteration_row,
    update_shadow_costs,
)
from scripts.ph_bundle_worker import run_bundle_job


def _save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _run_bundle_batch(mode: str, iteration: int, num_bundles: int, static_file: Path,
                      out_dir: Path, max_workers: int, gurobi_threads: int,
                      state_file: Path | None = None) -> list:
    iter_dir = out_dir / f"iter_{iteration:03d}"
    (iter_dir / "results").mkdir(parents=True, exist_ok=True)
    (iter_dir / "logs").mkdir(parents=True, exist_ok=True)

    workers = max(1, min(int(max_workers), num_bundles))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for b_idx in range(num_bundles):
            futures.append(
                executor.submit(
                    run_bundle_job,
                    mode,
                    b_idx,
                    iteration,
                    str(static_file),
                    str(out_dir),
                    str(state_file) if state_file is not None else "",
                    int(gurobi_threads),
                )
            )

        for fut in as_completed(futures):
            fut.result()

    results = []
    for b_idx in range(num_bundles):
        path = iter_dir / "results" / f"bundle_{b_idx:04d}.pkl"
        if not path.exists():
            results.append(None)
            continue
        results.append(_load_pickle(path))

    _write_iteration_timing_summary(iter_dir, iteration, num_bundles)
    return results


def _write_iteration_timing_summary(iter_dir: Path, iteration: int, num_bundles: int) -> None:
    logs_dir = iter_dir / "logs"
    summary_path = logs_dir / "timing_summary.csv"

    fieldnames = [
        "iteration",
        "bundle_index",
        "mode",
        "status",
        "elapsed_seconds",
        "gurobi_threads",
        "hostname",
        "pid",
        "sge_job_id",
        "sge_task_id",
        "start_utc",
        "end_utc",
        "error",
    ]

    rows = []
    for b_idx in range(num_bundles):
        log_path = logs_dir / f"bundle_{b_idx:04d}_timing.json"
        if not log_path.exists():
            rows.append(
                {
                    "iteration": iteration,
                    "bundle_index": b_idx,
                    "mode": "",
                    "status": "missing",
                    "elapsed_seconds": "",
                    "gurobi_threads": "",
                    "hostname": "",
                    "pid": "",
                    "sge_job_id": "",
                    "sge_task_id": "",
                    "start_utc": "",
                    "end_utc": "",
                    "error": "timing file missing",
                }
            )
            continue

        row = _load_pickle_or_json(log_path)
        rows.append({key: row.get(key, "") for key in fieldnames})

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_pickle_or_json(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        import json

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return _load_pickle(path)


def run_distributed_ph(time_str: str, n_total: int, n_per_bundle: int, num_bundles: int,
                       seed: int, alpha: float, epsilon: float, max_iter: int,
                       adaptive_alpha: bool, tau: float, mu: float, work_dir: Path,
                       max_workers: int, gurobi_threads_per_bundle: int) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)

    print("[PH-SGE] Loading data and building bundles...")
    input_data = read.load_parameters_from_parquet(time_str, n_total, seed)
    global_bounds = read.get_global_bounds_from_input_data(input_data)
    bundles = tree.build_scenario_bundles(input_data, n_per_bundle, num_bundles, seed=seed)
    market_products = get_market_products()

    static_file = work_dir / "static.pkl"
    _save_pickle(
        static_file,
        {
            "bundles": bundles,
            "global_bounds": global_bounds,
            "market_products": market_products,
        },
    )

    print_iteration_header()

    print(
        f"[PH-SGE] Local parallel execution with max_workers={max_workers}, "
        f"gurobi_threads_per_bundle={gurobi_threads_per_bundle}"
    )

    # Initial solves (k=0)
    results = _run_bundle_batch(
        mode="initial",
        iteration=0,
        num_bundles=num_bundles,
        static_file=static_file,
        out_dir=work_dir,
        max_workers=max_workers,
        gurobi_threads=gurobi_threads_per_bundle,
    )
    consensus = compute_consensus(results, verbose=False)
    w_shadow = initialize_shadow_costs(results, consensus, alpha=alpha, verbose=False)
    gap = compute_convergence_gap(results, consensus, market_products)
    k = 0

    print_iteration_row(k, gap, results, alpha=alpha)

    while gap > epsilon and k < max_iter:
        k += 1

        state_file = work_dir / f"state_iter_{k:03d}.pkl"
        _save_pickle(
            state_file,
            {
                "consensus": consensus,
                "W_shadow": w_shadow,
                "alpha": alpha,
            },
        )

        results = _run_bundle_batch(
            mode="augmented",
            iteration=k,
            num_bundles=num_bundles,
            static_file=static_file,
            out_dir=work_dir,
            max_workers=max_workers,
            gurobi_threads=gurobi_threads_per_bundle,
            state_file=state_file,
        )

        prev_consensus = consensus
        consensus = compute_consensus(results, verbose=False)
        w_shadow = update_shadow_costs(w_shadow, results, consensus, alpha)
        gap = compute_convergence_gap(results, consensus, market_products)

        if adaptive_alpha:
            dual_res = compute_dual_residual(consensus, prev_consensus, alpha)
            alpha = adapt_alpha(alpha, gap, dual_res, tau=tau, mu=mu)

        print_iteration_row(k, gap, results, alpha=alpha)

    status = "CONVERGED" if gap <= epsilon else f"MAX ITER ({max_iter})"
    print(f"{'':->82}")
    print(f"  Terminated: {status}  (gap={gap:.6f}, alpha={alpha:.4f})")
    print_final_consensus(consensus)

    _save_pickle(
        work_dir / "final_state.pkl",
        {
            "status": status,
            "iterations": k,
            "gap": gap,
            "alpha": alpha,
            "consensus": consensus,
            "results": results,
            "W_shadow": w_shadow,
        },
    )
    print(f"[PH-SGE] Final state written to: {work_dir / 'final_state.pkl'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-parallel PH coordinator for SGE jobs.")
    parser.add_argument("--time-str", type=str, required=True)
    parser.add_argument("--n-total", type=int, required=True)
    parser.add_argument("--n-per-bundle", type=int, required=True)
    parser.add_argument("--num-bundles", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--adaptive-alpha", type=int, default=1)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--mu", type=float, default=10.0)
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--gurobi-threads-per-bundle", type=int, required=True)
    args = parser.parse_args()

    run_distributed_ph(
        time_str=args.time_str,
        n_total=args.n_total,
        n_per_bundle=args.n_per_bundle,
        num_bundles=args.num_bundles,
        seed=args.seed,
        alpha=args.alpha,
        epsilon=args.epsilon,
        max_iter=args.max_iter,
        adaptive_alpha=bool(args.adaptive_alpha),
        tau=args.tau,
        mu=args.mu,
        work_dir=Path(args.work_dir),
        max_workers=args.max_workers,
        gurobi_threads_per_bundle=args.gurobi_threads_per_bundle,
    )


if __name__ == "__main__":
    main()
