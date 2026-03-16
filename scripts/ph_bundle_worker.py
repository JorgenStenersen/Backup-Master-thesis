import argparse
import json
import os
import pickle
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

from src.solvers.progressive_hedging import solve_bundles, solve_bundles_augmented


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pickle(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _resolve_bundle_index(explicit_idx: int | None) -> int:
    if explicit_idx is not None:
        return explicit_idx
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    if "SGE_TASK_ID" in os.environ:
        return int(os.environ["SGE_TASK_ID"]) - 1
    return 0


def run_bundle_job(
    mode: str,
    bundle_index: int,
    iteration: int,
    static_file: str,
    out_dir: str,
    state_file: str = "",
    gurobi_threads: int = 1,
) -> None:
    iter_dir = Path(out_dir) / f"iter_{iteration:03d}"
    results_dir = iter_dir / "results"
    logs_dir = iter_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    static_data = _load_pickle(Path(static_file))

    bundles = static_data["bundles"]
    if bundle_index < 0 or bundle_index >= len(bundles):
        raise IndexError(
            f"Bundle index {bundle_index} outside [0, {len(bundles) - 1}]"
        )

    bundle = bundles[bundle_index]
    global_bounds = static_data["global_bounds"]
    market_products = static_data["market_products"]

    start_ts = time.perf_counter()
    start_utc = datetime.now(timezone.utc).isoformat()
    status = "ok"
    error_message = ""

    try:
        if mode == "initial":
            result = solve_bundles(
                [bundle],
                global_bounds,
                market_products,
                models=None,
                base_objs=None,
                verbose=False,
                gurobi_threads=gurobi_threads,
            )[0]
        else:
            if not state_file:
                raise ValueError("--state-file is required for augmented mode")
            state = _load_pickle(Path(state_file))
            consensus = state["consensus"]
            w_shadow = state["W_shadow"][bundle_index]
            alpha = state["alpha"]

            result = solve_bundles_augmented(
                [bundle],
                global_bounds,
                [w_shadow],
                consensus,
                alpha,
                market_products,
                models=None,
                base_objs=None,
                verbose=False,
                gurobi_threads=gurobi_threads,
            )[0]
    except Exception as exc:
        status = "error"
        error_message = repr(exc)
        result = None

    end_ts = time.perf_counter()
    elapsed = end_ts - start_ts
    end_utc = datetime.now(timezone.utc).isoformat()

    out_path = results_dir / f"bundle_{bundle_index:04d}.pkl"
    _save_pickle(out_path, result)

    metrics = {
        "iteration": iteration,
        "mode": mode,
        "bundle_index": bundle_index,
        "status": status,
        "error": error_message,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "elapsed_seconds": elapsed,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "gurobi_threads": int(gurobi_threads),
        "sge_job_id": os.environ.get("JOB_ID", ""),
        "sge_task_id": os.environ.get("SGE_TASK_ID", ""),
    }

    metrics_path = logs_dir / f"bundle_{bundle_index:04d}_timing.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if status == "error":
        raise RuntimeError(
            f"Bundle {bundle_index} failed in iteration {iteration}: {error_message}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve one PH bundle.")
    parser.add_argument("--mode", choices=["initial", "augmented"], required=True)
    parser.add_argument("--bundle-index", type=int, default=None)
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--static-file", type=str, required=True)
    parser.add_argument("--state-file", type=str, default="")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--gurobi-threads", type=int, default=1)
    args = parser.parse_args()

    bundle_index = _resolve_bundle_index(args.bundle_index)
    run_bundle_job(
        mode=args.mode,
        bundle_index=bundle_index,
        iteration=args.iter,
        static_file=args.static_file,
        out_dir=args.out_dir,
        state_file=args.state_file,
        gurobi_threads=args.gurobi_threads,
    )


if __name__ == "__main__":
    main()
