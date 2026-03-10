import src.solvers.extensive_form as extensive_form
import src.solvers.progressive_hedging as progressive_hedging
#from experiments.robustness import run_robustness_experiment
#from experiments.benchmark import run_deterministic_benchmark

if __name__ == "__main__":
    path = "./input_data_10.csv"
    time_str = "2025-10-09 21:00:00+00:00"
    n = 5
    verbose = True
    seed = 15
    #number_of_runs = 20
    
    #extensive_form.run_model(time_str, n, seed=seed, verbose=verbose)

    # --- Progressive Hedging: solve bundles ---
    n_per_bundle = 2
    num_bundles = 6
    B, results, consensus, W_shadow = progressive_hedging.run_progressive_hedging(
        time_str, n_total=n, n_per_bundle=n_per_bundle,num_bundles=num_bundles, seed=seed, verbose=verbose
    )

    # --- Visualize the structure of results ---
    import json

    def summarize_results(results):
        """Pretty-print the structure and sample content of the results list."""
        print(f"\n{'='*60}")
        print(f"  RESULTS OVERVIEW  ({len(results)} bundles)")
        print(f"{'='*60}")
        for i, res in enumerate(results):
            if res is None:
                print(f"\n  Bundle {i}: NOT SOLVED")
                continue
            print(f"\n  Bundle {i}:")
            print(f"    Objective : {res['objective']:.4f}")
            print(f"    Stage 1 keys ({len(res['stage1'])}): {list(res['stage1'].keys())}")
            print(f"    Stage 2 keys ({len(res['stage2'])}): {list(res['stage2'].keys())[:6]}{'  ...' if len(res['stage2']) > 6 else ''}")
            print(f"    Stage 3 keys ({len(res['stage3'])}): {list(res['stage3'].keys())[:6]}{'  ...' if len(res['stage3']) > 6 else ''}")

        # Detailed view of first solved bundle
        first = next((r for r in results if r is not None), None)
        if first is None:
            return
        print(f"\n{'='*60}")
        print("  DETAILED VIEW — first solved bundle")
        print(f"{'='*60}")
        for stage_name in ("stage1", "stage2", "stage3"):
            stage = first[stage_name]
            print(f"\n  [{stage_name}]")
            first_key = next(iter(stage.keys()))
            vals = stage[first_key]
            print(f"    {str(first_key):>30s}  ->  x = {vals['x']:>10.4f},  r = {vals['r']:>10.4f}")
        print()

    summarize_results(results)

    # run_robustness_experiment(time_str, n, number_of_runs, 5)
    #run_deterministic_benchmark(time_str, n, seed)