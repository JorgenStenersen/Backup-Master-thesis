import src.solvers.extensive_form as extensive_form
import src.solvers.progressive_hedging as progressive_hedging
#from experiments.robustness import run_robustness_experiment
#from experiments.benchmark import run_deterministic_benchmark

if __name__ == "__main__":
    path = "./input_data_10.csv"
    time_str = "2025-10-09 21:00:00+00:00"
    n = 5
    verbose = True
    seed = 30
    #number_of_runs = 20
    
    #extensive_form.run_model(time_str, n, seed=seed, verbose=verbose)

    # --- Progressive Hedging: solve bundles ---
    n_per_bundle = 3
    num_bundles = 10
    alpha = 100
    epsilon = 1
    adaptive_alpha = True
    tau = 2.0
    mu = 10.0
    B, results, consensus, W_shadow = progressive_hedging.run_progressive_hedging(
        time_str, n_total=n, n_per_bundle=n_per_bundle,
        num_bundles=num_bundles, seed=seed, verbose=verbose,
        alpha=alpha, epsilon=epsilon,
        adaptive_alpha=adaptive_alpha, tau=tau, mu=mu
    )

    # run_robustness_experiment(time_str, n, number_of_runs, 5)
    #run_deterministic_benchmark(time_str, n, seed)