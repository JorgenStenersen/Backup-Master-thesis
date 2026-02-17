from src.solvers.extensive_form import run_model
from experiments.robustness import run_robustness_experiment
from experiments.benchmark import run_deterministic_benchmark

if __name__ == "__main__":
    path = "./input_data_10.csv"
    time_str = "2025-10-07 13:00:00+00:00"
    n = 3
    verbose = True
    seed = 15
    number_of_runs = 20
    
    run_model(time_str, n, seed=seed, verbose=verbose)

    # run_robustness_experiment(time_str, n, number_of_runs, 5)
    #run_deterministic_benchmark(time_str, n, seed)