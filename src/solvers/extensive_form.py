import src.utils as utils
from src.model import build_model


def run_model(time_str, n, seed=None, verbose=True):

    model_container = build_model(time_str, n, seed=seed)

    print("Starting to optimize model...")
    # --- OPTIMIZE MODEL ---
    model_container.model.optimize()
    
    
    print(f"Model optimized in {model_container.model.Runtime:.2f} seconds.")


    # --- PRINT RESULTS ---
    if verbose:
        utils.print_results(model_container)
