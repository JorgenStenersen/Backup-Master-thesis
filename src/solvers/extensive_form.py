import src.utils as utils
from src.model import build_model, initialize_run
from read import get_global_bounds_from_raw_data


def run_model(time_str, n, seed=None, verbose=True):

    # Bygg treet og hent globale grenseverdier
    full_scenario_tree, global_bounds = initialize_run(time_str, n, seed=seed)
    global_bounds = get_global_bounds_from_raw_data(time_str)

    # Bygg modell
    model_container = build_model(full_scenario_tree, global_bounds)

    # --- OPTIMIZE MODEL ---
    model_container.model.optimize()
    
    
    print(f"Model optimized in {model_container.model.Runtime:.2f} seconds.")


    # --- PRINT RESULTS ---
    if verbose:
        utils.print_results(model_container)






