import src.read as read
import src.utils as utils
from src.model import build_model, initialize_run
from src.read import get_global_bounds_from_raw_data
import src.tree as tree


def run_model(time_str, n, seed=None, verbose=True):

    input_data = read.load_parameters_from_parquet(time_str, n, seed)

    scenario_tree = tree.build_scenario_tree(input_data, n, seed)
    global_bounds = read.get_global_bounds_from_input_data(input_data)


    # Bygg modell
    model_container = build_model(scenario_tree, global_bounds)

    # --- OPTIMIZE MODEL ---
    model_container.model.optimize()
    
    
    print(f"Model optimized in {model_container.model.Runtime:.2f} seconds.")


    # --- PRINT RESULTS ---
    if verbose:
        utils.print_results(model_container)






