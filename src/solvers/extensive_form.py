import src.utils as utils
from src.model import build_model


def run_model(time_str, n, seed=None, verbose=True):

    model_container = build_model(time_str, n, seed=seed)

    model = model_container.model
    x = model_container.vars["x"]
    r = model_container.vars["r"]
    a = model_container.vars["a"]
    delta = model_container.vars["delta"]
    d = model_container.vars["d"]
    i = model_container.vars["i"]
    l = model_container.vars["l"]

    Q = model_container.params["Q"]
    P = model_container.params["P"]

    U = model_container.sets["U"]
    V = model_container.sets["V"]
    W = model_container.sets["W"]
    M_u = model_container.sets["M_u"]
    M_v = model_container.sets["M_v"]
    M_w = model_container.sets["M_w"]

    print("Added all constraints, starting to optimize model...")
    # --- OPTIMIZE MODEL ---
    
    model.optimize()
    
    runtime = model.Runtime
    print(f"Model optimized in {runtime:.2f} seconds.")


    # --- PRINT RESULTS ---
    if verbose:
        utils.print_results(model, x, r, a, delta, d, i, l, Q, P, U, V, W, M_u, M_v, M_w)

    output_dict = {
        "model": model,
        "x": x,
        "r": r,
        "a": a,
        "delta": delta,
        "d": d,
        "i": i,
        "l": l,
        "P": P,
        "Q": Q,
        "U": U,
        "V": V,
        "W": W,
        "M_u": M_u,
        "M_v": M_v,
        "M_w": M_w,
        "objective": model.ObjVal,   # <-- NEW
        "runtime": runtime          # <-- NEW
    }

    return output_dict