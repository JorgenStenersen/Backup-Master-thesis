import src.utils as utils
from src.model import build_model, get_market_products
import src.tree as tree
from src.read import get_global_bounds_from_raw_data
from gurobipy import GRB


def solve_bundles(B, global_bounds, verbose=False):
    """
    Solves the model for each scenario tree (bundle) in B and stores
    the first-, second-, and third-stage decision variables.

    Input:
        B:              list of scenario tree dicts (output from build_scenario_bundles)
        global_bounds:  dict with global Big-M bounds (from get_global_bounds_from_raw_data)
        verbose:        if True, print per-bundle summaries

    Returns:
        results: list of dicts, one per bundle, each containing:
            - "objective"   : optimal objective value (negated back to original maximization sense)
            - "stage2"      : dict  {m: {"x": val, "r": val}}  for m in M_u
                              (one representative value per market; NA enforced within each bundle)
            - "stage3"      : dict  {(m, u): {"x": val, "r": val}}  for m in M_v, u in U
                              (one representative per parent CM-node u)
            - "stage4"      : dict  {(m, v): {"x": val, "r": val}}  for m in M_w, v in V_all
                              (one representative per parent DA-node v)
    """

    M_u, M_v, M_w, M = get_market_products()
    results = []

    for b_idx, bundle_tree in enumerate(B):

        # Build and solve model for this bundle
        mc = build_model(bundle_tree, global_bounds, mode="progressive_hedging")
        mc.model.setParam("OutputFlag", 0)  # suppress Gurobi output per bundle
        mc.model.optimize()

        if mc.model.Status != GRB.OPTIMAL:
            print(f"[WARNING] Bundle {b_idx} not solved to optimality (status={mc.model.Status})")
            results.append(None)
            continue

        # Retrieve variable dicts
        x = mc.vars["x"]
        r = mc.vars["r"]

        # Retrieve sets
        U = mc.sets["U"]
        V = mc.sets["V"]
        W = mc.sets["W"]
        V_all = set().union(*V.values())

        # --- Stage 2 (first-stage decisions): CM markets ---
        # NA constraints make all u-nodes share the same value; pick any representative.
        u_rep = next(iter(U))
        stage2 = {}
        for m in M_u:
            stage2[m] = {
                "x": x[m, u_rep].X,
                "r": r[m, u_rep].X,
            }

        # --- Stage 3 (second-stage decisions): DA market ---
        # Within children of a given u, NA enforces the same value.
        # Store one representative per parent u.
        stage3 = {}
        for u in U:
            v_rep = next(iter(V[u]))  # representative v-node for this u
            for m in M_v:
                stage3[(m, u)] = {
                    "x": x[m, v_rep].X,
                    "r": r[m, v_rep].X,
                }

        # --- Stage 4 (third-stage decisions): EAM markets ---
        # Within children of a given v, NA enforces the same value.
        # Store one representative per parent v.
        stage4 = {}
        for v in V_all:
            w_rep = next(iter(W[v]))  # representative w-node for this v
            for m in M_w:
                stage4[(m, v)] = {
                    "x": x[m, w_rep].X,
                    "r": r[m, w_rep].X,
                }

        # Negate objective back to maximization sense
        obj_val = -mc.model.ObjVal

        bundle_result = {
            "objective": obj_val,
            "stage2": stage2,
            "stage3": stage3,
            "stage4": stage4,
        }
        results.append(bundle_result)

        if verbose:
            print(f"[INFO] Bundle {b_idx}: obj = {obj_val:.4f}")

    return results


def run_progressive_hedging(time_str, n, num_bundles, seed=0, verbose=True):
    """
    Entry point: builds bundles, computes global bounds, solves all bundles,
    and returns the per-bundle results.

    Input:
        time_str:     timestamp string
        n:            number of scenarios per bundle tree
        num_bundles:  how many bundles to generate
        seed:         base seed for bundle generation
        verbose:      print progress info

    Returns:
        B:       list of scenario tree dicts
        results: list of per-bundle result dicts (see solve_bundles)
    """

    # Build scenario bundles
    B = tree.build_scenario_bundles(time_str, n, num_bundles, seed=seed)
    print(f"[INFO] Built {num_bundles} scenario bundles.")

    # Global bounds (computed once from the full forecast data)
    global_bounds = get_global_bounds_from_raw_data(time_str)

    # Solve each bundle
    results = solve_bundles(B, global_bounds, verbose=verbose)

    return B, results