import math
import src.read as read

import gurobipy as gp
from gurobipy import GRB

import src.utils as utils
from src.model import build_model, get_market_products
import src.tree as tree
from src.read import get_global_bounds_from_raw_data


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
            - "stage1"      : dict  {m: {"x": val, "r": val}}  for m in M_u  (CM markets)
                              (one representative value per market; NA enforced within each bundle)
            - "stage2"      : dict  {(m, u): {"x": val, "r": val}}  for m in M_v, u in U
                              (DA market, one representative per parent CM-node u)
            - "stage3"      : dict  {(m, v): {"x": val, "r": val}}  for m in M_w, v in V_all
                              (EAM markets, one representative per parent DA-node v)
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

        # --- Stage 1 (first-stage decisions): CM markets ---
        # NA constraints make all u-nodes share the same value; pick any representative.
        u_rep = next(iter(U))
        stage1 = {}
        for m in M_u:
            stage1[m] = {
                "x": x[m, u_rep].X,
                "r": r[m, u_rep].X,
            }

        # --- Stage 2 (second-stage decisions): DA market ---
        # Within children of a given u, NA enforces the same value.
        # Store one representative per parent u.
        stage2 = {}
        for u in U:
            v_rep = next(iter(V[u]))  # representative v-node for this u
            for m in M_v:
                stage2[(m, u)] = {
                    "x": x[m, v_rep].X,
                    "r": r[m, v_rep].X,
                }

        # --- Stage 3 (third-stage decisions): EAM markets ---
        # Within children of a given v, NA enforces the same value.
        # Store one representative per parent v.
        stage3 = {}
        for v in V_all:
            w_rep = next(iter(W[v]))  # representative w-node for this v
            for m in M_w:
                stage3[(m, v)] = {
                    "x": x[m, w_rep].X,
                    "r": r[m, w_rep].X,
                }

        # Negate objective back to maximization sense
        obj_val = -mc.model.ObjVal

        bundle_result = {
            "objective": obj_val,
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
        }
        results.append(bundle_result)

        if verbose:
            print(f"[INFO] Bundle {b_idx}: obj = {obj_val:.4f}")

    return results


def compute_consensus(results, verbose=False):
    """
    Computes the consensus (weighted average) of decisions across all bundles
    for each node.  This corresponds to steps 6-8 of the PH algorithm:

        x̄_n = Σ_{b∈B} π_nb · x̂_nb

    where π_nb = 1 / |{bundles containing node n}|  (equal weight).

    For stage 1 (root): every bundle contributes → weight = 1/num_bundles.
    For stage 2 (u-nodes): only bundles that contain a given u-node contribute.
    For stage 3 (v-nodes): only bundles that contain a given v-node contribute.

    Input:
        results: list of per-bundle result dicts (from solve_bundles)
        verbose: if True, print summary

    Returns:
        consensus: dict with keys "stage1", "stage2", "stage3", each containing
                   the averaged decision values with the same key structure as
                   the individual bundle results.
    """

    # Filter out None results (unsolved bundles)
    solved = [r for r in results if r is not None]

    if not solved:
        raise ValueError("No bundles were solved successfully.")

    # ------------------------------------------------------------------
    # Stage 1 consensus  (root node — all bundles share it)
    # ------------------------------------------------------------------
    stage1_consensus = {}
    markets_stage1 = solved[0]["stage1"].keys()
    for m in markets_stage1:
        avg_x = sum(r["stage1"][m]["x"] for r in solved) / len(solved)
        avg_r = sum(r["stage1"][m]["r"] for r in solved) / len(solved)
        stage1_consensus[m] = {"x": avg_x, "r": avg_r}

    # ------------------------------------------------------------------
    # Stage 2 consensus  (u-nodes — may differ across bundles)
    # ------------------------------------------------------------------
    # For each (market, u-node) key, collect values from all bundles that
    # contain it, then average.
    stage2_accum = {}          # {(m, u): [{"x": ..., "r": ...}, ...]}
    for r in solved:
        for key, vals in r["stage2"].items():
            stage2_accum.setdefault(key, []).append(vals)

    stage2_consensus = {}
    for key, vals_list in stage2_accum.items():
        n_b = len(vals_list)
        avg_x = sum(v["x"] for v in vals_list) / n_b
        avg_r = sum(v["r"] for v in vals_list) / n_b
        stage2_consensus[key] = {"x": avg_x, "r": avg_r}

    # ------------------------------------------------------------------
    # Stage 3 consensus  (v-nodes — may differ across bundles)
    # ------------------------------------------------------------------
    stage3_accum = {}
    for r in solved:
        for key, vals in r["stage3"].items():
            stage3_accum.setdefault(key, []).append(vals)

    stage3_consensus = {}
    for key, vals_list in stage3_accum.items():
        n_b = len(vals_list)
        avg_x = sum(v["x"] for v in vals_list) / n_b
        avg_r = sum(v["r"] for v in vals_list) / n_b
        stage3_consensus[key] = {"x": avg_x, "r": avg_r}

    consensus = {
        "stage1": stage1_consensus,
        "stage2": stage2_consensus,
        "stage3": stage3_consensus,
    }

    if verbose:
        print(f"\n[CONSENSUS] Stage 1: {len(stage1_consensus)} market(s), "
              f"averaged over {len(solved)} bundles")
        print(f"[CONSENSUS] Stage 2: {len(stage2_consensus)} unique (market, u-node) pairs")
        print(f"[CONSENSUS] Stage 3: {len(stage3_consensus)} unique (market, v-node) pairs")

    return consensus


def initialize_shadow_costs(results, consensus, alpha=100, verbose=False):
    """
    Initialises the shadow costs (dual multipliers) for every node-bundle
    pair, corresponding to steps 9-10 of the PH algorithm:

        w_nb^(0) = alpha * (x_nb^(0) - x_bar_n^(0))

    Input:
        results:    list of per-bundle result dicts (from solve_bundles)
        consensus:  dict with "stage1", "stage2", "stage3" consensus values
        alpha:      penalty parameter (default 100)
        verbose:    if True, print summary

    Returns:
        W_shadow: list (one entry per bundle) of dicts, each with keys
                  "stage1", "stage2", "stage3".  Value structure mirrors
                  the results/consensus dicts:
                    stage1[m]      = {"x": w_val, "r": w_val}
                    stage2[(m, u)] = {"x": w_val, "r": w_val}
                    stage3[(m, v)] = {"x": w_val, "r": w_val}
    """

    W_shadow = []

    for b_idx, res in enumerate(results):
        if res is None:
            W_shadow.append(None)
            continue

        bundle_w = {"stage1": {}, "stage2": {}, "stage3": {}}

        # Stage 1
        for m, vals in res["stage1"].items():
            cons = consensus["stage1"][m]
            bundle_w["stage1"][m] = {
                "x": alpha * (vals["x"] - cons["x"]),
                "r": alpha * (vals["r"] - cons["r"]),
            }

        # Stage 2
        for key, vals in res["stage2"].items():
            cons = consensus["stage2"][key]
            bundle_w["stage2"][key] = {
                "x": alpha * (vals["x"] - cons["x"]),
                "r": alpha * (vals["r"] - cons["r"]),
            }

        # Stage 3
        for key, vals in res["stage3"].items():
            cons = consensus["stage3"][key]
            bundle_w["stage3"][key] = {
                "x": alpha * (vals["x"] - cons["x"]),
                "r": alpha * (vals["r"] - cons["r"]),
            }

        W_shadow.append(bundle_w)

    if verbose:
        n_init = sum(1 for w in W_shadow if w is not None)
        print(f"[SHADOW] Initialised shadow costs for {n_init} bundles (alpha={alpha})")

    return W_shadow


def compute_convergence_gap(results, consensus):
    """
    Computes the convergence gap (step 24 / while-condition 12):

        g^(k) = sum_{n in N} sum_{b in B} pi_nb * ||x_nb^(k) - xbar_n^(k)||

    where pi_nb = 1 / |{bundles containing node n}|.

    Input:
        results:    list of per-bundle result dicts
        consensus:  dict with "stage1", "stage2", "stage3" consensus values

    Returns:
        gap: float, the total weighted distance from consensus
    """

    solved = [(i, r) for i, r in enumerate(results) if r is not None]
    if not solved:
        return float('inf')

    M_u, M_v, M_w, _ = get_market_products()
    gap = 0.0
    num_solved = len(solved)

    # Stage 1 (root): all bundles contribute, pi = 1/num_solved
    pi = 1.0 / num_solved
    for _, res in solved:
        sq = sum(
            (res["stage1"][m]["x"] - consensus["stage1"][m]["x"])**2 +
            (res["stage1"][m]["r"] - consensus["stage1"][m]["r"])**2
            for m in M_u
        )
        gap += pi * math.sqrt(sq)

    # Stage 2 (u-nodes): only bundles containing u contribute
    u_to_bundles = {}
    for idx, res in solved:
        for (m, u) in res["stage2"]:
            u_to_bundles.setdefault(u, set()).add(idx)

    for u, bundle_set in u_to_bundles.items():
        pi_u = 1.0 / len(bundle_set)
        for idx in bundle_set:
            res = results[idx]
            sq = sum(
                (res["stage2"][(m, u)]["x"] - consensus["stage2"][(m, u)]["x"])**2 +
                (res["stage2"][(m, u)]["r"] - consensus["stage2"][(m, u)]["r"])**2
                for m in M_v
            )
            gap += pi_u * math.sqrt(sq)

    # Stage 3 (v-nodes): only bundles containing v contribute
    v_to_bundles = {}
    for idx, res in solved:
        for (m, v) in res["stage3"]:
            v_to_bundles.setdefault(v, set()).add(idx)

    for v, bundle_set in v_to_bundles.items():
        pi_v = 1.0 / len(bundle_set)
        for idx in bundle_set:
            res = results[idx]
            sq = sum(
                (res["stage3"][(m, v)]["x"] - consensus["stage3"][(m, v)]["x"])**2 +
                (res["stage3"][(m, v)]["r"] - consensus["stage3"][(m, v)]["r"])**2
                for m in M_w
            )
            gap += pi_v * math.sqrt(sq)

    return gap


def solve_bundles_augmented(B, global_bounds, W_shadow, consensus, alpha, verbose=False):
    """
    Solves each bundle with the augmented PH objective (steps 14-16):

        min  -f(b) + sum_{n in N} [ w_nb^(k-1) * x_nb
                                     + alpha * ||x_nb - xbar_n^(k-1)||^2 ]

    The linear shadow-cost term steers decisions toward consensus.
    The quadratic proximity term penalises deviation from the previous
    consensus.

    Input:
        B:              list of scenario tree dicts
        global_bounds:  dict with global Big-M bounds
        W_shadow:       list of shadow-cost dicts (one per bundle)
        consensus:      current consensus dict
        alpha:          penalty parameter
        verbose:        if True, print per-bundle summaries

    Returns:
        results: list of per-bundle result dicts (same format as solve_bundles)
    """

    M_u, M_v, M_w, M = get_market_products()
    results = []

    for b_idx, bundle_tree in enumerate(B):
        if W_shadow[b_idx] is None:
            results.append(None)
            continue

        # Build model with base objective min -f(b)
        mc = build_model(bundle_tree, global_bounds, mode="progressive_hedging")

        x = mc.vars["x"]
        r = mc.vars["r"]
        U = mc.sets["U"]
        V = mc.sets["V"]
        W = mc.sets["W"]
        V_all = set().union(*V.values())

        w_b = W_shadow[b_idx]

        # ---- Build penalty expression ----
        penalty = gp.QuadExpr()

        # Stage 1 penalties (root node)
        u_rep = next(iter(U))
        for m in M_u:
            w_x = w_b["stage1"][m]["x"]
            w_r = w_b["stage1"][m]["r"]
            xbar_x = consensus["stage1"][m]["x"]
            xbar_r = consensus["stage1"][m]["r"]

            penalty += w_x * x[m, u_rep] + w_r * r[m, u_rep]
            penalty += alpha * (x[m, u_rep] - xbar_x) * (x[m, u_rep] - xbar_x)
            penalty += alpha * (r[m, u_rep] - xbar_r) * (r[m, u_rep] - xbar_r)

        # Stage 2 penalties (u-nodes)
        for u in U:
            v_rep = next(iter(V[u]))
            for m in M_v:
                key = (m, u)
                w_x = w_b["stage2"][key]["x"]
                w_r = w_b["stage2"][key]["r"]
                xbar_x = consensus["stage2"][key]["x"]
                xbar_r = consensus["stage2"][key]["r"]

                penalty += w_x * x[m, v_rep] + w_r * r[m, v_rep]
                penalty += alpha * (x[m, v_rep] - xbar_x) * (x[m, v_rep] - xbar_x)
                penalty += alpha * (r[m, v_rep] - xbar_r) * (r[m, v_rep] - xbar_r)

        # Stage 3 penalties (v-nodes)
        for v in V_all:
            w_rep = next(iter(W[v]))
            for m in M_w:
                key = (m, v)
                w_x = w_b["stage3"][key]["x"]
                w_r = w_b["stage3"][key]["r"]
                xbar_x = consensus["stage3"][key]["x"]
                xbar_r = consensus["stage3"][key]["r"]

                penalty += w_x * x[m, w_rep] + w_r * r[m, w_rep]
                penalty += alpha * (x[m, w_rep] - xbar_x) * (x[m, w_rep] - xbar_x)
                penalty += alpha * (r[m, w_rep] - xbar_r) * (r[m, w_rep] - xbar_r)

        # Add penalty to the base objective
        mc.model.update()
        base_obj = mc.model.getObjective()
        mc.model.setObjective(base_obj + penalty, GRB.MINIMIZE)

        mc.model.setParam("OutputFlag", 0)
        mc.model.optimize()

        if mc.model.Status != GRB.OPTIMAL:
            print(f"[WARNING] Bundle {b_idx} not solved to optimality (status={mc.model.Status})")
            results.append(None)
            continue

        # Extract decisions (same structure as solve_bundles)
        u_rep = next(iter(U))
        stage1 = {}
        for m in M_u:
            stage1[m] = {"x": x[m, u_rep].X, "r": r[m, u_rep].X}

        stage2 = {}
        for u in U:
            v_rep = next(iter(V[u]))
            for m in M_v:
                stage2[(m, u)] = {"x": x[m, v_rep].X, "r": r[m, v_rep].X}

        stage3 = {}
        for v in V_all:
            w_rep = next(iter(W[v]))
            for m in M_w:
                stage3[(m, v)] = {"x": x[m, w_rep].X, "r": r[m, w_rep].X}

        bundle_result = {
            "objective": -mc.model.ObjVal,
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
        }
        results.append(bundle_result)

        if verbose:
            print(f"[INFO] Bundle {b_idx}: augmented obj = {mc.model.ObjVal:.4f}")

    return results


def run_progressive_hedging(time_str, n_total, n_per_bundle, num_bundles, seed=0, verbose=True,
                            alpha=100, epsilon=1e-2, max_iter=50):
    """
    Entry point for the Progressive Hedging algorithm.

    Steps 1-11:  Initial solve, consensus, shadow costs.
    Steps 12-16: While g > epsilon, re-solve augmented bundles.
                 (Steps 18-24 — consensus/shadow-cost updates — not yet
                  implemented; the loop currently runs a single iteration.)

    Input:
        time_str:     timestamp string
        n_total:      total number of scenarios
        n_per_bundle: number of scenarios per bundle tree
        num_bundles:  how many bundles to generate
        seed:         base seed for bundle generation
        verbose:      print progress info
        alpha:        PH penalty parameter
        epsilon:      convergence tolerance
        max_iter:     maximum number of PH iterations

    Returns:
        B:         list of scenario tree dicts
        results:   latest per-bundle result dicts
        consensus: latest consensus dict
        W_shadow:  latest shadow-cost dicts
    """

    input_data = read.load_parameters_from_parquet(time_str, n_total, seed)
    global_bounds = read.get_global_bounds_from_input_data(input_data)


    # Build scenario bundles

    

    B = tree.build_scenario_bundles(input_data, n_per_bundle, num_bundles, seed=seed)
    print(f"[INFO] Built {num_bundles} scenario bundles.")

    # ------------------------------------------------------------------
    # Steps 2-5: Initial solve (no penalty terms)
    # ------------------------------------------------------------------
    results = solve_bundles(B, global_bounds, verbose=verbose)

    # Steps 6-8: Compute consensus
    consensus = compute_consensus(results, verbose=verbose)

    # Steps 9-10: Initialise shadow costs
    W_shadow = initialize_shadow_costs(results, consensus, alpha=alpha, verbose=verbose)

    # Step 12: Compute initial convergence gap
    g = compute_convergence_gap(results, consensus)
    k = 0
    if verbose:
        print(f"\n[PH] Iteration {k}: convergence gap g = {g:.6f}")

    # ------------------------------------------------------------------
    # Steps 12-16: Iterative improvements
    # ------------------------------------------------------------------
    while g > epsilon and k < max_iter:
        # Step 13: increment iteration counter
        k += 1

        # Steps 14-16: solve augmented sub-problems for every bundle
        results = solve_bundles_augmented(
            B, global_bounds, W_shadow, consensus, alpha, verbose=verbose
        )

        if verbose:
            print(f"[PH] Iteration {k}: augmented solve complete")

        # Steps 18-24 (consensus update, shadow-cost update, gap update)
        # will be added in a subsequent step.  For now, break after one
        # augmented solve so we can inspect the penalised decisions.
        break

    return B, results, consensus, W_shadow