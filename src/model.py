import itertools
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import src.tree as tree
import src.utils as utils
import json
import statistics


def run_model(time_str: str, n:int, seed=None, det_policy_file=None, evaluate_deterministic_policy=False, only_da_and_eam=False, verbose=True):

    model = gp.Model()

    # --- SETS ---
    I = [1, 2, 3, 4]   # stages

    M_u = ["CM_up", "CM_down"]
    M_v = ["DA"]
    M_w = ["EAM_up", "EAM_down"]

    M  = M_u + M_v + M_w


    # Bygg treet
    scenario_tree = tree.build_scenario_tree(time_str, n, seed)
    print("[INFO] Built scenario tree.")
    # Lagre treet i modellen for tilgang
    model._scenario_tree = scenario_tree
    print("[INFO] Stored scenario tree in model.")


    # Bygg sett fra treet
    U, V, W, S = tree.build_sets_from_tree(scenario_tree)
    print("[INFO] Built sets from scenario tree.")

    # flate mengder for v- og w-noder:
    V_all = set().union(*V.values())
    W_all = set().union(*W.values())

    # bygg indeksmengder (m,s)
    idx_ms, idx_mw = tree.build_index_sets(U=U, V_all=V_all, W_all=W_all, M_u=M_u, M_v=M_v, M_w=M_w, M=M)
    print("[INFO] Built index sets.")

    # --- PARAMETERS ---
    print ("[INFO] Building scenario tree")
    

    P = utils.build_price_parameter(scenario_tree)
    Q = utils.build_production_capacity(scenario_tree)
    C = utils.build_cost_parameters(U, V, W, P)

    Pmax = {m: max(P[m, s] for s in S if (m, s) in idx_ms) for m in M}


    BIGM_1 = max(Pmax.values())
    BIGM_2 = max(Q.values())  # maksimal produksjonskapasitet
    BIGM_3 = 2*BIGM_2

    epsilon = 1e-3

    x_mFRR_min = 10  # minimum budstørrelse i mFRR-markedet

    r_MAX_EAM_up   = 0  # maks pris for EAM up
    r_MAX_EAM_down = 0    # maks pris for EAM down


    # --- VARIABLES ---

    # x_ms: bid quantity
    x = model.addVars(idx_ms, lb=0, vtype=GRB.INTEGER, name="x")
    # r_ms: bid price
    r = model.addVars(idx_ms, lb=0, name="r")
    # δ_ms: 1 hvis budet aktiveres
    delta = model.addVars(idx_ms, vtype=GRB.BINARY, name="delta")
    # a_ms: aktivert kvantum
    a = model.addVars(idx_ms, lb=0, vtype=GRB.INTEGER, name="a")
    # d_mw: avvik fra aktivert kvantum i terminale scenarier
    d = model.addVars(idx_mw, lb=0, ub=BIGM_2, name="d")
    # Binær variabel som indikerer om vi faktisk legger inn et bud (≠ 0)
    b = model.addVars([(m, s) for (m, s) in idx_ms if m in (M_u + M_w)], vtype=GRB.BINARY, name="b")
    
    # Mengde fysisk produksjon
    l = model.addVars([w for w in W_all], lb=0, name="l")
    # Imbalance. Differanse mellom faktisk produksjon og produksjonsforpliktelse
    i = model.addVars([w for w in W_all], lb=0, name="i")

    """
    # Overlappende avvik mellom CM og EAM for retning a
    d_overlap = model.addVars(idx_aw, lb=0, name="d_overlap")
    # Binær variabel for å håndtere overlappende avvik
    eta_overlap = model.addVars(idx_aw, vtype=GRB.BINARY, name="eta_overlap")
    # minimum av a_CM og d_EAM for retning a
    mu = model.addVars(idx_aw, lb=0, name="mu")
    # variabel for minimumsfunksjon. 1 dersom d_EAM < a_CM og 0 ellers
    lam = model.addVars(idx_aw, lb=0, name="lam")
    """


    # --- OBJECTIVE FUNCTION ---s


    nodes = scenario_tree["nodes"]  # fra build_scenario_tree
    # U, V, W: fra build_sets_from_tree(tree)
    #   U: set of u-noder
    #   V: dict u -> set of v-noder
    #   W: dict v -> set of w-noder

    obj = gp.LinExpr()

    for u in U:
        pi_u = nodes[u].cond_prob   # π_u

        # Inneste ledd for gitt u
        term_u = gp.quicksum(
            P[ (m, u) ] * a[m, u] for m in M_u
        )

        # Stage 3
        for v in V[u]:
            pi_v_u = nodes[v].cond_prob   # π_{v|u}

            term_v = gp.quicksum(
                P[ (m, v) ] * a[m, v] for m in M_v
            )

            # Stage 4
            for w in W[v]:
                pi_w_v = nodes[w].cond_prob   # π_{w|v}

                revenue_w = gp.quicksum(
                    P[ (m, w) ] * a[m, w] for m in M_w
                )

                imbalance_w = gp.quicksum(
                    P[ ("imb", w) ] * i[w] for w in W_all
                )

                penalty_w = gp.quicksum(
                    C[ (m, w) ] * d[m, w] for m in M_u + M_w
                )

                term_v += pi_w_v * (revenue_w + imbalance_w - penalty_w)

            term_u += pi_v_u * term_v

        obj += pi_u * term_u

    model.setObjective(obj, GRB.MAXIMIZE)


    # --- PRODUCTION CONSTRAINTS ---

    for w in W_all:
        # 1) l_w <= Q_w
        model.addConstr(
            l[w] <= Q[w],
            name=f"prod_cap[{w}]"
        )


    # --- ACTIVATION CONSTRAINTS ---


    # Aktiveringsgrenser (for alle gyldige (m,s))
    for (m, s) in idx_ms:
        # 1) a_ms <= x_ms
        model.addConstr(
            a[m, s] <= x[m, s],
            name=f"act_le_bid[{m},{s}]"
        )

        # 2) a_ms <= M * delta_ms
        model.addConstr(
            a[m, s] <= BIGM_2 * delta[m, s],
            name=f"act_le_Mdelta[{m},{s}]"
        )

        # 3) a_ms >= x_ms - (1 - M * delta_ms)
        model.addConstr(
            a[m, s] >= x[m, s] - BIGM_2 * (1 - delta[m, s]),
            name=f"act_ge_bid_bigM[{m},{s}]"
        )


    # Set aktiveringsvariabel delta
    for (m, s) in idx_ms:   

        # r_ms - P_ms <= M (1 - delta_ms)
        model.addConstr(
            r[m, s] - P[(m, s)] <= BIGM_1 * (1 - delta[m, s]),
            name=f"act_upper[{m},{s}]"
        )

        # P_ms - r_ms <= M delta_ms - eps
        model.addConstr(
            P[(m, s)] - r[m, s] <= BIGM_1 * delta[m, s] - epsilon,
            name=f"act_lower[{m},{s}]"
        )



    for w in W_all:
        # Ikke aktivert både opp- og nedregulering for EAM i samme scenario
        model.addConstr(
            delta["EAM_up", w] + delta["EAM_down", w] <= 1,
            name=f"no_up_and_down[{w}]"
        )



    # --- NON-ANTICIPATIVITY CONSTRAINTS ---

    # Stage 2 non-anticipativity
    u0 = next(iter(U))             # referansenode
    for m in M_u:
        for u in U:
            if u != u0:
                model.addConstr(x[m, u] == x[m, u0], name=f"NA_x_stage2[{m},{u}]")
                model.addConstr(r[m, u] == r[m, u0], name=f"NA_r_stage2[{m},{u}]")


    # Stage 3 non-anticipativity
    for u in U:
        V_u = V[u]                # alle v-noder som følger u
        v0 = next(iter(V_u))      # referanse-node for v-noder med denne historien

        for m in M_v:
            for v in V_u:
                if v == v0:
                    continue

                # x_{m,v} = x_{m,v0}
                model.addConstr(
                    x[m, v] == x[m, v0],
                    name=f"NA_x_stage3[{m},{u},{v}]"
                )

                # r_{m,v} = r_{m,v0}
                model.addConstr(
                    r[m, v] == r[m, v0],
                    name=f"NA_r_stage3[{m},{u},{v}]"
                )

    # Stage 4 non-anticipativity
    for v in V_all:
        W_v = W[v]                 # alle w-noder som følger v
        w0 = next(iter(W_v))       # referanse-node for denne historien

        for m in M_w:
            for w in W_v:
                if w == w0:
                    continue

                # x_{m,w} = x_{m,w0}
                model.addConstr(
                    x[m, w] == x[m, w0],
                    name=f"NA_x_stage4[{m},{v},{w}]"
                )

                # r_{m,w} = r_{m,w0}
                model.addConstr(
                    r[m, w] == r[m, w0],
                    name=f"NA_r_stage4[{m},{v},{w}]"
                )


    # --- MARKET CONSTRAINTS ---


    # any up or down regulation committed in market 1 must be followed by at least the same amount of up or down bidding in stage 3
    # Bygg W(u) fra V(u) og W(v). W(u) er mengden av alle w-noder som følger u
    W_u = {u: set().union(*(W[v] for v in V[u])) for u in U}


    # Forpliktelse i CM må følges opp i EAM, eller gi straff for brudd på CM
    for u in U:
        for w in W_u[u]:
            model.addConstr(
                d["CM_up", w] >= a["CM_up", u] - x["EAM_up", w]
            )
            model.addConstr(
                d["CM_down", w] >= a["CM_down", u] - x["EAM_down", w],  
            )

    
    # Balansere produksjon og produksjonsforpliktelse
    for v in V_all:
        for w in W[v]:
            # Imbalance definisjon
            model.addConstr(
                i[w] == l[w] - a["DA", v] - a["EAM_up", w] + a["EAM_down", w]
            )
    
    # Avvik i EAM-markedet
    for v in V_all:
        for w in W[v]:
            # EAM up
            model.addConstr(
                d["EAM_up", w] >= a["DA", v] + a["EAM_up", w] - l[w] - BIGM_2*(1 - delta["EAM_up", w])
            )
            model.addConstr(
                d["EAM_up", w] <= BIGM_2 * delta["EAM_up", w]
            )
            model.addConstr(
                d["EAM_up", w] <= a["EAM_up", w]
            )
            # EAM down
            model.addConstr(
                d["EAM_down", w] >= a["DA", v] + a["EAM_down", w] - l[w] - BIGM_2*(1 - delta["EAM_down", w])
            )
            model.addConstr(
                d["EAM_down", w] <= BIGM_2 * delta["EAM_down", w]
            )
            model.addConstr(
                d["EAM_down", w] <= a["EAM_down", w]
            )






            


    # Minimum bid quantity constraints for mFRR markets (CM and EAM)
    for (m, s) in b.keys():
        # Hvis b[m,s] = 1  ->  x[m,s] ≥ MIN_Q
        model.addConstr(
            x[m, s] >= x_mFRR_min * b[m, s],
            name=f"mFRR_min_lb[{m},{s}]"
        )

        # Hvis y_bid[m,s] = 0  ->  x[m,s] ≤ 0
        # (og generelt x[m,s] ≤ BIGM hvis y_bid = 1)
        model.addConstr(
            x[m, s] <= BIGM_2 * b[m, s],
            name=f"mFRR_min_ub[{m},{s}]"
        )


    """
    # Constraining bid price in the EAM markets
    #for w in W_all:
    #    model.addConstr(
    #        r["EAM_up", w] <= r_MAX_EAM_up,
    #        name=f"max_price_EAMup_{w}"
    #    )
    #    model.addConstr(
    #        r["EAM_down", w] <= r_MAX_EAM_down,
    #        name=f"max_price_EAMdown_{w}"
    #    )
    for w in W_all:
        model.addConstr(
            r["EAM_up", w] <= r_MAX_EAM_up,
            name=f"max_price_EAMup_{w}"
        )
        model.addConstr(
            r["EAM_down", w] <= r_MAX_EAM_down,
            name=f"max_price_EAMdown_{w}"
        )
    """

    # Constrain bid price within price interval
    for (m, s) in idx_ms:
        if Pmax[m] >= 0:
            model.addConstr(
                r[m, s] <= Pmax[m]
            )
        


    # --- EVALUATE DETERMINISTIC CM POLICY ---
    if evaluate_deterministic_policy:

        with open(det_policy_file, "r") as f:
            det_policy = json.load(f)

        # fiks x,r (CM) til den deterministiske policyen
        for m in det_policy if m in M_u else []:
            for s in S:
                if (m, s) in x:  # sjekk at paret finnes
                    model.addConstr(x[m, s] == det_policy[m]["x"],
                                    name=f"fix_x_{m}_{s}")
                    model.addConstr(r[m, s] == det_policy[m]["r"],
                                    name=f"fix_r_{m}_{s}")

    # --- EVALUATE MODEL WITH ONLY DA AND EAM ---
    if only_da_and_eam:
        for m in M_u:
            for u in U:
                if (m, u) in x:  # sjekk at paret finnes
                    model.addConstr(x[m, u] == 0,
                                    name=f"fix_x_zero_{m}_{s}")
                    model.addConstr(r[m, u] == 0,
                                    name=f"fix_r_zero_{m}_{s}")
    

    print("Added all constraints, starting to optimize model...")
    # --- OPTIMIZE MODEL ---
    
    model.optimize()
    runtime = model.Runtime
    print(f"Model optimized in {runtime:.2f} seconds.")

    model.optimize()

    print("Model is infeasible or unbounded, computing IIS...")
    model.setParam("DualReductions", 0)
    model.optimize()
    if model.Status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write("model_iis.ilp")
        print("Wrote IIS to model_iis.ilp")
    else:
        print("Model is unbounded or still INF_OR_UNBD")

    # --- PRINT RESULTS ---
    if verbose:
        if evaluate_deterministic_policy:
            utils.print_results_deterministic_policy(model, x, a, r, delta, d, Q, U, V, W, M_u, M_v, M_w)
        else:
            utils.print_results(model, x, r, a, delta, d, Q, U, V, W, M_u, M_v, M_w)

    output_dict = {
        "model": model,
        "x": x,
        "r": r,
        "a": a,
        "delta": delta,
        "d": d,
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





def run_robustness_experiment(
    time_str: str,
    n: int,
    num_runs: int = 20,
    base_seed: int | None = None,
    verbose_runs: bool = False,
    **run_model_kwargs,
):
    """
    Runs `run_model` num_runs times and returns averages and variances
    of objective values and runtimes.
    If base_seed is not None, seeds will be base_seed, base_seed+1, ...
    """

    objectives = []
    runtimes = []

    for k in range(num_runs):
        # Optional: different seeds per run
        seed = None
        if base_seed is not None:
            seed = base_seed + k

        res = run_model(
            time_str=time_str,
            n=n,
            verbose=verbose_runs,
            **run_model_kwargs,
            seed=seed,          # assumes run_model has a seed parameter
        )

        obj = res["objective"]
        rt = res["runtime"]

        objectives.append(obj)
        runtimes.append(rt)

        print(f"[RUN {k}] seed={seed}, obj={obj:.4f}, runtime={rt:.4f}s")

    # Use population statistics or sample statistics as you prefer:
    avg_obj = statistics.fmean(objectives)
    std_obj = statistics.pstdev(objectives)   # population std dev

    avg_rt = statistics.fmean(runtimes)
    std_rt = statistics.pstdev(runtimes)

    results = {
        "num_runs": num_runs,
        "n_scenarios": n,
        "avg_objective": avg_obj,
        "var_objective": std_obj,
        "avg_runtime": avg_rt,
        "var_runtime": std_rt,
        "objectives": objectives,
        "runtimes": runtimes,
    }

    print("\n=== ROBUSTNESS SUMMARY ===")
    print(f"Runs         : {num_runs}")
    print(f"Scenarios n  : {n}")
    print(f"Objective avg: {avg_obj:.4f}")
    print(f"Objective std: {std_obj:.4f}")
    print(f"Runtime avg  : {avg_rt:.4f} s")
    print(f"Runtime std  : {std_rt:.6f} s")

    return results





