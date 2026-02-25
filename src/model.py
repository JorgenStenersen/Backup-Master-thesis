import gurobipy as gp
from gurobipy import GRB
import src.tree as tree
import src.utils as utils
from src.model_container import ModelContainer



def build_model(scenario_tree, global_bounds, mode="extensive"): # Scenario tree is either the full tree for extensive form, or a subtree for progressive hedging

    if mode not in ["extensive", "progressive_hedging"]:
        raise ValueError("Invalid mode. Choose 'extensive' or 'progressive_hedging'.")
    
    model = gp.Model()

    # --- SETS ---

    # Sets of market products
    M_u, M_v, M_w, M = get_market_products()

    # Bygg sett fra treet
    U, V, W, S, L = tree.build_sets_from_tree(scenario_tree)

    # flate mengder for v- og w-noder:
    V_all = set().union(*V.values())
    W_all = set().union(*W.values())
    L_all = set().union(*L.values())

    # bygg indeksmengder (m,s)
    idx_ms, idx_mw = tree.build_index_sets(U=U, V_all=V_all, W_all=W_all, M_u=M_u, M_v=M_v, M_w=M_w, M=M)

    # --- PARAMETERS ---
    
    P = utils.build_price_parameter(scenario_tree)
    Q = utils.build_production_capacity(scenario_tree)
    C = utils.build_cost_parameters(U, V, W, P)


    Pmax_per_market = global_bounds["Pmax_per_market"] # Høyeste pris for hvert produkt
    BIGM_1 = global_bounds["Pmax"] # maksimal pris i inputdata
    BIGM_2 = global_bounds["Qmax"]  # maksimal produksjonskapasitet
    epsilon = 1e-3  # liten verdi for å sikre korrekt aktivering av delta
    x_mFRR_min = 10  # minimum budstørrelse i mFRR-markedet


    # --- VARIABLES ---

    # x_ms: bid quantity
    x = model.addVars(idx_ms, lb=0, vtype=GRB.INTEGER, name="x")
    # r_ms: bid price
    r = model.addVars(idx_ms, lb=-GRB.INFINITY, name="r")
    # δ_ms: 1 hvis budet aktiveres
    delta = model.addVars(idx_ms, vtype=GRB.BINARY, name="delta")
    # a_ms: aktivert kvantum
    a = model.addVars(idx_ms, lb=0, vtype=GRB.INTEGER, name="a")
    # d_mw: avvik fra aktivert kvantum i terminale scenarier
    d = model.addVars(idx_mw, lb=0, ub=BIGM_2, name="d")
    # Binær variabel som indikerer om vi faktisk legger inn et bud (≠ 0)
    b = model.addVars([(m, s) for (m, s) in idx_ms if m in (M_u + M_w)], vtype=GRB.BINARY, name="b")
    
    # Mengde fysisk produksjon
    q = model.addVars([w for w in W_all], lb=0, name="q")
    # Imbalance. Differanse mellom faktisk produksjon og produksjonsforpliktelse
    i = model.addVars([w for w in W_all], lb=0, name="i")



    # --- OBJECTIVE FUNCTION ---
    if mode == "extensive":
        obj = _build_objective_extensive_form(scenario_tree, U, V, W, L, M_u, M_v, M_w, P, C, a, d, i)
    elif mode == "progressive_hedging":
        obj = _build_objective_progressive_hedging(scenario_tree, U, V, W, L, M_u, M_v, M_w, P, C, a, d, i)

    model.setObjective(obj, GRB.MAXIMIZE)

    # --- PRODUCTION CONSTRAINTS ---
    _add_production_constraints(model, q, Q, W_all)

    # --- ACTIVATION CONSTRAINTS ---
    _add_activation_constraints(model, idx_ms, x, a, delta, r, P, BIGM_1, BIGM_2, epsilon)

    # --- MUTUAL EXCLUSION CONSTRAINTS FOR EAM ---
    _add_mutual_exclusion_eam(model, delta, W_all)

    # --- NON-ANTICIPATIVITY CONSTRAINTS ---
    _add_nonanticipativity_constraints(model, U, V, W, V_all, M_u, M_v, M_w, x, r)

    # --- MARKET CONSTRAINTS ---
    _add_market_constraints(model, U, V, W, L, V_all, x, a, d, q, i, delta, BIGM_2)

    # --- MINIMUM BID CONSTRAINTS FOR mFRR MARKETS ---
    _add_min_bid_constraints(model, b, x, x_mFRR_min, BIGM_2)

    # --- PRICE BOUNDS CONSTRAINTS ---
    _add_price_bounds(model, idx_ms, r, Pmax_per_market)




    # --- CREATE MODEL CONTAINER ---
    model_container = ModelContainer(
        model=model,
        vars={
            "x": x,
            "r": r,
            "delta": delta,
            "a": a,
            "d": d,
            "b": b,
            "q": q,
            "i": i
        },
        params={
            "P": P,
            "Q": Q,
            "C": C
        },
        sets={
            "U": U,
            "V": V,
            "W": W,
            "L": L,
            "M_u": M_u,
            "M_v": M_v,
            "M_w": M_w
        }
    )

    return model_container





def _build_objective_extensive_form(scenario_tree, U, V, W, L, M_u, M_v, M_w, P, C, a, d, i):
    nodes = scenario_tree["nodes"]
    obj = gp.LinExpr()

    # extensive: loop over all included U, then V[u], then W[v]
    # scenario: U/V/W er allerede trimmed til én path, så samme loop fungerer, men vi dropper sannsynlighetsvekting (eller setter dem = 1)
    for u in U:
        pi_u = nodes[u].cond_prob # π_u

        # Innerste ledd for gitt u
        term_u = gp.quicksum(
            P[ (m, u) ] * a[m, u] for m in M_u
        )

        # Stage 3
        for v in V[u]:
            pi_v_u = nodes[v].cond_prob# π_{v|u}

            term_v = gp.quicksum(
                P[ (m, v) ] * a[m, v] for m in M_v
            )

            # Stage 4
            for w in W[v]:
                pi_w_v = nodes[w].cond_prob # π_{w|v}

                revenue_w = gp.quicksum(
                    P[ (m, w) ] * a[m, w] for m in M_w
                )
                
                penalty_w = gp.quicksum(
                    C[ (m, w) ] * d[m, w] for m in M_u + M_w
                )

                term_w = revenue_w - penalty_w

                # Stage 5: Imbalance settlement
                for l in L[w]:
                    pi_l_w = nodes[l].cond_prob # π_{l|w}

                    imbalance = P[ ("imb", l) ] * i[w]

                
                    term_w += pi_l_w * imbalance
                    
                term_v += pi_w_v * term_w

            term_u += pi_v_u * term_v

        obj += pi_u * term_u

    return obj


def _build_objective_progressive_hedging(scenario_tree, U, V, W, L, M_u, M_v, M_w, P, C, a, d, i):
    # Implementer
    return


def _add_production_constraints(model, q, Q, W_all):
    # q_w <= Q_w
    for w in W_all:
        model.addConstr(
            q[w] <= Q[w], 
            name=f"prod_cap[{w}]"
        )

def _add_activation_constraints(model, idx_ms, x, a, delta, r, P, BIGM_1, BIGM_2, epsilon):
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



def _add_mutual_exclusion_eam(model, delta, W_all):
    for w in W_all:
        # Ikke aktivert både opp- og nedregulering for EAM i samme scenario
        model.addConstr(
            delta["EAM_up", w] + delta["EAM_down", w] <= 1,
            name=f"no_up_and_down[{w}]"
        )


def _add_nonanticipativity_constraints(model, U, V, W, V_all, M_u, M_v, M_w, x, r):
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


def _add_market_constraints(model, U, V, W, L, V_all, x, a, d, q, i, delta, BIGM_2):
    # any up or down regulation committed in market 1 must be followed by at least the same amount of up or down bidding in stage 3
    # Forpliktelse i CM må følges opp i EAM, eller gi straff for brudd på CM
    for u in U:
        for v in V[u]:
            for w in W[v]:
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
                i[w] == q[w] - a["DA", v] - a["EAM_up", w] + a["EAM_down", w]
            )
    
    # Avvik i EAM-markedet
    for v in V_all:
        for w in W[v]:
            
            # EAM up
            model.addConstr(
                d["EAM_up", w] >= a["DA", v] + a["EAM_up", w] - q[w] - BIGM_2*(1 - delta["EAM_up", w])
            )
            
            model.addConstr(
                d["EAM_up", w] <= BIGM_2 * delta["EAM_up", w]
            )
            model.addConstr(
                d["EAM_up", w] <= a["EAM_up", w]
            )


            # EAM down
            model.addConstr(
                d["EAM_down", w] >= a["DA", v] + a["EAM_down", w] - q[w] - BIGM_2*(1 - delta["EAM_down", w])
            )
            
            model.addConstr(
                d["EAM_down", w] <= BIGM_2 * delta["EAM_down", w]
            )
            model.addConstr(
                d["EAM_down", w] <= a["EAM_down", w]
            )

def _add_min_bid_constraints(model, b, x, x_mFRR_min, BIGM_2):
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

def _add_price_bounds(model, idx_ms, r, Pmax_per_market):
    # Constrain bid price within price interval
    for (m, s) in idx_ms:
        if Pmax_per_market[m] >= 0:
            model.addConstr(
                r[m, s] <= Pmax_per_market[m]
            )







def initialize_run(time_str, n, seed=None):
    # Bygg treet
    full_scenario_tree = tree.build_scenario_tree(time_str, n, seed)

    # Finn verdier for Big M
    P = utils.build_price_parameter(full_scenario_tree)
    Q = utils.build_production_capacity(full_scenario_tree)
    Pmax = max(P.values())
    Qmax = max(Q.values())

    # finn max-pris for hvert marked
    M_u, M_v, M_w, M = get_market_products()

    U, V, W, S, L = tree.build_sets_from_tree(full_scenario_tree)

    V_all = set().union(*V.values())
    W_all = set().union(*W.values())

    idx_ms, idx_mw = tree.build_index_sets(U=U, V_all=V_all, W_all=W_all, M_u=M_u, M_v=M_v, M_w=M_w, M=M)

    Pmax_per_market = {m: max(P[m, s] for s in S if (m, s) in idx_ms) for m in M} # Høyeste pris for hvert produkt

    # Dictionary med alle globale grenseverdier som trengs i modellen
    global_bounds = {"Pmax_per_market": Pmax_per_market, "Pmax": Pmax, "Qmax": Qmax}

    return full_scenario_tree, global_bounds



def get_market_products():
    M_u = ["CM_up", "CM_down"]
    M_v = ["DA"]
    M_w = ["EAM_up", "EAM_down"]
    M = M_u + M_v + M_w

    return M_u, M_v, M_w, M