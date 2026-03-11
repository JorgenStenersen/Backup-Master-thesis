from gurobipy import GRB
import pandas as pd
import numpy as np


def build_price_parameter(tree):
    """
    Lager P_ms som dictionary:
        P[(m, s)] = clearing price for product m in scenario s
    """

    nodes = tree["nodes"]

    P = {}  # (m, s) -> value

    for s, node in nodes.items():

        # --- Stage 2: CM prices ---
        if node.stage == 2:
            if "CM_up" in node.info:
                P[("CM_up", s)] = node.info["CM_up"]
            if "CM_down" in node.info:
                P[("CM_down", s)] = node.info["CM_down"]

        # --- Stage 3: DA price ---
        elif node.stage == 3:
            if "DA" in node.info:
                P[("DA", s)] = node.info["DA"]

        # --- Stage 4: EAM + wind ---
        elif node.stage == 4:
            if "EAM_up" in node.info:
                P[("EAM_up", s)] = node.info["EAM_up"]
            if "EAM_down" in node.info:
                P[("EAM_down", s)] = node.info["EAM_down"]
            if "imb" in node.info:
                P[("imb", s)] = node.info["imb"]

    return P


def build_production_capacity(tree):
    """
    Lager Q_w for alle terminale scenarier w i stage 4.
    Q_w baseres på node.info["wind_speed"].

    Returnerer:
        Q[w] = production capacity in scenario w
    """

    nodes = tree["nodes"]
    Q = {}

    for w, node in nodes.items():
        if node.stage == 4:
            wind = node.info["wind_speed"]

            prod_cap = wind
            Q[w] = prod_cap

    return Q



def build_cost_parameters(U, V, W, P):
    C = {}  # (m, w) -> cost coefficient
    for u in U:
        cm_up_price = P[("CM_up", u)]  # pris for EAM up i dette w-scenariet
        cm_down_price = P[("CM_down", u)]  # pris for EAM down i dette w-scenariet

        for v in V[u]:  # alle v som følger etter u
            for w in W[v]:  # alle w som følger etter v
                eam_up_price = P[("EAM_up", w)]  # pris for EAM up i dette w-scenariet
                eam_down_price = P[("EAM_down", w)]  # pris for EAM down i dette w-scenariet

                # her definerer vi kost for ALLE markeder i dette terminalscenariet
                C[("CM_up",    w)] = 2.0 * cm_up_price
                C[("CM_down",  w)] = 2.0 * cm_down_price
                C[("EAM_up",   w)] = 2.0 * eam_up_price
                C[("EAM_down", w)] = 2.0 * eam_down_price
    return C



def sort_nodes(node_set):
    """Sort nodes alphabetically by their content-based ID."""
    return sorted(node_set)



def print_results(model_container, max_u=5, max_v_per_u=6, max_w_per_v=6):
    """
    Skriver ut en komprimert oversikt over løsningen.

    max_u         : maks antall u-noder (stage 2) å skrive ut (None = alle)
    max_v_per_u   : maks antall v-noder per u (stage 3)
    max_w_per_v   : maks antall w-noder per v (stage 4)
    """

    model = model_container.model
    x = model_container.vars["x"]
    r = model_container.vars["r"]
    a = model_container.vars["a"]
    delta = model_container.vars["delta"]
    d = model_container.vars["d"]
    i = model_container.vars["i"]
    q_scheduled = model_container.vars["q_scheduled"]
    q_actual = model_container.vars["q_actual"]

    Q = model_container.params["Q"]
    P = model_container.params["P"]

    U = model_container.sets["U"]
    V = model_container.sets["V"]
    W = model_container.sets["W"]
    M_u = model_container.sets["M_u"]
    M_v = model_container.sets["M_v"]
    M_w = model_container.sets["M_w"]


    if model.Status != GRB.OPTIMAL:
        print("Model not solved to optimality. Status:", model.Status)
        return

    print("\n======================")
    print("   OPTIMAL SOLUTION")
    print("======================\n")

    print(f"Objective value: {model.ObjVal:,.4f}\n")

    # Sorter noder for ryddig utskrift
    U_sorted = sort_nodes(U)
    if max_u is not None:
        U_sorted = U_sorted[:max_u]

    # ---------- Stage 2: CM ----------
    print("--- Stage 2 (CM) – non-anticipative across u ---")
    for u in U_sorted:
        print(f"u = {u}:")
        for m in M_u:
            print(
                f"  {m}: x={x[m,u].X:.3f}, "
                f"a={a[m,u].X:.3f}, "
                f"r={r[m,u].X:.3f}, "
                f"δ={int(round(delta[m,u].X))}"
            )
        print()
    print()

    # Vi lagrer hvilke v-er vi skriver ut per u, så vi kan bruke de samme i stage 4
    V_samples = {}

    # ---------- Stage 3: DA ----------
    print("--- Stage 3 (DA) – per u and v ---")
    for u in U_sorted:
        V_u_sorted = sort_nodes(V[u])
        V_sample = V_u_sorted[:max_v_per_u]
        V_samples[u] = V_sample

        print(f"\nParent CM node u = {u}:")
        for v in V_sample:
            for m in M_v:
                print(
                    f"  {m} in {v}: "
                    f"x={x[m,v].X:.3f}, "
                    f"a={a[m,v].X:.3f}, "
                    f"r={r[m,v].X:.3f}, "
                    f"δ={int(round(delta[m,v].X))}"
                )
    print()


    # ---------- Stage 4: EAM ----------
    print("--- Stage 4 (EAM) – child w scenarios per v ---")
    for u in U_sorted:
        for v in V_samples[u]:
            W_v_sorted = sort_nodes(W[v])
            W_sample = W_v_sorted[:max_w_per_v]

            print(f"\nParent scenario v = {v} (from u = {u}):")
            for w in W_sample:
                for m in M_w:
                    print(
                        f"  {m} in {w}: "
                        f"x={x[m,w].X:.3f}, "
                        f"a={a[m,w].X:.3f}, "
                        f"r={r[m,w].X:.3f}, "
                        f"δ={int(round(delta[m,w].X))}, "
                        f"d={d[m,w].X:.3f}, "
                        f"d_CM_u={d['CM_up', w].X:.3f}, "
                        f"d_CM_d={d['CM_down', w].X:.3f}, "
                        f"Q={Q[w]:.3f}, "
                        f"q_sched={q_scheduled[w].X:.3f}, "
                        #f"q_actual={q_actual[w].X:.3f}, "
                        f"i={i[w].X:.3f}, "
                        f"P_imb={P[('imb', w)]:.3f}"
                    )
    print()


    print("=============================================")
    print("            END OF RESULTS")
    print("=============================================\n")



def select_scenarios(n, CM_up, CM_down, DA, EAM_up, EAM_down, wind_speed, seed=None):
    """
    Velger n scenarier tilfeldig og konsistent på tvers av alle lister.
    """

    # Antall scenarier
    m = len(DA)

    # Sjekk at alle lister har samme lengde
    assert all(len(lst) == m for lst in [CM_up, CM_down, EAM_up, EAM_down, wind_speed]), \
        "Alle lister må ha samme lengde"

    if n > m:
        raise ValueError(f"Kan ikke velge {n} scenarier når det bare finnes {m} scenarier.")

    # Bruk en RNG for renere seed-håndtering
    rng = np.random.default_rng(seed)

    # Velg n unike scenarie-indekser fra [0, m-1]
    picked_indices = rng.choice(m, size=n, replace=False)

    # Hjelpefunksjon for å plukke ut verdier
    def pick(lst):
        return [lst[i] for i in picked_indices]

    CM_up_sel      = pick(CM_up)
    CM_down_sel    = pick(CM_down)
    DA_sel         = pick(DA)
    EAM_up_sel     = pick(EAM_up)
    EAM_down_sel   = pick(EAM_down)
    wind_speed_sel = pick(wind_speed)

    # Returnerer også hvilke scenarie-indekser som ble valgt
    return CM_up_sel, CM_down_sel, DA_sel, EAM_up_sel, EAM_down_sel, wind_speed_sel, picked_indices.tolist()