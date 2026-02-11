from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Any, Optional
import src.read as read


@dataclass
class Node:
    name: str
    stage: int
    parent: Optional[str]
    info: Dict[str, Any]      # hvilke stokastiske variabler som blir kjent i noden
    cond_prob: float          # betinget sannsynlighet gitt foreldrenoden


def build_scenario_tree(time_str: str, n:int, seed=None) -> Dict[str, Any]:
    
    CM_up, CM_down, DA, EAM_up, EAM_down, wind_speed, picked_scenario_indices = read.load_parameters_from_parquet(time_str, n, seed)

    print("Read parameters from parquet.")
    print (CM_up, CM_down, DA, EAM_up, EAM_down, wind_speed)

    imb = ["up", "down"] # Imbalance er enten EAM_up eller EAM_down, med 50% sannsynlighet hver

    """
    Bygger scenariotre for:
      - Stage 1: root (før alt er kjent)
      - Stage 2: CM-priser (CM_up, CM_down)
      - Stage 3: DA-pris
      - Stage 4: EAM-priser + vind + imbalance (EAM_up, EAM_down, wind_speed, imb)

    Input kan være lister, numpy-arrays, etc.
    Antall alternativer i hver liste kan være vilkårlig.
    """

    nodes: Dict[str, Node] = {}
    children: Dict[Optional[str], List[str]] = {}

    def add_node(name, stage, parent, info, cond_prob):
        nodes[name] = Node(name, stage, parent, info, cond_prob)
        children.setdefault(parent, []).append(name)

    # --- Rotnode (stage 1) ---
    root = "root"
    add_node(root, stage=1, parent=None, info={}, cond_prob=1.0)
    print("[INFO] Added root node.")
    # --- Stage 2: CM (alle kombinasjoner av CM_up og CM_down) ---
    n_CM_up = len(CM_up)
    n_CM_down = len(CM_down)
    cm_cond_prob = 1.0 / (n_CM_up * n_CM_down)

    stage2_nodes: List[str] = []
    for idx, (p_up, p_down) in enumerate(product(CM_up, CM_down), start=1):
        name = f"u{idx}"
        info = {"CM_up": p_up, "CM_down": p_down}
        add_node(name, stage=2, parent=root, info=info, cond_prob=cm_cond_prob)
        stage2_nodes.append(name)
    print("[INFO] Added stage 2 CM nodes.")

    # --- Stage 3: DA (for hver CM-node alle DA-alternativer) ---
    n_DA = len(DA)
    da_cond_prob = 1.0 / n_DA

    stage3_nodes: List[str] = []
    for parent_u in stage2_nodes:
        for p_da in DA:
            name = f"v{len(stage3_nodes) + 1}"
            info = {"DA": p_da}
            add_node(name, stage=3, parent=parent_u, info=info, cond_prob=da_cond_prob)
            stage3_nodes.append(name)
    print("[INFO] Added stage 3 DA nodes.")

    # --- Stage 4: EAM + vind + imbalance (alle kombinasjoner) ---
    n_EAM_up = len(EAM_up)
    n_EAM_down = len(EAM_down)
    n_wind = len(wind_speed)
    n_imb = len(imb)
    
    leaf_cond_prob = 1.0 / (n_EAM_up * n_EAM_down * n_wind * n_imb)

    leaf_nodes: List[str] = []
    for parent_v in stage3_nodes:
        for p_eup, p_edown, w, i in product(EAM_up, EAM_down, wind_speed, imb):

            # For imbalance, vi antar at den er enten EAM_up eller EAM_down, med 50% sannsynlighet hver
            if i == "up":
                p_imb = p_eup
            elif i == "down":
                p_imb = p_edown

            name = f"w{len(leaf_nodes) + 1}"
            info = {
                "EAM_up": p_eup,
                "EAM_down": p_edown,
                "wind_speed": w,
                "imb": p_imb
            }
            add_node(
                name,
                stage=4,
                parent=parent_v,
                info=info,
                cond_prob=leaf_cond_prob,
            )
            leaf_nodes.append(name)
    print("[INFO] Added stage 4 EAM + wind + imbalance nodes.")
    
    # --- Bygg scenarier (én per løvnode) ---
    scenarios = []
    for leaf in leaf_nodes:
        path = []
        values: Dict[str, Any] = {}
        prob = 1.0
        cur = leaf

        # gå opp treet til roten
        while cur is not None:
            node = nodes[cur]
            prob *= node.cond_prob
            values.update(node.info)
            path.append(cur)
            cur = node.parent

        path.reverse()  # root -> ... -> leaf

        scenarios.append(
            {
                "leaf": leaf,
                "probability": prob,   # total sannsynlighet for scenariet
                "path": path,          # nodene langs denne historien
                "values": values,      # realiserte verdier (CM, DA, EAM, vind)
            }
        )

    tree = {
        "root": root,
        "nodes": nodes,        # dict: navn -> Node
        "children": children,  # dict: parent -> liste med barnenoder
        "leaves": leaf_nodes,
        "scenarios": scenarios,
    }
    return tree


def build_sets_from_tree(tree):
    """
    Input:
        tree: output fra build_scenario_tree()

    Output:
        U: set med alle stage-2 noder
        V: dict: V[u] = set med stage-3 noder barn av u
        W: dict: W[v] = set med stage-4 noder barn av v
        S: hele settet av noder i stage 2, 3 og 4
    """

    nodes = tree["nodes"]
    children = tree["children"]

    # --- 𝒰: scenarier i stage 2 ---
    U = {name for name, n in nodes.items() if n.stage == 2}

    # --- 𝒱(u): scenarier i stage 3 etter u ---
    V = {u: set(children.get(u, [])) for u in U}

    # --- 𝒲(v): scenarier i stage 4 etter v ---
    # Finn alle stage-3 noder:
    V_all = set().union(*V.values())
    W = {v: set(children.get(v, [])) for v in V_all}

    # --- 𝒮 = U ∪ V_all ∪ W_all ---
    W_all = set().union(*W.values()) if W else set()
    S = U.union(V_all).union(W_all)

    return U, V, W, S


def build_index_sets(U, V_all, W_all, M_u, M_v, M_w, M):
    """
    Build index sets for (m,s) and (m,w).

    Returns:
        idx_ms : list of (m, s) for all valid market-stage combinations
        idx_mw : list of (m, w) for all m in M and all w in W_all for d_{m,w}
    """

    idx_ms = []

    # Stage 2: CM markets (m in M_u, s in U)
    for u in U:
        for m in M_u:
            idx_ms.append((m, u))

    # Stage 3: DA market (m in M_v, s in V_all)
    for v in V_all:
        for m in M_v:
            idx_ms.append((m, v))

    # Stage 4: EAM markets (m in M_w, s in W_all)
    for w in W_all:
        for m in M_w:
            idx_ms.append((m, w))

    # d_{m,w}: only scenarios w, but all products m
    idx_mw = []
    for w in W_all:
        for m in M:
            idx_mw.append((m, w))


    return idx_ms, idx_mw