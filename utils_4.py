from __future__ import annotations
import os
from matplotlib.patches import Wedge
import networkx as nx
from pandas import read_csv
import math
from networkx.algorithms.shortest_paths.weighted import _weight_function
from prompts_final import *
from heapq import heappop, heappush
from itertools import count
from collections import deque
import re, ast, os, subprocess, base64, json
# from models import ChatGPT #ASTAR work
from openai import OpenAI
from pydantic import BaseModel
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import csv
from datetime import datetime
from itertools import islice
import numpy as np
from collections import defaultdict
from collections import Counter
import landmark_picker
import time
import re
from collections import deque
# from A_search import *
import pickle
LAYER_NODE_RE = re.compile(r"^L\d+_\d+$")
import json
from pathlib import Path

def append_csv_row(path, row):
    with open(path+".csv", mode="a", newline="", encoding="utf-8") as f:
        print(len(row))
        csv.writer(f).writerow(row)

def valid_nodes(lst):
    return all(
        isinstance(x, int) or
        (isinstance(x, str) and LAYER_NODE_RE.fullmatch(x))
        for x in lst
    )

def get_layer(node, nodes_per_layer=None):
    if isinstance(node, str):
        # expects "L<num>_<num>"
        if node.startswith("L") and "_" in node:
            return int(node[1:node.index("_")])
        return None

    if isinstance(node, int) and nodes_per_layer is not None:
        # 1-based encoding: X*nodes + i, i in 1..nodes
        return (node - 1) // nodes_per_layer

    return None

def pick_valid_src_goal(
    G,
    LayerGraphs,
    nodes: int,
    rng,
    num_layers: int,
    layer_format: bool,
    layered: bool,
    max_tries: int = 20,
    flat = False,
    functions = None,
):
    """
    Returns (source, goal) such that:
      - they exist in G
      - there is a path source -> goal in G
    If not found within max_tries, returns (None, None).
    """

    # If NOT using string node names, define how to pick goal range
    # if not layer_format:
    #     if layered:
    #         start_last_layer = num_layers * nodes
    #         end_last_layer = (num_layers + 1) * nodes - 1
    #     else:
    #         start_last_layer = max(0, nodes // 2)   # was int(nodes-1/2) (buggy)
    #         end_last_layer = nodes - 1

    for _ in range(max_tries):
        if flat:
            source,goal = rng.sample(list(LayerGraphs[0]["layer_graph"].nodes),k=2)
            if layer_format :
                layered_goal = f"L{len(functions)}_{goal[3:]}"
            else:
                layered_goal = (len(functions)) * nodes + goal
        else:
            source = rng.choice(list(LayerGraphs[0]["layer_graph"].nodes))
            goal = rng.choice(list(LayerGraphs[-1]["layer_graph"].nodes))
            layered_goal=None

        # Existence + reachability
        if source in G and goal in G and nx.has_path(G, source, goal):
            if not flat and not nx.has_path(G, source, layered_goal):
                pass
            return source, goal,layered_goal

    return None, None,None

def pick_valid_src_goal_flat_graph(
    G,
    max_tries: int = 20,
):
    """
    Returns (source, goal) such that:
      - they exist in G
      - there is a path source -> goal in G
    If not found within max_tries, returns (None, None).
    """
    for _ in range(max_tries):
        source,goal = random.sample(list(G.nodes),k=2)
        # Existence + reachability
        if source in G and goal in G and nx.has_path(G, source, goal):
            if not nx.has_path(G, source,goal):
                pass
            return source, goal

    return None, None

def longest_path(G,max_tries=50, k_min=5, k_max=8):
    # Works for directed graphs
    largest_comp = max(nx.weakly_connected_components(G), key=len)
    H = G.subgraph(largest_comp)

    # Convert to undirected just for distance calculation
    H_undirected = H.to_undirected()

    # All-pairs shortest paths
    lengths = dict(nx.all_pairs_shortest_path_length(H_undirected))

    # Find the pair with the maximum shortest-path distance
    source, dest, dist = max(
        ((u, v, d) for u, vds in lengths.items() for v, d in vds.items()),
        key=lambda x: x[2]
    )
    return source, dest, dist

def pick_valid_src_goal_component_based(G, max_tries=50, k_min=5, k_max=8):
    # components
    if G.is_directed():
        comps = [list(c) for c in nx.weakly_connected_components(G)]
    else:
        comps = [list(c) for c in nx.connected_components(G)]

    comps = [c for c in comps if len(c) >= 2]
    if not comps:
        return None, None, None

    # pick the largest component
    comp = max(comps, key=len)
    H = G.subgraph(comp)

    # We want the 5th–8th node on the path => distance 4..7 edges from s
    min_dist = k_min - 1
    max_dist = k_max - 1

    for _ in range(max_tries):
        s = random.choice(list(H.nodes))

        # shortest-path distances from s up to max_dist edges
        lengths = nx.single_source_shortest_path_length(H, s, cutoff=max_dist)

        # candidates at distances 4..7 (so path has 5..8 nodes)
        candidates = [v for v, d in lengths.items() if min_dist <= d <= max_dist]
        if not candidates:
            continue

        g = random.choice(candidates)

        # shortest path gives you the actual node sequence
        path = nx.shortest_path(H, s, g)

        # pick the 5th–8th node on this path (index 4..7)
        idx_min = 4
        idx_max = min(7, len(path) - 1)
        mid = path[random.randint(idx_min, idx_max)]

        first = path[0]
        return first, mid, path

    return None, None, None


# def pick_valid_src_goal_component_based(G, max_tries=50):
#     if G.is_directed():
#         comps = [list(c) for c in nx.weakly_connected_components(G)]
#     else:
#         comps = [list(c) for c in nx.connected_components(G)]

#     comps = [c for c in comps if len(c) >= 2]
#     if not comps:
#         return None, None

#     for _ in range(max_tries):
#         comp = random.choice(comps)
#         s = random.choice(comp)

#         # directed reachability still matters
#         reachable = set(nx.descendants(G, s)) if G.is_directed() else set(comp)
#         reachable.discard(s)
#         if not reachable:
#             continue

#         g = random.choice(list(reachable))
#         return s, g

#     return None, None


def pick_src_goal_from_largest_component_long_path(G, k_range=(5, 8), seed=42):
    """
    1) Pick the largest component (weakly connected if directed, connected if undirected).
    2) Find a long shortest-path inside it (approx diameter path via 2 BFS sweeps).
    3) Return (first_node, kth_node, path) where kth_node is the 5th-8th node on that path.

    k_range=(5,8) means choose one of {5,6,7,8} if possible.
    """
    # --- components ---
    if G.is_directed():
        comps = [list(c) for c in nx.weakly_connected_components(G)]
    else:
        comps = [list(c) for c in nx.connected_components(G)]

    comps = [c for c in comps if len(c) >= 2]
    if not comps:
        return None, None, None

    largest_comp = max(comps, key=len)
    H = G.subgraph(largest_comp).copy()

    # --- helper: farthest node by shortest-path distance (within cutoff=None) ---
    def farthest_node(graph, start):
        lengths = nx.single_source_shortest_path_length(graph, start)
        # pick node with maximum distance
        far_node = max(lengths, key=lengths.get)
        return far_node, lengths[far_node]

    # --- two-sweep to get a long shortest path (approx diameter path) ---
    a = random.choice(list(H.nodes))
    b, _ = farthest_node(H, a)
    c, dist_bc = farthest_node(H, b)

    # shortest path from b to c is long (in terms of hops)
    path = nx.shortest_path(H, b, c)
    path_len_nodes = len(path)

    # --- pick the kth node: 5th-8th node means indices 4..7 ---
    min_k, max_k = k_range
    min_idx = min_k - 1
    max_idx = max_k - 1

    if path_len_nodes <= min_idx:
        # path too short to have a 5th node
        return None, None, path

    # clamp max_idx to path end
    max_idx = min(max_idx, path_len_nodes - 1)

    kth_idx = random.randint(min_idx, max_idx)
    first_node = path[0]
    kth_node = path[kth_idx]

    return first_node, kth_node, path

def print_adj_list_with_cost(G, cost_key="cost"):
    for u in G.nodes():
        neighbors = list(G.successors(u))
        if not neighbors:
            continue

        print(f"Node {u} connects to:")
        for v in neighbors:
            cost = G[u][v].get(cost_key, "N/A")
            print(f"  - Node {v} (cost: {cost})")
    # TODO return it

def suboptimal_by_removing_edge(G, source, goal, weight="Cost"):
    opt = nx.shortest_path(G, source, goal, weight=weight)

    # pick an interior edge to remove (avoid disconnecting too often)
    u, v = opt[len(opt)//2 - 1], opt[len(opt)//2]
    
    H = G.copy()
    if H.has_edge(u, v):
        H.remove_edge(u, v)

    sub = nx.shortest_path(H, source, goal, weight=weight)
    return opt, sub, (u, v)
    return top2[1]  # sub-optimal (2nd best)


open_AI_key = ""
client = OpenAI(api_key = open_AI_key)
def get_adj_list_cost(graph):
    # Get the adjacency list
    adjacency_list = {}
    for node in graph.nodes():
        for neighbor in list(graph.successors(node)):
            adjacency_list[node] = (neighbor, graph[node][neighbor]["Cost"])
    return adjacency_list


def get_adj_list(graph):
    # Get the adjacency list
    adjacency_list = {}
    for node in graph.nodes():
        adjacency_list[node] = list(graph.successors(node))
    return adjacency_list


def get_adj_list_bfs(graph, source):
    adjacency_list = {}
    visited = set()
    queue = deque([source])
    visited.add(source)

    while queue:
        node = queue.popleft()

        # successors in BFS order
        neighbors = []
        for nbr in graph.successors(node):
            neighbors.append(nbr)
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)

        adjacency_list[node] = neighbors

    # Optional: include unreachable nodes at the end
    for node in graph.nodes():
        if node not in adjacency_list:
            adjacency_list[node] = list(graph.successors(node))

    return adjacency_list


def path_cost(G, path, weight="weight"):
    return sum(
        G.get_edge_data(path[i], path[i+1]).get(weight, 0)
        for i in range(len(path) - 1)
    )
    
def extract_nodes(json_response):
    # json_response = [{'node': 0, 'function': 1}, {'node': 5, 'function': 2}, {'node': 8, 'function': 3}]
    # nodes = [row['node'] for row in json_response["generated_path"]]
    nodes = [node for node in json_response["node"]]
    
    return nodes

def is_valid_path(G, path):
    return all(G.has_edge(path[i], path[i+1]) for i in range(len(path)-1))




def freeze_layout_in_graph(G, layout="spring", seed=42):
    # Pick ONE layout and keep it consistent across all runs/graphs
    # pos = degree_shell_pos(G)
    # pos = degree_radial_pos(G)
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        raise ValueError("Unknown layout")

    # Store as attributes so it persists on disk
    nx.set_node_attributes(G, {n: float(pos[n][0]) for n in G.nodes}, "x")
    nx.set_node_attributes(G, {n: float(pos[n][1]) for n in G.nodes}, "y")
    return G

import networkx as nx

def freeze_layout_in_graph(G, seed=42):
    # Graphviz sfdp (scales better, less overlap than spring)
    pos = nx.nx_agraph.graphviz_layout(G, prog="sfdp")  # needs pygraphviz
    nx.set_node_attributes(G, {n: float(pos[n][0]) for n in G.nodes}, "x")
    nx.set_node_attributes(G, {n: float(pos[n][1]) for n in G.nodes}, "y")
    return G


def compute_pos_spread(G, method="sfdp", seed=42, k=5.5, iterations=300):
    """
    Returns a pos dict {node: (x,y)} with more spread.
    method: "sfdp" (graphviz) or "spring"
    """
    if method == "sfdp":
        # Requires pygraphviz
        return nx.nx_agraph.graphviz_layout(G, prog="sfdp")
    elif method == "spring":
        # Pure NetworkX fallback (increase k to spread more)
        return nx.spring_layout(G, seed=seed, k=k, iterations=iterations)
    else:
        raise ValueError(f"Unknown method: {method}")



def stacked_layer_pos(
    G: nx.DiGraph,
    num_nodes_per_layer: int,
    layer_attr: str = "Layer",
    phys_attr: str = "node_id",
    base_layout: str = "spring",
    seed: int = 42,
    layer_gap: float = 30.0,   # vertical spacing between layers
    x_scale: float = 8.0,
    y_scale: float = 20.0,
):
    """
    Creates a position dict for a layered graph where each layer is placed
    under the previous one, reusing the same x-position for the same physical node.
    """

    # --- pick one representative node per physical id (prefer layer 0) ---
    # We'll build a "physical" graph layout on the original node ids [0..num_nodes_per_layer-1]
    # by extracting edges from layer 0 (or any single layer).
    H = nx.DiGraph()
    H.add_nodes_from(range(num_nodes_per_layer))

    # Use edges from layer 0 only to avoid duplicates
    for u, v in G.edges():
        lu = G.nodes[u].get(layer_attr, None)
        lv = G.nodes[v].get(layer_attr, None)
        pu = G.nodes[u].get(phys_attr, None)
        pv = G.nodes[v].get(phys_attr, None)

        # Only horizontal edges within the same layer (choose layer 0 as reference)
        if lu == lv == 0 and pu is not None and pv is not None and pu != pv:
            H.add_edge(pu, pv)

    # If layer 0 had no edges (rare), fall back to using any-layer horizontal edges
    if H.number_of_edges() == 0:
        for u, v in G.edges():
            lu = G.nodes[u].get(layer_attr, None)
            lv = G.nodes[v].get(layer_attr, None)
            pu = G.nodes[u].get(phys_attr, None)
            pv = G.nodes[v].get(phys_attr, None)
            if lu == lv and pu is not None and pv is not None and pu != pv:
                H.add_edge(pu, pv)

    # --- compute base positions for physical nodes ---
    if base_layout == "spring":
        base_pos = nx.spring_layout(H, seed=seed)
    elif base_layout == "kamada_kawai":
        base_pos = nx.kamada_kawai_layout(H)
    elif base_layout == "spectral":
        base_pos = nx.spectral_layout(H)
    elif base_layout == "circular":
        base_pos = nx.circular_layout(H)
    else:
        raise ValueError(f"Unknown base_layout: {base_layout}")

    # --- map to layered nodes: same x per physical node, y shifted by layer ---
    pos = {}
    for n, data in G.nodes(data=True):
        layer = int(data.get(layer_attr, 0))
        phys = data.get(phys_attr, None)
        if phys is None:
            # fallback: infer physical id from your encoding i*num_nodes + phys
            phys = int(n) % num_nodes_per_layer

        x0, y0 = base_pos.get(phys, (0.0, 0.0))
        x = x_scale * float(x0)
        y = y_scale * float(y0) - layer * layer_gap
        pos[n] = (x, y)

    return pos

def freeze_pos_into_graph(G, pos, x_attr="x", y_attr="y"):
    for n in G.nodes:
        G.nodes[n][x_attr] = float(pos[n][0])
        G.nodes[n][y_attr] = float(pos[n][1])



# def freeze_layout_in_graph_layered(
#     G,
#     layout="spring",
#     seed=42,
#     layered=False,
#     num_nodes_per_layer=None,
#     layer_gap=6.0,
# ):
#     if not layered:
#         if layout == "spring":
#             pos = nx.spring_layout(G, seed=seed)
#         elif layout == "kamada_kawai":
#             pos = nx.kamada_kawai_layout(G)
#         elif layout == "spectral":
#             pos = nx.spectral_layout(G)
#         else:
#             raise ValueError("Unknown layout")
#     else:
#         if num_nodes_per_layer is None:
#             raise ValueError("num_nodes_per_layer required for layered layout")
#         pos = stacked_layer_pos(
#             G,
#             num_nodes_per_layer=num_nodes_per_layer,
#             layer_gap=layer_gap,
#             seed=seed,
#         )

#     for n in G.nodes:
#         G.nodes[n]["x"] = float(pos[n][0])
#         G.nodes[n]["y"] = float(pos[n][1])

#     return G


def load_pos_from_graph(G):
    return {n: (float(G.nodes[n]["x"]), float(G.nodes[n]["y"])) for n in G.nodes}

import matplotlib.pyplot as plt
import networkx as nx

def draw_astar_graph(G, pos):
    node_colors = []
    node_edgecolors = []
    node_linewidths = []

    for n in G.nodes:
        status = G.nodes[n].get("status", "unseen")
        Type = G.nodes[n].get("type", "None")

        if status == "path":
            node_colors.append("forestgreen")
        elif status == "expanded":
            node_colors.append("goldenrod")
        elif status == "enqueued":
            node_colors.append("wheat")
        elif status == "source":
            node_colors.append("lightsalmon")
        
        else:
            node_colors.append("lightsteelblue")

        # addd stroke around landmark nodes
        if Type == "landmark":
            node_edgecolors.append("black")
            node_linewidths.append(2.5)
        elif Type == "waypoint":
            # node_edgecolors.append("royalblue")
            node_edgecolors.append("plum")
            node_linewidths.append(1.0)
        else:
            node_edgecolors.append("cornsilk")
            node_linewidths.append(1.0)


    edges = list(G.edges)
    edge_colors = []
    for u, v in edges:
        u_status = G.nodes[u].get("status", "unseen")
        v_status = G.nodes[v].get("status", "unseen")
        if (u_status== "path" or u_status=="source" ) and (v_status == "path"):
            edge_colors.append("darkgreen")
        else:
            edge_colors.append("lightgray")
    plt.figure(figsize=(10, 8))
    nx.draw_networkx(
        G,
        pos,
        node_color=node_colors,
        edgecolors=node_edgecolors,
        linewidths=node_linewidths,
        edge_color=edge_colors,
        with_labels=True,
        node_size=600,
        font_size=8,
    )
    plt.axis("off")
    plt.show()


def degree_radial_pos(G, center=(0, 0), power=1.0, jitter=0.02, use_undirected=True):
    """
    High-degree nodes close to center, low-degree farther away.
    power > 1 increases separation of low-degree nodes.
    """
    H = G.to_undirected() if use_undirected and G.is_directed() else G

    deg = dict(H.degree())
    nodes = list(H.nodes())

    dvals = np.array([deg[n] for n in nodes], dtype=float)
    dmin, dmax = float(dvals.min()), float(dvals.max())
    denom = (dmax - dmin) if (dmax - dmin) > 1e-9 else 1.0

    # Normalize degree to [0,1]
    dn = {n: (deg[n] - dmin) / denom for n in nodes}

    # Radius: high degree -> small radius
    r = {n: (1.0 - dn[n])**power for n in nodes}

    # Angles: stable order (by degree, then id)
    ordered = sorted(nodes, key=lambda n: (-deg[n], str(n)))
    angles = {n: 2*np.pi*i/len(ordered) for i, n in enumerate(ordered)}

    cx, cy = center
    pos = {}
    for n in nodes:
        rr = r[n] + np.random.uniform(-jitter, jitter)  # tiny jitter helps overlaps
        th = angles[n]
        pos[n] = (cx + rr*np.cos(th), cy + rr*np.sin(th))
    return pos


def degree_shell_pos(G, shells=5, use_undirected=True):
    H = G.to_undirected() if use_undirected and G.is_directed() else G
    deg = dict(H.degree())
    nodes = list(H.nodes())

    dvals = np.array([deg[n] for n in nodes], dtype=float)
    # split degrees into shells (quantiles)
    cuts = np.quantile(dvals, np.linspace(0, 1, shells+1))

    shell_list = []
    for i in range(shells):
        lo, hi = cuts[i], cuts[i+1]
        # include hi in last bin
        if i == shells-1:
            shell_nodes = [n for n in nodes if (deg[n] >= lo and deg[n] <= hi)]
        else:
            shell_nodes = [n for n in nodes if (deg[n] >= lo and deg[n] < hi)]
        shell_list.append(shell_nodes)

    # Put highest degree shell first (center)
    shell_list = shell_list[::-1]
    return nx.shell_layout(H, nlist=shell_list)


def draw_astar_graph_V2(G, pos, ax=None, title=None, with_labels=False, node_size=80, font_size=6, edge_alpha=0.35):
    if ax is None:
        ax = plt.gca()

    # ---- node colors ----
    node_colors = []
    node_edgecolors = []
    node_linewidths = []
    for n in G.nodes:
        status = G.nodes[n].get("status", "unseen")
        Type = G.nodes[n].get("type", "None")

        if status == "path":
            node_colors.append("forestgreen")
        elif status == "expanded":
            node_colors.append("goldenrod")
        elif status == "enqueued":
            node_colors.append("wheat")
        elif status == "source":
            node_colors.append("lightsalmon")
        
        else:
            node_colors.append("lightsteelblue")

        # addd stroke around landmark nodes
        if Type == "landmark":
            node_edgecolors.append("black")
            node_linewidths.append(2.5)
        elif Type == "waypoint":
            # node_edgecolors.append("royalblue")
            node_edgecolors.append("plum")
            node_linewidths.append(2.5)
        else:
            node_edgecolors.append("cornsilk")
            node_linewidths.append(1.0)

    # ---- edges ----
    edges = list(G.edges)
    edge_colors = []
    edge_styles = []  # "solid" or "dashed" (we'll draw in two passes)

    for u, v in edges:
        lu = G.nodes[u].get("Layer", 0)
        lv = G.nodes[v].get("Layer", 0)

        is_vertical = (G.nodes[u].get("node_id") == G.nodes[v].get("node_id")) and (lv == lu + 1)

        u_status = G.nodes[u].get("status", "unseen")
        v_status = G.nodes[v].get("status", "unseen")

        if (u_status in ("path", "source")) and (v_status == "path"):
            edge_colors.append("darkgreen")
        else:
            edge_colors.append("lightgray" if not is_vertical else "gray")

        edge_styles.append("dashed" if is_vertical else "solid")
    
    solid_edges  = [e for e, s in zip(edges, edge_styles) if s == "solid"]
    solid_colors = [c for c, s in zip(edge_colors, edge_styles) if s == "solid"]

    dash_edges   = [e for e, s in zip(edges, edge_styles) if s == "dashed"]
    dash_colors  = [c for c, s in zip(edge_colors, edge_styles) if s == "dashed"]

    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=solid_edges, edge_color=solid_colors, arrows=True)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=dash_edges,  edge_color=dash_colors,  arrows=True, style="dashed")

    if title:
        ax.set_title(title)

    nx.draw_networkx(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        edgecolors=node_edgecolors,
        linewidths=node_linewidths,
        edge_color=edge_colors,
        alpha=edge_alpha,
        with_labels=with_labels,
        node_size=node_size,
        font_size=font_size,
    )


    ax.axis("off")




def draw_astar_graph_V2(G, pos, ax=None, title=None, with_labels=False, node_size=80, font_size=6, edge_alpha=0.35):
    if ax is None:
        ax = plt.gca()

    # ---- node colors ----
    node_colors = []
    node_edgecolors = []
    node_linewidths = []
    for n in G.nodes:
        status = G.nodes[n].get("status", "unseen")
        Type = G.nodes[n].get("type", "None")

        if status == "path":
            node_colors.append("forestgreen")
        elif status == "expanded":
            node_colors.append("goldenrod")
        elif status == "enqueued":
            node_colors.append("wheat")
        elif status == "source":
            node_colors.append("lightsalmon")
        
        else:
            node_colors.append("lightsteelblue")

        # addd stroke around landmark nodes
        if Type == "landmark":
            node_edgecolors.append("black")
            node_linewidths.append(1.4)
        elif Type == "waypoint":
            # node_edgecolors.append("royalblue")
            node_edgecolors.append("plum")
            node_linewidths.append(1.4)
        else:
            node_edgecolors.append("cornsilk")
            node_linewidths.append(1.4)

    # ---- edges ----
    edges = list(G.edges)
    edge_colors = []
    for u, v in edges:
        u_status = G.nodes[u].get("status", "unseen")
        v_status = G.nodes[v].get("status", "unseen")
        if (u_status== "path" or u_status=="source" ) and (v_status == "path"):
            edge_colors.append("darkgreen")
        else:
            edge_colors.append("lightgray")

    nx.draw_networkx(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        edgecolors=node_edgecolors,
        linewidths=node_linewidths,
        edge_color=edge_colors,
        # alpha=edge_alpha,
        with_labels=False,
        node_size=node_size,
        font_size=font_size,
    )

    if title:
        ax.set_title(title)

    ax.axis("off")


def draw_astar_graph_V3(
    G,
    pos,
    ax=None,
    title=None,
    with_labels=False,
    node_size=80,
    font_size=6,
    edge_alpha=0.35,
):
    import networkx as nx
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    # ---- node styles ----
    style_map = {}
    labels = {}

    for n in G.nodes:
        status = G.nodes[n].get("status", "unseen")
        node_type = G.nodes[n].get("type", "None")

        lw = 0.4
        # lw = 1.4

        if status == "path":
            fill = "#73A2E7"
            edge = "#1b4f9d"
            lw = 1.0
            lw = 2.4
        elif status == "expanded":
            fill = "#ffe169"
            edge = "#1b4f9d"
        elif status == "source":
            labels[n] = "S"
            fill = "#1b4f9d"
            edge = "#1b4f9d"
            lw = 1.0
        elif status == "dest":
            labels[n] = "D"
            fill = "#1b4f9d"
            edge = "#1b4f9d"
            lw = 1.0
        else:
            fill = "#ffffff"
            edge = "#1b4f9d"

        # waypoint styling override
        if node_type == "waypoint":
            fill = "#ff8787"
            lw = 1.0
            lw = 2.4

        style_map[n] = {
            "fill": fill,
            "edge": edge,
            "lw": lw,
            "status": status,
            "type": node_type,
        }

    # ---- split nodes by draw priority ----
    base_nodes = []
    waypoint_nodes = []
    top_nodes = []

    for n in G.nodes:
        status = style_map[n]["status"]
        node_type = style_map[n]["type"]

        if status in {"path", "source", "dest"}:
            top_nodes.append(n)
        elif node_type == "waypoint":
            waypoint_nodes.append(n)
        else:
            base_nodes.append(n)

    def get_node_styles(nodelist):
        return (
            [style_map[n]["fill"] for n in nodelist],
            [style_map[n]["edge"] for n in nodelist],
            [style_map[n]["lw"] for n in nodelist],
        )

    # ---- edges ----
    edges = list(G.edges)
    edge_colors = []
    edge_widths = []

    for u, v in edges:
        u_status = G.nodes[u].get("status", "unseen")
        v_status = G.nodes[v].get("status", "unseen")

        if u_status in {"path", "source", "dest"} and v_status in {"path", "source", "dest"}:
            edge_colors.append("#1b4f9d")
            edge_widths.append(2.5)
        else:
            edge_colors.append("#8d959c")
            edge_widths.append(1.0)

    if title:
        ax.set_title(title)

    # ---- draw edges first ----
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=edge_alpha,
    )

    # ---- draw nodes in layers ----
    if base_nodes:
        c, e, w = get_node_styles(base_nodes)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=base_nodes,
            node_color=c,
            edgecolors=e,
            linewidths=w,
            node_size=node_size,
            ax=ax,
        )

    if waypoint_nodes:
        c, e, w = get_node_styles(waypoint_nodes)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=waypoint_nodes,
            node_color=c,
            edgecolors=e,
            linewidths=w,
            node_size=node_size,
            ax=ax,
        )

    if top_nodes:
        c, e, w = get_node_styles(top_nodes)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=top_nodes,
            node_color=c,
            edgecolors=e,
            linewidths=w,
            node_size=node_size,
            ax=ax,
        )

    # ---- optional full node labels ----
    if with_labels:
        nx.draw_networkx_labels(
            G,
            pos,
            font_size=font_size,
            font_color="black",
            ax=ax,
        )

    # ---- source/dest labels on top ----
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=10,
        font_color="white",
        ax=ax,
    )

    ax.axis("off")

def draw_astar_graph_V4(G, pos, ax=None, title=None, with_labels=False, node_size=80, font_size=6, edge_alpha=0.35):
    if ax is None:
        ax = plt.gca()

    # ---- node colors ----
    node_colors = []
    node_edgecolors = []
    node_linewidths = []
    labels = {}

    for n in G.nodes:
        status = G.nodes[n].get("status", "unseen")
        node_type = G.nodes[n].get("type", "None")

        lw = 0.4
        # Default fill color from status
        if status == "path":
            fill = "#73A2E7"
            edge = "#1b4f9d"
            lw = 2.4
            # lw = 1.0
        elif status == "expanded":
            # fill = "#CBDAF6"
            fill = "#ffe169"
            edge = "#1b4f9d"
            lw = 1.4
        elif status == "source":
            labels[n] = "S"
            fill = "#1b4f9d"
            edge = "#1b4f9d"
            # lw = 2.4
            lw = 1.0
        elif status == "dest":
            labels[n] = "D"
            fill = "#1b4f9d"
            edge = "#1b4f9d"
            # lw = 2.4
            lw = 1.0
        else:
            # fill = "#96b9ed"
            fill = "#ffffff"
            edge = "#1b4f9d"
            # lw = 1.4
            lw = 1.0

        # Override or modify style based on node type
        # if node_type == "landmark":
        #     edge = "black"
        #     lw = 1.4
        if node_type == "waypoint":
            # edge = "#ff8787"
            fill = "#ff8787"
            lw = 2.4
            # lw = 1.0

        node_colors.append(fill)
        node_edgecolors.append(edge)
        node_linewidths.append(lw)


    # ---- edges ----
    edges = list(G.edges)
    edge_colors = []
    edge_widths = []
    for u, v in edges:
        u_status = G.nodes[u].get("status", "unseen")
        v_status = G.nodes[v].get("status", "unseen")
        if (
            u_status in {"path", "source", "dest"} and v_status in {"path", "dest","source"}
        ):
            edge_colors.append("#1b4f9d")
            edge_widths.append(2.5)
        else:
            edge_widths.append(1.0)
            edge_colors.append("#8d959c")

        # if (u_status in {"path", "source"}) and (v_status == "path"):
        #     edge_colors.append("#1b4f9d")
        # else:
        #     edge_colors.append("#8d959c")

    if title:
        ax.set_title(title)

    nx.draw_networkx(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        edgecolors=node_edgecolors,
        linewidths=node_linewidths,
        edge_color=edge_colors,
        width=edge_widths,
        # alpha=edge_alpha,
        with_labels=False,
        node_size=node_size,
        font_size=font_size,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=10,
        font_color="white",  # good contrast for dark nodes
        ax=ax,
    )

    ax.axis("off")


# Cisco_dataset_func
def read_gt(gt_file):
    f = open(gt_file)
    node_gt = {}
    gt_to_nodes = defaultdict(set)
    i = 0
    for line in f:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        i += 1
        parts = line.split(',')
        for nid in parts:
           node_gt[nid] = i
           gt_to_nodes[i].add(nid)
    f.close()
    print( '\n# num gt sets=%d  size(node_to_gt)=%d' % (len(gt_to_nodes), len(node_gt)))
    sizes = Counter()
    ss = []
    for i, s in gt_to_nodes.items():
        sizes[len(s)] += 1
        ss.append(len(s))
    print('# gt sizes histo:', sizes)
    ss.sort()
    ss.reverse()
    print('# group sizes descending: %s\n' % ss)
    sys.stdout.flush()
    return node_gt, gt_to_nodes






def node_attrs(G,nodes):
    colors, edges, widths = [], [], []
    for n in nodes:
        status = G.nodes[n].get("status", "unseen")
        Type = G.nodes[n].get("type", "None")

        if status == "path":
            colors.append("forestgreen")
        elif status == "expanded":
            colors.append("goldenrod")
        elif status == "enqueued":
            colors.append("wheat")
        elif status == "source":
            colors.append("lightsalmon")
        else:
            colors.append("lightsteelblue")

        if Type == "landmark":
            edges.append("black")
            widths.append(1.4)
        elif Type == "waypoint":
            edges.append("plum")
            widths.append(1.6)
        elif Type == "high_deg":
            edges.append("blue")
        else:
            edges.append("cornsilk")
            widths.append(0.8)

    return colors, edges, widths


def draw_astar_graph_V2(G, pos, ax=None, title=None, with_labels=False, node_size=80, font_size=6, edge_alpha=0.35):
    node_colors = []
    node_edgecolors = []
    node_linewidths = []
    # =========================
    # ---- separate waypoint nodes ----
    waypoint_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "waypoint"]
    other_nodes = [n for n in G.nodes if n not in waypoint_nodes]


    # ---- edges ----
    edges = list(G.edges)
    edge_colors = []
    for u, v in edges:
        u_status = G.nodes[u].get("status", "unseen")
        v_status = G.nodes[v].get("status", "unseen")
        if (u_status== "path" or u_status=="source" ) and (v_status == "path"):
            edge_colors.append("darkgreen")
        else:
            edge_colors.append("lightgray")

    # ---- draw edges (transparent) ----
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color=edge_colors,
        alpha=edge_alpha,
        width=0.4,
    )

    # draw non-waypoints first
    colors, edges, widths = node_attrs(G,other_nodes)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=other_nodes,
        ax=ax,
        node_color=colors,
        edgecolors=edges,
        linewidths=widths,
        node_size=node_size,
    )

    # draw waypoints last (on top)
    colors, edges, widths = node_attrs(G,waypoint_nodes)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=waypoint_nodes,
        ax=ax,
        node_color=colors,
        edgecolors=edges,
        linewidths=widths,
        node_size=node_size,
    )

def abstract_graph_with_endpoints_fast(G, source, goal, percentile=95):
    """
    Fast abstraction using BFS instead of all-pairs shortest paths.
    """
    degrees = [G.degree(n) for n in G.nodes()]
    threshold = np.percentile(degrees, percentile)
    high_degree_nodes = {n for n in G.nodes() if G.degree(n) >= threshold}
    
    # Start with high-degree nodes
    result_nodes = set(high_degree_nodes)
    
    # Single BFS from each high-degree node to find bridges
    # (Much faster than all-pairs shortest path)
    for start_node in high_degree_nodes:
        visited = {start_node}
        queue = [(start_node, [start_node])]
        
        while queue:
            node, path = queue.pop(0)
            
            # Stop exploring if we hit another high-degree node
            if node != start_node and node in high_degree_nodes:
                result_nodes.update(path)
                continue
            
            # Don't explore too far
            if len(path) > 10:  # Adjust based on your graph structure
                continue
            
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
    
    abstracted = G.subgraph(result_nodes).copy()
    abstracted.add_node(source)
    abstracted.add_node(goal)
    
    # Connect source
    for neighbor in G.neighbors(source):
        if neighbor in abstracted.nodes():
            abstracted.add_edge(source, neighbor, weight=G[source][neighbor].get('weight', 1))
        else:
            # Find closest abstracted node via BFS
            closest = _bfs_find_closest(G, neighbor, abstracted, max_depth=5)
            if closest:
                weight = nx.shortest_path_length(G, source, closest)
                abstracted.add_edge(source, closest, weight=weight)
    
    # Connect goal
    for neighbor in G.neighbors(goal):
        if neighbor in abstracted.nodes():
            abstracted.add_edge(neighbor, goal, weight=G[neighbor][goal].get('weight', 1))
        else:
            closest = _bfs_find_closest(G, neighbor, abstracted, max_depth=5)
            if closest:
                weight = nx.shortest_path_length(G, neighbor, goal)
                abstracted.add_edge(closest, goal, weight=weight)
    
    return abstracted

def _bfs_find_closest(G, start, abstracted, max_depth=5):
    """Quick BFS to find nearest abstracted node."""
    visited = {start}
    queue = [(start, 0)]
    
    while queue:
        node, depth = queue.pop(0)
        
        if node in abstracted.nodes():
            return node
        
        if depth >= max_depth:
            continue
        
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    
    return None

def abstract_graph_collapse_bridges_fast(G, source, goal, percentile=99):
    """
    Fast version: only connect high-degree nodes that are actually reachable.
    """
    degrees = [G.degree(n) for n in G.nodes()]
    threshold = np.percentile(degrees, percentile)
    high_degree_nodes = {n for n in G.nodes() if G.degree(n) >= threshold}
    
    abstracted = nx.DiGraph() if isinstance(G, nx.DiGraph) else nx.Graph()
    
    # Add high-degree nodes and copy attributes
    for node in high_degree_nodes:
        abstracted.add_node(node)
        for attr, value in G.nodes[node].items():
            abstracted.nodes[node][attr] = value
    
    abstracted.add_node(source)
    abstracted.add_node(goal)
    if 'h' in G.nodes[source]:
        abstracted.nodes[source]['h'] = G.nodes[source]['h']
    if 'h' in G.nodes[goal]:
        abstracted.nodes[goal]['h'] = G.nodes[goal]['h']
    
    # Fast: only connect to neighbors of high-degree nodes
    # Don't compute all-pairs shortest paths
    for hd_node in high_degree_nodes:
        # Look at neighbors of this high-degree node
        for neighbor in G.neighbors(hd_node):
            if neighbor in high_degree_nodes:
                # Direct edge to another high-degree node
                weight = G[hd_node][neighbor].get('weight', 1)
                abstracted.add_edge(hd_node, neighbor, weight=weight)
            else:
                # Find the next high-degree node reachable from neighbor
                next_hd = _bfs_find_next_hd_node(G, neighbor, high_degree_nodes, max_depth=10)
                if next_hd and next_hd != hd_node:
                    # Compute distance only once per pair
                    if not abstracted.has_edge(hd_node, next_hd):
                        dist = nx.shortest_path_length(G, hd_node, next_hd, weight='weight')
                        abstracted.add_edge(hd_node, next_hd, weight=dist)
    
    # Connect source
    for neighbor in G.neighbors(source):
        if neighbor in high_degree_nodes:
            abstracted.add_edge(source, neighbor, weight=G[source][neighbor].get('weight', 1))
        else:
            closest = _find_closest_high_degree(G, neighbor, high_degree_nodes, max_depth=10)
            if closest:
                dist = nx.shortest_path_length(G, source, closest, weight='weight')
                abstracted.add_edge(source, closest, weight=dist)
    
    # Connect goal
    for neighbor in G.neighbors(goal):
        if neighbor in high_degree_nodes:
            abstracted.add_edge(neighbor, goal, weight=G[neighbor][goal].get('weight', 1))
        else:
            closest = _find_closest_high_degree(G, neighbor, high_degree_nodes, max_depth=10)
            if closest:
                dist = nx.shortest_path_length(G, closest, goal, weight='weight')
                abstracted.add_edge(closest, goal, weight=dist)
    
    return abstracted

def _bfs_find_next_hd_node(G, start, high_degree_nodes, max_depth=10):
    """Find the next high-degree node reachable from start."""
    visited = {start}
    queue = [(start, 0)]
    
    while queue:
        node, depth = queue.pop(0)
        
        if node in high_degree_nodes:
            return node
        
        if depth >= max_depth:
            continue
        
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    
    return None

def _find_closest_high_degree(G, start, high_degree_nodes, max_depth=10):
    """Find nearest high-degree node."""
    visited = {start}
    queue = [(start, 0)]
    
    while queue:
        node, depth = queue.pop(0)
        
        if node in high_degree_nodes:
            return node
        
        if depth >= max_depth:
            continue
        
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    
    return None





# =======================
import networkx as nx
from typing import Any, Dict, Iterable, List, Optional, Tuple
import heapq


def build_intersection_only_corridor_graph(
    G: nx.Graph,
    intersection_nodes: Iterable[Any],
    weight: Optional[str] = None,
) -> Tuple[nx.Graph, Dict[Tuple[Any, Any], List[Any]]]:
    """
    Build corridor-collapsed graph between intersection nodes only:
      - kept nodes = intersection_nodes
      - edge u->v exists if walking from u hits next intersection v with only non-kept nodes inside
    """
    GI = nx.DiGraph() if G.is_directed() else nx.Graph()
    keep = set(intersection_nodes)
    GI.add_nodes_from(keep)

    def out_neighbors(u):
        return G.successors(u) if G.is_directed() else G.neighbors(u)

    def edge_cost(u, v):
        if weight is None:
            return 1.0
        if G.is_multigraph():
            return min((edata.get(weight, 1.0) for edata in G[u][v].values()), default=1.0)
        return G[u][v].get(weight, 1.0)

    expand: Dict[Tuple[Any, Any], List[Any]] = {}

    for u in keep:
        if u not in G:
            continue
        for nbr in out_neighbors(u):
            prev, cur = u, nbr
            path = [u, nbr]
            total = edge_cost(u, nbr)
            visited = {u, nbr}

            # corridor walk: only valid if interior nodes behave like degree-2 chains
            while cur not in keep:
                nxts = [x for x in out_neighbors(cur) if x != prev]
                if not nxts:
                    break
                if len(nxts) != 1:
                    break
                nxt = nxts[0]
                if nxt in visited:
                    break
                visited.add(nxt)
                path.append(nxt)
                total += edge_cost(cur, nxt)
                prev, cur = cur, nxt

            v = cur
            if v in keep and v != u:
                if GI.has_edge(u, v):
                    if total < GI[u][v].get("weight", float("inf")):
                        GI[u][v]["weight"] = total
                        expand[(u, v)] = path
                else:
                    GI.add_edge(u, v, weight=total)
                    expand[(u, v)] = path

    return GI, expand


def _edge_weight(G: nx.Graph, u: Any, v: Any, weight: Optional[str]) -> float:
    if weight is None:
        return 1.0
    if G.is_multigraph():
        return min((edata.get(weight, 1.0) for edata in G[u][v].values()), default=1.0)
    return G[u][v].get(weight, 1.0)


def connect_endpoint_to_intersections(
    G: nx.Graph,
    GI: nx.Graph,
    expand: Dict[Tuple[Any, Any], List[Any]],
    endpoint: Any,
    intersections: Iterable[Any],
    weight: Optional[str] = None,
    k: int = 3,                      # connect to the first k intersections found
    max_cost: Optional[float] = None, # optional radius cutoff
    mode: str = "out",               # "out" (src->ints), "in" (ints->goal), "both"
) -> None:
    """
    Robustly attach an endpoint using Dijkstra/BFS until we hit intersection nodes.
    - For directed graphs:
        mode="out": follow outgoing edges from endpoint (good for src)
        mode="in":  follow incoming reachability (good for goal), implemented via reverse graph
        mode="both": do both
    Updates GI and expand in-place.

    Adds edges endpoint->i (or i->endpoint) with weight = shortest path cost in original graph,
    and expand[(u,v)] = original shortest path node list.
    """

    ints = set(intersections)
    GI.add_node(endpoint)

    def run_search(Gr: nx.Graph, forward: bool):
        if endpoint not in Gr:
            return

        # Dijkstra (or BFS if weight is None) with early stopping on k targets
        pq: List[Tuple[float, Any]] = [(0.0, endpoint)]
        dist: Dict[Any, float] = {endpoint: 0.0}
        parent: Dict[Any, Any] = {endpoint: None}
        found: List[Any] = []

        while pq and len(found) < k:
            d, u = heapq.heappop(pq)
            if d != dist.get(u, float("inf")):
                continue
            if max_cost is not None and d > max_cost:
                break

            # if we reached an intersection (not counting the endpoint itself)
            if u in ints and u != endpoint:
                found.append(u)
                # don't return immediately; continue until we have k (or pq empty)
                # Note: we do NOT expand from this node if you want "first encountered intersections only"
                # But keeping expansion is fine; we still early-stop on k.
                # If you want strictly "first layer" intersections, uncomment the next line:
                # continue

            for v in (Gr.successors(u) if Gr.is_directed() else Gr.neighbors(u)):
                nd = d + _edge_weight(Gr, u, v, weight)
                if max_cost is not None and nd > max_cost:
                    continue
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))

        def reconstruct_path(target: Any) -> List[Any]:
            path = []
            cur = target
            while cur is not None:
                path.append(cur)
                cur = parent.get(cur)
            path.reverse()
            return path

        for i in found:
            cost = dist[i]
            path = reconstruct_path(i)

            if GI.is_directed():
                u_, v_ = (endpoint, i) if forward else (i, endpoint)
            else:
                u_, v_ = endpoint, i

            # if we searched on reversed graph, the reconstructed path is reversed wrt original direction
            if not forward:
                path = list(reversed(path))

            # add/relax
            if GI.has_edge(u_, v_):
                if cost < GI[u_][v_].get("weight", float("inf")):
                    GI[u_][v_]["weight"] = cost
                    expand[(u_, v_)] = path
            else:
                GI.add_edge(u_, v_, weight=cost)
                expand[(u_, v_)] = path

    if mode in ("out", "both"):
        run_search(G, forward=True)

    if mode in ("in", "both"):
        if not G.is_directed():
            # undirected: "in" is same as "out"
            run_search(G, forward=False)  # forward flag doesn't matter for undirected GI edges
        else:
            Grev = G.reverse(copy=False)
            run_search(Grev, forward=False)


def build_abstract_graph_with_endpoints(
    G: nx.Graph,
    cutoff: int,
    source: Any,
    goal: Any,
    weight: Optional[str] = None,
    k_attach: int = 3,
) -> Tuple[nx.Graph, Dict[Tuple[Any, Any], List[Any]], List[Any]]:
    """
    1) intersection_nodes from degree cutoff
    2) build intersection-only corridor graph
    3) attach source (outgoing) + goal (incoming) robustly via Dijkstra
    """
    intersection_nodes = [n for n, d in G.degree() if d >= cutoff]

    GI, expand = build_intersection_only_corridor_graph(G, intersection_nodes, weight=weight)

    # Attach source: src -> intersections it can reach
    connect_endpoint_to_intersections(
        G, GI, expand,
        endpoint=source,
        intersections=intersection_nodes,
        weight=weight,
        k=k_attach,
        mode="out",
    )

    # Attach goal: intersections -> goal (for directed graphs, use incoming reachability)
    connect_endpoint_to_intersections(
        G, GI, expand,
        endpoint=goal,
        intersections=intersection_nodes,
        weight=weight,
        k=k_attach,
        mode="in",
    )

    return GI, expand, intersection_nodes


# =======================
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _safe_json(obj: Any) -> Any:
    """
    Make objects JSON-serializable (e.g., numpy types, sets, Path).
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    if isinstance(obj, set):
        return [_safe_json(x) for x in sorted(obj)]
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}

    # Try best-effort conversion
    try:
        return obj.item()  # numpy scalar
    except Exception:
        return str(obj)


def atomic_write_json(path: Path, data: Dict[str, Any], indent: int = 2) -> None:
    """
    Atomic JSON write: write to .tmp then replace.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    payload = _safe_json(data)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False, sort_keys=False)

    os.replace(tmp, path)


@dataclass
class RunLogger:
    json_path: Path
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.json_path = Path(self.json_path)
        if self.json_path.exists():
            try:
                self.data = json.loads(self.json_path.read_text(encoding="utf-8"))
            except Exception:
                # If corrupted, start fresh but keep a note
                self.data = {"_note": "Previous JSON could not be read; restarted."}

        # Ensure structure
        self.data.setdefault("meta", {})
        self.data["meta"].setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        self.data["meta"].setdefault("updated_at", datetime.now().isoformat(timespec="seconds"))
        self.data.setdefault("graphs", {})
        self.data.setdefault("runs", {})

    def register_graph(
        self,
        graph_key: str,
        graph_file: Optional[str] = None,
        graph_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {}
        if graph_file is not None:
            entry["file"] = graph_file
        if graph_info:
            entry.update(graph_info)

        # Don’t overwrite if already registered (unless you want to)
        self.data["graphs"].setdefault(graph_key, entry)
        self.flush()

    def add_run(self, run_key: str, run_payload: Dict[str, Any]) -> None:
        self.data["runs"][run_key] = run_payload
        self.flush()

    def flush(self) -> None:
        self.data["meta"]["updated_at"] = datetime.now().isoformat(timespec="seconds")
        atomic_write_json(self.json_path, self.data, indent=2)




def graph_key_for(size: int, version_i: int, g_type=None):
    return f"size_{size}_ver_{version_i}_{g_type}"



def load_registry(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return {"graphs": {}, "src_dest_sets": {}}

def save_registry(path: Path, registry: dict):
    path.write_text(json.dumps(registry, indent=2))


def load_graph_from_registry(registry, gkey):
    gpath = registry["graphs"][gkey]["file"]
    with open(gpath, "rb") as f:
        G = pickle.load(f)
        
    # G = nx.read_edgelist(gpath, data=(("Cost", int),), create_using=nx.DiGraph())
    # G = nx.read_gpickle(gpath)
    # If your saved graphs are undirected sometimes, adjust create_using accordingly.
    return G


def load_snap_road_network(filepath, directed=True,type_="other",multigraph = False):
    """
    Load Stanford Large Network Dataset (SNAP) road network from TXT file.
    Format: edge list with comments starting with '#'
    
    Example file (roadNet-TX.txt):
        # Directed graph...
        # FromNodeId	ToNodeId
        0	1
        0	2
        1	0
        ...
    
    Args:
        filepath: Path to the .txt file (e.g., "roadNet-TX.txt")
        directed: If True, create DiGraph; else undirected Graph
    
    Returns:
        NetworkX graph object
    """
    # print(f"Loading SNAP road network from: {filepath}")
    if type_== "colt_tel":
        node_ids = set()
        edges = []

        in_node = False
        in_edge = False
        cur_node_id = None
        cur_source = None
        cur_target = None

        def finalize_node():
            nonlocal cur_node_id
            if cur_node_id is not None:
                node_ids.add(cur_node_id)
            cur_node_id = None

        def finalize_edge():
            nonlocal cur_source, cur_target
            if cur_source is not None and cur_target is not None:
                edges.append((cur_source, cur_target))
            cur_source = None
            cur_target = None

        with open(filepath, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()

                if not line or line.startswith("#"):
                    continue

                # Enter blocks
                if line.startswith("node ["):
                    in_node = True
                    cur_node_id = None
                    continue

                if line.startswith("edge ["):
                    in_edge = True
                    cur_source = None
                    cur_target = None
                    continue

                # Exit blocks
                if line == "]":
                    if in_node:
                        finalize_node()
                        in_node = False
                    elif in_edge:
                        finalize_edge()
                        in_edge = False
                    continue

                # Parse fields inside node block
                if in_node:
                    # Example: id 123
                    if line.startswith("id "):
                        cur_node_id = int(line.split()[1])
                    continue

                # Parse fields inside edge block
                if in_edge:
                    # Example: source 0
                    if line.startswith("source "):
                        cur_source = int(line.split()[1])
                    elif line.startswith("target "):
                        cur_target = int(line.split()[1])
                    continue

        # Build SIMPLE graph and ignore duplicates
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(sorted(node_ids))

        seen = set()
        if directed:
            for u, v in edges:
                if (u, v) in seen:
                    continue
                seen.add((u, v))
                G.add_edge(u, v)
                G.add_edge(v, u)
        else:
            for u, v in edges:
                a, b = (u, v) if u <= v else (v, u)
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                G.add_edge(a, b)

        return G
    elif type_ == "real":
        G = nx.DiGraph() if directed else nx.Graph()

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = list(map(int, line.split()))
                u = parts[0]
                neighbors = parts[1:]

                for v in neighbors:
                    G.add_edge(u, v)
        largest_wcc = max(nx.weakly_connected_components(G), key=len)
        G_main = G.subgraph(largest_wcc).copy()
        return G_main
    else:
        try:
            # Read the file, skipping comment lines (starting with #)
            G = nx.DiGraph() if directed else nx.Graph()
            
            with open(filepath, 'r') as f:
                edge_count = 0
                for line in f:
                    # Skip comment lines and empty lines
                    if line.startswith('#') or line.strip() == '':
                        continue
                    
                    # Parse edge
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        try:
                            from_node = int(parts[0])
                            to_node = int(parts[1])
                            G.add_edge(from_node, to_node)
                            edge_count += 1
                        except ValueError:
                            # Skip lines that can't be parsed
                            continue
                    else:
                        parts = line.strip().split(' ')
                        if len(parts) >= 2:
                            try:
                                from_node = int(parts[0])
                                to_node = int(parts[1])
                                G.add_edge(from_node, to_node)
                                edge_count += 1
                            except ValueError:
                                # Skip lines that can't be parsed
                                continue

            
            print(f"Loaded network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G
        
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}")
            return None
        except Exception as e:
            print(f"Error loading SNAP network: {e}")
            return None
