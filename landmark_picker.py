
import random
import re
import networkx as nx
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple




def _reachable_fraction(G: nx.DiGraph, source: Any, weight: str) -> float:
    """
    For directed graphs: rough measure of how much of the graph is reachable from source.
    Uses Dijkstra reachability (weighted). If graph is undirected, returns 1.0.
    """
    if not G.is_directed():
        return 1.0
    dist = nx.single_source_dijkstra_path_length(G, source, weight=weight)
    return len(dist) / max(1, G.number_of_nodes())


def pick_landmarks_random(
    G: nx.Graph,
    k: int,
    rng,
    seed: Optional[int] = None,
    candidates: Optional[Sequence[Any]] = None,
) -> List[Any]:
    # rng = rng.Random(seed)
    nodes = list(candidates) if candidates is not None else list(G.nodes())
    if k <= 0:
        return []
    if k >= len(nodes):
        return nodes
    return rng.sample(nodes, k)


def pick_landmarks_top_degree(
    G: nx.Graph,
    k: int,
    rng,
    mode: str = "total",          # "total", "in", "out"
    candidates: Optional[Sequence[Any]] = None,
    use_undirected: bool = True,  # for directed graphs, you often want total degree in undirected view
) -> List[Any]:
    if k <= 0:
        return []

    nodes = list(candidates) if candidates is not None else list(G.nodes())
    if not nodes:
        return []

    H = G.to_undirected() if (use_undirected and G.is_directed()) else G

    if mode == "total":
        deg = dict(H.degree(nodes))
    elif mode == "in":
        if not G.is_directed():
            deg = dict(H.degree(nodes))
        else:
            deg = dict(G.in_degree(nodes))
    elif mode == "out":
        if not G.is_directed():
            deg = dict(H.degree(nodes))
        else:
            deg = dict(G.out_degree(nodes))
    else:
        raise ValueError("mode must be one of: 'total', 'in', 'out'")

    # sort high -> low, stable tie-breaker by node id string
    ranked = sorted(nodes, key=lambda n: (deg.get(n, 0), str(n)), reverse=True)
    return ranked[: min(k, len(ranked))]


def pick_landmarks_farthest_point(
    G: nx.Graph,
    k: int,
    rng,
    weight: str = "Cost",
    seed: Optional[int] = None,
    start: Optional[Any] = None,
    candidates: Optional[Sequence[Any]] = None,
    candidate_sample: Optional[int] = None,
    require_reachability_frac: float = 0.0,
) -> List[Any]:
    """
    Greedy k-center / farthest-point landmark selection using weighted shortest-path distances.

    How it works:
      - Choose start landmark (given or random)
      - Maintain minDist[v] = min distance from v to any chosen landmark
      - Next landmark = argmax_v minDist[v]
      - Distances computed via Dijkstra from the newly chosen landmark each iteration

    Notes:
      - For directed graphs, distances are from landmark -> v (forward reachability).
        This pairs naturally with ALT's d(L, v) tables. If you want node->landmark,
        run this on G.reverse(copy=False) instead.
      - candidate_sample: if set (e.g., 2000), it will only consider a sampled subset
        of nodes as possible landmarks (faster, weaker but often fine).
      - require_reachability_frac: for directed graphs, skip start/candidates with
        reachable fraction below this (e.g., 0.2).
    """
    G = G.reverse(copy=False) 
    all_nodes = list(candidates) if candidates is not None else list(G.nodes())
    if k <= 0:
        return []
    if not all_nodes:
        return []

    # Optional sampling of candidate landmark pool (speeds up on huge graphs)
    if candidate_sample is not None and candidate_sample < len(all_nodes):
        pool = rng.sample(all_nodes, candidate_sample)
    else:
        pool = all_nodes

    # Pick a start landmark
    if start is None:
        # try a few times to satisfy reachability constraint if directed
        for _ in range(20):
            s = rng.choice(pool)
            if _reachable_fraction(G, s, weight) >= require_reachability_frac:
                start = s
                break
        if start is None:
            start = rng.choice(pool)

    landmarks: List[Any] = [start]

    # Initialize minDist with distances from first landmark
    dist0 = nx.single_source_dijkstra_path_length(G, start, weight=weight)
    INF = float("inf")
    minDist: Dict[Any, float] = {v: dist0.get(v, INF) for v in all_nodes}

    # Greedily add farthest points
    while len(landmarks) < k:
        # Pick node with maximum minDist (farthest from current landmark set)
        # Restrict choice to pool to avoid picking weird nodes if you sampled
        next_L = max(pool, key=lambda v: minDist.get(v, INF))

        # If everything is unreachable (inf), stop early
        if minDist.get(next_L, INF) == INF:
            break

        # If directed, optionally enforce reachability
        if _reachable_fraction(G, next_L, weight) < require_reachability_frac:
            # remove it from pool and try again
            pool = [v for v in pool if v != next_L]
            if not pool:
                break
            continue

        landmarks.append(next_L)

        # Update minDist using distances from the new landmark
        distL = nx.single_source_dijkstra_path_length(G, next_L, weight=weight)
        for v in all_nodes:
            d = distL.get(v, INF)
            if d < minDist[v]:
                minDist[v] = d

    return landmarks

