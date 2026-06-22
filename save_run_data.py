import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


def _avg(x: Iterable[float]) -> float:
    x = list(x) if x is not None else []
    return float(np.mean(x)) if len(x) > 0 else 0.0


def _percent_reduction(baseline: float, other: float) -> float:
    return ((baseline - other) / baseline * 100.0) if baseline > 0 else 0.0


def _safe_json(obj: Any) -> str:
    """Serialize to a compact JSON string (never raises)."""
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False, separators=(",", ":"))


def _get_metrics_averages(counter_dict: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Expects keys like:
      expanded_count_sum
      neighbors_enqueued_sum
      pushed_count_revised_paths_sum
      explored_neighbors_sum
    Each value is a list across repetitions, so we average them.
    """
    return {
        "expanded_avg": _avg(counter_dict.get("expanded_count_sum", [])),
        "neighbors_enq_avg": _avg(counter_dict.get("neighbors_enqueued_sum", [])),
        "pushed_avg": _avg(counter_dict.get("pushed_count_revised_paths_sum", [])),
        "generated_avg": _avg(counter_dict.get("explored_neighbors_sum", [])),
    }


def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_run_results(
    out_dir: Union[str, Path],
    filename_stem: str,
    *,
    # --- identifiers ---
    test_no: int,
    # --- graph config ---
    graph_edges_total: int,
    num_nodes: int,
    edges_added: Optional[int],
    graph_type: str,
    graph_struct: str,
    node_id_format: str,
    cost_mode: str,
    seed: Optional[int] = None,
    # --- SFC config ---
    service_DR: Optional[float] = None,
    functions: Optional[List[Dict[str, Any]]] = None,
    # --- prompt/LLM ---
    prompt_name: str,
    prompt_text: Optional[str] = None,  # store for debug file only (can be big)
    model_name: Optional[str] = None,
    t_list_flat: Optional[Any] = None,   # waypoints list (any json-serializable)
    llm_info: Optional[Dict[str, Any]] = None,
    llm_output_raw: Optional[str] = None,
    query_flat: Optional[str] = None,
    # --- src/goal ---
    src: Optional[Any] = None,
    goal: Optional[Any] = None,
    # --- counters (dict of lists) ---
    NC_A_dict: Dict[str, List[float]] = None,
    NC_LLM_dict: Dict[str, List[float]] = None,
    SRC_goal_dict: Dict[str, List[float]] = None,

    # --- file behavior ---
    stats_csv_name: Optional[str] = None,
    debug_jsonl_name: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Writes:
      1) A stats CSV (1 row per test_no / iteration)
      2) A debug JSONL file (one JSON object per iteration with richer info)

    This mirrors your format but also logs extra config to make runs reproducible.

    Returns: (stats_csv_path, debug_jsonl_path)
    """
    out_dir = _ensure_dir(out_dir)

    if stats_csv_name is None:
        stats_csv_name = f"stats_{filename_stem}_final.csv"
    if debug_jsonl_name is None:
        debug_jsonl_name = f"debug_{filename_stem}.jsonl"

    stats_path = out_dir / stats_csv_name
    debug_path = out_dir / debug_jsonl_name

    # ---- defaults / safety ----
    NC_A_dict = NC_A_dict or {}
    NC_LLM_dict = NC_LLM_dict or {}
    SRC_goal_dict = SRC_goal_dict or {}


    # ---- averages ----
    A = _get_metrics_averages(NC_A_dict)
    L = _get_metrics_averages(NC_LLM_dict)
    S = _get_metrics_averages(SRC_goal_dict)


    # ---- percent reductions vs A (expanded) ----
    pct_LLM_reduction = _percent_reduction(A["expanded_avg"], L["expanded_avg"])
    pct_SRC_reduction = _percent_reduction(A["expanded_avg"], S["expanded_avg"])


    # ---- CSV header (kept close to yours, with a few added identifiers) ----
    header = [
        "Test No.",
        "# Nodes",
        "total_edges",
        "# Edges added",
        "graph_type",
        "graph_struct",
        "node_id_format",
        "cost_mode",
        "seed",
        "service_DR",
        "functions_count",
        "prompt_name",
        "model_name",
        "src",
        "goal",
        "waypoints_count",

        "%LLM_expanded_reduction",
        "%SRC_GOAL_expanded_reduction",


        "AVG_A_neighbors_enqueued",
        "AVG_A_pushed_count_revised_paths",
        "AVG_A_generated_neighbor_count",
        "AVG_A_expanded_count",

        "AVG_LLM_neighbors_enqueued",
        "AVG_LLM_pushed_count_revised_paths",
        "AVG_LLM_generated_neighbor_count",
        "AVG_LLM_expanded_count",

        "AVG_SRC_GOAL_neighbors_enqueued",
        "AVG_SRC_GOAL_pushed_count_revised_paths",
        "AVG_SRC_GOAL_generated_neighbor_count",
        "AVG_SRC_GOAL_expanded_count",
    ]

    waypoints_count = 0
    try:
        waypoints_count = len(t_list_flat) if t_list_flat is not None else 0
    except Exception:
        waypoints_count = 0

    row = [
        test_no,
        num_nodes,
        graph_edges_total,
        edges_added if edges_added is not None else "",
        graph_type,
        graph_struct,
        node_id_format,
        cost_mode,
        seed if seed is not None else "",
        service_DR if service_DR is not None else "",
        len(functions) if functions is not None else "",
        prompt_name,
        model_name if model_name is not None else "",
        src if src is not None else "",
        goal if goal is not None else "",
        waypoints_count,

        pct_LLM_reduction,
        pct_SRC_reduction,


        A["neighbors_enq_avg"],
        A["pushed_avg"],
        A["generated_avg"],
        A["expanded_avg"],

        L["neighbors_enq_avg"],
        L["pushed_avg"],
        L["generated_avg"],
        L["expanded_avg"],

        S["neighbors_enq_avg"],
        S["pushed_avg"],
        S["generated_avg"],
        S["expanded_avg"],

    ]

    # ---- write stats csv ----
    file_exists = stats_path.exists()
    with open(stats_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if (not file_exists) or f.tell() == 0:
            writer.writerow(header)
        writer.writerow(row)

    # ---- write debug jsonl (rich per-iteration record) ----
    debug_obj = {
        "ts": time.time(),
        "test_no": test_no,
        "graph": {
            "type": graph_type,
            "struct": graph_struct,
            "node_id_format": node_id_format,
            "cost_mode": cost_mode,
            "seed": seed,
            "num_nodes": num_nodes,
            "total_edges": graph_edges_total,
            "edges_added": edges_added,
        },
        "sfc": {
            "service_DR": service_DR,
            "functions": functions,
        },
        "prompt": {
            "name": prompt_name,
            "model": model_name,
            "text": prompt_text,      # optional
            "query_flat": query_flat, # optional
        },
        "llm": {
            "waypoints": t_list_flat,
            "info": llm_info,
            "output_raw": llm_output_raw,
        },
        "instance": {"src": src, "goal": goal},
        "averages": {"A": A, "LLM": L, "SRC_GOAL": S, "MOD": M},
        "pct_reductions_vs_A_expanded": {
            "LLM": pct_LLM_reduction,
            "SRC_GOAL": pct_SRC_reduction,
        },
        "raw_counters": {
            "NC_A": NC_A_dict,
            "NC_LLM": NC_LLM_dict,
            "SRC_GOAL": SRC_goal_dict,
        },
    }

    with open(debug_path, mode="a", encoding="utf-8") as f:
        f.write(_safe_json(debug_obj) + "\n")

    return stats_path, debug_path



# # save_run_data.py
# from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Helpers
# -----------------------------
def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _append_row(csv_path: Path, header: List[str], row: List[Any]) -> None:
    _ensure_parent(csv_path)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if (not file_exists) or (csv_path.stat().st_size == 0):
            w.writerow(header)
        w.writerow(row)

def _avg(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0

def _percent_reduction(baseline: float, other: float) -> float:
    return ((baseline - other) / baseline * 100.0) if baseline > 0 else 0.0


# -----------------------------
# Public API
# -----------------------------
@dataclass
class RunFiles:
    per_run_csv: Path
    per_graph_csv: Path
    debug_jsonl: Path


def init_run_files(out_dir: str | Path, filename_stem: str) -> RunFiles:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_csv = out_dir / f"{filename_stem}_per_run.csv"
    per_graph_csv = out_dir / f"{filename_stem}_per_graph.csv"
    debug_jsonl = out_dir / f"{filename_stem}_debug.jsonl"
    return RunFiles(per_run_csv=per_run_csv, per_graph_csv=per_graph_csv, debug_jsonl=debug_jsonl)


def write_per_run_row(
    files: RunFiles,
    *,
    run_id: str,
    timestamp: str,
    overall_iter: int,
    graph_iter: int,
    # graph metadata
    graph_type: str,
    graph_struct: str,
    node_id_format: str,
    cost_mode: str,
    seed: Optional[int],
    num_nodes: int,
    graph_edges_total: int,
    edges_added: Any,
    service_DR: Any,
    functions: Any,
    # prompt/LLM metadata
    prompt_name: str,
    model_name: str,
    t_n_weight: Optional[float],
    src: Any,
    goal: Any,
    # outputs
    NC_A: Tuple[Any, float, Dict[str, Any]],
    NC_LLM: Tuple[Any, float, Dict[str, Any]],
    SRC_goal: Tuple[Any, float, Dict[str, Any]],
    # optional debugging
    query_flat: Optional[str] = None,
    llm_info: Optional[Dict[str, Any]] = None,
    llm_output_raw: Optional[Any] = None,
    t_list_flat: Optional[Any] = None,
) -> None:
    """
    Writes ONE row per inner iteration (per src/goal run).
    NC_* expected format: (path, total_cost, counters_dict)
    counters_dict keys expected: expanded_count, generated_count, enqueued_size, pushed count
    """

    # Normalize counter access (avoid KeyError)
    def cnt(stats: Dict[str, Any], key: str, default: float = 0.0) -> float:
        v = stats.get(key, default)
        try:
            return float(v)
        except Exception:
            return default

    A_path, A_cost, A_stats = NC_A
    L_path, L_cost, L_stats = NC_LLM
    S_path, S_cost, S_stats = SRC_goal

    header = [
        "run_id", "timestamp",
        "overall_iter", "graph_iter",
        "graph_type", "graph_struct", "node_id_format", "cost_mode", "seed",
        "num_nodes", "graph_edges_total", "edges_added",
        "service_DR", "functions",
        "prompt_name", "model_name", "t_n_weight",
        "src", "goal",

        # costs
        "A_cost", "LLM_cost", "SRC_cost",
        "LLM_is_optimal", "SRC_is_optimal",

        # counters A
        "A_expanded", "A_generated", "A_enqueued", "A_pushed_revised",

        # counters LLM
        "LLM_expanded", "LLM_generated", "LLM_enqueued", "LLM_pushed_revised",

        # counters SRC_goal
        "SRC_expanded", "SRC_generated", "SRC_enqueued", "SRC_pushed_revised",

        # optional: path lengths (keeps file small; paths go in debug jsonl)
        "A_path_len", "LLM_path_len", "SRC_path_len",
    ]

    row = [
        run_id, timestamp,
        overall_iter, graph_iter,
        graph_type, graph_struct, node_id_format, cost_mode, seed,
        num_nodes, graph_edges_total, edges_added,
        service_DR, json.dumps(functions, ensure_ascii=False),
        prompt_name, model_name, t_n_weight,
        src, goal,

        float(A_cost), float(L_cost), float(S_cost),
        1 if A_cost == L_cost else 0,
        1 if A_cost == S_cost else 0,

        cnt(A_stats, "expanded_count"),
        cnt(A_stats, "generated_count"),
        cnt(A_stats, "enqueued_size"),
        cnt(A_stats, "pushed count"),

        cnt(L_stats, "expanded_count"),
        cnt(L_stats, "generated_count"),
        cnt(L_stats, "enqueued_size"),
        cnt(L_stats, "pushed count"),

        cnt(S_stats, "expanded_count"),
        cnt(S_stats, "generated_count"),
        cnt(S_stats, "enqueued_size"),
        cnt(S_stats, "pushed count"),

        (len(A_path) if hasattr(A_path, "__len__") else 0),
        (len(L_path) if hasattr(L_path, "__len__") else 0),
        (len(S_path) if hasattr(S_path, "__len__") else 0),
    ]

    _append_row(files.per_run_csv, header, row)

    # --- Debug JSONL (one object per run, keeps giant strings out of CSV) ---
    debug_obj = {
        "run_id": run_id,
        "timestamp": timestamp,
        "overall_iter": overall_iter,
        "graph_iter": graph_iter,
        "src": src,
        "goal": goal,
        "prompt_name": prompt_name,
        "model_name": model_name,
        "t_n_weight": t_n_weight,
        "query_flat": query_flat,
        "t_list_flat": t_list_flat,
        "llm_info": llm_info,
        "llm_output_raw": llm_output_raw,
        "paths": {
            "A": A_path,
            "LLM": L_path,
            "SRC": S_path,
        },
        "costs": {
            "A": A_cost,
            "LLM": L_cost,
            "SRC": S_cost,
        },
        "counters": {
            "A": A_stats,
            "LLM": L_stats,
            "SRC": S_stats,
        },
    }
    _ensure_parent(files.debug_jsonl)
    with files.debug_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(debug_obj, ensure_ascii=False) + "\n")


def write_per_graph_summary_row(
    files: RunFiles,
    *,
    run_id: str,
    timestamp: str,
    overall_iter: int,
    # graph metadata
    graph_type: str,
    graph_struct: str,
    node_id_format: str,
    cost_mode: str,
    seed: Optional[int],
    num_nodes: int,
    graph_edges_total: int,
    edges_added: Any,
    service_DR: Any,
    functions: Any,
    prompt_name: str,
    model_name: str,
    t_n_weight: Optional[float],
    # aggregated arrays (length = itr_graph)
    A_expanded: List[float],
    A_generated: List[float],
    A_enqueued: List[float],
    A_pushed: List[float],
    A_costs: List[float],

    L_expanded: List[float],
    L_generated: List[float],
    L_enqueued: List[float],
    L_pushed: List[float],
    L_costs: List[float],
    L_optimal_flags: List[int],

    S_expanded: List[float],
    S_generated: List[float],
    S_enqueued: List[float],
    S_pushed: List[float],
    S_costs: List[float],
    S_optimal_flags: List[int],
) -> None:
    """
    Writes ONE row per outer iteration (per generated graph).
    """

    A_exp = _avg(A_expanded)
    L_exp = _avg(L_expanded)
    S_exp = _avg(S_expanded)

    header = [
        "run_id", "timestamp",
        "overall_iter",
        "graph_type", "graph_struct", "node_id_format", "cost_mode", "seed",
        "num_nodes", "graph_edges_total", "edges_added",
        "service_DR", "functions",
        "prompt_name", "model_name", "t_n_weight",

        # optimality rates vs A*
        "LLM_optimal_rate",
        "SRC_optimal_rate",

        # percent reductions vs A* (expanded is usually the headline)
        "pct_LLM_expanded_reduction",
        "pct_SRC_expanded_reduction",

        # avg costs
        "AVG_A_cost",
        "AVG_LLM_cost",
        "AVG_SRC_cost",

        # avg counters (A)
        "AVG_A_neighbors_enqueued",
        "AVG_A_pushed_count_revised_paths",
        "AVG_A_generated_neighbor_count",
        "AVG_A_expanded_count",

        # avg counters (LLM)
        "AVG_LLM_neighbors_enqueued",
        "AVG_LLM_pushed_count_revised_paths",
        "AVG_LLM_generated_neighbor_count",
        "AVG_LLM_expanded_count",

        # avg counters (SRC)
        "AVG_SRC_neighbors_enqueued",
        "AVG_SRC_pushed_count_revised_paths",
        "AVG_SRC_generated_neighbor_count",
        "AVG_SRC_expanded_count",
    ]

    row = [
        run_id, timestamp,
        overall_iter,
        graph_type, graph_struct, node_id_format, cost_mode, seed,
        num_nodes, graph_edges_total, edges_added,
        service_DR, json.dumps(functions, ensure_ascii=False),
        prompt_name, model_name, t_n_weight,

        float(np.mean(L_optimal_flags)) if L_optimal_flags else 0.0,
        float(np.mean(S_optimal_flags)) if S_optimal_flags else 0.0,

        _percent_reduction(A_exp, L_exp),
        _percent_reduction(A_exp, S_exp),

        _avg(A_costs),
        _avg(L_costs),
        _avg(S_costs),

        _avg(A_enqueued),
        _avg(A_pushed),
        _avg(A_generated),
        _avg(A_expanded),

        _avg(L_enqueued),
        _avg(L_pushed),
        _avg(L_generated),
        _avg(L_expanded),

        _avg(S_enqueued),
        _avg(S_pushed),
        _avg(S_generated),
        _avg(S_expanded),
    ]

    _append_row(files.per_graph_csv, header, row)
