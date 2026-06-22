from utils_4 import *
from test_graph_generation import *

import os
import networkx as nx
from pandas import read_csv
import math
from networkx.algorithms.shortest_paths.weighted import _weight_function
from prompt import *
from prompt_test import *
from prompt_basic import *
from prompts_v2 import *
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
import pickle


class SFCGraphBuilder:
    def __init__(self, functions ,service_DR, graph_path =None):
        """
        functions: [ {'ID':func_ID, 'req': resource_req},{...}...]
        service_DR: int

        """
        self.graph_path = graph_path
        self.functions = functions #[ {function_id: processing_requirement}, ... ]
        self.service_DR = service_DR
        self.llm = 'gpt'
        # self.model = ChatGPT(method=self.GPT_LLMASTAR_METHOD, sysprompt="", example=None) #ASTAR work
        
        # graph, gateways
        self.layers = []

    def get_info(self):
        print("Graph info:")
        print("Number of nodes:", self.G.number_of_nodes())
        print("Number of edges:", self.G.number_of_edges())
        degrees = [val for (node, val) in self.G.degree()]
        print("max, min, average degree:", max(degrees), min(degrees), sum(degrees)/len(degrees))

    def read_graph(self):
        topo_path = os.path.join(self.graph_path, 'topo.csv')
        node_capacity_path = os.path.join(self.graph_path, 'node_capacity.csv')
        topo_df = read_csv(topo_path)
        node_capacity_df = read_csv(node_capacity_path)
        topo_df['Bandwidth'] = topo_df['Bandwidth'].astype(float) # / 1e9
        node_capacity_df['Capacity'] = node_capacity_df['Capacity'].astype(float) # / 1e9
        self.G = nx.DiGraph()
        for idx, row in node_capacity_df.iterrows():
            self.G.add_node(int(row['NodeID']), Capacity=row['Capacity'], Delay=row['Delay'], Cost=row['Cost'], UtilRate = 0)
        for idx, row in topo_df.iterrows():
            self.G.add_edge(int(row['SourceID']), int(row['DestinationID']), Bandwidth=row['Bandwidth'], Delay=row['Delay'], \
                        Cost=row['Cost'], UtilRate = 0)
        # return self.G

    def create_BA_model_graph(self,nodes,edges,seed=42):
        # https://www.geeksforgeeks.org/dsa/barabasi-albert-graph-scale-free-models/
        self.physical_graph= nx.barabasi_albert_graph(nodes,edges,seed)
        # print("number of nodes: ",nodes)
        # convert to directed graph

    def create_graph(self,nodes,edges,rng,type_="BA",seed=42,k =30):
        """create graphs, possible types:
            Supported types:
                BA        : Barabási–Albert (scale-free, hubs)
                ER        : Erdős–Rényi (random)
                SBM       : Stochastic Block Model (community structure)
                SFN       : Scale-Free Network (directed)
                COMPLETE  : Complete graph
                STAR      : Star graph
                CLCL      : Clustered / Watts–Strogatz small-world
                llp       : lollipop
                road_network: 
            """
        if (type_=="BA"):
            # create graph
            self.physical_graph= nx.barabasi_albert_graph(nodes,edges,seed)
        elif type_ == "ER":
            # Erdős–Rényi: random connectivity
            # edges interpreted as expected average degree
            if edges is None:
                p = 0.1 #if no edge parameter is proivded, default to 10% chance that any two nodes are connected
            else:
                p = min(1.0, edges / nodes)
            self.physical_graph = nx.erdos_renyi_graph(
                n=nodes,
                p=p,
                seed=seed
            )
        elif type_=="road_network":
            self.physical_graph = load_snap_road_network("/Users/noufabbasi/Downloads/roadNet-TX.txt")

            nodes_ = list(self.physical_graph.nodes)
            start_node = nodes_[rng.randrange(len(nodes_))]

            self.physical_graph = self.get_subset(self.physical_graph,start_node,k=k,num_nodes=nodes)
            print(f"there are {len(self.physical_graph.nodes)} in the subset with k = {k}")


        elif type_=="advogato":
            self.physical_graph = load_snap_road_network("/Users/noufabbasi/Library/Mobile  Documents/com~apple~CloudDocs/khalifa_2023-/Thesis/IRL_summer_2025/ToT_/SFC_LLM/network-corpus/networks/advogato.txt")
            # self.physical_graph = load_snap_road_network("/Users/noufabbasi/Library/Mobile Documents/com~apple~CloudDocs/khalifa_2023-/Thesis/IRL_summer_2025/ToT_/SFC_LLM/network-corpus/networks/AS-oregon-1.txt")
            # self.physical_graph = load_snap_road_network("/Users/noufabbasi/Library/Mobile Documents/com~apple~CloudDocs/khalifa_2023-/Thesis/IRL_summer_2025/ToT_/SFC_LLM/network-corpus/networks/BioGrid-Co-Localization.txt")
            self.physical_graph = self.get_subset(self.physical_graph , 10,k=k)
            print(f"there are {len(self.physical_graph.nodes)} in the subset with k = {k}")
        elif type_ == "colt_tel":
            self.physical_graph = load_snap_road_network("/Users/noufabbasi/Library/Mobile Documents/com~apple~CloudDocs/khalifa_2023-/Thesis/IRL_summer_2025/ToT_/SFC_LLM/network-corpus/colt_tel.txt",type_=type_)
        elif type_ == "real":
            self.physical_graph = load_snap_road_network("/Users/noufabbasi/Library/Mobile Documents/com~apple~CloudDocs/khalifa_2023-/Thesis/IRL_summer_2025/ToT_/SFC_LLM/real/fungi/graphs/basidiomycetes_lbc_net.txt",type_=type_)
        elif type_ == "llp":
            self.physical_graph = nx.lollipop_graph(500, 500)
            nx.path_graph(1000)
        elif type_ == "SBM":
            # Stochastic Block Model: communities
            num_blocks = 3
            block_size = nodes // num_blocks
            sizes = [block_size] * num_blocks
            sizes[-1] += nodes - sum(sizes)

            p_in = 0.3
            p_out = 0.05

            probs = [[p_in if i == j else p_out
                    for j in range(num_blocks)]
                    for i in range(num_blocks)]

            self.physical_graph = nx.stochastic_block_model(
                sizes,
                probs,
                seed=seed
            )

        elif type_ == "SFN":
            # Scale-Free Network (directed)
            self.physical_graph = nx.scale_free_graph(
                n=nodes,
                seed=seed
            ).to_undirected()

        elif type_ == "COMPLETE":
            # Fully connected graph
            self.physical_graph = nx.complete_graph(nodes)

        elif type_ == "STAR":
            # Star graph (1 hub, N-1 leaves)
            self.physical_graph = nx.star_graph(nodes - 1)

        elif type_ == "CLCL":
            # Clustered / small-world (Watts–Strogatz)
            # edges = k = neighbors per node
            if edges is None:
                edges = max(2, nodes // 10)
            self.physical_graph = nx.watts_strogatz_graph(
                n=nodes,
                k=edges,
                p=0.1,
                seed=seed
            )
        nx.set_node_attributes(self.physical_graph, "none","type")
        
    def conver_to_weighted_graph(self,cost,rng,seed = 42):

        for node_id, attrs in self.physical_graph.nodes(data=True):
            attrs["capacity"] = self.physical_graph.degree(node_id) * 100

        # ========= node weights
        # normalize node cost into [0.1, 1.0] (avoid extreme tiny values)
        caps = [self.physical_graph.nodes[n]["capacity"] for n in self.physical_graph.nodes()]
        cmin, cmax = min(caps), max(caps)


        # def norm_inv(x, xmin, xmax, lo=50, hi=100):
        def norm_inv(x, xmin, xmax, lo=0.1, hi=1.0):
            # bigger x => smaller cost
            if xmax == xmin:
                return (lo + hi) / 2
            z = (x - xmin) / (xmax - xmin)     # 0..1
            inv = 1.0 - z                      # 1..0
            return lo + inv * (hi - lo)
        

        for n in self.physical_graph.nodes():
            cap = self.physical_graph.nodes[n]["capacity"]
            if cost == "constant":
                self.physical_graph.nodes[n]["Cost"] = 1
            else:
                self.physical_graph.nodes[n]["Cost"] = norm_inv(cap, cmin, cmax)

        # ========= edge weights -> tied to bandwidth
        bws = []
        for u, v, w in self.physical_graph.edges(data=True):
            w["Bandwidth"] = rng.randint(1000, 5000)
            bws.append(w["Bandwidth"])

        bwmin, bwmax = min(bws), max(bws)

        for u, v, w in self.physical_graph.edges(data=True):
            bw = w["Bandwidth"]
            if cost == "constant":
                w["Cost"] = 1
            else:
                w["Cost"] = norm_inv(bw, bwmin, bwmax)

        
        self.physical_graph = self.physical_graph.to_directed()
        self.G = self.physical_graph


    def get_subset(self,G, start_node, k=2,num_nodes=10000):
        # # Get all nodes within k hops
        # visited = {start_node}
        # current_layer = {start_node}
        
        # print(num_nodes)
        # while len(visited)<num_nodes:
        #     # for hop in range(k):
        #         next_layer = set()
        #         for node in current_layer:
        #             # Add successors (outgoing edges)
        #             next_layer.update(G.successors(node))
        #             # Add predecessors (incoming edges) for undirected view
        #             if not G.is_directed():
        #                 next_layer.update(G.predecessors(node))
                
        #         visited.update(next_layer)
        #         current_layer = next_layer

        visited = {start_node}
        current_layer = {start_node}

        while len(visited) < num_nodes:
            next_layer = set()

            for node in current_layer:
                next_layer.update(G.successors(node))
                if not G.is_directed():
                    next_layer.update(G.predecessors(node))

            # Only consider new nodes
            new_nodes = next_layer - visited

            if not new_nodes:
                break  # graph exhausted

            # Add only as many as needed
            remaining = num_nodes - len(visited)
            nodes_to_add = set(list(new_nodes)[:remaining])

            visited.update(nodes_to_add)
            current_layer = nodes_to_add
        
        # Create induced subgraph
        subgraph = G.subgraph(visited).copy()

        # relabel nodes to 0..n-1
        mapping = {node: i for i, node in enumerate(subgraph.nodes())}
        subgraph = nx.relabel_nodes(subgraph, mapping)

        return subgraph
    def print_graph(self):
        nx.draw(self.G, with_labels=True)
        plt.show()

    def calculate_heuristic_accuracy(self, nodes_evaluated):
        """
        For each evaluated node n:
        - h(n)  = heuristic estimate from self.h(n)
        - h*(n) = true cost-to-go stored in G.nodes[n]["Cost"]

        Notes:
            heuristic accuracy -> error(n)=∣h(n)−true cost∣
            checks
                Is my heuristic informative?
                Is it tight or loose?
                How much does it underestimate the real cost?

            
        """

        errors = []

        for n in nodes_evaluated:
            # Skip nodes missing ground-truth cost
            if "Cost" not in self.BA_G.nodes[n]:
                continue

            true_cost = self.BA_G.nodes[n]["Cost"]          # h*(n)
            heuristic = self.h(n)                         # h(n)

            if true_cost is None:
                continue

            absolute_error = abs(heuristic - true_cost)

            relative_error = absolute_error / max(true_cost, 1e-9)
            errors.append({
                "absolute": absolute_error,
                "relative": relative_error,
            })

        violations = [
            n for n in nodes_evaluated
            if self.h(n) > self.BA_G.nodes[n]["Cost"]
        ]

        if not errors:
            return None

        return {
            # off by X units, lower is better
            "mean_absolute_error": np.mean([e["absolute"] for e in errors]),
            # Mean Relative Error (MRE)
            "mean_relative_error": np.mean([e["relative"] for e in errors]),

            # Penalizes larger errors more than MAE.
            # Root Mean Squared Error
            # RMSE ≈ MAE → errors are fairly uniform
            # x% of the true remaining cost
            "rmse": np.sqrt(np.mean([e["absolute"] ** 2 for e in errors])),
            "violations":violations,

        }
    
    def check_hueristic_tightness(self, tolerance=0.1):
        """
        tolerance: relative slack threshold (e.g., 0.1 = within 10% of optimal)
        """

        # True distances to goal
        true_dist = nx.single_source_dijkstra_path_length(
            self.BA_G, self.goal, weight="Cost"
        )

        slacks = {}
        rel_slacks = {}

        for n in self.BA_G.nodes:
            h_star = true_dist.get(n, float("inf"))
            if h_star == float("inf") or h_star == 0:
                continue

            h_n = self.h(n, self.goal)
            slack = h_star - h_n
            rel_slack = slack / h_star

            slacks[n] = slack
            rel_slacks[n] = rel_slack

        # ---- admissibility check ----
        violations = [n for n, s in slacks.items() if s < -1e-9]

        # ---- statistics ----
        rel_vals = np.array(list(rel_slacks.values()))

        stats = {
            "num_nodes_checked": len(rel_vals),
            "num_admissibility_violations": len(violations),
            "mean_relative_slack": float(np.mean(rel_vals)),
            "median_relative_slack": float(np.median(rel_vals)),
            "max_relative_slack": float(np.max(rel_vals)),
            "pct_within_tolerance": float(np.mean(rel_vals <= tolerance)),
        }

        return stats, violations

    def get_function_to_node_csv(self, function_num=6):
        """
        function_num: total number of functions in the SFC
        Returns a dictionary mapping each function to the list of nodes that can host it
            and a dictionary of node capacities.
        """
        node_capacity_path = os.path.join(self.graph_path, 'node_capacity.csv')
        node_capacity_df = read_csv(node_capacity_path)
        node_capacity_df['Function'] = node_capacity_df['Function'].astype(str)
        function_to_node = {i: [] for i in range(function_num)}
        node_capacities = {}

        # iterate over each node, check its capacity against each function's requirement
        for idx,row in node_capacity_df.iterrows():
            nodeID = row['NodeID']
            node_capacities[row['NodeID']] = row['Capacity']

            for func in self.functions:
                func_ID = func['ID']
                func_r = func['requirements']
                # print("NodeID:",nodeID, " Node capacity:", row['Capacity'], "func:", func_ID, "with Function req:", func_r*self.service_DR)
                if row['Capacity'] >= func_r*self.service_DR:
                    # print(f"node {nodeID}, can host {func_ID}")
                    function_to_node[func_ID].append(nodeID)
                # set the functions each node can accept in the function_to_node dict
                # eg.
                    # {
                    #   function: nodes that can host it
                    #   0: [0, 3],
                    #   1: [2],
                    #   2: [0, 2],
                    #   3: [],
                    #   4: [1],
                    #   5: []
                    # }
        return function_to_node,node_capacities
    
    def get_function_to_node_graph(self, function_num=6):
        """
        Build mapping: function_id -> list of physical nodes that can host it,
        based on node capacity stored in self.BA_G.nodes[n]["capacity"].

        Args:
            function_num: total number of functions in the SFC (for initializing keys)

        Returns:
            function_to_node: {func_id: [node_ids that can host]}
            node_capacities:  {node_id: capacity}
        """
        G = self.physical_graph

        # init outputs
        function_to_node = {i: [] for i in range(function_num)}
        node_capacities = {}

        # iterate over graph nodes and compare capacity vs requirements
        for node_id, attrs in G.nodes(data=True):
            cap = attrs.get("capacity", None)
            if cap is None:
                # If you prefer: cap = 0 instead of skipping
                # cap = 0
                continue

            node_capacities[node_id] = cap

            for func in self.functions:
                func_id = func['ID']
                req = func['requirements']
                required = req * self.service_DR  # keep same scaling logic as before

                if cap >= required:
                    # print(f"\n=============================\n node {node_id} can host function {func_id}")
                    function_to_node[func_id].append(node_id)

        return function_to_node, node_capacities
    
    def get_layered_graph(self,layered=True,remove=False):
    # def get_layer_graph(G, sfc_request, num_node, function_to_node,R_serviceB, sfc_request_num=3,remove=False):
        """
        inputs:
            remove: whether to remove edges that cannot support the service bandwidth
        Constructs a layered graph based on the original graph and SFC request."""

        # function_to_node = self.get_function_to_node_csv()[0]
        function_to_node = self.get_function_to_node_graph()[0]
        G = self.physical_graph
        num_node = G.number_of_nodes()
        if layered:
            sfc_request_num = len(self.functions)
        else:
            sfc_request_num = 0 #len(self.functions)

        # setup, build graph(layer 0)
        self.layer_graph = nx.DiGraph()
        origin_nodes = list(G.nodes)
        origin_edges = list(G.edges)
        pruned_count = 0


        # remove edges that cannot support the service bandwidth
        if remove:
            for u, v in origin_edges:
                if G.edges[(u,v)]['Bandwidth'] < self.service_DR: #bandwidth in Mbps
                    print("bandwidth for edge" , (u,v),"is less than the service req:", ":", end=" ")
                    print(G.edges[(u,v)]['Bandwidth'],"<", self.service_DR, "X remove")
                    G.remove_edge(u,v)
                    origin_edges = list(G.edges)
                    pruned_count+=1
                pass
            
        
        print(f"pruned {pruned_count} edges")
        # create i independent layers, one for eahc VNF in the request
        for i in range(sfc_request_num + 1):
            nodes = [(i * num_node + n, {'Cost':G.nodes[n]['Cost'],'Layer': i, 'node_id':n, 'capacity':G.nodes[n]['capacity'], 'function_layer':self.functions[i-1]['ID'], 'Node_v2': i * num_node + n}) for n in origin_nodes]
            edges = [(i * num_node + u, i * num_node + v,{'Cost':G.edges[(u,v)]['Cost'],'Bandwidth':G.edges[(u,v)]['Bandwidth']}) for u, v in origin_edges]

            self.layer_graph.add_nodes_from(nodes)
            self.layer_graph.add_edges_from(edges)

        # print(f"there are {sfc_request_num} VNF")



        # eg. of what we have at this stage
            # Layer 0: 0 → 1 → 2
            # Layer 1: 3 → 4 → 5
            # Layer 2: 6 → 7 → 8

        # vertical edges between layers (if the node can host the function)
        for i in range(sfc_request_num):
                        # get all nodes that can host the function sfc_request[i]
            for node_id in function_to_node[self.functions[i]['ID']]:
                self.layer_graph.add_edge(i * num_node + node_id,        #physical node in layer i
                                    (i + 1) * num_node + node_id,        #physical node in layer i+1
                                    Cost=G.nodes[node_id]['Cost'])


        # Store layers -- used for LLM output
        self.LayerGraphs = [] #stores nx graph for each layer
        num_layers = sfc_request_num + 1
        num_nodes = num_node

        for i in range(num_layers):
            layer_nodes = list(range(i * num_nodes, (i+1) * num_nodes))
            # induced subgraph
            layer_subG = self.layer_graph.subgraph(layer_nodes).copy()
            gateways = []
            # all gateway nodes out of this layer (vertical edges only)
            if i < sfc_request_num:  # only layers before the last have gateways
                func_id = self.functions[i]["ID"]
                for node_id in function_to_node[func_id]:
                    src = i * num_nodes + node_id
                    dst = (i + 1) * num_nodes + node_id
                    # sanity check—only treat it as gateway if the edge exists
                    if self.layer_graph.has_edge(src, dst):
                        gateways.append((src,self.layer_graph[src][dst]["Cost"] ,dst))


            self.LayerGraphs.append({"layer_graph":layer_subG,"layer_gateways":gateways})

    def get_layered_graph_v2(self, layered=True, remove=False):
        """
        Constructs a layered graph based on the original graph and SFC request.

        Node IDs in the layered graph are strings: "L{layer}_{node}".
        Example: L0_7, L3_12
        """
        def lname(layer: int, node_id: int) -> str:
            return f"L{layer}_{node_id}"

        function_to_node = self.get_function_to_node_graph()[0]
        G = self.G
        num_node = G.number_of_nodes()

        if layered:
            sfc_request_num = len(self.functions)
        else:
            sfc_request_num = 0

        # Build base
        self.layer_graph = nx.DiGraph()
        origin_nodes = list(G.nodes)
        origin_edges = list(G.edges)

        # Remove edges that cannot support service bandwidth
        if remove:
            for (u, v) in list(origin_edges):
                if G.edges[(u, v)]['Bandwidth'] < self.service_DR:
                    G.remove_edge(u, v)
            origin_edges = list(G.edges)

        # Create layers (0..sfc_request_num)
        for i in range(sfc_request_num + 1):
            nodes = []
            for n in origin_nodes:
                node_attrs = {
                    'Cost': G.nodes[n]['Cost'],
                    # 'Delay': G.nodes[n]['Delay'],
                    'Layer': i,
                    'node_id': n,                  # physical node id
                    'capacity': G.nodes[n]['capacity'],
                    # keep old numeric layered id if you still want it
                    'Node_v2_int': i * num_node + n,
                }

                # Only layers 1..sfc_request_num correspond to a function placement
                if i > 0:
                    node_attrs['function_layer'] = self.functions[i - 1]['ID']
                else:
                    node_attrs['function_layer'] = None

                nodes.append((lname(i, n), node_attrs))

            edges = []
            for (u, v) in origin_edges:
                edges.append((
                    lname(i, u),
                    lname(i, v),
                    {
                        'Cost': G.edges[(u, v)]['Cost'],
                        # 'Delay': G.edges[(u, v)]['Delay'],
                        'Bandwidth': G.edges[(u, v)]['Bandwidth'],
                        'edge_type': 'intra'
                    }
                ))

            self.layer_graph.add_nodes_from(nodes)
            self.layer_graph.add_edges_from(edges)

        # Vertical edges between layers (if node can host function)
        for i in range(sfc_request_num):
            func_id = self.functions[i]['ID']
            for node_id in function_to_node[func_id]:
                self.layer_graph.add_edge(
                    lname(i, node_id),
                    lname(i + 1, node_id),
                    Cost=G.nodes[node_id]['Cost'],
                    # Delay=G.nodes[node_id]['Delay'],
                    Bandwidth=float('inf'),   # optional; helps if your code assumes it exists
                    edge_type='inter'
                )

        # Store layers -- used for LLM output
        self.LayerGraphs = [] #stores nx graph for each layer
        num_layers = sfc_request_num + 1
        num_nodes = num_node

        for i in range(num_layers):
            layer_nodes = [f'L{i}_{v}' for v in range(num_nodes-1)]
            # list(range(i * num_nodes, (i+1) * num_nodes))
            # induced subgraph
            layer_subG = self.layer_graph.subgraph(layer_nodes).copy()
            gateways = []
            # all gateway nodes out of this layer (vertical edges only)
            if i < sfc_request_num:  # only layers before the last have gateways
                func_id = self.functions[i]["ID"]
                for node_id in function_to_node[func_id]:
                    src = f"L{i}_{node_id}"
                    dst = f"L{i+1}_{node_id}"
                    # sanity check—only treat it as gateway if the edge exists
                    if self.layer_graph.has_edge(src, dst):
                        gateways.append((src,self.layer_graph[src][dst]["Cost"] ,dst))


            self.LayerGraphs.append({"layer_graph":layer_subG,"layer_gateways":gateways})

        return self.layer_graph


    def save_graph(self, output_dir="graph/"):
        """
        Save the layered graph in multiple formats for inspection.
        :param output_dir: Directory to save the graph files.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nx.write_graphml(self.layer_graph, os.path.join(output_dir, "layered.graphml"))
        nx.write_gml(self.layer_graph, os.path.join(output_dir, "layered.gml"))
        nx.write_edgelist(self.layer_graph, os.path.join(output_dir, "layered.edgelist"), data=True)

    def get_graph(self,type_):
        if (type_ == "layered"):
            return self.layer_graph
        elif (type_ == "graph_layers"):
            return self.LayerGraphs
        else:
            return self.physical_graph



def build_and_save_graph(size, version_i, edge_degree, graph_type, cost, graph_struct, node_ID_format,
                         functions, service_DR, base_seed, out_dir: Path,rng):

    builder = SFCGraphBuilder(functions, service_DR=service_DR)
    builder.create_graph(size, edge_degree, rng, type_=graph_type, k=edge_degree)
    builder.conver_to_weighted_graph(cost, rng)

    if graph_struct == "layered":
        if node_ID_format == "L0_3":
            builder.get_layered_graph_v2()
        else:
            builder.get_layered_graph()

    G = builder.get_graph(graph_struct)


    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{graph_key_for(size, version_i,graph_type)}.edgelist"
    
    # nx.write_edgelist(G, path, delimiter=" ", data=edge_attrs)
    # nx.write_edgelist(G, path, delimiter=" ", data=["Cost"])
    # nx.write_gpickle(G, path)

    with open(path, "wb") as f:
        pickle.dump(G, f)
    return G, path
