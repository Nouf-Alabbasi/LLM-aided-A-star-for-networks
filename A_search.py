from utils_4 import *

from pydantic_core import ValidationError
import os
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
from sortedcontainers import SortedList

class A_search:
    def __init__(self,graph,model_name,functions,service_DR,print_ = False):
        self.graph = graph
        self.print_ = print_
        self.model_name = model_name
        self.functions = functions
        self.service_DR = service_DR

    def get_min_edge_delay(self):
        """
        Get the minimum edge delay in the graph.
        :return: Minimum edge delay.
        """

        cost_list = [data.get("Cost", 1) for _, _, data in self.graph.edges(data=True)]
        self.d_min = min(
            (data.get("Cost", 1) for _, _, data in self.graph.edges(data=True)),
            default=1
        )
        # self.d_min = sum(cost_list) / len(cost_list)
        if self.d_min <= 0:
            # If zero/negative weights exist, clamp to a tiny positive to keep a valid lower bound
            d_min = 1e-12
        self.d_min

        self.H = self.graph.reverse(copy=False) if self.graph.is_directed() else self.graph
        self.hop_dist_to_goal = nx.single_source_shortest_path_length(self.H, self.goal)

    def h(self, u, target=0):
        """
        triangle-inequality ALT heuristic
        """
        # If you didn't preprocess landmarks, fall back to 0  
        if not hasattr(self, "landmarks") or not self.landmarks:
            return 0.0

        best = 0.0
        for L in self.landmarks:
            dL_t = self.L_from[L].get(target)  # d(L -> t)
            dL_u = self.L_from[L].get(u)       # d(L -> u)
            du_L = self.L_to[L].get(u)         # d(u -> L)
            dt_L = self.L_to[L].get(target)    # d(t -> L)

            # Directed ALT lower bound:
            # h_L(u,t) = max( d(L,t)-d(L,u), d(u,L)-d(t,L), 0 )
            cand = 0.0
            if dL_t is not None and dL_u is not None:
                cand = max(cand, dL_t - dL_u)
            if du_L is not None and dt_L is not None:
                cand = max(cand, du_L - dt_L)

            if cand > best:
                best = cand

        return float(best)

    """Shortest paths and path lengths using the A* ("A star") algorithm."""
    __all__ = ["astar_path", "astar_path_length"]
    # @nx._dispatchable(edge_attrs="weight", preserve_node_attrs="heuristic")

    def astar_path(self, weight="Cost", *, test_name = None, cutoff=None, constrain=True, service_DR=0,cap_check=True,g_n=1):
        """Returns a list of nodes in a shortest path between source and goal
        using the A* ("A-star") algorithm.

        There may be more than one shortest path.  This returns only one.

        Parameters
        ----------
        G : NetworkX graph

        source : node
        Starting node for path

        goal : node
        Ending node for path

        heuristic : function
        A function to evaluate the estimate of the distance
        from the a node to the goal.  The function takes
        two nodes arguments and must return a number.
        If the heuristic is inadmissible (if it might
        overestimate the cost of reaching the goal from a node),
        the result may not be a shortest path.
        The algorithm does not support updating heuristic
        values for the same node due to caching the first
        heuristic calculation per node.

        weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.
        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

        cutoff : float, optional
        If this is provided, the search will be bounded to this value. I.e. if
        the evaluation function surpasses this value for a node n, the node will not
        be expanded further and will be ignored. More formally, let h'(n) be the
        heuristic function, and g(n) be the cost of reaching n from the source node. Then,
        if g(n) + h'(n) > cutoff, the node will not be explored further.
        Note that if the heuristic is inadmissible, it is possible that paths
        are ignored even though they satisfy the cutoff.

        Raises
        ------
        NetworkXNoPath
            If no path exists between source and goal.

        Examples
        --------
        >>> G = nx.path_graph(5)
        >>> print(nx.astar_path(G, 0, 4))
        [0, 1, 2, 3, 4]
        >>> G = nx.grid_graph(dim=[3, 3])  # nodes are two-tuples (x,y)
        >>> nx.set_edge_attributes(G, {e: e[1][0] * 2 for e in G.edges()}, "cost")
        >>> def dist(a, b):
        ...     (x1, y1) = a
        ...     (x2, y2) = b
        ...     return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        >>> print(nx.astar_path(G, (0, 0), (2, 2), heuristic=dist, weight="cost"))
        [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

        Notes
        -----
        Edge weight attributes must be numerical.
        Distances are calculated as sums of weighted edges traversed.

        The weight function can be used to hide edges by returning None.
        So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
        will find the shortest red path.

        See Also
        --------
        shortest_path, dijkstra_path

        """

        
        # target_list = get_target_list(G, goal)

        goal = self.goal
        source = self.source
        G = self.graph
        function_req = {func['ID']: func['requirements'] * self.service_DR for func in self.functions}
        # TO KEEP TRACK OF MEMORY
        expanded_count = 0
        generated_count = 0          # neighbors you consider inserting
        pushed_count = 0             # actually pushed into heap
        max_queue_size = 1
        nx.set_node_attributes(G, "unseen", "status")
        nx.set_node_attributes(G, "none", "type")


        self.get_min_edge_delay()
        expanded_count = 0

        if source not in G:
            raise nx.NodeNotFound(f"Source {source} is not in G")

        if goal not in G:
            raise nx.NodeNotFound(f"goal {goal} is not in G")

        
        weight = _weight_function(G, "Cost")
        # og_node_capacities = node_capacities
        

        G_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)

        # ---- precompute base capacity per physical node (no mutation during search) ----
        base_capacity = {}  # added this
        for n, data in G.nodes(data=True):  # added this
            phys_id = data.get("node_id")  # added this
            cap = data.get("capacity", float("inf"))  # added this
            if phys_id not in base_capacity:  # added this
                base_capacity[phys_id] = cap  # added this
        # ---- end base capacity precomputation ----


        # The queue stores priority, node, cost to reach, parent, true_node_id
        # Uses Python heapq to keep in priority order.
        # Add a counter to the queue to prevent the underlying heap from
        # attempting to compare the nodes themselves. The hash breaks ties in the
        # priority and is guaranteed unique for all nodes in the graph.
        c = count()
        queue = [(0, next(c), source, 0, None)] 

        # Maps enqueued nodes to distance of discovered paths and the
        # computed heuristics to goal. We avoid computing the heuristics
        # more than once and inserting the node into the queue too many times.
        enqueued = {}
        # Maps explored nodes to parent closest to the source.
        explored = {}

        while queue:
            # Pop the smallest item from queue.
            _, __, curnode, dist, parent = heappop(queue)



            if curnode == goal:
                if self.print_:
                    print(f"path found with total cost: {dist}")
                    print(f"source {source}, and source: {goal}")
                    print(f"the number of generated nodes: {len(enqueued) + 1} and number of explored nodes {expanded_count}")
                    print(f"nodes in the graph {len(G.nodes)}")
                path = [curnode]
                G.nodes[curnode]["status"] = "path"
                node = parent
                while node is not None:
                    G.nodes[node]["status"] = "path"
                    path.append(node)
                    node = explored[node]
                self.color_landmarks()
                G.nodes[self.source]["status"] = "source"
                G.nodes[self.goal]["status"] = "dest"
                path.reverse()

                # for saving the graph purposes
                for n, d in G.nodes(data=True):
                    if d.get("function_layer") is None:
                        d["function_layer"] = -1
                                    # "STAT-expanded_count",       "STAT-explored_neighbors ",   "STAT-neighbors_enqueued","STAT-pushed_count_revised_paths"])
                return path,self.get_path_cost(path),{"enqueued_size":len(enqueued), "pushed count":pushed_count, "generated_count":generated_count,"expanded_count":expanded_count}

            if curnode in explored:
                # Do not override the parent of starting node
                if explored[curnode] is None:
                    continue

                # Skip bad paths that were enqueued before finding a better one
                qcost, h = enqueued[curnode]
                if qcost < dist:
                    continue
            
            G.nodes[curnode]["status"] = "expanded"
            expanded_count += 1
            explored[curnode] = parent

            if (self.print_):
                print("++++++")
                print("on node:",curnode)
                print("exploring neighbors")
            for neighbor, w in G_succ[curnode].items():

                # keep track of the number of neighbors it considers
                generated_count += 1

                if "Bandwidth" in G.edges[(curnode, neighbor)]:
                    bw = G.edges[(curnode, neighbor)]["Bandwidth"]

                # if the link can't handle the service req, skip
                if constrain and bw < service_DR:
                    if(self.print_):
                        print("X skipped edge (u,v):","(",curnode,",",neighbor,") bw:",bw,"and R_service",service_DR)
                    continue

                cost = weight(curnode, neighbor, w)
                if cost is None:
                    continue
                ncost = (dist + cost)*g_n
                if neighbor in enqueued:
                    qcost, h = enqueued[neighbor]
                    # if qcost <= ncost, a less costly path from the
                    # neighbor to the source was already determined.
                    # Therefore, we won't attempt to push this neighbor
                    # to the queue
                    if qcost <= ncost:
                        continue
                else:
                    h = self.h(neighbor, goal)

                if (self.print_):
                    print("neighbors:",neighbor,"with h(n):",h,"and g(n):",ncost)
                if cutoff and ncost + h > cutoff:
                    continue

                enqueued[neighbor] = ncost, h
                G.nodes[neighbor]["status"] = "enqueued"
                # ncost=0
                heappush(queue, (ncost + h, next(c), neighbor, ncost, curnode))

                # Keep track of nodes pushed to heap
                pushed_count += 1
                if len(queue) > max_queue_size:
                    max_queue_size = len(queue)
        raise nx.NetworkXNoPath(f"Node {goal} not reachable from {source}")

        
    def _generate_llm_query(self, start, goal,h,_,node_ID_format,prompt_,__,graph=None):
        """Generate the query for the LLM."""       
        if graph == None:
            G = self.graph
        else:
            G = graph


        if self.adj_list_format == "BFS":
            adj_list = get_adj_list_bfs(G,start)
        else:
            adj_list = get_adj_list(G)

        if h:
            prompt = prompt_get_waypoints_v4_h
        else:
            prompt = prompt_

        label_to_edges, edge_to_label = self.get_labels(G)
        h_label_to_edges, h_edge_to_label = self.get_h_label(G)
        # adjacency_list_v2 = print_adj_list_with_cost(self.BA_G,cost_key = "Cost")
        adjacency_list_v2 = ['placeholder list']

        gateway_nodes = ""
        if node_ID_format == "L0_3":
            output_example_path = "['L0_10', 'L0_23', 'L1_23', 'L1_45', 'L2_45', 'L2_67', 'L3_67']"
            interpetation = ""
        else:
            output_example_path = "['10', '100', '150','200', '350', '400', '530']"
            interpetation = ""

        base_prompt = prompt.format(
                    adjacency_list= adj_list if "{adjacency_list}" in prompt else "",
                    adjacency_list_v2=adjacency_list_v2 if "{adjacency_list_v2}" in prompt else "",
                    gateway_nodes=gateway_nodes if "{gateway_nodes}" in prompt else "",
                    num_waypoints = self.num_waypoints if "{num_waypoints}" in prompt else "",
                    src = start if "{src}" in prompt else "",
                    goal = goal,
                    edge_costs=nx.get_edge_attributes(G, "Cost") if "{edge_costs}" in prompt else "",
                    edge_to_label=edge_to_label if "{edge_to_label}" in prompt else "",
                    label_to_edges=label_to_edges if "{label_to_edges}" in prompt else "",
                    output_example = output_example_path if "{output_example}" in prompt else "",
                    interpetation = interpetation if "{interpetation}" in prompt else "",

                    h_n_values = self.get_h_n() if "{h_n_values}" in prompt else "",
                    h_n_values_labels = h_edge_to_label if "{h_n_values_labels}" in prompt else "", # self.get_h_n() if "{h_n_values}" in prompt else "",
                    adj_list_eg_1 = examples.Adjacent_list_larger_graph if "{adj_list_eg_1}" in prompt else "",
                    steps_eg_1 = examples.graph_2_long_example_4_waypoints if "{steps_eg_1}" in prompt else "",
                    
                    adj_list_eg_2 = examples.Adjacent_list_smaller_graph if "{adj_list_eg_2}" in prompt else "",
                    steps_eg_2 = examples.longer_path_example_2_waypoints if "{steps_eg_2}" in prompt else "",
                )               

        return base_prompt

    def get_h_label(self,G):
        LABELS = ["excellent", "good", "ok", "bad", "terrible"]
        edge_h_n = {
            u: self.h(u,self.goal)
            for u in G.nodes
        }
        # create thresholds
        cost_values = list(edge_h_n.values())
        thresholds = np.quantile(
            cost_values,
            q=[i / len(LABELS) for i in range(1, len(LABELS))]
        )

        # ======================================================
        # returns {(0,1): "good", (u,v): label...}
        edge_to_label = {
            edge: self.cost_to_label(cost, thresholds, LABELS)
            for edge, cost in edge_h_n.items()
        }

        # ======================================================
        # returns {
        #   "excellent": [(5,7), (2,3)],
        label_to_edges = defaultdict(list)
        for edge, label in edge_to_label.items():
            label_to_edges[label].append(edge)
        label_to_edges = dict(label_to_edges)

        return label_to_edges, edge_to_label

    def get_labels(self,G):
        LABELS = ["excellent", "good", "ok", "bad", "terrible"]
        edge_costs = {
            (u, v): data["Cost"]
            for u, v, data in G.edges(data=True)
        }


        # create thresholds
        cost_values = list(edge_costs.values())
        thresholds = np.quantile(
            cost_values,
            q=[i / len(LABELS) for i in range(1, len(LABELS))]
        )

        # ======================================================
        # returns {(0,1): "good", (u,v): label...}
        edge_to_label = {
            edge: self.cost_to_label(cost, thresholds, LABELS)
            for edge, cost in edge_costs.items()
        }

        # ======================================================
        # returns {
        #   "excellent": [(5,7), (2,3)],
        label_to_edges = defaultdict(list)
        for edge, label in edge_to_label.items():
            label_to_edges[label].append(edge)
        label_to_edges = dict(label_to_edges)

        return label_to_edges, edge_to_label

    def cost_to_label(self,cost, thresholds, labels):
        for t, label in zip(thresholds, labels):
            if cost <= t:
                return label
        return labels[-1]        

    def _initialize_llm_paths_limited(self,node_ID_format,prompt_,__,review=False, h=False,adj_list_format="other",graph=None):
        self.adj_list_format = adj_list_format
        query = self._generate_llm_query(self.source, self.goal,h,"_",node_ID_format,prompt_,"_",graph=graph)
        if review:
            query = query+" Please ensure that the output is in the format specified."



        path_ = path

        desired_count = self.num_waypoints
        max_retries = 5

        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()            
                response = client.responses.parse(
                    model=self.model_name,
                    input=[
                        {
                            "role": "system",
                            "content": "you are a helpful network optimization assistant.",
                        },
                        {"role": "user", "content": query},
                    ],
                    text_format=path_,
                )
                latency_s = time.perf_counter() - t0

                # compliance = response.output_parsed
                counter = 0
            except ValidationError:
                print("Validation error occurred. Retrying... Attempt", attempt + 1)
                continue

            try:
                full_output = response.output[0].content[0].text
                self.json_response = json.loads(full_output)
                cleaned = extract_nodes(self.json_response)
            except Exception:
                full_output = msg0 = response.output[0] if response.output else None
                c0 = msg0.content[0] if (msg0 and getattr(msg0, "content", None)) else None

                if c0 is None:
                    text = None
                elif getattr(c0, "type", None) == "refusal":
                    text = getattr(c0, "refusal", None)  # not .text
                else:
                    text = getattr(c0, "text", None)

                try:
                    # self.target_list = ast.literal_eval(text)
                    cleaned = ast.literal_eval(text)
                except ValueError:
                    cleaned = [x.strip() for x in text.strip("[]").split(",")]

            # =========================== eval infomration
            usage = getattr(response, "usage", None)
            record = {
                "latency_s": latency_s,
                "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
                "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
                "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            }
            


            if (self.print_):
                print("LLM response:", self.json_response)

            # ensure that the source and goal nodes are included in the list
            if (self.source not in cleaned):
                if(self.print_):
                    print("The source was not included in the list")
                cleaned.insert(0, self.source)

            if (self.goal not in cleaned):
                if(self.print_):
                    print("The goal was not included in the list")
                cleaned.append(self.goal)

            if len(cleaned) == desired_count and valid_nodes(cleaned):
                self.target_list = cleaned
                break
            print("Retrying... Attempt", attempt, "waypoint length:", len(cleaned))

        # self.target_list = [2443,2565,2637,6497,9800]
        print("the LLM generated waypoint list:",self.target_list)
        
        self.s_target = self.target_list[1]
        self.i = 1

        # for graphing purposes
        if graph!= None:
            for node in graph:
                self.graph.nodes[node]["type"] = "high_deg"
        for waypoint in self.target_list:
            self.graph.nodes[waypoint]["type"] = "waypoint"

        return query,full_output, self.target_list,record
    

    def preprocess_landmarks(self, landmarks, weight="Cost"):
        """
        -> compute distance from all vertices to nodes
        Precompute distances needed for ALT / differential landmark heuristic.
        Stores:
        self.L_from[L][v] = d(L -> v)
        self.L_to[L][v]   = d(v -> L)   (computed via reverse graph)
        """
        G = self.graph
        R = G.reverse(copy=False) if G.is_directed() else G
        
        self.landmarks = list(landmarks)
        self.L_from = {}  # distances from landmark to all nodes
        self.L_to = {}    # distances to landmark from all nodes (via reverse)

        for L in self.landmarks:
            self.L_from[L] = nx.single_source_dijkstra_path_length(G, L, weight=weight)
            # On reversed graph: distances from L in R are distances to L in G
            self.L_to[L] = nx.single_source_dijkstra_path_length(R, L, weight=weight)


    def hops_to_set(self):
        """
        Compute hop distances sequentially to each target in target_list.
        Returns a list of dicts: one distance map per target.
        """
        G = self.graph
        R = G.reverse(copy=False) #reverse graph, in order to discover all nodes that can reach it
        results = []

        for t in self.target_list: #preform BFS for each target
            # Initialize distances to infinity
            dist = {u: math.inf for u in G.nodes()}
            dist[t] = 0  # distance to itself = 0
            q = deque([t]) #double ended queue

            while q:
                u = q.popleft()
                for v in R[u]:  # reverse edges: v → u in original
                    if dist[v] == math.inf:
                        # dist[v] = 0.5*(dist[u] + 1)*self.d_min
                        dist[v] = (dist[u] + 1)*self.d_min
                        q.append(v)

            results.append(dist)

        return results
        
    def _update_queue(self,t_dist,t_n_weight=1):
        """Update the priority queue based on new target."""
        queue = []
        for f,temp,nodeID,g,parent in self.queue:
            heappush(queue, (g +  self.h(nodeID, self.s_target)+t_n_weight*self.h(nodeID,self.goal),temp,nodeID,g, parent )) 
        self.queue = queue

    def _update_queue_FOCAL(self,t_dist,FOCAL,t_n_weight=1):
        """Update the priority queue based on new target."""
        queue = []
        for f,temp,nodeID,g,parent in FOCAL:
            heappush(queue, (g +  self.h(nodeID, self.s_target)+t_n_weight*self.h(nodeID,self.goal),temp,nodeID,g, parent )) 
        return FOCAL

    def _update_queue_sorted_list(self, t_dist, t_n_weight=1):
        new_queue = SortedList()
        for _, temp, nodeID, g, parent in self.queue:
            new_priority = g + self.h(nodeID, self.s_target) + t_n_weight * self.h(nodeID, self.goal)
            new_queue.add((new_priority, temp, nodeID, g, parent))

        self.queue = new_queue

    def _update_target(self):
        """Update the current target in the path."""
        self.i += 1
        if self.i < len(self.target_list):
            self.s_target = self.target_list[self.i]  

    def astar_path_LLM(self, weight="Cost", *,t_n_weight, test_name=None, cutoff=None, constrain=False, service_DR=0,cap_check=False):
            """Returns a list of nodes in a shortest path between source and goal
            using the A* ("A-star") algorithm.

            There may be more than one shortest path.  This returns only one.

            Parameters
            ----------
            G : NetworkX graph

            source : node
            Starting node for path

            goal : node
            Ending node for path

            heuristic : function
            A function to evaluate the estimate of the distance
            from the a node to the goal.  The function takes
            two nodes arguments and must return a number.
            If the heuristic is inadmissible (if it might
            overestimate the cost of reaching the goal from a node),
            the result may not be a shortest path.
            The algorithm does not support updating heuristic
            values for the same node due to caching the first
            heuristic calculation per node.

            weight : string or function
            If this is a string, then edge weights will be accessed via the
            edge attribute with this key (that is, the weight of the edge
            joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
            such edge attribute exists, the weight of the edge is assumed to
            be one.
            If this is a function, the weight of an edge is the value
            returned by the function. The function must accept exactly three
            positional arguments: the two endpoints of an edge and the
            dictionary of edge attributes for that edge. The function must
            return a number or None to indicate a hidden edge.

            cutoff : float, optional
            If this is provided, the search will be bounded to this value. I.e. if
            the evaluation function surpasses this value for a node n, the node will not
            be expanded further and will be ignored. More formally, let h'(n) be the
            heuristic function, and g(n) be the cost of reaching n from the source node. Then,
            if g(n) + h'(n) > cutoff, the node will not be explored further.
            Note that if the heuristic is inadmissible, it is possible that paths
            are ignored even though they satisfy the cutoff.

            Raises
            ------
            NetworkXNoPath
                If no path exists between source and goal.

            Examples
            --------
            >>> G = nx.path_graph(5)
            >>> print(nx.astar_path(G, 0, 4))
            [0, 1, 2, 3, 4]
            >>> G = nx.grid_graph(dim=[3, 3])  # nodes are two-tuples (x,y)
            >>> nx.set_edge_attributes(G, {e: e[1][0] * 2 for e in G.edges()}, "cost")
            >>> def dist(a, b):
            ...     (x1, y1) = a
            ...     (x2, y2) = b
            ...     return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            >>> print(nx.astar_path(G, (0, 0), (2, 2), heuristic=dist, weight="cost"))
            [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

            Notes
            -----
            Edge weight attributes must be numerical.
            Distances are calculated as sums of weighted edges traversed.

            The weight function can be used to hide edges by returning None.
            So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
            will find the shortest red path.

            See Also
            --------
            shortest_path, dijkstra_path

            """

            
            # target_list = get_target_list(G, goal)

            goal = self.goal
            source = self.source
            G = self.graph
            service_DR = self.service_DR
            # function_req = [func['requirements']*service_DR for func in self.functions]
            function_req = {func['ID']: func['requirements'] * self.service_DR for func in self.functions}
            
            # TO KEEP TRACK OF MEMORY
            expanded_count = 0
            generated_count = 0          # neighbors you consider inserting
            pushed_count = 0             # actually pushed into heap
            max_queue_size = 1
            nx.set_node_attributes(G, "unseen", "status")
            nx.set_node_attributes(G, "none", "type")
    
            # self._initialize_llm_paths()
            self.s_target = self.target_list[1]
            self.i = 1

            # remove
            self.get_min_edge_delay()
            h_T_dist = None

            expanded_count = 0

            if source not in G:
                raise nx.NodeNotFound(f"Source {source} is not in G")

            if goal not in G:
                raise nx.NodeNotFound(f"goal {goal} is not in G")

            
            weight = _weight_function(G, 'Cost')
            # og_node_capacities = node_capacities
            

            G_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)

            # ---- precompute base capacity per physical node (no mutation during search) ----
            base_capacity = {}  # added this
            for n, data in G.nodes(data=True):  # added this
                phys_id = data.get("node_id")  # added this
                cap = data.get("capacity", float("inf"))  # added this
                if phys_id not in base_capacity:  # added this
                    base_capacity[phys_id] = cap  # added this
            # ---- end base capacity precomputation ----

            # The queue stores priority, node, cost to reach, parent, true_node_id
            # Uses Python heapq to keep in priority order.
            # Add a counter to the queue to prevent the underlying heap from
            # attempting to compare the nodes themselves. The hash breaks ties in the
            # priority and is guaranteed unique for all nodes in the graph.
            c = count()
            self.queue = [(0, next(c), source, 0, None)] 


            # Maps enqueued nodes to distance of discovered paths and the
            # computed heuristics to goal. We avoid computing the heuristics
            # more than once and inserting the node into the queue too many times.
            enqueued = {}
            # Maps explored nodes to parent closest to the source.
            explored = {}

            while self.queue:
                # Pop the smallest item from queue.
                _, __, curnode, dist, parent = heappop(self.queue)


                if curnode == goal:
                    if self.print_:
                        print(f"path found with total cost: {dist}")
                        print(f"source {source}, and source: {goal}")
                        print(f"the number of generated nodes: {len(enqueued) + 1} and number of explored nodes {expanded_count}")
                        print(f"nodes in the graph {len(G.nodes)}")
                        # print("Total delay of the path:", total_delay)
                    path = [curnode]
                    G.nodes[curnode]["status"] = "path"
                    node = parent
                    while node is not None:
                        G.nodes[node]["status"] = "path"
                        path.append(node)
                        node = explored[node]
                    self.color_landmarks()
                    self.color_waypoints()
                    G.nodes[self.source]["status"] = "source"
                    path.reverse()

                    # calculate total delay
                    total_delay = 0 
                    for i in range(len(path)-1):
                        u = path[i]
                        v = path[i+1]
                        if G.has_edge(u, v):
                            total_delay += G.edges[u, v].get("Cost", 0)

                    # # =============saving the graph
                    return path,self.get_path_cost(path),{"enqueued_size":len(enqueued), "pushed count":pushed_count, "generated_count":generated_count,"expanded_count":expanded_count}


                if curnode in explored:
                    # Do not override the parent of starting node
                    if explored[curnode] is None:
                        continue

                    # Skip bad paths that were enqueued before finding a better one
                    qcost, h = enqueued[curnode]
                    if qcost < dist:
                        continue

                G.nodes[curnode]["status"] = "expanded"
                expanded_count += 1
                explored[curnode] = parent

                if(self.print_):
                    print("++++++")
                    print("on node:",curnode)
                    print("exploring neighbors")
                for neighbor, w in G_succ[curnode].items():
                    generated_count += 1


                    if "Bandwidth" in G.edges[(curnode, neighbor)]:
                        bw = G.edges[(curnode, neighbor)]["Bandwidth"]
                    # if the link can't handle the service req, skip
                    if constrain and bw < service_DR:
                        if (self.print_):
                            print("X skipped edge (u,v):","(",curnode,",",neighbor,") bw:",bw,"and R_service",service_DR)
                        continue

                    if neighbor == self.s_target and self.goal != self.s_target :
                        self._update_target()
                        self._update_queue(h_T_dist,t_n_weight)
                        if (self.print_):
                            print(neighbor, self.s_target)

                    cost = weight(curnode, neighbor, w)

                    if cost is None:
                        continue
                    ncost = dist + cost
                    if neighbor in enqueued:
                        qcost, h = enqueued[neighbor]
                        # if qcost <= ncost, a less costly path from the
                        # neighbor to the source was already determined.
                        # Therefore, we won't attempt to push this neighbor
                        # to the queue
                        if qcost <= ncost:
                            continue
                    else:
                        h = self.h(neighbor, goal)

                    if(self.print_):
                        print("neighbors:",neighbor,"with h(n):",h,"and g(n):",ncost)
                    if cutoff and ncost + h > cutoff:
                        continue

                    # ASTAR
                    
                    enqueued[neighbor] = ncost, h
                    G.nodes[neighbor]["status"] = "enqueued"
                    
                    heappush(self.queue, (ncost + h + t_n_weight*self.h(neighbor, self.s_target), next(c), neighbor, ncost, curnode))


                    
                    pushed_count += 1
                    if len(self.queue) > max_queue_size:
                        max_queue_size = len(self.queue)

            raise nx.NetworkXNoPath(f"Node {goal} not reachable from {source}")


    def print_to_CSV(
        self,
        f_N,
        filename,
        printing_path,
        prompt,
        NC_A_dict,
        NC_LLM_dict,
        SRC_goal_dict,
        # mod_dict,
        i,
        nodes,
        edges_added,
        waypoint_lists,
        t_list_flat=None,
        waypoint=None
    ):
        def avg(lst):
            return float(np.mean(lst)) if len(lst) > 0 else 0.0


        # ---- Averages number of waypoints ----
        lengths = [len(w) for w in waypoint_lists]
        avg_waypoint_lengths = avg(lengths)
        

        # ---- Averages for NC_A (baseline) ----
        A_expanded_avg = avg(NC_A_dict["expanded_count_sum"])
        A_neighbors_enq_avg = avg(NC_A_dict["neighbors_enqueued_sum"])
        A_pushed_avg = avg(NC_A_dict["pushed_count_revised_paths_sum"])
        A_generated_avg = avg(NC_A_dict["explored_neighbors_sum"])

        # ---- Averages for LLM ----
        LLM_expanded_avg = avg(NC_LLM_dict["expanded_count_sum"])
        LLM_neighbors_enq_avg = avg(NC_LLM_dict["neighbors_enqueued_sum"])
        LLM_pushed_avg = avg(NC_LLM_dict["pushed_count_revised_paths_sum"])
        LLM_generated_avg = avg(NC_LLM_dict["explored_neighbors_sum"])

        # ---- Averages for SRC_GOAL ----
        SRC_expanded_avg = avg(SRC_goal_dict["expanded_count_sum"])
        SRC_neighbors_enq_avg = avg(SRC_goal_dict["neighbors_enqueued_sum"])
        SRC_pushed_avg = avg(SRC_goal_dict["pushed_count_revised_paths_sum"])
        SRC_generated_avg = avg(SRC_goal_dict["explored_neighbors_sum"])

        # # ---- Averages for MOD ----
        # MOD_expanded_avg = avg(mod_dict["expanded_count_sum"])
        # MOD_neighbors_enq_avg = avg(mod_dict["neighbors_enqueued_sum"])
        # MOD_pushed_avg = avg(mod_dict["pushed_count_revised_paths_sum"])
        # MOD_generated_avg = avg(mod_dict["explored_neighbors_sum"])

        # ---- Percent reductions vs NC_A ----
        def percent_reduction(baseline, other):
            return ((baseline - other) / baseline * 100.0) if baseline > 0 else 0.0

        pct_LLM_reduction = percent_reduction(A_expanded_avg, LLM_expanded_avg)
        pct_SRC_reduction = percent_reduction(A_expanded_avg, SRC_expanded_avg)
        # pct_MOD_reduction = percent_reduction(A_expanded_avg, MOD_expanded_avg)

        # ---- CSV Header ----
        header = [
            "Test No.", "# Nodes", "total edges", "# Edges added", "prompt",

            "num_waypoints",
            "%LLM_expanded_reduction",
            "%SRC_GOAL_expanded_reduction",
            # "%mod_expanded_reduction",

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
            
            "AVG waypoint lengths",

            # "AVG_mod_neighbors_enqueued",
            # "AVG_mod_pushed_count_revised_paths",
            # "AVG_mod_generated_neighbor_count",
            # "AVG_mod_expanded_count"
        ]

        # ---- CSV Row ----
        row = [
            i,
            nodes,
            len(self.graph.edges),
            edges_added,
            prompt,
            waypoint,

            pct_LLM_reduction,
            pct_SRC_reduction,
            # pct_MOD_reduction,

            A_neighbors_enq_avg,
            A_pushed_avg,
            A_generated_avg,
            A_expanded_avg,

            LLM_neighbors_enq_avg,
            LLM_pushed_avg,
            LLM_generated_avg,
            LLM_expanded_avg,

            SRC_neighbors_enq_avg,
            SRC_pushed_avg,
            SRC_generated_avg,
            SRC_expanded_avg,

            avg_waypoint_lengths,

            # MOD_neighbors_enq_avg,
            # MOD_pushed_avg,
            # MOD_generated_avg,
            # MOD_expanded_avg
        ]

        # ---- Write to CSV ----
        with open(f"{printing_path}/stats_per_overall_iter_{filename}_{f_N}_final.csv", mode="a", newline="") as file:
            writer = csv.writer(file)

            # Write header only if file is empty
            if file.tell() == 0:
                writer.writerow(header)

            writer.writerow(row)

    def print_to_CSV_2(
        self,
            filename,
            printing_path,
            prompt,
            accumulators,       # dict: {alg_name: accumulator_dict}
            baseline_alg,       # which algorithm is the baseline for % reduction (e.g. "A*")
            i,
            nodes,
            edges_added,
            waypoint_lists,
            waypoint=None,
        ):
            """
            Write a summary CSV row with averages and % reduction for each algorithm.
    
            Parameters
            ----------
            accumulators : dict
                {alg_name: {"expanded_count_sum": [...], "neighbors_enqueued_sum": [...],
                            "pushed_count_revised_paths_sum": [...], "explored_neighbors_sum": [...]}}
            baseline_alg : str
                The algorithm to compute % reduction against (typically "A*").
            """
    
            def avg(lst):
                return float(np.mean(lst)) if len(lst) > 0 else 0.0
    
            def percent_reduction(baseline, other):
                return ((baseline - other) / baseline * 100.0) if baseline > 0 else 0.0
    
            alg_names = list(accumulators.keys())
    
            # ---- compute averages per algorithm ----
            alg_avgs = {}
            for alg, acc in accumulators.items():
                alg_avgs[alg] = {
                    "enqueued":  avg(acc["neighbors_enqueued_sum"]),
                    "pushed":    avg(acc["pushed_count_revised_paths_sum"]),
                    "generated": avg(acc["explored_neighbors_sum"]),
                    "expanded":  avg(acc["expanded_count_sum"]),
                    "opt":       acc["opt"],
                }
    
            # ---- compute % reduction vs baseline ----
            baseline_expanded = alg_avgs.get(baseline_alg, {}).get("expanded", 0)
            pct_reductions = {}
            for alg in alg_names:
                if alg == baseline_alg:
                    continue
                pct_reductions[alg] = percent_reduction(baseline_expanded, alg_avgs[alg]["expanded"])
    
            # ---- average waypoint lengths ----
            lengths = [len(w) for w in waypoint_lists]
            avg_waypoint_lengths = avg(lengths)
    
            # ---- build header ----
            header = [
                "Test No.", "# Nodes", "total edges", "# Edges added",
                "prompt", "num_waypoints",
            ]
            # % reduction columns
            for alg in alg_names:
                if alg == baseline_alg:
                    continue
                header.append(f"%{alg}_expanded_reduction")
            # per-algorithm stat columns
            for alg in alg_names:
                header.extend([
                    f"AVG_{alg}_enqueued",
                    f"AVG_{alg}_pushed",
                    f"AVG_{alg}_generated",
                    f"AVG_{alg}_expanded",
                    f"{alg}_optimal_count",
                ])
            header.append("AVG_waypoint_lengths")
    
            # ---- build row ----
            row = [
                i, nodes, len(self.graph.edges), edges_added,
                prompt, waypoint,
            ]
            for alg in alg_names:
                if alg == baseline_alg:
                    continue
                row.append(pct_reductions[alg])
            for alg in alg_names:
                a = alg_avgs[alg]
                row.extend([a["enqueued"], a["pushed"], a["generated"], a["expanded"], a["opt"]])
            row.append(avg_waypoint_lengths)
    
            # ---- write ----
            filepath = f"{printing_path}/stats_summary_{filename}.csv"
            with open(filepath, mode="a", newline="") as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(header)
                writer.writerow(row) 



    def generate_landmarks(self,type_,selection_tech,k_landmarks,rng):


        if selection_tech == "furthest":
            # landmarks = landmark_picker.pick_landmarks_top_degree(
            landmarks = landmark_picker.pick_landmarks_farthest_point(
                self.graph,
                k=k_landmarks,
                rng = rng,
                # weight="Cost",
                # seed=42,
                # require_reachability_frac=0.2,   # optional, helps on directed graphs
            )
        else: 
            landmarks = landmark_picker.pick_landmarks_random(
                self.graph,
                k=k_landmarks,
                rng = rng,
            )

        self.preprocess_landmarks(landmarks, weight="Cost")
    
    def color_landmarks(self):
        for landmark in self.landmarks:
            self.graph.nodes[landmark]["type"] = "landmark"
    def color_waypoints(self):
        for waypoint in self.target_list:
            self.graph.nodes[waypoint]["type"] = "waypoint"

    def set_landmarks(self,landmarks):
        """
        hardcode a set of landmarks
        """
        self.preprocess_landmarks(landmarks, weight="Cost")
        # self.landmarks=landmarks


    def get_landmarks(self):
        return self.landmarks
    
    def set_waypoints(self,waypoints):
        """
        hardcode a set of waypoints
        """
        self.target_list=waypoints

    def get_waypoints(self):
        return self.target_list
    
    def get_source_goal_nodes(self,method,graph_struct,node_ID_format,LayerGraphs,Last_layer,rng,k_min=5, k_max=8,max_tries=50):
        if method == "component_based":
            # components
            if self.graph.is_directed():
                comps = [list(c) for c in nx.weakly_connected_components(self.graph)]
            else:
                comps = [list(c) for c in nx.connected_components(self.graph)]

            comps = [c for c in comps if len(c) >= 2]
            if not comps:
                return None, None, None

            # pick the largest component
            comp = max(comps, key=len)
            H = self.graph.subgraph(comp)

            # We want the 5th–8th node on the path => distance 4..7 edges from s
            min_dist = 6 #k_min - 1
            max_dist = 20 #k_max - 1

            for _ in range(max_tries):
                s = rng.choice(list(H.nodes))

                # shortest-path distances from s up to max_dist edges
                lengths = nx.single_source_shortest_path_length(H, s, cutoff=max_dist)

                # candidates at distances 4..7 (so path has 5..8 nodes)
                candidates = [v for v, d in lengths.items() if min_dist <= d <= max_dist]
                if not candidates:
                    continue

                g = rng.choice(candidates)

                # shortest path gives you the actual node sequence
                path = nx.shortest_path(H, s, g)

                # pick the 5th–8th node on this path (index 4..7)
                idx_min = 4
                idx_max = min(7, len(path) - 1)
                mid = path[rng.randint(idx_min, idx_max)]

                first = path[0]
                self.source = first
                self.goal = mid
                return first, mid, path

            return None, None, None
        elif method == "longest_path":
            # Works for directed graphs
            largest_comp = max(nx.weakly_connected_components(self.graph), key=len)
            H = self.graph.subgraph(largest_comp)

            # Convert to undirected just for distance calculation
            H_undirected = H.to_undirected()

            # All-pairs shortest paths
            lengths = dict(nx.all_pairs_shortest_path_length(H_undirected))

            # Find the pair with the maximum shortest-path distance
            source, dest, dist = max(
                ((u, v, d) for u, vds in lengths.items() for v, d in vds.items()),
                key=lambda x: x[2]
            )
            self.source = source
            self.goal = dest
            return source, dest, dist
        elif method == "random_from_max":
            # Largest connected component
            if self.graph.is_directed():
                comp = max(nx.weakly_connected_components(self.graph), key=len)
            else:
                comp = max(nx.connected_components(self.graph), key=len)

            H = self.graph.subgraph(comp)

            # --- Step 1: approximate diameter (two BFS / Dijkstra sweeps) ---
            start = rng.choice(list(H.nodes))

            # First sweep
            dist1 = nx.single_source_shortest_path_length(H, start)
            far1 = max(dist1, key=dist1.get)

            # Second sweep
            dist2 = nx.single_source_shortest_path_length(H, far1)
            far2 = max(dist2, key=dist2.get)

            D_hat = dist2[far2]   # approximate diameter

            # Threshold for "near-maximum"
            alpha = 0.85
            min_dist = int(alpha * D_hat)

            # --- Step 2: random sampling near diameter ---
            nodes = list(H.nodes)

            for _ in range(max_tries):
                s = rng.choice(nodes)
                lengths = nx.single_source_shortest_path_length(H, s, cutoff=D_hat)

                candidates = [v for v, d in lengths.items() if d >= min_dist]
                if not candidates:
                    continue

                g = rng.choice(candidates)
                path = nx.shortest_path(H, s, g)

                self.source = s
                self.goal = g
                return s, g, path

            return None, None, None

        else:
            for _ in range(max_tries):

                source,goal = rng.sample(list(self.graph.nodes),k=2)

                # Existence + reachability
                if source in self.graph and goal in self.graph and nx.has_path(self.graph, source, goal):
                    self.source = source
                    self.goal = goal
                    return source, goal,""

            return None, None,None

    def get_src_goal(self):
        return [self.source,self.goal]


    def print_graph(self,title_):
        freeze_layout_in_graph(self.graph)
        fig, axes = plt.subplots(1, 1, figsize=(20, 14))
        def load_pos_from_graph(graph, x_attr="x", y_attr="y"):
            return {n: (float(graph.nodes[n][x_attr]), float(graph.nodes[n][y_attr])) for n in graph.nodes}

        # pos = load_pos_from_graph(self.graph)
        pos = compute_pos_spread(self.graph)
        # draw_astar_graph_V2(self.graph, pos, ax=axes, title=title_, with_labels=False, node_size=60)
        draw_astar_graph_V3(self.graph, pos, ax=axes, title=title_, with_labels=False, node_size=100) #100
        plt.tight_layout()
        plt.show()

    def get_h_n(self,nodes = None):
        h_n = {}
        if nodes == None:
            for node in self.graph.nodes():
                h_n[node] = self.h(node,self.goal)
            return h_n
        else:
            for node in nodes:
                h_n[node] = self.h(node,self.goal)
            return h_n

    def get_path_cost(self,path):
        total_cost = 0
        for node_idx in range(len(path)-1):
            node_1 = path[node_idx]
            node_2 =path[node_idx+1]
            total_cost += self.graph[node_1][node_2]["Cost"]

        return total_cost


    def save_print_graph(self,file_path, title_, graph_size=None):
        freeze_layout_in_graph(self.graph)

        fig, axes = plt.subplots(1, 1, figsize=(20, 14))

        def load_pos_from_graph(graph, x_attr="x", y_attr="y"):
            return {n: (float(graph.nodes[n][x_attr]), float(graph.nodes[n][y_attr])) for n in graph.nodes}

        # pos = load_pos_from_graph(self.graph)
        pos = compute_pos_spread(self.graph)

        draw_astar_graph_V3(
            self.graph,
            pos,
            ax=axes,
            title=title_,
            with_labels=False,
            node_size=100
        )

        plt.tight_layout()

        printing_path = f"{file_path}/graph_prints_{graph_size}/"
        Path(printing_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(printing_path+title_+".png", dpi=300, bbox_inches="tight")
        plt.close(fig)  # important to avoid memory issues

    def set_num_waypoints(self,num_waypoints):
        self.num_waypoints = num_waypoints
    def set_source(self,source):
        self.source = source
    def set_goal(self,goal):
        self.goal = goal



def ensure_src_dest_sets_for_graph(registry, gkey, set_count, pairs_per_set,
                                   model_name, functions, service_DR,
                                   graph_struct, node_ID_format, prompt_for_init,
                                   adj_list_format, len_layers,rng):
    if gkey not in registry["src_dest_sets"]:
        registry["src_dest_sets"][gkey] = {}

    for set_i in range(set_count):
        set_key = f"set_{set_i}"
        if set_key in registry["src_dest_sets"][gkey]:
            continue

        G = load_graph_from_registry(registry, gkey)



        path_finder = A_search(G, model_name, functions, service_DR, print_=False)

        pairs = []
        for _ in range(pairs_per_set):
            source, dest, _ = path_finder.get_source_goal_nodes(
                "random_from_max",
                graph_struct,
                node_ID_format,
                "network_graph",
                Last_layer=len_layers,
                rng=rng
            )
            while source is None:
                source, dest, _ = path_finder.get_source_goal_nodes(
                    "random_from_max",
                    graph_struct,
                    node_ID_format,
                    "network_graph",
                    Last_layer=len_layers,
                    rng=rng
                )
            pairs.append([source, dest])


        registry["src_dest_sets"][gkey][set_key] = {
            "method": "random_from_max",
            "pairs": pairs,
            "note": f"{pairs_per_set} pairs for itr_graph={pairs_per_set}"
        }









