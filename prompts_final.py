# ==================== adj_list | - , CoT, few_shot, limited waypoints

adj_list= """
You are a network optimization assistant tasked with finding the shortest path in a directed graph.

PROBLEM SETUP:
- Network representation: Directed graph as adjacency list
- Input: Start node {src}, goal node {goal}
- Output: Shortest path as ordered list of node IDs

AVAILABLE INFORMATION:
Adjacency list (graph structure):
{adjacency_list}


CRITICAL STRATEGY - TIE-BREAKING WITH LOOKAHEAD:
When choosing the next node, if two or more candidates have the same number of edges traversed:
1. Do NOT explore multiple branches
2. Instead, look ahead to their immediate neighbors
3. For each tied candidate, identify its neighbors
4. This lookahead ensures you pick the node that opens the best path forward

INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Find the path with the minimum total edges traversed
3. If there is a unique minimum, proceed to that node
4. ONLY in the case of ties at the total edges traversed until the current nodes, apply lookahead:
   - For each tied node, list its unvisited neighbors and number of hops
   - Find the minimum number of hops among each tied node's neighbors
   - Proceed to the tied node whose neighbors have the lowest minimum number of hops
5. Continue until reaching {goal}


INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Return the {num_waypoints} essential nodes from the path. The nodes must be from the begining of the path, from the middle, and from the end.


From the identified path, select a set of {num_waypoints} key waypoints.
Answer: [node_ID, node_ID, ..., {goal}]
"""

adj_list_CoT= """
You are a network optimization assistant tasked with finding the shortest path in a directed graph.

PROBLEM SETUP:
- Network representation: Directed graph as adjacency list
- Input: Start node {src}, goal node {goal}
- Output: Shortest path as ordered list of node IDs

AVAILABLE INFORMATION:
Adjacency list (graph structure):
{adjacency_list}


CRITICAL STRATEGY - TIE-BREAKING WITH LOOKAHEAD:
When choosing the next node, if two or more candidates have the same number of edges traversed:
1. Do NOT explore multiple branches
2. Instead, look ahead to their immediate neighbors
3. For each tied candidate, identify its neighbors
4. This lookahead ensures you pick the node that opens the best path forward

INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Find the path with the minimum total edges traversed
3. If there is a unique minimum, proceed to that node
4. ONLY in the case of ties at the total edges traversed until the current nodes, apply lookahead:
   - For each tied node, list its unvisited neighbors and number of hops
   - Find the minimum number of hops among each tied node's neighbors
   - Proceed to the tied node whose neighbors have the lowest minimum number of hops
5. Continue until reaching {goal}


INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Return the {num_waypoints} essential nodes from the path. The nodes must be from the begining of the path, from the middle, and from the end.


From the identified path, select a set of {num_waypoints} key waypoints.
SHOW your step-by-step reasoning, then provide your final answer as:
Answer: [node_ID, node_ID, ..., {goal}]
"""

adj_list_few_shot= """
You are a network optimization assistant tasked with finding the shortest path in a directed graph.

PROBLEM SETUP:
- Network representation: Directed graph as adjacency list
- Input: Start node {src}, goal node {goal}
- Output: Shortest path as ordered list of node IDs


CRITICAL STRATEGY - TIE-BREAKING WITH LOOKAHEAD:
When choosing the next node, if two or more candidates have the same number of edges traversed:
1. Do NOT explore multiple branches
2. Instead, look ahead to their immediate neighbors
3. For each tied candidate, identify its neighbors
4. This lookahead ensures you pick the node that opens the best path forward

INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Find the path with the minimum total edges traversed
3. If there is a unique minimum, proceed to that node
4. ONLY in the case of ties at the total edges traversed until the current nodes, apply lookahead:
   - For each tied node, list its unvisited neighbors and number of hops
   - Find the minimum number of hops among each tied node's neighbors
   - Proceed to the tied node whose neighbors have the lowest minimum number of hops
5. Continue until reaching {goal}


INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Return the {num_waypoints} essential nodes from the path. The nodes must be from the begining of the path, from the middle, and from the end.

WORKED EXAMPLE 1:
Adjacency list: {adj_list_eg_1}

Steps: {steps_eg_1}

--------
WORKED EXAMPLE 2:
Adjacency list: {adj_list_eg_2}

Steps: {steps_eg_2}
--------
--------

NOW SOLVE THIS PROBLEM:

Adjacency list:
{adjacency_list}

From the identified path, select a set of {num_waypoints} key waypoints.
SHOW your step-by-step reasoning, then provide your final answer as:
Answer: [node_ID, node_ID, ..., {goal}]
"""

adj_list_no_limit= """
You are a network optimization assistant tasked with finding the shortest path in a directed graph.

PROBLEM SETUP:
- Network representation: Directed graph as adjacency list
- Input: Start node {src}, goal node {goal}
- Output: Shortest path as ordered list of node IDs

AVAILABLE INFORMATION:
Adjacency list (graph structure):
{adjacency_list}


CRITICAL STRATEGY - TIE-BREAKING WITH LOOKAHEAD:
When choosing the next node, if two or more candidates have the same number of edges traversed:
1. Do NOT explore multiple branches
2. Instead, look ahead to their immediate neighbors
3. For each tied candidate, identify its neighbors
4. This lookahead ensures you pick the node that opens the best path forward

INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Find the path with the minimum total edges traversed
3. If there is a unique minimum, proceed to that node
4. ONLY in the case of ties at the total edges traversed until the current nodes, apply lookahead:
   - For each tied node, list its unvisited neighbors and number of hops
   - Find the minimum number of hops among each tied node's neighbors
   - Proceed to the tied node whose neighbors have the lowest minimum number of hops
5. Continue until reaching {goal}


INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Return the essential nodes from the path. The nodes must be from the begining of the path, from the middle, and from the end.


SHOW your step-by-step reasoning, then provide your final answer as:
Answer: [node_ID, node_ID, ..., {goal}]
"""

# ==================== adj_list_h_values | - , CoT, few_shot, limited waypoints
adj_list_h_values= """
You are a network optimization assistant tasked with finding the shortest path in a directed graph.

PROBLEM SETUP:
- Network representation: Directed graph as adjacency list
- Input: Start node {src}, goal node {goal}
- Output: Shortest path as ordered list of node IDs

AVAILABLE INFORMATION:
Adjacency list (graph structure):
{adjacency_list}

Heuristic values h(n) - estimated cost from each node to {goal}:
{h_n_values}

CRITICAL STRATEGY - TIE-BREAKING WITH LOOKAHEAD:
When choosing the next node, if two or more candidates have the same lowest h(n):
1. Do NOT explore multiple branches
2. Instead, look ahead to their immediate neighbors
3. For each tied candidate, identify its neighbors and their h(n) values
4. Choose the candidate whose neighbors have the LOWEST minimum h(n)
5. This lookahead ensures you pick the node that opens the best path forward

INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Refer to h(n) values to guide your search (lower h(n) suggests nodes closer to goal)
3. Find the path with minimum h(n) values and minimum total edges traversed
4. If there is a unique minimum, proceed to that node
5. ONLY in the case of ties at the h(n) values of the current nodes, apply lookahead:
   - For each tied node, list its unvisited neighbors and their h(n) values
   - Find the minimum h(n) among each tied node's neighbors
   - Proceed to the tied node whose neighbors have the lowest minimum h(n)
6. Continue until reaching {goal}


INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Refer to h(n) values to guide your search (lower h(n) suggests nodes closer to goal)
3. Find the path with minimum total edges traversed and minimum h(n) values
4. Return the {num_waypoints} essential nodes from the path. The nodes must be from the begining of the path, from the middle, and from the end.


From the identified path, select a set of {num_waypoints} key waypoints.
Answer: [node_ID, node_ID, ..., {goal}]
"""

adj_list_h_values_CoT= """
You are a network optimization assistant tasked with finding the shortest path in a directed graph.

PROBLEM SETUP:
- Network representation: Directed graph as adjacency list
- Input: Start node {src}, goal node {goal}
- Output: Shortest path as ordered list of node IDs

AVAILABLE INFORMATION:
Adjacency list (graph structure):
{adjacency_list}

Heuristic values h(n) - estimated cost from each node to {goal}:
{h_n_values}

CRITICAL STRATEGY - TIE-BREAKING WITH LOOKAHEAD:
When choosing the next node, if two or more candidates have the same lowest h(n):
1. Do NOT explore multiple branches
2. Instead, look ahead to their immediate neighbors
3. For each tied candidate, identify its neighbors and their h(n) values
4. Choose the candidate whose neighbors have the LOWEST minimum h(n)
5. This lookahead ensures you pick the node that opens the best path forward

INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Refer to h(n) values to guide your search (lower h(n) suggests nodes closer to goal)
3. Find the path with minimum h(n) values and minimum total edges traversed
4. If there is a unique minimum, proceed to that node
5. ONLY in the case of ties at the h(n) values of the current nodes, apply lookahead:
   - For each tied node, list its unvisited neighbors and their h(n) values
   - Find the minimum h(n) among each tied node's neighbors
   - Proceed to the tied node whose neighbors have the lowest minimum h(n)
6. Continue until reaching {goal}


INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node from node {src} to goal node {goal}
2. Refer to h(n) values to guide your search (lower h(n) suggests nodes closer to goal)
3. Find the path with minimum total edges traversed and minimum h(n) values
4. Return the {num_waypoints} essential nodes from the path. The nodes must be from the begining of the path, from the middle, and from the end.


From the identified path, select a set of {num_waypoints} key waypoints.
SHOW your step-by-step reasoning, then provide your final answer as:
Answer: [node_ID, node_ID, ..., {goal}]
"""


adj_list_h_values_few_shot= """
You are a network optimization assistant tasked with finding the shortest path in a directed graph.

PROBLEM SETUP:
- Network representation: Directed graph as adjacency list
- Input: Start node {src}, goal node {goal}
- Output: Shortest path as ordered list of node IDs

CRITICAL STRATEGY - TIE-BREAKING WITH LOOKAHEAD:
When choosing the next node, if two or more candidates have the same lowest h(n):
1. Do NOT explore multiple branches
2. Instead, look ahead to their immediate neighbors
3. For each tied candidate, identify its neighbors and their h(n) values
4. Choose the candidate whose neighbors have the LOWEST minimum h(n)
5. This lookahead ensures you pick the node that opens the best path forward

INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Refer to h(n) values to guide your search (lower h(n) suggests nodes closer to goal)
3. Find the path with minimum h(n) values and minimum total edges traversed
4. If there is a unique minimum, proceed to that node
5. ONLY in the case of ties at the h(n) values of the current nodes, apply lookahead:
   - For each tied node, list its unvisited neighbors and their h(n) values
   - Find the minimum h(n) among each tied node's neighbors
   - Proceed to the tied node whose neighbors have the lowest minimum h(n)
6. Continue until reaching {goal}


INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Refer to h(n) values to guide your search (lower h(n) suggests nodes closer to goal)
3. Find the path with minimum total edges traversed and minimum h(n) values
4. Return the {num_waypoints} essential nodes from the path. The nodes must be from the begining of the path, from the middle, and from the end.

WORKED EXAMPLE 1:
Adjacency list: {adj_list_eg_1}

Steps: {steps_eg_1}

--------
WORKED EXAMPLE 2:
Adjacency list: {adj_list_eg_2}

Steps: {steps_eg_2}
--------
--------

NOW SOLVE THIS PROBLEM:

Adjacency list:
{adjacency_list}

Heuristic values h(n) - estimated cost from each node to {goal}:
{h_n_values}

From the identified path, select a set of {num_waypoints} key waypoints.
SHOW your step-by-step reasoning, then provide your final answer as:
Answer: [node_ID, node_ID, ..., {goal}]
"""


adj_list_h_values_unlimited= """
You are a network optimization assistant tasked with finding the shortest path in a directed graph.

PROBLEM SETUP:
- Network representation: Directed graph as adjacency list
- Input: Start node {src}, goal node {goal}
- Output: Shortest path as ordered list of node IDs

AVAILABLE INFORMATION:
Adjacency list (graph structure):
{adjacency_list}

Heuristic values h(n) - estimated cost from each node to {goal}:
{h_n_values}

CRITICAL STRATEGY - TIE-BREAKING WITH LOOKAHEAD:
When choosing the next node, if two or more candidates have the same lowest h(n):
1. Do NOT explore multiple branches
2. Instead, look ahead to their immediate neighbors
3. For each tied candidate, identify its neighbors and their h(n) values
4. Choose the candidate whose neighbors have the LOWEST minimum h(n)
5. This lookahead ensures you pick the node that opens the best path forward

INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Refer to h(n) values to guide your search (lower h(n) suggests nodes closer to goal)
3. Find the path with minimum h(n) values and minimum total edges traversed
4. If there is a unique minimum, proceed to that node
5. ONLY in the case of ties at the h(n) values of the current nodes, apply lookahead:
   - For each tied node, list its unvisited neighbors and their h(n) values
   - Find the minimum h(n) among each tied node's neighbors
   - Proceed to the tied node whose neighbors have the lowest minimum h(n)
6. Continue until reaching {goal}


INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Refer to h(n) values to guide your search (lower h(n) suggests nodes closer to goal)
3. Find the path with minimum total edges traversed and minimum h(n) values
4. Return the essential nodes from the path. The nodes must be from the begining of the path, from the middle, and from the end.


SHOW your step-by-step reasoning, then provide your final answer as:
Answer: [node_ID, node_ID, ..., {goal}]
"""


adj_list_h_values_label = """
You are a network optimization assistant tasked with finding the shortest path in a directed graph.

PROBLEM SETUP:
- Network representation: Directed graph as adjacency list
- Input: Start node {src}, goal node {goal}
- Output: Shortest path as ordered list of node IDs

AVAILABLE INFORMATION:
Adjacency list (graph structure):
{adjacency_list}

- Directed edge quality groups (label -> list of directed edges):
{edge_to_label}

Edge quality order (best → worst):
excellent > good > ok > bad > terrible

CRITICAL STRATEGY - TIE-BREAKING WITH LOOKAHEAD:
When choosing the next node, if two or more candidates have the same lowest h(n):
1. Do NOT explore multiple branches
2. Instead, look ahead to their immediate neighbors
3. For each tied candidate, identify its neighbors and their h(n) values
4. Choose the candidate whose neighbors have the LOWEST minimum h(n)
5. This lookahead ensures you pick the node that opens the best path forward

INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Refer to h(n) values to guide your search (lower h(n) suggests nodes closer to goal)
3. Find the path with minimum h(n) values and minimum total edges traversed
4. If there is a unique minimum, proceed to that node
5. ONLY in the case of ties at the h(n) values of the current nodes, apply lookahead:
   - For each tied node, list its unvisited neighbors and their h(n) values
   - Find the minimum h(n) among each tied node's neighbors
   - Proceed to the tied node whose neighbors have the lowest minimum h(n)
6. Continue until reaching {goal}


INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Refer to h(n) values to guide your search (lower h(n) suggests nodes closer to goal)
3. Find the path with minimum total edges traversed and minimum h(n) values
4. Return the {num_waypoints} essential nodes from the path. The nodes must be from the begining of the path, from the middle, and from the end.


From the identified path, select a set of {num_waypoints} key waypoints.
Answer: [node_ID, node_ID, ..., {goal}]
"""

adj_list_h_values_label_set = """
You are a network optimization assistant tasked with finding the shortest path in a directed graph.

PROBLEM SETUP:
- Network representation: Directed graph as adjacency list
- Input: Start node {src}, goal node {goal}
- Output: Shortest path as ordered list of node IDs

AVAILABLE INFORMATION:
Adjacency list (graph structure):
{adjacency_list}

- Directed edge quality groups (label -> list of directed edges):
{label_to_edges}

Edge quality order (best → worst):
excellent > good > ok > bad > terrible

CRITICAL STRATEGY - TIE-BREAKING WITH LOOKAHEAD:
When choosing the next node, if two or more candidates have the same lowest h(n):
1. Do NOT explore multiple branches
2. Instead, look ahead to their immediate neighbors
3. For each tied candidate, identify its neighbors and their h(n) values
4. Choose the candidate whose neighbors have the LOWEST minimum h(n)
5. This lookahead ensures you pick the node that opens the best path forward

INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Refer to h(n) values to guide your search (lower h(n) suggests nodes closer to goal)
3. Find the path with minimum h(n) values and minimum total edges traversed
4. If there is a unique minimum, proceed to that node
5. ONLY in the case of ties at the h(n) values of the current nodes, apply lookahead:
   - For each tied node, list its unvisited neighbors and their h(n) values
   - Find the minimum h(n) among each tied node's neighbors
   - Proceed to the tied node whose neighbors have the lowest minimum h(n)
6. Continue until reaching {goal}


INSTRUCTIONS:
1. Use the adjacency list to identify valid moves from each node
2. Refer to h(n) values to guide your search (lower h(n) suggests nodes closer to goal)
3. Find the path with minimum total edges traversed and minimum h(n) values
4. Return the {num_waypoints} essential nodes from the path. The nodes must be from the begining of the path, from the middle, and from the end.


From the identified path, select a set of {num_waypoints} key waypoints.
Answer: [node_ID, node_ID, ..., {goal}]
"""

