# Example 1 - short
short_example_1 = """on node: 7
exploring neighbors
neighbors: 11 with h(n): 2.0 and g(n): 1
++++++
on node: 11
exploring neighbors
neighbors: 1 with h(n): 3.0 and g(n): 2
neighbors: 7 with h(n): 3.0 and g(n): 2
neighbors: 12 with h(n): 1.0 and g(n): 2
++++++
on node: 12
exploring neighbors
neighbors: 9 with h(n): 0.0 and g(n): 3
path found with total cost: 3
source 7, and destination: 9
the number of explored neighbours: 6 and number of generated nodes 3

Final answer: [7,11,12,9]
"""



# Example 2 - longer path
longer_path_example_2 = """on node: 4
exploring neighbors
neighbors: 2 with h(n): 4.0 and g(n): 1
neighbors: 5 with h(n): 6.0 and g(n): 1
neighbors: 10 with h(n): 6.0 and g(n): 1
++++++
on node: 2
exploring neighbors
neighbors: 0 with h(n): 3.0 and g(n): 2
neighbors: 13 with h(n): 5.0 and g(n): 2
neighbors: 4 with h(n): 5.0 and g(n): 2
++++++
on node: 0
exploring neighbors
neighbors: 1 with h(n): 2.0 and g(n): 3
neighbors: 14 with h(n): 2.0 and g(n): 3
++++++
on node: 1
exploring neighbors
neighbors: 11 with h(n): 1.0 and g(n): 4
neighbors: 3 with h(n): 3.0 and g(n): 4
++++++
on node: 14
exploring neighbors
neighbors: 9 with h(n): 1.0 and g(n): 4
neighbors: 6 with h(n): 3.0 and g(n): 4
++++++
on node: 11
exploring neighbors
neighbors: 7 with h(n): 2.0 and g(n): 5
neighbors: 12 with h(n): 0.0 and g(n): 5
++++++
on node: 9
exploring neighbors
neighbors: 8 with h(n): 2.0 and g(n): 5
path found with total cost: 5
source 4, and destination: 12
the number of explored neighbours: 16 and number of generated nodes 7

Final answer: [4,3,0,1,11,12]
"""

longer_path_example_2_waypoints = """on node: 4
exploring neighbors
neighbors: 2 with h(n): 4.0 and g(n): 1
neighbors: 5 with h(n): 6.0 and g(n): 1
neighbors: 10 with h(n): 6.0 and g(n): 1
++++++
on node: 2
exploring neighbors
neighbors: 0 with h(n): 3.0 and g(n): 2
neighbors: 13 with h(n): 5.0 and g(n): 2
neighbors: 4 with h(n): 5.0 and g(n): 2
++++++
on node: 0
exploring neighbors
neighbors: 1 with h(n): 2.0 and g(n): 3
neighbors: 14 with h(n): 2.0 and g(n): 3
++++++
on node: 1
exploring neighbors
neighbors: 11 with h(n): 1.0 and g(n): 4
neighbors: 3 with h(n): 3.0 and g(n): 4
++++++
on node: 14
exploring neighbors
neighbors: 9 with h(n): 1.0 and g(n): 4
neighbors: 6 with h(n): 3.0 and g(n): 4
++++++
on node: 11
exploring neighbors
neighbors: 7 with h(n): 2.0 and g(n): 5
neighbors: 12 with h(n): 0.0 and g(n): 5
++++++
on node: 9
exploring neighbors
neighbors: 8 with h(n): 2.0 and g(n): 5
path found with total cost: 5
source 4, and destination: 12
the number of explored neighbours: 16 and number of generated nodes 7

full path: [4,2,0,1,11,9,12]
final answer: [4,2,11,12]
"""


# Example 3 -short
short_example_3 = """on node: 13
exploring neighbors
neighbors: 2 with h(n): 1.0 and g(n): 1
neighbors: 5 with h(n): 3.0 and g(n): 1
++++++
on node: 2
exploring neighbors
neighbors: 0 with h(n): 0.0 and g(n): 2
neighbors: 13 with h(n): 2.0 and g(n): 2
neighbors: 4 with h(n): 2.0 and g(n): 2
path found with total cost: 2
source 13, and destination: 0
nodes: [13,2,0]
the number of explored neighbours: 6 and number of generated nodes 2

Example 4 also short:
++++++
on node: 12
exploring neighbors
neighbors: 11 with h(n): 3.0 and g(n): 1
neighbors: 9 with h(n): 1.0 and g(n): 1
++++++
on node: 9
exploring neighbors
neighbors: 12 with h(n): 2.0 and g(n): 2
neighbors: 14 with h(n): 0.0 and g(n): 2
neighbors: 8 with h(n): 2.0 and g(n): 2
path found with total cost: 2
source 12, and destination: 14
the number of explored neighbours: 6 and number of generated nodes 2

Final answer: [12,9,14]
"""

Adjacent_list_smaller_graph = """{{0: [1, 2, 14], 1: [0, 11, 3], 2: [0, 13, 4], 3: [1], 4: [2, 5, 10], 5: [13, 4], 6: [14], 7: [11], 8: [9], 9: [12, 14, 8], 10: [4], 11: [1, 7, 12], 12: [11, 9], 13: [2, 5], 14: [0, 9, 6]}}"""


# Example - with generated nodes not in final path:
graph_2_long_example_4 = """on node: 11
exploring neighbors
neighbors: 1 with h(n): 4.0 and g(n): 1
++++++
on node: 1
exploring neighbors
neighbors: 0 with h(n): 3.0 and g(n): 2
neighbors: 7 with h(n): 3.0 and g(n): 2
neighbors: 11 with h(n): 5.0 and g(n): 2
++++++
on node: 0
exploring neighbors
neighbors: 2 with h(n): 4.0 and g(n): 3
neighbors: 10 with h(n): 2.0 and g(n): 3
++++++
on node: 7
exploring neighbors
neighbors: 5 with h(n): 4.0 and g(n): 3
neighbors: 8 with h(n): 2.0 and g(n): 3
++++++
on node: 10
exploring neighbors
neighbors: 21 with h(n): 1.0 and g(n): 4
neighbors: 22 with h(n): 3.0 and g(n): 4
++++++
on node: 8
exploring neighbors
neighbors: 20 with h(n): 1.0 and g(n): 4
++++++
on node: 21
exploring neighbors
neighbors: 19 with h(n): 0.0 and g(n): 5
++++++
on node: 20
exploring neighbors
path found with total cost: 5
source 11, and destination: 19
the number of expanded neighbours: 13 and number of generated nodes 8

Final answer:[11,1,0,10,21,19]"""


graph_2_long_example_4_waypoints = """on node: 11
exploring neighbors
neighbors: 1 with h(n): 4.0 and g(n): 1
++++++
on node: 1
exploring neighbors
neighbors: 0 with h(n): 3.0 and g(n): 2
neighbors: 7 with h(n): 3.0 and g(n): 2
neighbors: 11 with h(n): 5.0 and g(n): 2
++++++
on node: 0
exploring neighbors
neighbors: 2 with h(n): 4.0 and g(n): 3
neighbors: 10 with h(n): 2.0 and g(n): 3
++++++
on node: 7
exploring neighbors
neighbors: 5 with h(n): 4.0 and g(n): 3
neighbors: 8 with h(n): 2.0 and g(n): 3
++++++
on node: 10
exploring neighbors
neighbors: 21 with h(n): 1.0 and g(n): 4
neighbors: 22 with h(n): 3.0 and g(n): 4
++++++
on node: 8
exploring neighbors
neighbors: 20 with h(n): 1.0 and g(n): 4
++++++
on node: 21
exploring neighbors
neighbors: 19 with h(n): 0.0 and g(n): 5
++++++
on node: 20
exploring neighbors
path found with total cost: 5
source 11, and destination: 19
the number of expanded neighbours: 13 and number of generated nodes 8

full path:[11,1,0,10,21,19]
final answer: [11,10,19]
"""


Adjacent_list_larger_graph = """{{0: [1, 2, 10], 1: [0, 7, 11], 2: [0, 9, 13], 3: [19], 4: [5], 5: [4, 6, 7], 6: [5], 7: [1, 5, 8], 8: [7, 20, 21], 9: [2, 12], 10: [0, 21, 22], 11: [1], 12: [9, 13, 14], 13: [2, 12, 18], 14: [12], 15: [18], 16: [18], 17: [18], 18: [13, 17, 15, 16], 19: [20, 21, 3], 20: [8, 19], 21: [8, 10, 19], 22: [10]}}"""


hueristic_list_larger_graph = """{{0: 0.0, 1: 1.0, 2: 1.0, 3: 4.0, 4: 4.0, 5: 3.0, 6: 4.0, 7: 2.0, 8: 3.0, 9: 2.0, 10: 1.0, 11: 2.0, 12: 3.0, 13: 2.0, 14: 4.0, 15: 4.0, 16: 4.0, 17: 4.0, 18: 3.0, 19: 3.0, 20: 4.0, 21: 2.0, 22: 2.0}}"""


# graph_2_long_example_4="""
# on node: 3
# exploring neighbors
# neighbors: 19 with h(n): 3.0

# on node: 19
# neighbors: 20 with h(n): 4.0 and 21 with h(n): 2.0

# on node 21:
# neighbors: 8 with h(n): 3.0 and 10 with h(n): 1.0

# on node 10:
# neighbors: 0 with h(n): 0.0 and 22 with h(n): 2.0

# on node 0:
# neighbors: 1 with h(n): 

# """