'''
base run - create graphs, src/dst pairs, run A* and LLM-A* with waypoints, save results to CSV and JSON
'''
from save_run_data import *
from graph_builder import *
from utils_4 import *
from A_search import *
import networkx as nx
from pathlib import Path
from prompts_final import *
import csv
from pathlib import Path
from tqdm import tqdm


itr_overall = 5
itr_graph = 10


prompts = [adj_list,adj_list_CoT,adj_list_no_limit,adj_list_few_shot,adj_list_h_values,adj_list_h_values_CoT,adj_list_h_values_unlimited,adj_list_h_values_few_shot]
prompt_names = ["adj_list","adj_list_CoT","adj_list_no_limit","adj_list_few_shot","adj_list_h_values","adj_list_h_values_CoT","adj_list_h_values_unlimited","adj_list_h_values_few_shot"]

prompt =  adj_list_h_values
prompt_name =  "adj_list_h_values"

# go back and rerun waypoints 7,8 for 1000
# graph_size = [3000,4000,5000]
graph_size =[750,1000]#,1500,2000] #,1000,2000] #,2500]
# graph_size = [1500,2000]
num_waypoints = [5] # [8,12]
# 5,8,12

model_name = "gpt-4.1-2025-04-14"
# model_name = "gpt-4o-mini-2024-07-18" 
k_landmarks = 20
# num_waypoints = 5
landmark_sel_tech = "furthest"
promptIDX = -1
# landmark_sel_tech = "ranrandom"
graph_type = "road_network"
# graph_type = "advogato"
# graph_type = "colt_tel"
# graph_type = "real"
# graph_type = "COMPLETE"
# graph_type = "BA"
# graph_type = "ER"

adj_list_format = "BFS" #"other" #"BFS"
node_ID_format = "int"#"int"#"L0_3"
graph_struct = "flat" #"flat" #layered
cost = "constant"# "constant" #"relative_to_degree"
g_n = 0

service_DR=200
functions=[{'ID':0, 'requirements':4}, {'ID':1, 'requirements':3}, {'ID':2, 'requirements':4}, {'ID':3, 'requirements':4}, {'ID':5, 'requirements':3}]
# functions=[{'ID':0, 'requirements':30}, {'ID':1, 'requirements':20}, {'ID':2, 'requirements':0.5}, {'ID':3, 'requirements':20}, {'ID':5, 'requirements':10}]
num_nodes = 1000
edge_degree = 4
t_n_weight=1

# ===================================  =================================== setup
printing_path = f"/Users/noufabbasi/Library/Mobile Documents/com~apple~CloudDocs/khalifa_2023-/Thesis/IRL_summer_2025/ToT_/SFC_LLM/output/tests_final_waypoints_test_v4_{graph_type}_cleaned_wapoints_{num_nodes}_{edge_degree}"
Path(printing_path).mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
total_filename = f"graph_{graph_type}_iterG{itr_graph}_iterO{itr_overall}_{timestamp}_node{num_nodes}_edge{edge_degree}_{graph_struct}_{node_ID_format}_{model_name}_landmarks{k_landmarks}_g_n_{g_n}_{adj_list_format}_num_wapoints{num_waypoints[0]}"
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
source_dest = {}
base_seed = 43
rng = random.Random(base_seed)

# =================================== Make one JSON file per experiment config (matches your CSV naming style)
json_filename = f"{total_filename}.json"  # or build it once per (num_nodes, waypoint)
json_path = Path(printing_path) / json_filename

logger = RunLogger(json_path=json_path)

# Optional: save experiment-level metadata once
logger.data["meta"].update({
    "model_name": model_name,
    "graph_type": graph_type,
    "graph_struct": graph_struct,
    "node_ID_format": node_ID_format,
    "adj_list_format": adj_list_format,
    "edge_degree": edge_degree,
    "k_landmarks": k_landmarks,
    "t_n_weight": t_n_weight,
})
logger.flush()
# ===================================


# # =================================== create graphs and src dest pairs
Path(printing_path+"/graph").mkdir(parents=True, exist_ok=True)
REGISTRY_PATH = Path(printing_path+"/graph") / "registry.json"
registry = load_registry(REGISTRY_PATH)
registry["landmarks"] = {}

graphs_dir = Path(printing_path) / "graphs"

for size in graph_size:            
    for version_i in range(itr_overall):      
        gkey = graph_key_for(size, version_i)
        if gkey in registry["graphs"]:
            continue  # already saved

        G, gpath = build_and_save_graph(
            size=size,
            version_i=version_i,
            edge_degree=edge_degree,
            graph_type=graph_type,
            cost=cost,
            graph_struct=graph_struct,
            node_ID_format=node_ID_format,
            functions=functions,
            service_DR=service_DR,
            base_seed=base_seed,
            out_dir=graphs_dir,
            rng=rng,
        )

        registry["graphs"][gkey] = {
            "file": str(gpath),
            "num_nodes": len(G.nodes),
            "num_edges": len(G.edges),
            "edge_degree_param": edge_degree,
            "graph_type": graph_type,
            "graph_struct": graph_struct,
            "cost_mode": cost,
            "seed_note": base_seed,
        }

save_registry(REGISTRY_PATH, registry)
# =======
registry = load_registry(REGISTRY_PATH)

for size in graph_size:
    for version_i in range(itr_overall):
        gkey = graph_key_for(size, version_i)
        ensure_src_dest_sets_for_graph(
            registry=registry,
            gkey=gkey,
            set_count=2,         
            pairs_per_set=itr_graph,
            model_name=model_name,
            functions=functions,
            service_DR=service_DR,
            graph_struct=graph_struct,
            node_ID_format=node_ID_format,
            prompt_for_init=prompt,    # not used for src/dest, but kept if you later want it
            adj_list_format=adj_list_format,
            len_layers=len(functions),
            rng=rng
        )

save_registry(REGISTRY_PATH, registry)
# ===================================


# for graph size
    # test 3,5,7,8 waypoints
    # repeat for each permutaion 30 times 

    # total of 120 tests per graph size and 4 different graph sizes (480 runs)


landmarks = {}
Prev_results_path_2 = Path("/Users/noufabbasi/Library/Mobile Documents/com~apple~CloudDocs/khalifa_2023-/Thesis/IRL_summer_2025/ToT_/SFC_LLM/output/tests_final_waypoints_test_v5_same_landmarks_smaller_sets_1000_100") / "graph_road_network_iterG2_iterO5_2026_02_05_14_05_59_node1000_edge100_flat_int_gpt-4.1-2025-04-14_landmarks20_g_n_0_BFS_num_wapoints5.json"
Prev_results_2 = json.loads(Prev_results_path_2.read_text())

for waypoint in num_waypoints:
    print("=================================== running for ",waypoint," waypoints")
    for num_nodes in graph_size: 
        print("=================================== setting graph size ",num_nodes)
        filename = f"waypoints_{waypoint}_graph_{graph_type}_iterG{itr_graph}_iterO{itr_overall}_{timestamp}_node{num_nodes}_edge{edge_degree}_{graph_struct}_{node_ID_format}_{prompt_name}_{model_name}_landmarks{k_landmarks}_g_n_{g_n}_{adj_list_format}_num_wapoints{num_waypoints[0]}"


        # ===================================  =================================== 
        with open(printing_path + "/" + filename+".csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            # writer.writerow(["Test No.", "# Nodes", "# Edges added", "total edges", "Source","goal","pompt","full output", "LLM_waypoints", "A* path", "A* explored nodes", "A* cost","cap_LLM A* path","cap_LLM A* explored nodes", "cap_LLMA* cost","optimal?","explore_gap"])  
            CSV_print = ["waypoint","overall_i","graph_i", "# Nodes", "total edges","# Edges added", "Source","goal"]

            CSV_print.extend(["waypoints_generated_provided_full_graph"]) #,"prompt_and_output_for_waypoints_generated_provided_full_graph"])
            CSV_print.extend(["A* path","A*cost","A_neighbors_enqueued", "A_pushed_count_revised_paths", "generated_neighbor_count", "A_expanded_count"])
            CSV_print.extend(["LLMA* path","LLMA*cost","LLMA*_neighbors_enqueued", "LLMA*_pushed_count_revised_paths", "LLMA*generated_neighbor_count", "LLMA*_expanded_count"])
            # CSV_print.extend(["LLMmode* path","LLMmode*cost","LLMmode*_neighbors_enqueued", "LLMmode*_pushed_count_revised_paths", "LLMmode*generated_neighbor_count", "LLMmode*_expanded_count"])
            CSV_print.extend(["SRC_goal* path","SRC_goal*cost","SRC_goal*_neighbors_enqueued", "SRC_goal*_pushed_count_revised_paths", "SRC_goal*generated_neighbor_count", "SRC_goal*_expanded_count"])
            CSV_print.extend(["LLM latency","input_tokens","output_tokens"])
            CSV_print.extend(["STAT-Cost inc.","STAT_neighbors_enqueued", "STAT_pushed_count_revised_paths", "STAT_generated_neighbor_count", "STAT_expanded_count"])
            # CSV_print.extend(["S_G_onlyA* path","S_G_onlyA*cost","S_G_only*_neighbors_enqueued", "S_G_only*_pushed_count_revised_paths", "S_G_only*generated_neighbor_count", "S_G_only*_expanded_count"])
            # CSV_print.extend(["S_G_only_STAT-Cost inc.","S_G_only_STAT_neighbors_enqueued", "S_G_only_STAT_pushed_count_revised_paths", "S_G_only_STAT_generated_neighbor_count", "S_G_only_STAT_expanded_count"])
            writer.writerow(CSV_print)
            # print(len(CSV_print))



        NC_A_dict={"path":[], "opt":0, "expanded_count_sum":[], "explored_neighbors_sum":[], "neighbors_enqueued_sum":[], "pushed_count_revised_paths_sum":[]}
        NC_LLM_dict={"path":[], "opt":0, "expanded_count_sum":[], "explored_neighbors_sum":[], "neighbors_enqueued_sum":[], "pushed_count_revised_paths_sum":[]}
        SRC_goal_dict={"path":[], "opt":0, "expanded_count_sum":[], "explored_neighbors_sum":[], "neighbors_enqueued_sum":[], "pushed_count_revised_paths_sum":[]}
        waypoint_lists = []
 

        for overall_i in range(itr_overall):
            # get graphs
            gkey = graph_key_for(num_nodes, overall_i)

            # check if we already have landmarks for this graph from prev runs
            if gkey in Prev_results_2["graphs"]:
                landmarks[gkey] = Prev_results_2["graphs"][gkey]["landmarks"]

            graph = load_graph_from_registry(registry, gkey)
            graph_layers = None



            path_finder = A_search(graph, model_name, functions, service_DR, print_=False)
            # path_finder.print_graph("graph_v1")
            
            # =================================== get landmarks
            if gkey in landmarks:
                landmark = landmarks[gkey]
                path_finder.set_landmarks(landmark)

            else:
                print("=================================== generating landmarks")
                landmark_sel_tech = "furthest"
                path_finder.generate_landmarks(graph_struct, landmark_sel_tech, k_landmarks, graph_layers, rng)
                landmark=path_finder.get_landmarks()
                landmarks[gkey] = landmark

            registry["landmarks"][gkey] = landmark



            # choose which of the 2 sets you want this run to use
            chosen_set = "set_0"  # or "set_1" or loop over both
            pairs = registry["src_dest_sets"][gkey][chosen_set]["pairs"]

            # ======================================================================
            # ============================================= JSON logging of graph===
            # graph_key = f"size_{num_nodes}_overall_{overall_i}"  
            # Path(printing_path+"/graph").mkdir(parents=True, exist_ok=True)
            # graph_edgelist_path = Path(printing_path+"/graph") / f"{graph_key}.edgelist"

            # Save graph once per overall_i (avoid overwriting every graph_i run)
            # nx.write_edgelist(graph, graph_edgelist_path, delimiter=" ", data=["weight"])

            logger.register_graph(
                graph_key=gkey,
                graph_file=str(REGISTRY_PATH),
                graph_info={
                    "num_nodes": len(graph.nodes),
                    "num_edges": len(graph.edges),
                    "edge_degree_param": edge_degree,
                    "graph_type": graph_type,
                    "graph_struct": graph_struct,
                    "cost_mode": cost,
                    "landmarks": path_finder.get_landmarks(),
                }
            )
            # ======================================================================



            # # rerun the search on the same graph (diff source and dest)
            for graph_i, (source, dest) in enumerate(pairs):
                # =================================== set source and dest
                run_key = f"run_{overall_i}_{graph_i}_waypoint_{waypoint}_nodes_{num_nodes}"
                path_finder.set_source(source)
                path_finder.set_goal(dest)

                    
                # =================================== generate waypoints
                print("=================================== generating waypoints")
                # query_flat,full_output_flat, t_list_flat,LLM_info = ["","",[1],{"latency_s":3, "input_tokens":3 ,"output_tokens":3 ,"total_tokens": 3}] #self._initialize_llm_paths(False)
                path_finder.set_num_waypoints(waypoint)
                # if gkey in Prev_results["runs"] and Prev_results["runs"][gkey]["waypoint_k"] == waypoint and source == Prev_results["runs"][gkey]["source"] and dest == Prev_results["runs"][gkey]["dest"]:
                #     query_flat,full_output_flat, t_list_flat,LLM_info = [Prev_results["runs"][gkey]["prompt"],Prev_results["runs"][gkey]["output"],Prev_results["generated_waypoints"][gkey]["output"],Prev_results["generated_waypoints"][gkey]["llm_info"]] #self._initialize_llm_paths(False)
                #     path_finder.set_waypoints(Prev_results["runs"][gkey]["generated_waypoints"])

                # else:
                # query_flat,full_output_flat, t_list_flat,LLM_info = path_finder._initialize_llm_paths(graph_struct=="layered", graph_layers,node_ID_format,prompt,len_layers=len(functions),adj_list_format=adj_list_format)
                query_flat,full_output_flat, t_list_flat,LLM_info = path_finder._initialize_llm_paths_limited(graph_struct=="layered", graph_layers,node_ID_format,prompt,len_layers=len(functions),adj_list_format=adj_list_format)
                waypoint_lists.append(t_list_flat)


                # =================================== run shortest path_alg
                NC_A= [0,0,0]
                NC_LLMA= [0,0,0]
                SRC_goal = [0,0,0]

                # Note: the "graph saving" process happens within the astar function
                # ====== normal A*
                NC_A = path_finder.astar_path_constrained_flat(test_name="A_star")
                # path_finder.print_graph(f"A* graph{overall_i}_{graph_i}_{source},{dest}")

                # ====== LLM A*
                # NC_LLMA = path_finder.astar_path_constrained_flat(test_name="A_star",g_n=g_n)
                NC_LLMA = path_finder.astar_path_constrained_LLM(test_name="LLM_A",t_n_weight=t_n_weight)
                # path_finder.print_graph(f"LLM A* graph{overall_i}_{graph_i}_{source},{dest}")

                # ====== normal A*
                path_finder.set_waypoints([source,dest])
                SRC_goal = path_finder.astar_path_constrained_LLM(test_name="Src_Dest",t_n_weight=t_n_weight)
                
                # =================================== print to file
                CSV_print = [waypoint,overall_i,graph_i, len(graph.nodes), len(graph.edges),edge_degree, source, dest]
                CSV_print.append(t_list_flat)

                NC_A_dict["path"].append(NC_A[0])
                NC_A_dict["opt"] += 1 if NC_A[1] == NC_LLMA[1] else 0
                NC_A_dict["expanded_count_sum"].append(NC_A[2]['expanded_count'])
                NC_A_dict["explored_neighbors_sum"].append(NC_A[2]['generated_count'])
                NC_A_dict["neighbors_enqueued_sum"].append(NC_A[2]['enqueued_size'])
                NC_A_dict["pushed_count_revised_paths_sum"].append(NC_A[2]['pushed count'])

                NC_LLM_dict["path"].append(NC_LLMA[0])
                NC_LLM_dict["opt"] += 1 if NC_A[1] == NC_LLMA[1] else 0
                NC_LLM_dict["expanded_count_sum"].append(NC_LLMA[2]['expanded_count'])
                NC_LLM_dict["explored_neighbors_sum"].append(NC_LLMA[2]['generated_count'])
                NC_LLM_dict["neighbors_enqueued_sum"].append(NC_LLMA[2]['enqueued_size'])
                NC_LLM_dict["pushed_count_revised_paths_sum"].append(NC_LLMA[2]['pushed count'])

                SRC_goal_dict["path"].append(SRC_goal[0])
                SRC_goal_dict["opt"] += 1 if NC_A[1] == SRC_goal[1] else 0
                SRC_goal_dict["expanded_count_sum"].append(SRC_goal[2]['expanded_count'])
                SRC_goal_dict["explored_neighbors_sum"].append(SRC_goal[2]['generated_count'])
                SRC_goal_dict["neighbors_enqueued_sum"].append(SRC_goal[2]['enqueued_size'])
                SRC_goal_dict["pushed_count_revised_paths_sum"].append(SRC_goal[2]['pushed count'])


                # store all the values
                CSV_print.extend(NC_A[:2])
                for key,value in NC_A[2].items():
                    CSV_print.append(value)

                CSV_print.extend(NC_LLMA[:2])
                for key,value in NC_LLMA[2].items():
                    CSV_print.append(value)



                CSV_print.extend(SRC_goal[:2])
                for key,value in SRC_goal[2].items():
                    CSV_print.append(value)
                
                # SAVE STAT
                CSV_print.extend([LLM_info["latency_s"],LLM_info["input_tokens"],LLM_info["output_tokens"]])
                CSV_print.extend([NC_LLMA[1]-NC_A[1]])
                for key,value in NC_A[2].items():
                    CSV_print.append(NC_LLMA[2][key]-value)

                # print(len(CSV_print))
                print("writing to: ",printing_path, "/", filename)
                append_csv_row(printing_path + "/"+ filename,CSV_print)

                # ======================================================================
                # ========================================== loggin current iteration===
                

                run_payload = {
                    "waypoint_graph_key": run_key,
                    "graph_key": gkey,
                    "waypoint_k": waypoint,
                    "source": source,
                    "dest": dest,


                    "prompt_name": prompt_name,
                    "prompt": query_flat,                 # or query_flat if you want the fully-rendered prompt text
                    "output": full_output_flat,        # model raw output text
                    "generated_waypoints": t_list_flat,

                    "AStar_generated_path": NC_A[0],
                    "AStar_path_cost": NC_A[1],
                    "AStar_stats": NC_A[2],

                    "LLMAStar_generated_path": NC_LLMA[0],
                    "LLMAStar_path_cost": NC_LLMA[1],
                    "LLMAStar_stats": NC_LLMA[2],

                    "SrcDest_generated_path": SRC_goal[0],
                    "SrcDest_path_cost": SRC_goal[1],
                    "SrcDest_stats": SRC_goal[2],

                    "llm_info": LLM_info,
                    "landmarks": path_finder.get_landmarks(),
                }

                logger.add_run(run_key, run_payload)


        # print("writing final results for graph size ",num_nodes," with ",waypoint," waypoints to: ",printing_path, "/", filename)
        # writer.writerow(CSV_print)
        path_finder.print_to_CSV(total_filename, printing_path, prompt_name, NC_A_dict, NC_LLM_dict,SRC_goal_dict, overall_i, num_nodes, len(graph.edges),waypoint_lists,waypoint=waypoint)
REGISTRY_PATH = Path(printing_path+"/graph") / "registry.json"
save_registry(REGISTRY_PATH, registry)






# import json
# import pickle
# from pathlib import Path
# from tqdm import tqdm

# from A_search import A_search  # your class

# def load_graph(graph_path: str):
#     with open(graph_path, "rb") as f:
#         return pickle.load(f)

# def replay_runs(json_path: str, functions, service_DR, model_name, t_n_weight=1):
#     data = json.loads(Path(json_path).read_text())

#     graphs = data["graphs"]
#     runs = data["runs"]

#     results = {}  # store rerun outputs if you want

    
#     for run_key, run in tqdm(runs.items(), desc="Replaying runs"):
#         gkey = run["graph_key"]

#         graph_file = "/Users/noufabbasi/Library/Mobile Documents/com~apple~CloudDocs/khalifa_2023-/Thesis/IRL_summer_2025/ToT_/SFC_LLM/output/tests_final_waypoints_test_v3_1000_100/graphs/"+gkey + ".edgelist"
#         if graph_file.endswith("registry.json"):
#             raise ValueError(
#                 f"Graph file path for {gkey} points to registry.json, not a graph: {graph_file}"
#             )

#         G = load_graph(graph_file)

#         # build path_finder
#         path_finder = A_search(G, model_name, functions, service_DR, print_=False)

#         # set src/dst
#         source = run["source"]
#         dest = run["dest"]
#         path_finder.set_source(source)
#         path_finder.set_goal(dest)

#         # set waypoints from stored LLM output
#         waypoints = run["generated_waypoints"]
#         path_finder.set_waypoints(waypoints)

#         # ---- rerun the exact section you highlighted ----
#         NC_A = path_finder.astar_path_constrained_flat(test_name="A_star")
#         NC_LLMA = path_finder.astar_path_constrained_LLM(test_name="LLM_A", t_n_weight=t_n_weight)

#         # Src/Dest baseline (force waypoints=[src,dst])
#         path_finder.set_waypoints([source, dest])
#         SRC_goal = path_finder.astar_path_constrained_LLM(test_name="Src_Dest", t_n_weight=t_n_weight)

#         results[run_key] = {
#             "AStar_generated_path": NC_A[0],
#             "AStar_path_cost": NC_A[1],
#             "AStar_stats": NC_A[2],
#             "LLMAStar_generated_path": NC_LLMA[0],
#             "LLMAStar_path_cost": NC_LLMA[1],
#             "LLMAStar_stats": NC_LLMA[2],
#             "SrcDest_generated_path": SRC_goal[0],
#             "SrcDest_path_cost": SRC_goal[1],
#             "SrcDest_stats": SRC_goal[2],
#         }

#     return results

# # Example usage:

# service_DR=200
# functions=[{'ID':0, 'requirements':4}, {'ID':1, 'requirements':3}, {'ID':2, 'requirements':4}, {'ID':3, 'requirements':4}, {'ID':5, 'requirements':3}]
# json_path = "/Users/noufabbasi/Library/Mobile Documents/com~apple~CloudDocs/khalifa_2023-/Thesis/IRL_summer_2025/ToT_/SFC_LLM/output/tests_final_waypoints_test_v3_1000_100/json_test.json"
# model_name = "none"
# results = replay_runs(json_path, functions, service_DR, model_name, t_n_weight=1)
# Path("rerun_results.json").write_text(json.dumps(results, indent=2))


# ++++++++++
# ++++++++++
# ++++++++++
# ++++++++++
# ++++++++++
# ++++++++++



# from save_run_data import *
# from graph_builder import *
# from utils_4 import *
# from A_search import *
# import networkx as nx
# from pathlib import Path
# from prompts_final import *
# import csv
# from pathlib import Path
# from tqdm import tqdm

# itr_overall = 6
# itr_graph = 2
# # itr_overall = 1
# # itr_graph = 1

# prompts = [adj_list,adj_list_CoT,adj_list_no_limit,adj_list_few_shot,adj_list_h_values,adj_list_h_values_CoT,adj_list_h_values_unlimited,adj_list_h_values_few_shot]
# prompt_names = ["adj_list","adj_list_CoT","adj_list_no_limit","adj_list_few_shot","adj_list_h_values","adj_list_h_values_CoT","adj_list_h_values_unlimited","adj_list_h_values_few_shot"]

# prompt =  adj_list_h_values
# prompt_name =  "adj_list_h_values"

# # go back and rerun waypoints 7,8 for 1000
# # graph_size = [3000,4000,5000]
# graph_size =[500,750,1000]
# # graph_size = [1500,2000]
# num_waypoints = [5,7,8,12]

# model_name = "gpt-4.1-2025-04-14"
# # model_name = "gpt-4o-mini-2024-07-18" 
# k_landmarks = 20
# # num_waypoints = 5
# landmark_sel_tech = "furthest"
# promptIDX = -1
# # landmark_sel_tech = "ranrandom"
# graph_type = "road_network"
# # graph_type = "advogato"
# # graph_type = "colt_tel"
# # graph_type = "real"
# # graph_type = "COMPLETE"
# # graph_type = "BA"
# # graph_type = "ER"

# adj_list_format = "BFS" #"other" #"BFS"
# node_ID_format = "int"#"int"#"L0_3"
# graph_struct = "flat" #"flat" #layered
# cost = "constant"# "constant" #"relative_to_degree"
# g_n = 0

# service_DR=200
# functions=[{'ID':0, 'requirements':4}, {'ID':1, 'requirements':3}, {'ID':2, 'requirements':4}, {'ID':3, 'requirements':4}, {'ID':5, 'requirements':3}]
# # functions=[{'ID':0, 'requirements':30}, {'ID':1, 'requirements':20}, {'ID':2, 'requirements':0.5}, {'ID':3, 'requirements':20}, {'ID':5, 'requirements':10}]
# num_nodes = 1000
# edge_degree = 100
# t_n_weight=1

# # ===================================  =================================== setup
# printing_path = f"/Users/noufabbasi/Library/Mobile Documents/com~apple~CloudDocs/khalifa_2023-/Thesis/IRL_summer_2025/ToT_/SFC_LLM/output/tests_final_waypoints_test_{num_nodes}_{edge_degree}"
# Path(printing_path).mkdir(parents=True, exist_ok=True)
# timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# total_filename = f"graph_{graph_type}_iterG{itr_graph}_iterO{itr_overall}_{timestamp}_node{num_nodes}_edge{edge_degree}_{graph_struct}_{node_ID_format}_{model_name}_landmarks{k_landmarks}_g_n_{g_n}_{adj_list_format}_num_wapoints{num_waypoints}"
# timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# source_dest = {}

# # =================================== Make one JSON file per experiment config (matches your CSV naming style)
# json_filename = f"{total_filename}.json"  # or build it once per (num_nodes, waypoint)
# json_path = Path(printing_path) / json_filename

# logger = RunLogger(json_path=json_path)

# # Optional: save experiment-level metadata once
# logger.data["meta"].update({
#     "model_name": model_name,
#     "graph_type": graph_type,
#     "graph_struct": graph_struct,
#     "node_ID_format": node_ID_format,
#     "adj_list_format": adj_list_format,
#     "edge_degree": edge_degree,
#     "k_landmarks": k_landmarks,
#     "t_n_weight": t_n_weight,
# })
# logger.flush()
# # ===================================


# # =================================== create graphs and src dest pairs
# registry = load_registry(REGISTRY_PATH)

# graphs_dir = Path(printing_path) / "graphs"
# base_seed = 43

# for size in graph_size:            
#     for version_i in range(itr_overall):      
#         gkey = graph_key_for(size, version_i)
#         if gkey in registry["graphs"]:
#             continue  # already saved

#         G, gpath = build_and_save_graph(
#             size=size,
#             version_i=version_i,
#             edge_degree=edge_degree,
#             graph_type=graph_type,
#             cost=cost,
#             graph_struct=graph_struct,
#             node_ID_format=node_ID_format,
#             functions=functions,
#             service_DR=service_DR,
#             base_seed=base_seed,
#             out_dir=graphs_dir,
#         )

#         registry["graphs"][gkey] = {
#             "file": str(gpath),
#             "num_nodes": len(G.nodes),
#             "num_edges": len(G.edges),
#             "edge_degree_param": edge_degree,
#             "graph_type": graph_type,
#             "graph_struct": graph_struct,
#             "cost_mode": cost,
#             "seed_note": f"seed = base_seed + ver + 10000*size",
#         }

# save_registry(REGISTRY_PATH, registry)
# # =======
# registry = load_registry(REGISTRY_PATH)

# for size in graph_size:
#     for version_i in range(6):
#         gkey = graph_key_for(size, version_i)
#         ensure_src_dest_sets_for_graph(
#             registry=registry,
#             gkey=gkey,
#             set_count=2,          # two sets
#             pairs_per_set=itr_graph,  # usually 2
#             model_name=model_name,
#             functions=functions,
#             service_DR=service_DR,
#             graph_struct=graph_struct,
#             node_ID_format=node_ID_format,
#             prompt_for_init=prompt,    # not used for src/dest, but kept if you later want it
#             adj_list_format=adj_list_format,
#             len_layers=len(functions),
#         )

# save_registry(REGISTRY_PATH, registry)

# # ===================================


# # for graph size
#     # test 3,5,7,8 waypoints
#     # repeat for each permutaion 30 times 

#     # total of 120 tests per graph size and 4 different graph sizes (480 runs)
# for waypoint in num_waypoints:
#     print("=================================== running for ",waypoint," waypoints")
#     rng = random.Random(43)
#     for num_nodes in graph_size:
#         print(num_nodes)
#         if (num_nodes <=1000 and waypoint ==5) or (num_nodes ==1500 and waypoint ==5):
#             continue
#         print("=================================== setting graph size ",num_nodes)
#         filename = f"waypoints_{waypoint}_graph_{graph_type}_iterG{itr_graph}_iterO{itr_overall}_{timestamp}_node{num_nodes}_edge{edge_degree}_{graph_struct}_{node_ID_format}_{prompt_name}_{model_name}_landmarks{k_landmarks}_g_n_{g_n}_{adj_list_format}_num_wapoints{num_waypoints[0]}"
#     #     print("writing to: ",printing_path,filename)


#         # ===================================  =================================== 
#         with open(printing_path + "/" + filename+".csv", mode='w', newline='') as file:
#             writer = csv.writer(file)
#             # writer.writerow(["Test No.", "# Nodes", "# Edges added", "total edges", "Source","goal","pompt","full output", "LLM_waypoints", "A* path", "A* explored nodes", "A* cost","cap_LLM A* path","cap_LLM A* explored nodes", "cap_LLMA* cost","optimal?","explore_gap"])  
#             CSV_print = ["waypoint","overall_i","graph_i", "# Nodes", "total edges","# Edges added", "Source","goal"]

#             CSV_print.extend(["waypoints_generated_provided_full_graph"]) #,"prompt_and_output_for_waypoints_generated_provided_full_graph"])
#             CSV_print.extend(["A* path","A*cost","A_neighbors_enqueued", "A_pushed_count_revised_paths", "generated_neighbor_count", "A_expanded_count"])
#             CSV_print.extend(["LLMA* path","LLMA*cost","LLMA*_neighbors_enqueued", "LLMA*_pushed_count_revised_paths", "LLMA*generated_neighbor_count", "LLMA*_expanded_count"])
#             # CSV_print.extend(["LLMmode* path","LLMmode*cost","LLMmode*_neighbors_enqueued", "LLMmode*_pushed_count_revised_paths", "LLMmode*generated_neighbor_count", "LLMmode*_expanded_count"])
#             CSV_print.extend(["SRC_goal* path","SRC_goal*cost","SRC_goal*_neighbors_enqueued", "SRC_goal*_pushed_count_revised_paths", "SRC_goal*generated_neighbor_count", "SRC_goal*_expanded_count"])
#             CSV_print.extend(["LLM latency","input_tokens","output_tokens"])
#             CSV_print.extend(["STAT-Cost inc.","STAT_neighbors_enqueued", "STAT_pushed_count_revised_paths", "STAT_generated_neighbor_count", "STAT_expanded_count"])
#             # CSV_print.extend(["S_G_onlyA* path","S_G_onlyA*cost","S_G_only*_neighbors_enqueued", "S_G_only*_pushed_count_revised_paths", "S_G_only*generated_neighbor_count", "S_G_only*_expanded_count"])
#             # CSV_print.extend(["S_G_only_STAT-Cost inc.","S_G_only_STAT_neighbors_enqueued", "S_G_only_STAT_pushed_count_revised_paths", "S_G_only_STAT_generated_neighbor_count", "S_G_only_STAT_expanded_count"])
#             writer.writerow(CSV_print)
#             # print(len(CSV_print))



#         NC_A_dict={"path":[], "opt":0, "expanded_count_sum":[], "explored_neighbors_sum":[], "neighbors_enqueued_sum":[], "pushed_count_revised_paths_sum":[]}
#         NC_LLM_dict={"path":[], "opt":0, "expanded_count_sum":[], "explored_neighbors_sum":[], "neighbors_enqueued_sum":[], "pushed_count_revised_paths_sum":[]}
#         SRC_goal_dict={"path":[], "opt":0, "expanded_count_sum":[], "explored_neighbors_sum":[], "neighbors_enqueued_sum":[], "pushed_count_revised_paths_sum":[]}
#         waypoint_lists = []


#         for overall_i in range(itr_overall):
#             print("=================================== overall iteration ",overall_i)
#             # =================================== setup
#             SFC_graph_builder = SFCGraphBuilder(functions,service_DR=service_DR)

#             # =================================== Create graph
#             # SFC_graph_builder.create_graph(num_nodes, edge_degree,rng,type="ER")
#             # SFC_graph_builder.create_graph(num_nodes, edge_degree,rng,type_="BA")
#             SFC_graph_builder.create_graph(num_nodes, edge_degree,rng,type_=graph_type,k=edge_degree)

#             SFC_graph_builder.conver_to_weighted_graph(cost,rng)
#             if graph_struct == "layered":
#                 if node_ID_format == "L0_3":
#                     SFC_graph_builder.get_layered_graph_v2()
#                 else:
#                     SFC_graph_builder.get_layered_graph()
#             graph = SFC_graph_builder.get_graph(graph_struct)

#             # ======================================================================
#             # ============================================= JSON logging of graph===
#             graph_key = f"size_{num_nodes}_overall_{overall_i}"  
#             Path(printing_path+"/graph").mkdir(parents=True, exist_ok=True)
#             graph_edgelist_path = Path(printing_path+"/graph") / f"{graph_key}.edgelist"

#             # Save graph once per overall_i (avoid overwriting every graph_i run)
#             nx.write_edgelist(graph, graph_edgelist_path, delimiter=" ", data=["weight"])

#             logger.register_graph(
#                 graph_key=graph_key,
#                 graph_file=str(graph_edgelist_path),
#                 graph_info={
#                     "num_nodes": len(graph.nodes),
#                     "num_edges": len(graph.edges),
#                     "edge_degree_param": edge_degree,
#                     "graph_type": graph_type,
#                     "graph_struct": graph_struct,
#                     "cost_mode": cost,
#                 }
#             )
#             # ======================================================================

#             print(f"=================================== built a graph -{graph_type} with {len(graph.nodes)} nodes and {len(graph.edges)}")
#             if graph_struct == "flat":
#                 graph_layers = None
#             else:
#                 graph_layers = SFC_graph_builder.get_graph("graph_layers")
#             path_finder = A_search(graph,model_name,functions,service_DR,print_=False)
#             # path_finder.print_graph(f"A* graph{overall_i}_{4}_{3},{1}")


#             # =================================== get landmarks
#             print("=================================== generating landmarks")
#             # path_finder.generate_landmarks(graph_struct,landmark_sel_tech,k_landmarks,graph_layers,rng)
#             path_finder.generate_landmarks(graph_struct,landmark_sel_tech,k_landmarks,graph_layers,rng)

#             # rerun the search on the same graph (diff source and dest)
#             for graph_i in range(itr_graph):   
#                 print("=================================== graph iteration ",graph_i)
#                 print(waypoint, "waypoints test - iteration ",graph_i)
                
#                 # =================================== generate source and goal nodes
#                 if f"{overall_i}_{graph_i}_{num_nodes}" not in source_dest:
#                     print("generating source and destination pair")
#                     print("=================================== generating source and destination pair")
#                     # source,dest,_ = path_finder.get_source_goal_nodes("source_goal",graph_struct, node_ID_format,graph_layers,Last_layer=len(functions),rng=rng)
#                     # source,dest,_ = path_finder.get_source_goal_nodes("component_based",graph_struct, node_ID_format,graph_layers,Last_layer=len(functions),rng=rng)
#                     # source,dest,_ = path_finder.get_source_goal_nodes("longest_path",graph_struct, node_ID_format,graph_layers,Last_layer=len(functions),rng=rng)
#                     source,dest,_ = path_finder.get_source_goal_nodes("random_from_max",graph_struct, node_ID_format,graph_layers,Last_layer=len(functions),rng=rng)
#                     # source,dest,_ = path_finder.get_source_goal_nodes("source_goal",graph_struct, node_ID_format,graph_layers,Last_layer=len(functions),rng=rng)
#                     while source == None:
#                         source,dest,_ = path_finder.get_source_goal_nodes("source_goal",graph_struct, node_ID_format,graph_layers,Last_layer=len(functions),rng=rng)
#                         print("random points not found")
#                         pass
#                     source_dest[f"{overall_i}_{graph_i}_{num_nodes}"] = (source,dest)
#                 else:
#                     print("using cashed source and destination pair")
#                     source,dest = source_dest[f"{overall_i}_{graph_i}_{num_nodes}"]
                    
#                 # =================================== generate waypoints
#                 print("=================================== generating waypoints")
#                 # query_flat,full_output_flat, t_list_flat,LLM_info = ["","",[1],{"latency_s":3, "input_tokens":3 ,"output_tokens":3 ,"total_tokens": 3}] #self._initialize_llm_paths(False)
#                 path_finder.set_num_waypoints(waypoint)
#                 query_flat,full_output_flat, t_list_flat,LLM_info = path_finder._initialize_llm_paths(graph_struct=="layered", graph_layers,node_ID_format,prompt,len_layers=len(functions),adj_list_format=adj_list_format)
#                 waypoint_lists.append(t_list_flat)
#                 # path_finder.set_waypoints([source,dest])
#                 # print(full_output_flat)
#                 # if graph_i < 2:
#                 #     print(query_flat)
#                 # path_finder.print_graph(f"graph{overall_i}_{graph_i}_{source},{dest}")

#                 # =================================== run shortest path_alg
#                 NC_A= [0,0,0]
#                 NC_LLMA= [0,0,0]
#                 SRC_goal = [0,0,0]

#                 # Note: the "graph saving" process happens within the astar function
#                 # ====== normal A*
#                 NC_A = path_finder.astar_path_constrained_flat(test_name="A_star")
#                 # path_finder.print_graph(f"A* graph{overall_i}_{graph_i}_{source},{dest}")

#                 # ====== LLM A*
#                 # NC_LLMA = path_finder.astar_path_constrained_flat(test_name="A_star",g_n=g_n)
#                 NC_LLMA = path_finder.astar_path_constrained_LLM(test_name="LLM_A",t_n_weight=t_n_weight)
#                 # path_finder.print_graph(f"LLM A* graph{overall_i}_{graph_i}_{source},{dest}")

#                 # ====== normal A*
#                 path_finder.set_waypoints([source,dest])
#                 SRC_goal = path_finder.astar_path_constrained_LLM(test_name="Src_Dest",t_n_weight=t_n_weight)
                
#                 # =================================== print to file
#                 CSV_print = [waypoint,overall_i,graph_i, len(graph.nodes), len(graph.edges),edge_degree, source, dest]
#                 CSV_print.append(t_list_flat)

#                 NC_A_dict["path"].append(NC_A[0])
#                 NC_A_dict["opt"] += 1 if NC_A[1] == NC_LLMA[1] else 0
#                 NC_A_dict["expanded_count_sum"].append(NC_A[2]['expanded_count'])
#                 NC_A_dict["explored_neighbors_sum"].append(NC_A[2]['generated_count'])
#                 NC_A_dict["neighbors_enqueued_sum"].append(NC_A[2]['enqueued_size'])
#                 NC_A_dict["pushed_count_revised_paths_sum"].append(NC_A[2]['pushed count'])

#                 NC_LLM_dict["path"].append(NC_LLMA[0])
#                 NC_LLM_dict["opt"] += 1 if NC_A[1] == NC_LLMA[1] else 0
#                 NC_LLM_dict["expanded_count_sum"].append(NC_LLMA[2]['expanded_count'])
#                 NC_LLM_dict["explored_neighbors_sum"].append(NC_LLMA[2]['generated_count'])
#                 NC_LLM_dict["neighbors_enqueued_sum"].append(NC_LLMA[2]['enqueued_size'])
#                 NC_LLM_dict["pushed_count_revised_paths_sum"].append(NC_LLMA[2]['pushed count'])

#                 SRC_goal_dict["path"].append(SRC_goal[0])
#                 SRC_goal_dict["opt"] += 1 if NC_A[1] == SRC_goal[1] else 0
#                 SRC_goal_dict["expanded_count_sum"].append(SRC_goal[2]['expanded_count'])
#                 SRC_goal_dict["explored_neighbors_sum"].append(SRC_goal[2]['generated_count'])
#                 SRC_goal_dict["neighbors_enqueued_sum"].append(SRC_goal[2]['enqueued_size'])
#                 SRC_goal_dict["pushed_count_revised_paths_sum"].append(SRC_goal[2]['pushed count'])


#                 # store all the values
#                 CSV_print.extend(NC_A[:2])
#                 for key,value in NC_A[2].items():
#                     CSV_print.append(value)

#                 CSV_print.extend(NC_LLMA[:2])
#                 for key,value in NC_LLMA[2].items():
#                     CSV_print.append(value)



#                 CSV_print.extend(SRC_goal[:2])
#                 for key,value in SRC_goal[2].items():
#                     CSV_print.append(value)
                
#                 # SAVE STAT
#                 CSV_print.extend([LLM_info["latency_s"],LLM_info["input_tokens"],LLM_info["output_tokens"]])
#                 CSV_print.extend([NC_LLMA[1]-NC_A[1]])
#                 for key,value in NC_A[2].items():
#                     CSV_print.append(NC_LLMA[2][key]-value)

#                 # print(len(CSV_print))
#                 print("writing to: ",printing_path, "/", filename)
#                 append_csv_row(printing_path + "/"+ filename,CSV_print)

#                 # ======================================================================
#                 # ========================================== loggin current iteration===
#                 run_key = f"run_{overall_i}_{graph_i}_waypoint_{waypoint}_nodes_{num_nodes}"

#                 run_payload = {
#                     "graph_key": graph_key,
#                     "waypoint_k": waypoint,
#                     "source": source,
#                     "dest": dest,

#                     # what you asked for
#                     "prompt_name": prompt_name,
#                     "prompt": query_flat,                 # or query_flat if you want the fully-rendered prompt text
#                     "output": full_output_flat,        # model raw output text
#                     "generated_waypoints": t_list_flat,

#                     "AStar_generated_path": NC_A[0],
#                     "AStar_path_cost": NC_A[1],
#                     "AStar_stats": NC_A[2],

#                     "LLMAStar_generated_path": NC_LLMA[0],
#                     "LLMAStar_path_cost": NC_LLMA[1],
#                     "LLMAStar_stats": NC_LLMA[2],

#                     "SrcDest_generated_path": SRC_goal[0],
#                     "SrcDest_path_cost": SRC_goal[1],
#                     "SrcDest_stats": SRC_goal[2],

#                     "llm_info": LLM_info,
#                 }

#                 logger.add_run(run_key, run_payload)


#         # print("writing final results for graph size ",num_nodes," with ",waypoint," waypoints to: ",printing_path, "/", filename)
#         # writer.writerow(CSV_print)
#         path_finder.print_to_CSV(total_filename, printing_path, prompt_name, NC_A_dict, NC_LLM_dict,SRC_goal_dict, overall_i, num_nodes, len(graph.edges),waypoint_lists,waypoint=waypoint)




# # from save_run_data import *
# # from graph_builder import *
# # from utils_4 import *
# # from A_search import *
# # import networkx as nx
# # from pathlib import Path
# # from prompts_final import *
# # import csv
# # import numpy as np

# # itr_overall = 2
# # itr_graph = 10

# # # prompts = [select_gateways,select_gateways_v2,base_graph_list_shortest_CoT_v4_5waypoints,prompt_get_waypoints_v7, prompt_get_waypoints_label,prompt_get_waypoints_label_set,prompt_get_waypoints_cost_1]
# # # prompt_names = ["select_gateways","select_gateways_v2", "base_graph_list_shortest_CoT_v4_5waypoints","prompt_get_waypoints_v7", "prompt_get_waypoints_label","prompt_get_waypoints_label_set","prompt_get_waypoints_cost_1"]
# # # prompt = base_graph_list_shortest_CoT_v4_5waypoints_no_repeated_guided
# # # prompt_name = "base_graph_list_shortest_CoT_v4_5waypoints_no_repeated_guided"

# # # prompts = [adj_list,adj_list_CoT,adj_list_no_limit,adj_list_few_shot,adj_list_h_values,adj_list_h_values_CoT,adj_list_h_values_unlimited,adj_list_h_values_few_shot]
# # # prompt_names = ["adj_list","adj_list_CoT","adj_list_no_limit","adj_list_few_shot","adj_list_h_values","adj_list_h_values_CoT","adj_list_h_values_unlimited","adj_list_h_values_few_shot"]

# # # prompts = [adj_list_few_shot,adj_list_h_values,adj_list_h_values_CoT,adj_list_h_values_unlimited,adj_list_h_values_few_shot]
# # # prompt_names = ["adj_list_few_shot","adj_list_h_values","adj_list_h_values_CoT","adj_list_h_values_unlimited","adj_list_h_values_few_shot"]


# # prompts = [adj_list_h_values_CoT]
# # prompt_names = ["adj_list_h_values_CoT"]

# # # prompt = select_gateways_v2_no_cost
# # # prompt_name = "select_gateways_v2_no_cost"
# # model_name = "gpt-4.1-2025-04-14"
# # # model_name = "gpt-4o-mini-2024-07-18" 
# # k_landmarks = 20
# # num_waypoints = 5
# # landmark_sel_tech = "furthest"
# # promptIDX = -1
# # # landmark_sel_tech = "ranrandom"
# # graph_type = "road_network"
# # # graph_type = "advogato"
# # # graph_type = "colt_tel"
# # # graph_type = "real"
# # # graph_type = "COMPLETE"
# # # graph_type = "BA"
# # # graph_type = "ER"

# # adj_list_format = "other" #"other" #"BFS"
# # node_ID_format = "int"#"int"#"L0_3"
# # graph_struct = "flat" #"flat" #layered
# # cost = "constant"# "constant" #"relative_to_degree"
# # g_n = 0

# # service_DR=200
# # functions=[{'ID':0, 'requirements':4}, {'ID':1, 'requirements':3}, {'ID':2, 'requirements':4}, {'ID':3, 'requirements':4}, {'ID':5, 'requirements':3}]
# # # functions=[{'ID':0, 'requirements':30}, {'ID':1, 'requirements':20}, {'ID':2, 'requirements':0.5}, {'ID':3, 'requirements':20}, {'ID':5, 'requirements':10}]
# # num_nodes = 10000
# # edge_degree = 100
# # t_n_weight=1

# # # ===================================  =================================== setup
# # printing_path = f"/Users/noufabbasi/Library/Mobile Documents/com~apple~CloudDocs/khalifa_2023-/Thesis/IRL_summer_2025/ToT_/SFC_LLM/output/tests_final_{num_nodes}_{edge_degree}"
# # Path(printing_path).mkdir(parents=True, exist_ok=True)
# # timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# # total_filename = f"graph_{graph_type}_iterG{itr_graph}_iterO{itr_overall}_{timestamp}_node{num_nodes}_edge{edge_degree}_{graph_struct}_{node_ID_format}_{model_name}_landmarks{k_landmarks}_g_n_{g_n}_{adj_list_format}_num_wapoints{num_waypoints}"


# # for promptIDX in range(len(prompts)):
# #     rng = random.Random(43)
# #     prompt =  prompts[promptIDX]
# #     prompt_name =  prompt_names[promptIDX]
# #     timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# #     filename = f"graph_{graph_type}_iterG{itr_graph}_iterO{itr_overall}_{timestamp}_node{num_nodes}_edge{edge_degree}_{graph_struct}_{node_ID_format}_{prompt_name}_{model_name}_landmarks{k_landmarks}_g_n_{g_n}_{adj_list_format}_num_wapoints{num_waypoints}"
# #     print("writing to: ",printing_path,filename)


# #     # ===================================  =================================== 
# #     with open(printing_path + "/" + filename+".csv", mode='w', newline='') as file:
# #         writer = csv.writer(file)
# #         # writer.writerow(["Test No.", "# Nodes", "# Edges added", "total edges", "Source","goal","pompt","full output", "LLM_waypoints", "A* path", "A* explored nodes", "A* cost","cap_LLM A* path","cap_LLM A* explored nodes", "cap_LLMA* cost","optimal?","explore_gap"])  
# #         CSV_print = ["overall_i","graph_i", "# Nodes", "total edges","# Edges added", "Source","goal"]

# #         CSV_print.extend(["waypoints_generated_provided_full_graph"]) #,"prompt_and_output_for_waypoints_generated_provided_full_graph"])
# #         CSV_print.extend(["A* path","A*cost","A_neighbors_enqueued", "A_pushed_count_revised_paths", "generated_neighbor_count", "A_expanded_count"])
# #         CSV_print.extend(["LLMA* path","LLMA*cost","LLMA*_neighbors_enqueued", "LLMA*_pushed_count_revised_paths", "LLMA*generated_neighbor_count", "LLMA*_expanded_count"])
# #         # CSV_print.extend(["LLMmode* path","LLMmode*cost","LLMmode*_neighbors_enqueued", "LLMmode*_pushed_count_revised_paths", "LLMmode*generated_neighbor_count", "LLMmode*_expanded_count"])
# #         CSV_print.extend(["SRC_goal* path","SRC_goal*cost","SRC_goal*_neighbors_enqueued", "SRC_goal*_pushed_count_revised_paths", "SRC_goal*generated_neighbor_count", "SRC_goal*_expanded_count"])
# #         CSV_print.extend(["LLM latency","input_tokens","output_tokens"])
# #         CSV_print.extend(["STAT-Cost inc.","STAT_neighbors_enqueued", "STAT_pushed_count_revised_paths", "STAT_generated_neighbor_count", "STAT_expanded_count"])
# #         # CSV_print.extend(["S_G_onlyA* path","S_G_onlyA*cost","S_G_only*_neighbors_enqueued", "S_G_only*_pushed_count_revised_paths", "S_G_only*generated_neighbor_count", "S_G_only*_expanded_count"])
# #         # CSV_print.extend(["S_G_only_STAT-Cost inc.","S_G_only_STAT_neighbors_enqueued", "S_G_only_STAT_pushed_count_revised_paths", "S_G_only_STAT_generated_neighbor_count", "S_G_only_STAT_expanded_count"])
# #         writer.writerow(CSV_print)



# #     NC_A_dict={"path":[], "opt":0, "expanded_count_sum":[], "explored_neighbors_sum":[], "neighbors_enqueued_sum":[], "pushed_count_revised_paths_sum":[]}
# #     NC_LLM_dict={"path":[], "opt":0, "expanded_count_sum":[], "explored_neighbors_sum":[], "neighbors_enqueued_sum":[], "pushed_count_revised_paths_sum":[]}
# #     SRC_goal_dict={"path":[], "opt":0, "expanded_count_sum":[], "explored_neighbors_sum":[], "neighbors_enqueued_sum":[], "pushed_count_revised_paths_sum":[]}
# #     waypoint_lists = []

# #     for overall_i in range(itr_overall):
# #         # num_nodes+=5000
# #         # edge_degree +=1
# #         # num_nodes=num_nodes + 100

# #         # =================================== setup
# #         SFC_graph_builder = SFCGraphBuilder(functions,service_DR=service_DR)

# #         # =================================== Create graph
# #         # SFC_graph_builder.create_graph(num_nodes, edge_degree,rng,type="ER")
# #         # SFC_graph_builder.create_graph(num_nodes, edge_degree,rng,type_="BA")
# #         SFC_graph_builder.create_graph(num_nodes, edge_degree,rng,type_=graph_type,k=edge_degree)

# #         SFC_graph_builder.conver_to_weighted_graph(cost,rng)
# #         if graph_struct == "layered":
# #             if node_ID_format == "L0_3":
# #                 SFC_graph_builder.get_layered_graph_v2()
# #             else:
# #                 SFC_graph_builder.get_layered_graph()
# #         graph = SFC_graph_builder.get_graph(graph_struct)
# #         print(f"=================================== built a graph -{graph_type} with {len(graph.nodes)} nodes and {len(graph.edges)}")
# #         if graph_struct == "flat":
# #             graph_layers = None
# #         else:
# #             graph_layers = SFC_graph_builder.get_graph("graph_layers")
# #         path_finder = A_search(graph,model_name,functions,service_DR,print_=False)
# #         path_finder.set_num_waypoints(num_waypoints)
# #         # path_finder.print_graph(f"A* graph{overall_i}_{4}_{3},{1}")


# #         # =================================== get landmarks
# #         print("=================================== generating landmarks")
# #         # path_finder.generate_landmarks(graph_struct,landmark_sel_tech,k_landmarks,graph_layers,rng)
# #         path_finder.generate_landmarks(graph_struct,landmark_sel_tech,k_landmarks,graph_layers,rng)


# #         deg_vals = np.array([d for _, d in graph.degree()])
# #         cutoff = np.percentile(deg_vals, 95)

# #         # # intersection_nodes = [n for n, d in graph.degree() if d >= cutoff]
# #         # # subgraph_intersection_nodes = graph.subgraph(intersection_nodes) 

# #         # rerun the search on the same graph (diff source and dest)
# #         for graph_i in range(itr_graph):
# #             # =================================== generate source and goal nodes
# #             print("=================================== generating source and destination pair")
# #             # source,dest,_ = path_finder.get_source_goal_nodes("source_goal",graph_struct, node_ID_format,graph_layers,Last_layer=len(functions),rng=rng)
# #             # source,dest,_ = path_finder.get_source_goal_nodes("component_based",graph_struct, node_ID_format,graph_layers,Last_layer=len(functions),rng=rng)
# #             # source,dest,_ = path_finder.get_source_goal_nodes("longest_path",graph_struct, node_ID_format,graph_layers,Last_layer=len(functions),rng=rng)
# #             source,dest,_ = path_finder.get_source_goal_nodes("random_from_max",graph_struct, node_ID_format,graph_layers,Last_layer=len(functions),rng=rng)
# #             # source,dest,_ = path_finder.get_source_goal_nodes("source_goal",graph_struct, node_ID_format,graph_layers,Last_layer=len(functions),rng=rng)
# #             while source == None:
# #                 source,dest,_ = path_finder.get_source_goal_nodes("source_goal",graph_struct, node_ID_format,graph_layers,Last_layer=len(functions),rng=rng)
# #                 print("random points not found")
# #                 pass
            
# #             # intersection_nodes = [n for n, d in graph.degree() if d >= cutoff]
# #             # subgraph = abstract_graph_with_endpoints_fast(graph, source=source, goal=dest, percentile=99.9)

# #             # subgraph = abstract_graph_collapse_bridges_fast(graph, source=source, goal=dest, percentile=99)
# #             subgraph, expand, intersection_nodes = build_abstract_graph_with_endpoints(
# #             graph,
# #             cutoff=cutoff,
# #             source=source,
# #             goal=dest,
# #             weight="Cost",   # or None
# #             # k_attach=3         # connect to the 3 nearest intersections
# #             )

# #             # path_finder_2 = A_search(subgraph,model_name,functions,service_DR,print_=False)
# #             # path_finder_2.print_graph(f"A* graph")
# #             # print(f"Abstracted graph size: {len(subgraph.nodes())} nodes, {len(subgraph.edges())} edges")

# #             print(len(subgraph))
# #     #         # =================================== generate waypoints
# #     #         print("=================================== generating waypoints")
# #     #         # query_flat,full_output_flat, t_list_flat,LLM_info = ["","",[1],{"latency_s":3, "input_tokens":3 ,"output_tokens":3 ,"total_tokens": 3}] #self._initialize_llm_paths(False)
# #             query_flat,full_output_flat, t_list_flat,LLM_info = path_finder._initialize_llm_paths(graph_struct=="layered", graph_layers,node_ID_format,prompt,len_layers=len(functions),adj_list_format=adj_list_format,graph=subgraph)
# #             waypoint_lists.append(f"the waypoints: {t_list_flat}")
# #             # path_finder.print_graph(f"A* graph_{source},{dest}")
# #             # path_finder.print_graph(f"LLM output: {full_output_flat}")
            
# #             # if graph_i < 2:
# #             #     print(query_flat)
# #             # path_finder.print_graph(f"graph{overall_i}_{graph_i}_{source},{dest}")

# #             # =================================== run shortest path_alg
# #             NC_A= [0,0,0]
# #             NC_LLMA= [0,0,0]
# #             SRC_goal = [0,0,0]

# #             # Note: the "graph saving" process happens within the astar function
# #             # ====== normal A*
# #             NC_A = path_finder.astar_path_constrained_flat(test_name="A_star")
# #             path_finder.print_graph(f"A* graph{overall_i}_{graph_i}_{source},{dest}")

# #             # ====== LLM A*
# #             # NC_LLMA = path_finder.astar_path_constrained_flat(test_name="A_star",g_n=g_n)
# #             NC_LLMA = path_finder.astar_path_constrained_LLM(test_name="LLM_A",t_n_weight=t_n_weight)
# #             path_finder.print_graph(f"LLM A* graph{overall_i}_{graph_i}_{source},{dest}")

# #             # ====== normal A*
# #             path_finder.set_waypoints([source,dest])
# #             SRC_goal = path_finder.astar_path_constrained_LLM(test_name="Src_Dest",t_n_weight=t_n_weight)
            

# #             # =================================== print to file
# #             CSV_print = [overall_i,graph_i, len(graph.nodes), len(graph.edges),edge_degree, source, dest]
# #             CSV_print.append(t_list_flat)

# #             NC_A_dict["path"].append(NC_A[0])
# #             NC_A_dict["opt"] += 1 if NC_A[1] == NC_LLMA[1] else 0
# #             NC_A_dict["expanded_count_sum"].append(NC_A[2]['expanded_count'])
# #             NC_A_dict["explored_neighbors_sum"].append(NC_A[2]['generated_count'])
# #             NC_A_dict["neighbors_enqueued_sum"].append(NC_A[2]['enqueued_size'])
# #             NC_A_dict["pushed_count_revised_paths_sum"].append(NC_A[2]['pushed count'])

# #             NC_LLM_dict["path"].append(NC_LLMA[0])
# #             NC_LLM_dict["opt"] += 1 if NC_A[1] == NC_LLMA[1] else 0
# #             NC_LLM_dict["expanded_count_sum"].append(NC_LLMA[2]['expanded_count'])
# #             NC_LLM_dict["explored_neighbors_sum"].append(NC_LLMA[2]['generated_count'])
# #             NC_LLM_dict["neighbors_enqueued_sum"].append(NC_LLMA[2]['enqueued_size'])
# #             NC_LLM_dict["pushed_count_revised_paths_sum"].append(NC_LLMA[2]['pushed count'])

# #             SRC_goal_dict["path"].append(SRC_goal[0])
# #             SRC_goal_dict["opt"] += 1 if NC_A[1] == SRC_goal[1] else 0
# #             SRC_goal_dict["expanded_count_sum"].append(SRC_goal[2]['expanded_count'])
# #             SRC_goal_dict["explored_neighbors_sum"].append(SRC_goal[2]['generated_count'])
# #             SRC_goal_dict["neighbors_enqueued_sum"].append(SRC_goal[2]['enqueued_size'])
# #             SRC_goal_dict["pushed_count_revised_paths_sum"].append(SRC_goal[2]['pushed count'])


# #             # store all the values
# #             CSV_print.extend(NC_A[:2])
# #             for key,value in NC_A[2].items():
# #                 CSV_print.append(value)

# #             CSV_print.extend(NC_LLMA[:2])
# #             for key,value in NC_LLMA[2].items():
# #                 CSV_print.append(value)



# #             CSV_print.extend(SRC_goal[:2])
# #             for key,value in SRC_goal[2].items():
# #                 CSV_print.append(value)
            
# #             # SAVE STAT
# #             CSV_print.extend([LLM_info["latency_s"],LLM_info["input_tokens"],LLM_info["output_tokens"]])
# #             CSV_print.extend([NC_LLMA[1]-NC_A[1]])
# #             for key,value in NC_A[2].items():
# #                 CSV_print.append(NC_LLMA[2][key]-value)

# #             append_csv_row(printing_path + "/" + filename,CSV_print)

# #     # writer.writerow(CSV_print)
# #     path_finder.print_to_CSV(total_filename, printing_path, prompt_name, NC_A_dict, NC_LLM_dict,SRC_goal_dict, overall_i, num_nodes, len(graph.edges),waypoint_lists)


