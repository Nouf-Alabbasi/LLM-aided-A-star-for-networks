'''

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

graph_size =[750]
num_waypoints = [5,8,12]

model_name = "gpt-4.1-2025-04-14"
k_landmarks = 20
landmark_sel_tech = "furthest"
promptIDX = -1
graph_type = "road_network"
# graph_type = "BA"

adj_list_format = "BFS"
node_ID_format = "int"
graph_struct = "flat"
cost = "constant"
g_n = 0

service_DR=200
functions=[{'ID':0, 'requirements':4}, {'ID':1, 'requirements':3}, {'ID':2, 'requirements':4}, {'ID':3, 'requirements':4}, {'ID':5, 'requirements':3}]
# functions=[{'ID':0, 'requirements':30}, {'ID':1, 'requirements':20}, {'ID':2, 'requirements':0.5}, {'ID':3, 'requirements':20}, {'ID':5, 'requirements':10}]
num_nodes = 1000
edge_degree = 4
t_n_weight=1

# ===================================  =================================== setup
printing_path = f"LLM_aided_A*_results"
Path(printing_path).mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
full_filename = f"graph_{graph_type}_iterG{itr_graph}_iterO{itr_overall}_{timestamp}_node{num_nodes}_edge{edge_degree}_{node_ID_format}_{model_name}_landmarks{k_landmarks}_g_n_{g_n}_{adj_list_format}_num_wapoints{num_waypoints[0]}"
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
source_dest = {}
base_seed = 43
rng = random.Random(base_seed)

# =================================== Make one JSON file per experiment config (matches your CSV naming style)
json_filename = f"{full_filename}.json"  # or build it once per (num_nodes, waypoint)
json_path = Path(printing_path) / json_filename

logger = RunLogger(json_path=json_path)

# Optional: save experiment-level metadata once
logger.data["meta"].update({
    "model_name": model_name,
    "graph_type": graph_type,
    "graph_struct": "network_graph",
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
            graph_struct="network_graph",
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
            "graph_struct": "network_graph",
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
            graph_struct="network_graph",
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
        filename = full_filename


        # ===================================  =================================== 
        with open(printing_path + "/" + filename+".csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            # writer.writerow(["Test No.", "# Nodes", "# Edges added", "total edges", "Source","goal","pompt","full output", "LLM_waypoints", "A* path", "A* explored nodes", "A* cost","cap_LLM A* path","cap_LLM A* explored nodes", "cap_LLMA* cost","optimal?","explore_gap"])  
            CSV_print = ["waypoint","overall_i","graph_i", "# Nodes", "total edges","# Edges added", "Source","goal"]

            CSV_print.extend(["waypoints_generated_provided_full_graph"]) #,"prompt_and_output_for_waypoints_generated_provided_full_graph"])
            CSV_print.extend(["A* path","A*cost","A_neighbors_enqueued", "A_pushed_count_revised_paths", "generated_neighbor_count", "A_expanded_count"])
            CSV_print.extend(["LLMA* path","LLMA*cost","LLMA*_neighbors_enqueued", "LLMA*_pushed_count_revised_paths", "LLMA*generated_neighbor_count", "LLMA*_expanded_count"])
            # CSV_print.extend(["LLMmode* path","LLMmode*cost","LLMmode*_neighbors_enqueued", "LLMmode*_pushed_count_revised_paths", "LLMmode*generated_neighbor_count", "LLMmode*_expanded_count"])
            CSV_print.extend(["LLM latency","input_tokens","output_tokens"])
            CSV_print.extend(["STAT-Cost inc.","STAT_neighbors_enqueued", "STAT_pushed_count_revised_paths", "STAT_generated_neighbor_count", "STAT_expanded_count"])
            # CSV_print.extend(["S_G_onlyA* path","S_G_onlyA*cost","S_G_only*_neighbors_enqueued", "S_G_only*_pushed_count_revised_paths", "S_G_only*generated_neighbor_count", "S_G_only*_expanded_count"])
            # CSV_print.extend(["S_G_only_STAT-Cost inc.","S_G_only_STAT_neighbors_enqueued", "S_G_only_STAT_pushed_count_revised_paths", "S_G_only_STAT_generated_neighbor_count", "S_G_only_STAT_expanded_count"])
            writer.writerow(CSV_print)
            # print(len(CSV_print))



        NC_A_dict={"path":[], "opt":0, "expanded_count_sum":[], "explored_neighbors_sum":[], "neighbors_enqueued_sum":[], "pushed_count_revised_paths_sum":[]}
        NC_LLM_dict={"path":[], "opt":0, "expanded_count_sum":[], "explored_neighbors_sum":[], "neighbors_enqueued_sum":[], "pushed_count_revised_paths_sum":[]}
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
                path_finder.generate_landmarks(graph_struct, landmark_sel_tech, k_landmarks, rng)
                landmark=path_finder.get_landmarks()
                landmarks[gkey] = landmark

            registry["landmarks"][gkey] = landmark



            # choose which of the 2 sets you want this run to use
            chosen_set = "set_0"  # or "set_1" or loop over both
            pairs = registry["src_dest_sets"][gkey][chosen_set]["pairs"]


            logger.register_graph(
                graph_key=gkey,
                graph_file=str(REGISTRY_PATH),
                graph_info={
                    "num_nodes": len(graph.nodes),
                    "num_edges": len(graph.edges),
                    "edge_degree_param": edge_degree,
                    "graph_type": graph_type,
                    "graph_struct": "network_graph",
                    "cost_mode": cost,
                    "landmarks": path_finder.get_landmarks(),
                }
            )
            # ======================================================================



            for graph_i, (source, dest) in enumerate(pairs):
                # =================================== set source and dest
                run_key = f"run_{overall_i}_{graph_i}_waypoint_{waypoint}_nodes_{num_nodes}"
                path_finder.set_source(source)
                path_finder.set_goal(dest)

                    
                # =================================== generate waypoints
                print("=================================== generating waypoints")
                path_finder.set_num_waypoints(waypoint)

                query_flat,full_output_flat, t_list_flat,LLM_info = path_finder._initialize_llm_paths_limited(node_ID_format,prompt,"_",adj_list_format=adj_list_format)
                waypoint_lists.append(t_list_flat)


                # =================================== run shortest path_alg
                NC_A= [0,0,0]
                NC_LLMA= [0,0,0]

                # ====== normal A*
                NC_A = path_finder.astar_path(test_name="A_star")

                # ====== LLM A*
                NC_LLMA = path_finder.astar_path_LLM(test_name="LLM_A",t_n_weight=t_n_weight)

                
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


                # store all the values
                CSV_print.extend(NC_A[:2])
                for key,value in NC_A[2].items():
                    CSV_print.append(value)

                CSV_print.extend(NC_LLMA[:2])
                for key,value in NC_LLMA[2].items():
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


                    "llm_info": LLM_info,
                    "landmarks": path_finder.get_landmarks(),
                }

                logger.add_run(run_key, run_payload)


        path_finder.print_to_CSV(full_filename, printing_path, prompt_name, NC_A_dict, NC_LLM_dict, overall_i, num_nodes, len(graph.edges),waypoint_lists,waypoint=waypoint)
REGISTRY_PATH = Path(printing_path+"/graph") / "registry.json"
save_registry(REGISTRY_PATH, registry)




