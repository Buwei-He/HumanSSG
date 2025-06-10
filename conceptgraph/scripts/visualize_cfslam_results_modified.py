import cv2
import os
import copy
import json
import pickle
import gzip
import argparse

import matplotlib
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import open_clip

from conceptgraph.utils.pointclouds import Pointclouds
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.utils.vis import LineMesh

def create_ball_mesh(center, radius, color=(0, 1, 0)):
    # ... (as before)
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.translate(center)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere

def get_parser():
    # ... (as before)
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--rgb_pcd_path", type=str, default=None)
    parser.add_argument("--edge_file", type=str, default=None)
    parser.add_argument("--no_clip", action="store_true")
    parser.add_argument("--merge_overlap_thresh", type=float, default=-1)
    parser.add_argument("--merge_visual_sim_thresh", type=float, default=-1)
    parser.add_argument("--merge_text_sim_thresh", type=float, default=-1)
    parser.add_argument("--obj_min_points", type=int, default=0)
    parser.add_argument("--obj_min_detections", type=int, default=0)
    return parser

def load_result(result_path):
    # ... (as before, ensure 'objects' is a MapObjectList instance)
    potential_path = os.path.realpath(result_path)
    if potential_path != result_path:
        print(f"Resolved symlink for result_path: {result_path} -> \n{potential_path}")
        result_path = potential_path
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)

    if not isinstance(results, dict):
        raise ValueError("Results should be a dictionary!")
    
    objects_data = results["objects"]
    objects = MapObjectList()
    if isinstance(objects_data, MapObjectList):
        objects = objects_data
    elif isinstance(objects_data, list): 
        objects.load_serializable(objects_data)
    elif hasattr(objects_data, 'get_serializable_list'): 
        objects.load_serializable(objects_data.get_serializable_list())
    else:
        raise ValueError(f"Unsupported type for results['objects']: {type(objects_data)}")

    class_colors = results['class_colors']
    return objects, None, class_colors

def add_edge_labels(vis_param, loaded_edges, obj_id_to_node_data_map):
    """
    Adds labels to the visualizer for each edge in the scene graph using 3D text meshes.
    """
    for edge_idx, edge_info in enumerate(loaded_edges):
        id1_from_edge = edge_info.get("object_1_id")
        id2_from_edge = edge_info.get("object_2_id")
        label_text = edge_info.get("label", f"Edge {edge_idx}")  # Default label if none provided

        if id1_from_edge is None or id2_from_edge is None:
            continue

        data1 = obj_id_to_node_data_map.get(id1_from_edge)
        data2 = obj_id_to_node_data_map.get(id2_from_edge)

        if data1 is None or data2 is None:
            continue

        # Compute the midpoint between the two nodes for label placement
        midpoint = (data1['center_3d'] + data2['center_3d']) / 2.0

        # Create the 3D text mesh for the label
        text_mesh = o3d.t.geometry.TriangleMesh.create_text(
            text=edge_info.get("relationship", f"Edge {edge_idx}"),
            depth=1      # Thickness of the text
        ).to_legacy()  # Convert to legacy Open3D format
        flipped_text_mesh = o3d.t.geometry.TriangleMesh.create_text(
            text=edge_info.get("relationship", f"Edge {edge_idx}"),
            depth=1      # Thickness of the text
        ).to_legacy() 

        text_mesh.translate(midpoint)  # Position the text at the midpoint
        text_mesh.scale(0.01, midpoint)  # Scale the text down if necessary
        text_mesh.paint_uniform_color([0.1, 0.1, 0.1])  # White color for the text
        vis_param.add_geometry(text_mesh)

        # Create a flipped version of the text mesh
        flipped_text_mesh.transform(np.diag([1, 1, -1, 1]))  # Flip along the Y-axis
        flipped_text_mesh.translate(midpoint) 
        flipped_text_mesh.scale(0.01, midpoint)  # Scale the text down if necessary
        flipped_text_mesh.paint_uniform_color([0.1, 0.1, 0.1])  # White color for the text
        vis_param.add_geometry(flipped_text_mesh)

def main(args):
    result_path = args.result_path
    rgb_pcd_path = args.rgb_pcd_path
    
    assert not (result_path is None and rgb_pcd_path is None), \
        "Either result_path or rgb_pcd_path must be provided."

    global_pcd = None
    if rgb_pcd_path is not None:        
        pointclouds = Pointclouds.load_pointcloud_from_h5(rgb_pcd_path)
        global_pcd = pointclouds.open3d(0, include_colors=True)
        if result_path is None:
            o3d.visualization.draw_geometries([global_pcd])
            exit()
        
    objects, _, class_colors = load_result(result_path)
    
    main.scene_graph_nodes_geometries = [] 
    main.scene_graph_edges_geometries = [] 
    main.scene_graph_node_list_for_picking = [] 

    if args.edge_file is not None and result_path is not None:
        print(f"Loading scene graph data...")
        with open(args.edge_file, "r") as f:
            edges_json_data = json.load(f)
            loaded_edges = []
            if isinstance(edges_json_data, dict): loaded_edges = list(edges_json_data.values())
            elif isinstance(edges_json_data, list): loaded_edges = edges_json_data
            else: print(f"Warning: Edge file format not recognized: {type(edges_json_data)}.")

        obj_id_to_node_data_map = {} # Map from 'curr_obj_num' to node_data
        temp_node_list_for_picking = [] 

        print(f"Processing {len(objects)} objects for scene graph nodes...")
        print("  --- Object ID (curr_obj_num) Debugging (from .pkl.gz) ---")
        for i, obj_dict in enumerate(objects): 
            # YOUR FIX: Use 'curr_obj_num' as the key for matching with edges
            curr_obj_num_from_pkl = obj_dict.get('curr_obj_num') 
            print(f"    Object_List_Index {i}: Retrieved 'curr_obj_num' = {curr_obj_num_from_pkl} (type: {type(curr_obj_num_from_pkl)})")
            
            if curr_obj_num_from_pkl is None:
                print(f"      Warning: Object at pkl_index {i} is missing 'curr_obj_num'. Won't be a graph node for edges.")
                # Still create a node for visualization if other fields exist, but it won't link via edges
                # Or decide to skip it entirely if curr_obj_num is essential for any node display
                # For now, let's make it eligible for scene_graph_node_list_for_picking if it has a PCD
                # but it won't be in obj_id_to_node_data_map for edge linking.
                pass # It will be skipped for obj_id_to_node_data_map if curr_obj_num_from_pkl is None

            pcd = obj_dict.get('pcd')
            if pcd is None or not hasattr(pcd, 'points') or len(pcd.points) == 0:
                # print(f"  Warning: Object with curr_obj_num {curr_obj_num_from_pkl} has no valid PCD. Skipping node display.")
                continue # Skip if no PCD for the node sphere
            
            points = np.asarray(pcd.points); center_3d = np.mean(points, axis=0)
            obj_classes_ids = np.asarray(obj_dict.get('class_id', [0])); obj_class_id = 0
            if len(obj_classes_ids) > 0:
                values, counts = np.unique(obj_classes_ids, return_counts=True)
                if len(values) > 0: obj_class_id = values[np.argmax(counts)]
            
            color = class_colors.get(str(obj_class_id), (0.5, 0.5, 0.5))
            caption_id_for_display = curr_obj_num_from_pkl if curr_obj_num_from_pkl is not None else f"PklIdx_{i}"
            caption = obj_dict.get('consolidated_caption', obj_dict.get('class_name', f"Obj_{caption_id_for_display}"))

            node_data = {
                'display_id': caption_id_for_display, # ID used for display/picking list
                'linking_id': curr_obj_num_from_pkl, # ID used for edges (curr_obj_num)
                'center_3d': center_3d, 'color': color, 'caption': caption, 'mesh': None
            }
            temp_node_list_for_picking.append(node_data)
            if curr_obj_num_from_pkl is not None: # Only add to map if it has the linking ID
                 obj_id_to_node_data_map[curr_obj_num_from_pkl] = node_data
        print("  --- End Object ID Debugging ---")
        main.scene_graph_node_list_for_picking = temp_node_list_for_picking

        node_radius = 0.15
        for node_data_ref in main.scene_graph_node_list_for_picking:
            # Only create mesh if center_3d was successfully computed (i.e., PCD was valid)
            if 'center_3d' in node_data_ref:
                ball_mesh = create_ball_mesh(node_data_ref['center_3d'], node_radius, node_data_ref['color'])
                node_data_ref['mesh'] = ball_mesh 
                main.scene_graph_nodes_geometries.append(ball_mesh)
        print(f"  Created {len(main.scene_graph_nodes_geometries)} node sphere meshes.")

        edge_radius = 0.03; default_edge_color = [1, 0, 0]; successful_edges_count = 0
        print(f"Processing {len(loaded_edges)} potential edges...")
        if loaded_edges:
            print("  --- Edge Linking Debugging (curr_obj_num from edge file vs. pkl) ---")
            for edge_idx, edge_info in enumerate(loaded_edges):
                # These IDs from edge file should match 'curr_obj_num' from pkl
                id1_from_edge = edge_info.get("object_1_id") 
                id2_from_edge = edge_info.get("object_2_id")
                print(f"    Edge {edge_idx}: id1_from_edge={id1_from_edge} (type {type(id1_from_edge)}), id2_from_edge={id2_from_edge} (type {type(id2_from_edge)})")

                if id1_from_edge is None or id2_from_edge is None: continue

                data1 = obj_id_to_node_data_map.get(id1_from_edge) 
                data2 = obj_id_to_node_data_map.get(id2_from_edge)

                if data1 is None:
                    print(f"      LOOKUP FAILED for id1={id1_from_edge} (curr_obj_num). Not found as key in obj_id_to_node_data_map.")
                    # print(f"        Available keys (curr_obj_num values from pkl): {list(obj_id_to_node_data_map.keys())[:10]}")
                    continue
                if data2 is None:
                    print(f"      LOOKUP FAILED for id2={id2_from_edge} (curr_obj_num). Not found as key in obj_id_to_node_data_map.")
                    continue
                
                print(f"      LOOKUP SUCCESS for edge {edge_idx} between curr_obj_nums {id1_from_edge} and {id2_from_edge}.")
                if np.array_equal(data1['center_3d'], data2['center_3d']): continue

                try:
                    line_mesh_creator = LineMesh(
                        points=np.array([data1['center_3d'], data2['center_3d']]),
                        lines=np.array([[0, 1]]), colors=default_edge_color, radius=edge_radius
                    )
                    main.scene_graph_edges_geometries.extend(line_mesh_creator.cylinder_segments)
                    successful_edges_count += 1
                except Exception as e:
                    print(f"    Error creating LineMesh for edge: {e}")
            print("  --- End Edge Linking Debugging ---")
        print(f"  Created {successful_edges_count} edge meshes.")

    # ... (CLIP init, pcds_list, bboxes_list, object_display_data setup as before)
    if not args.no_clip:
        print("Initializing CLIP model...")
        try:
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
            clip_device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_model = clip_model.to(clip_device)
            clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
            main.clip_model = clip_model 
            main.clip_tokenizer = clip_tokenizer
            main.clip_device = clip_device
            print(f"Done initializing CLIP model on {clip_device}.")
        except Exception as e:
            print(f"Could not initialize CLIP model: {e}")
            args.no_clip = True

    cmap = matplotlib.colormaps.get_cmap("turbo")
    
    pcds_list = [] 
    if result_path: pcds_list = copy.deepcopy(objects.get_values("pcd"))

    raw_bboxes = []
    if result_path: raw_bboxes = objects.get_values("bbox")
    bboxes_list = [] 
    for bb in raw_bboxes:
        if isinstance(bb, (o3d.geometry.OrientedBoundingBox, o3d.geometry.AxisAlignedBoundingBox)):
            bboxes_list.append(copy.deepcopy(bb))

    object_display_data = [] 
    if result_path:
        for i in range(len(objects)):
            obj = objects[i] 
            obj_classes_ids = np.asarray(obj.get('class_id', [0]))
            obj_class_id = 0
            if len(obj_classes_ids) > 0 :
                values, counts = np.unique(obj_classes_ids, return_counts=True)
                if len(values) > 0: obj_class_id = values[np.argmax(counts)]
            original_colors = None
            current_pcd = pcds_list[i] if i < len(pcds_list) and pcds_list[i] is not None else None
            if current_pcd and hasattr(current_pcd, 'colors'):
                 original_colors = copy.deepcopy(current_pcd.colors)
            object_display_data.append({'class_id': obj_class_id, 'original_pcd_colors': original_colors})


    vis = o3d.visualization.VisualizerWithKeyCallback()

    window_title = 'Open3D Visualizer' # ... (window title setup)
    if result_path: window_title = f'Open3D - {os.path.basename(result_path)}'
    elif rgb_pcd_path: window_title = f'Open3D - {os.path.basename(rgb_pcd_path)}'
    vis.create_window(window_name=window_title, width=1920, height=1080)

    for pcd_geom in pcds_list: 
        if pcd_geom: vis.add_geometry(pcd_geom, reset_bounding_box=False)
    for bbox_geom in bboxes_list:
        if bbox_geom: vis.add_geometry(bbox_geom, reset_bounding_box=False)
    if pcds_list or bboxes_list: vis.reset_view_point(True)

    # After processing edges and creating text meshes
    add_edge_labels(vis, loaded_edges, obj_id_to_node_data_map)


    # --- TOGGLE FIX: Using a separate flag for visualizer state ---
    main.graph_geometries_currently_in_visualizer = False
    # main.show_scene_graph is now more of an "intent" or logical state

    def toggle_global_pcd(vis_param): # ... (as before)
        if global_pcd is None: print("No RGB pcd path provided."); return
        if main.show_global_pcd: vis_param.remove_geometry(global_pcd, reset_bounding_box=False)
        else: vis_param.add_geometry(global_pcd, reset_bounding_box=False)
        main.show_global_pcd = not main.show_global_pcd # This flag is fine for single geoms

    def toggle_scene_graph(vis_param, show_on_startup=False): # Added new parameter
        all_graph_geometries = main.scene_graph_nodes_geometries + main.scene_graph_edges_geometries
        if not all_graph_geometries and not show_on_startup: # Don't print if just starting up and no graph
            print("No scene graph data (nodes or edges) to display.")
            return

        # Determine desired state: if called on startup, we want to show. Otherwise, toggle.
        desired_to_show = False
        if show_on_startup:
            desired_to_show = True
        else: # Normal toggle behavior
            desired_to_show = not main.graph_geometries_currently_in_visualizer

        if desired_to_show:
            if not main.graph_geometries_currently_in_visualizer: # Only add if not already there
                if not show_on_startup: print("\nShowing scene graph...") # Avoid print on startup if already doing it
                # ... (print node list to console - this part is fine to do on startup show too)
                print("--- Scene Graph Nodes (for picking by list #) ---")
                if not main.scene_graph_node_list_for_picking: print("  No nodes available.")
                else:
                    for list_idx, node_data in enumerate(main.scene_graph_node_list_for_picking):
                        print(f"  {list_idx + 1}: [ID {node_data['display_id']}] {node_data['caption'][:80]}...")
                print("-------------------------------------------------")
                
                # print(f"Adding {len(all_graph_geometries)} graph geometries to visualizer.")
                for geometry in all_graph_geometries:
                    vis_param.add_geometry(geometry, reset_bounding_box=False)
                
                if main.scene_graph_edges_geometries: 
                    print(f"  ({len(main.scene_graph_edges_geometries)} Edges added)")
                elif args.edge_file and not show_on_startup : 
                    print(f"  (No edge geometries were created. Check startup logs.)")
                main.graph_geometries_currently_in_visualizer = True
        else: # We want to hide
            if main.graph_geometries_currently_in_visualizer: # Only remove if they are there
                if not show_on_startup: print("Hiding scene graph...")
                for geometry in all_graph_geometries:
                    vis_param.remove_geometry(geometry, reset_bounding_box=False)
                main.graph_geometries_currently_in_visualizer = False
        
        if not show_on_startup: # Don't update renderer twice if called on startup then again by key
            vis_param.poll_events() 
            vis_param.update_renderer()

    # ... (all coloring functions and save_view_params as before, ensure they use pcds_list)
    def color_by_class(vis_param):
        if not result_path: return
        for i in range(len(pcds_list)): 
            pcd = pcds_list[i] 
            if not pcd: continue
            obj_data = object_display_data[i]
            color = class_colors.get(str(obj_data['class_id']), (0.5, 0.5, 0.5)) 
            pcd.paint_uniform_color(color)
            vis_param.update_geometry(pcd)
            
    def color_by_rgb(vis_param):
        if not result_path: return
        for i in range(len(pcds_list)): 
            pcd = pcds_list[i]
            if not pcd: continue
            original_colors = object_display_data[i]['original_pcd_colors']
            if original_colors is not None and hasattr(pcd, 'points') and len(original_colors) == len(pcd.points):
                pcd.colors = original_colors
            else: pcd.paint_uniform_color([0.5,0.5,0.5])
            vis_param.update_geometry(pcd)

    def color_by_instance(vis_param):
        if not result_path or len(pcds_list) == 0: return 
        try: import distinctipy
        except ImportError:
            class distinctipy_fallback:
                def get_colors(self, n): cm = matplotlib.colormaps.get_cmap("turbo"); return [cm(j/max(1,n-1))[:3] for j in range(n)]
            distinctipy = distinctipy_fallback()
        instance_colors_rgb = distinctipy.get_colors(len(pcds_list)) 
        for i in range(len(pcds_list)): 
            pcd = pcds_list[i]; 
            if not pcd: continue
            color = instance_colors_rgb[i] if i < len(instance_colors_rgb) else [0.5,0.5,0.5]
            pcd.paint_uniform_color(color)
            vis_param.update_geometry(pcd)
        
    def color_by_clip_sim(vis_param):
        if not result_path or args.no_clip or not hasattr(main, 'clip_model'):
            print("CLIP coloring prerequisites not met."); return
        text_query = input("Enter CLIP query: ")
        if not text_query: return
        text_queries_tokenized = main.clip_tokenizer([text_query]).to(main.clip_device)
        with torch.no_grad():
            text_query_ft = main.clip_model.encode_text(text_queries_tokenized).squeeze(0).float()
            text_query_ft /= text_query_ft.norm(keepdim=True)
            obj_clip_fts_list, valid_indices = [], []
            for i, obj_item in enumerate(objects): 
                ft_data = obj_item.get('clip_ft') 
                if ft_data is None: continue
                if isinstance(ft_data, list): ft_data = ft_data[0] if ft_data else None
                if ft_data is None: continue
                if isinstance(ft_data, np.ndarray): ft = torch.from_numpy(ft_data)
                elif isinstance(ft_data, torch.Tensor): ft = ft_data
                else: continue
                ft = ft.to(main.clip_device).float()
                if ft.ndim == 1: ft = ft.unsqueeze(0)
                if ft.shape[0] == 1 and ft.shape[1] > 1: obj_clip_fts_list.append(ft); valid_indices.append(i)
            if not obj_clip_fts_list: print("No objects with suitable CLIP features."); return
            objects_clip_fts = torch.cat(obj_clip_fts_list, dim=0)
            similarities = F.cosine_similarity(text_query_ft.unsqueeze(0), objects_clip_fts)
        if similarities.numel() == 0: print("No similarities computed."); return
        norm_sim = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-6)
        if similarities.numel() > 0:
            max_prob_idx_in_sim = torch.argmax(F.softmax(similarities * 10, dim=0))
            original_obj_idx = valid_indices[max_prob_idx_in_sim]
            obj_data = objects[original_obj_idx] 
            # Use display_id from the node_list if available and matches, or construct
            node_info_for_print = "N/A"
            for node in main.scene_graph_node_list_for_picking:
                if node.get('linking_id') == obj_data.get('curr_obj_num'): # Match by linking_id
                    node_info_for_print = f"ID {node['display_id']}"
                    break
            if node_info_for_print == "N/A": # Fallback if not in graph node list
                node_info_for_print = f"curr_obj_num {obj_data.get('curr_obj_num', original_obj_idx)}"

            print(f"Most probable: {node_info_for_print}, Class '{obj_data.get('class_name', 'N/A')}', Sim: {similarities[max_prob_idx_in_sim]:.4f}")
        sim_colors = cmap(norm_sim.cpu().numpy())[..., :3]
        for i_pcd_idx, pcd_geom in enumerate(pcds_list): 
            if not pcd_geom: continue
            if i_pcd_idx in valid_indices: pcd_geom.paint_uniform_color(sim_colors[valid_indices.index(i_pcd_idx)])
            else: pcd_geom.paint_uniform_color([0.5, 0.5, 0.5])
            vis_param.update_geometry(pcd_geom)

    def save_view_params(vis_param): # ... (as before)
        param_path = "view_params.json"; ctr = vis_param.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(param_path, param)
        print(f"View params saved to {param_path}")


    vis.register_key_callback(ord("S"), toggle_global_pcd)
    vis.register_key_callback(ord("C"), color_by_class)
    vis.register_key_callback(ord("R"), color_by_rgb)
    vis.register_key_callback(ord("I"), color_by_instance)
    vis.register_key_callback(ord("F"), color_by_clip_sim)
    vis.register_key_callback(ord("V"), save_view_params)
    vis.register_key_callback(ord("G"), toggle_scene_graph)

    # --- Show scene graph by default if data is available ---
    if main.scene_graph_nodes_geometries or main.scene_graph_edges_geometries:
        print("Showing scene graph by default...")
        toggle_scene_graph(vis, show_on_startup=True) # Call to show it initially
    # --- End default show ---

    if pcds_list or bboxes_list or main.graph_geometries_currently_in_visualizer : # if anything was added
        vis.reset_view_point(True) # Reset viewpoint after all initial geometries are added
    
    print("\n--- Key Bindings ---") # (as before)
    print("  G: Toggle Scene Graph (Nodes/Edges). Prints node list to console.")
    print("  P: Pick node by list number (prompts in console, graph must be visible).")
    print("  S: Toggle Global Scene PCD"); print("  C: Color objects by Class")
    print("  R: Color objects by original RGB"); print("  I: Color objects by Instance ID")
    print("  F: Color objects by CLIP similarity"); print("  V: Save current view parameters")
    print("--------------------\n")

    opt = vis.get_render_option(); opt.point_size = 3.0; opt.line_width = 5.0 
    vis.run(); vis.destroy_window()
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)