import gzip
import os
import json
import pickle
import numpy as np
import argparse
from omegaconf import DictConfig, OmegaConf
import hydra
from conceptgraph.slam.slam_classes import MapObjectList

def load_result(result_path):
    """
    Loads the result file and returns objects and class colors.
    """
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

    return objects

def process_config(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True) # convert to a regular dict
    fungraph_cfg = {
        # when adding new node pointcloud we use this values to decide if we merge the object with another
        'merge_box_thresh': cfg.get('merge_overlap_thresh', 0.5),
        'merge_point_thresh': 0.5, # not exists
        'merge_semantic_thresh': cfg.get('merge_visual_sim_thresh', 0.7),
        'merge_small_thresh': 0.5, # not exists
        'merge_small_single': 0.5, # not exists
        'process_small_pcd': True, 
        # pcd processing
        'downsample_voxel_size': cfg.get('downsample_voxel_size', 0.025),
        'downsample_voxel_size_small': 0.025,
        'dbscan_remove_noise': cfg.get('dbscan_remove_noise', True),
        'dbscan_eps': cfg.get('dbscan_eps', 0.1),  # epsilon for DBSCAN
        'dbscan_min_points': cfg.get('dbscan_min_points', 10),  # minimum points for DBSCAN
        'min_points_threshold': cfg.get('min_points_threshold', 16),
        'mask_padding': True,
        # clipping
        'use_clip': True,
        'max_distance': cfg.get('max_distance', 10.0),  # maximum distance for clipping
        # device
        'device': cfg.get('device', 'cuda'),
    }
    return fungraph_cfg

@hydra.main(version_base=None, config_path=os.path.join(os.path.dirname(__file__), "../hydra_configs/"), config_name="base_mapping")

def main(cfg: DictConfig):
    dataset_root = "/home/hackathon/Programming/nice-slam/Datasets/Custom/"
    scene_id = "Map2"
    is_behavior = True  # Set to True if you want to use behavior mapping
    scene_root = os.path.join(dataset_root, scene_id, "exps", f"r_mapping_{scene_id}_stride5")
    result_path = os.path.join(scene_root, f"pcd_r_mapping_{scene_id}_stride5.pkl.gz")
    # edge_file = os.path.join(scene_root, "edge_json_r_mapping_" + scene_id + "_stride5.json")
    edge_file = os.path.join(scene_root, f"edge_interactive_{scene_id}.json") if is_behavior else os.path.join(scene_root, f"edge_json_r_mapping_{scene_id}_stride5.json")
    # Load objects and class colors
    objects = load_result(result_path)

    # Load edges
    with open(edge_file, "r") as f:
        edges_json_data = json.load(f)
        loaded_edges = edges_json_data if isinstance(edges_json_data, list) else list(edges_json_data.values())
        if loaded_edges is None:
            raise ValueError("Edges data should be a list or a dictionary of edges.")

    graph = {
        'nodes': [],
        'edges': [],
        'num_objects': len(objects),
        'small_objects': None,
        'args': process_config(cfg) # convert to a regular dict
    }

    # Create node geometries
    for obj in objects:
        node = obj.copy()  # Ensure we work with a copy of the object
        node_id = node.get("curr_obj_num")
        obj_name = node.get("class_name", "Unknown")
        pcd = node.get("pcd")
        pcd_points = np.asarray(pcd.points).tolist()
        pcd_colors = np.asarray(pcd.colors).tolist()
        graph['nodes'].append(
            {
                'node_id': node_id,
                'label': [obj_name],
                'pcd_points': pcd_points,
                'pcd_colors': pcd_colors,
                'processed_last': 0,
                'features': [],
            }
        )


    # Create edge geometries
    for edge in loaded_edges:
        id1 = edge.get("object_1_id")
        id2 = edge.get("object_2_id")
        full_desc = edge.get("edge_description", "")
        desc = edge.get("relationship", ""),
        graph['edges'].append(
            [id1, id2, {'desc': desc}]
        )

    # Save the graph to a JSON file
    output_file = os.path.join(os.path.dirname(edge_file), f"scene_graph_{scene_id}_behavior.json" if is_behavior else f"scene_graph_{scene_id}.json")
    with open(output_file, "w") as f:
        json.dump(graph, f, indent=4)


if __name__ == "__main__":
    main()