import gzip
import os
import json
import pickle
import numpy as np
import open3d as o3d
from conceptgraph.utils.pointclouds import Pointclouds
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.utils.vis import LineMesh


def create_ball_mesh(center, radius, color=(0, 1, 0)):
    """
    Creates a sphere mesh for a node in the scene graph.
    """
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.translate(center)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere


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

    class_colors = results['class_colors']
    return objects, class_colors


def visualize_scene_graph(result_path, edge_file):
    """
    Visualizes the scene graph using O3DVisualizer.
    """
    # Load objects and class colors
    objects, class_colors = load_result(result_path)

    # Load edges
    with open(edge_file, "r") as f:
        edges_json_data = json.load(f)
        loaded_edges = edges_json_data if isinstance(edges_json_data, list) else list(edges_json_data.values())

    # Prepare geometries
    node_radius = 0.15
    edge_radius = 0.03
    default_edge_color = [1, 0, 0]

    node_geometries ,edge_geometries, pcds, labels = [], [], [], []
    registered_points = []
    obj_id_to_node_data_map = {}

    # Create node geometries
    for obj in objects:
        curr_obj_num = obj.get("curr_obj_num")
        pcd = obj.get("pcd")
        obj_name = obj.get("class_name", "Unknown")
        if pcd is None or not hasattr(pcd, "points") or len(pcd.points) == 0:
            continue
        pcds.append(pcd)

        points = np.asarray(pcd.points)
        center_3d = np.mean(points, axis=0)
        class_id = obj.get("class_id", [0])[0]
        color = class_colors.get(str(class_id), (0.5, 0.5, 0.5))

        ball_mesh = create_ball_mesh(center_3d, node_radius, color)
        node_geometries.append(ball_mesh)

        obj_id_to_node_data_map[curr_obj_num] = {"center_3d": center_3d, 
                                                 "class_name": obj_name,
                                                 "class_id": class_id}

    # Create edge geometries
    for edge in loaded_edges:
        id1 = edge.get("object_1_id")
        id2 = edge.get("object_2_id")
        description = edge.get("edge_description", "")
        relation = edge.get("relationship", ""),

        if id1 not in obj_id_to_node_data_map or id2 not in obj_id_to_node_data_map:
            continue

        center1 = obj_id_to_node_data_map[id1]["center_3d"]
        center2 = obj_id_to_node_data_map[id2]["center_3d"]

        if np.array_equal(center1, center2):
            continue

        line_mesh_creator = LineMesh(
            points=np.array([center1, center2]),
            lines=np.array([[0, 1]]),
            colors=default_edge_color,
            radius=edge_radius,
        )
        edge_geometries.extend(line_mesh_creator.cylinder_segments)

        # add 3d label for edges on the midpoint
        midpoint = (center1 + center2) / 2
        if np.any([np.all(row == midpoint) for row in registered_points]):
            midpoint[1] += 0.1  # Adjust y-coordinate to avoid overlap
            print(f"Adjusted midpoint to avoid overlap: {midpoint}")
        registered_points.append(midpoint)

        # label_tuple = (midpoint, relation[0])
        label_tuple = (midpoint, description)
        labels.append(label_tuple)

    return node_geometries, edge_geometries, pcds, labels


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize Scene Graph")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result file")
    parser.add_argument("--edge_file", type=str, required=True, help="Path to the edge file")
    args = parser.parse_args()

    node_geometries, edge_geometries, rgb_pcd, labels = visualize_scene_graph(
        args.result_path, args.edge_file
    )

    app = o3d.visualization.gui.Application.instance
    app.initialize()

    # Visualize using O3DVisualizer
    vis = o3d.visualization.O3DVisualizer("Scene Graph Visualization", 1920, 1080)
    vis.show_settings = True

    # Add node geometries with unique names
    for i, node in enumerate(node_geometries):
        vis.add_geometry(f"Node_{i}", node)

    # Add edge geometries with unique names
    for i, edge in enumerate(edge_geometries):
        vis.add_geometry(f"Edge_{i}", edge)

    # Add RGB point cloud if available
    if rgb_pcd:
        for i, pcd in enumerate(rgb_pcd):
            vis.add_geometry(f"RGB_PCD_{i}", pcd)

    for i, (midpoint, relation) in enumerate(labels):
        l = vis.add_3d_label(midpoint, relation)
        try:
            l.color = o3d.visualization.gui.Color(0.9, 0.1, 0.1)
            l.scale = 2
        except Exception as e:
            continue

    vis.reset_camera_to_default()
    # Add the visualizer to the application window
    app.add_window(vis)

    # Start the application
    app.run()