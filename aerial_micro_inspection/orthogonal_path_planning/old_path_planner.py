import yaml
import numpy as np
import networkx as nx
from networkx.algorithms import approximation as approx
import trimesh
import os
import math
import time


def load_config(config_path):
    """
    Loads the config file provided in path
    
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ---------------------------
# Load viewpoints (your format)
# ---------------------------
def load_viewpoints(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    viewpoints = []
    for vp in data["viewpoints"]:
        viewpoints.append({
            "position": np.array(vp["position"]),
            "yaw": vp["yaw"],
            "pitch": vp["pitch"],
            "target":   np.array(vp["target"]),
            "normal":   np.array(vp["normal"]) if "normal" in vp else None
        })

    return viewpoints


# ---------------------------
# Save path (same format, reordered)
# ---------------------------
def save_path(file_path, ordered_viewpoints):
    data = {
        "viewpoints": [
            {
                "position": vp["position"].tolist(),
                "yaw": vp["yaw"].tolist(),
                "pitch": vp["pitch"].tolist(),
                "target": vp["target"].tolist()
            }
            for vp in ordered_viewpoints
        ]
    }
    with open(file_path, 'w') as f:
        yaml.dump(data, f)

def transform_viewpoints_to_ned(viewpoints, translation_object, translation_origin, translation_camera, enu_to_ned=False):
    """
    Transforms a list of viewpoints from ENU to NED frame.
    Parameters:
        viewpoints (list of dicts): Each dict has 'position', 'target', 'normal' in ENU
        translation_enu (array-like): [x, y, z] offset of ENU origin w.r.t NED origin, expressed in ENU
        enu_to_ned (Boolean): Whether to trasnform from ENU to NED or not
    Returns:
        list of dicts with the same structure but coordinates in NED
    """
    R = np.array([
        [1, 0,  0],
        [0, 1,  0],
        [0, 0, 1]
        ])
    if enu_to_ned:
        R = np.array([
            [0, 1,  0],
            [1, 0,  0],
            [0, 0, -1]
            ])
    t = np.array(translation_origin) - np.array(translation_object)  

    transformed = []
    for vp in viewpoints:
        transformed.append({
            "position": (R @ (vp["position"] - t)) - np.array(translation_camera),
            "yaw": (R @ np.array([0.0,0.0,vp["yaw"]]))[2] + math.pi/2,
            "pitch": np.array([-vp["pitch"]]),
            "target":   R @ (vp["target"] - t)
        })

    return transformed


# ---------------------------
# Collision checking
# ---------------------------
def collision_free(mesh, p1, p2):
    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length == 0:
        return True

    direction = direction / length

    locations, _, _ = mesh.ray.intersects_location(
        ray_origins=[p1],
        ray_directions=[direction]
    )

    if len(locations) == 0:
        return True

    # nearest intersection
    dist = np.linalg.norm(locations[0] - p1)

    return dist > length


# ---------------------------
# Build graph
# ---------------------------
def build_graph(viewpoints, mesh, penalty_factor=50.0):
    G = nx.Graph()

    # Add nodes
    for i, vp in enumerate(viewpoints):
        pos = np.array(vp["position"])
        G.add_node(i, pos=pos)

    # Add edges
    for i in range(len(viewpoints)):
        for j in range(i + 1, len(viewpoints)):
            p1 = G.nodes[i]['pos']
            p2 = G.nodes[j]['pos']

            dist = np.linalg.norm(p1 - p2)

            if collision_free(mesh, p1, p2):
                weight = dist
            else:
                weight = dist * penalty_factor

            G.add_edge(i, j, weight=weight)

    return G


# ---------------------------
# Solve TSP
# ---------------------------
def solve_tsp(G):
    cycle = approx.traveling_salesman_problem(G, weight='weight')

    # remove closing node
    if cycle[0] == cycle[-1]:
        cycle = cycle[:-1]

    return cycle

def get_path_colors(n):
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        # gradient: blau → verd → vermell
        r = int(255 * t)
        g = int(255 * (1 - abs(t - 0.5) * 2))
        b = int(255 * (1 - t))
        colors.append([r, g, b, 255])
    return colors

def create_path_lines(viewpoints):
    lines = []
    colors = get_path_colors(len(viewpoints))

    for i in range(len(viewpoints) - 1):
        p1 = viewpoints[i]["position"]
        p2 = viewpoints[i + 1]["position"]

        line = trimesh.load_path(np.array([p1, p2]))
        line.colors = np.array([colors[i]])

        lines.append(line)

    return lines

def visualize_path(mesh, viewpoints):
    scene = trimesh.Scene()

    # Mesh
    mesh_vis = mesh.copy()
    mesh_vis.visual.face_colors = [200, 200, 200, 100]
    scene.add_geometry(mesh_vis)

    # Viewpoints
    colors = get_path_colors(len(viewpoints))

    for i, vp in enumerate(viewpoints):
        sphere = trimesh.creation.icosphere(radius=0.08)
        sphere.visual.face_colors = colors[i]
        sphere.apply_translation(vp["position"])
        scene.add_geometry(sphere)

    # Path lines
    lines = create_path_lines(viewpoints)
    for line in lines:
        scene.add_geometry(line)

    # Optional: axes (debug)
    origin = np.array([0.0, 0.0, 0.0])
    axes = [
        (np.array([1.3,0,0]), [255,0,0,255]),
        (np.array([0,1.3,0]), [0,255,0,255]),
        (np.array([0,0,1.3]), [0,0,255,255])
    ]

    for direction, color in axes:
        line = trimesh.load_path(np.array([origin, origin + direction]))
        line.colors = np.array([color])
        scene.add_geometry(line)

    scene.show()


def weld_vertices(mesh, tolerance=1e-5):
    """
    Merges vertices that are within tolerance distance of each other,
    fixing meshes where patches were tessellated independently.
    """
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()

    from scipy.spatial import cKDTree
    tree = cKDTree(vertices)

    # For each vertex, find the lowest-index vertex within tolerance
    # and remap to it
    pairs = tree.query_pairs(r=tolerance)
    
    remap = np.arange(len(vertices))
    for i, j in pairs:
        root = min(remap[i], remap[j])
        remap[max(i, j)] = root

    # Flatten transitive remappings (e.g. A->B->C becomes A->C)
    for i in range(len(remap)):
        while remap[remap[i]] != remap[i]:
            remap[i] = remap[remap[i]]

    new_faces = remap[faces]
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=new_faces, process=False)
    
    return new_mesh
# ---------------------------
# Main
# ---------------------------
def main():

    #Check execution time
    start = time.time()

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../../config/viewpoint_config.yaml")
    config = load_config(config_path)

    mesh_file = os.path.join(script_dir,config["mesh_path"])
    viewpoints_file = os.path.join(script_dir,config["output_path"])
    output_file = os.path.join(script_dir,config["planning_path"])

    print("Loading mesh...")
    mesh = trimesh.load(mesh_file)
    mesh.apply_scale(config["mesh_scale"])
    if config["weld_mesh"]:
        mesh = weld_vertices(mesh, tolerance=config["weld_tolerance"])
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    print("Loading viewpoints...")
    viewpoints = load_viewpoints(viewpoints_file)

    print(f"{len(viewpoints)} viewpoints loaded")

    print("Building graph...")
    G = build_graph(viewpoints, mesh,penalty_factor=config["penalty_factor"])

    print("Solving TSP...")
    order = solve_tsp(G)

    ordered_viewpoints = [viewpoints[i] for i in order]

    end = time.time()
    print(f"Elapsed: {end - start:.4f}s")

    print("Visualizing path...")
    visualize_path(mesh, ordered_viewpoints)

    transformed = transform_viewpoints_to_ned(ordered_viewpoints,config["object_offset"],config["origin_offset"],config["camera_offset"],enu_to_ned=config["enu_to_ned"])

    print("Saving path...")
    save_path(output_file, transformed)

    print("Done!")


if __name__ == "__main__":
    main()