import yaml
import numpy as np
import trimesh
import os
import colorsys
from collections import defaultdict


def load_config(config_path):
    """
    Loads the config file provided in path
    
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_camera(camera_yaml_path, camera_name="ai_camera"):
    """
    Loads the camera parameters from the simulated file, only use for matching file layout

    """
    with open(camera_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    cam = data[camera_name]

    K = cam["camera_matrix"]

    fx = K[0]
    fy = K[4]

    width = cam["image_width"]
    height = cam["image_height"]

    return fx, fy, width, height

def diagnose_adjacency(mesh):
    avg_adjacency = len(mesh.face_adjacency) / len(mesh.faces)
    print(f"Faces: {len(mesh.faces)}")
    print(f"Adjacency pairs: {len(mesh.face_adjacency)}")
    print(f"Average neighbors per face: {avg_adjacency:.2f}")
    print(f"Watertight mesh: {mesh.is_watertight}")

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

def cluster_surfaces_old(mesh, threshold=0.95):#TODO Remove or use for comparison without adjacency
    """
    From a mesh, clusters triangles into surfaces by checking similarity normal angles

    """

    normals=mesh.face_normals

    clusters = []

    for i,n in enumerate(normals):
        assigned=False

        for cluster in clusters:
            if np.dot(n,cluster["normal"])>threshold:
                cluster["faces"].append(i) 
                assigned=True
                break

        if not assigned:
            clusters.append({
            "normal": n,
            "faces": [i]
            })
    
    return clusters

def cluster_surfaces(mesh, threshold=0.95):
    """
    Clusters triangles into surfaces by checking normal similarity
    AND adjacency between faces, using a region growing approach.
    """
    normals = mesh.face_normals

    # Build adjacency lookup
    adj = defaultdict(set)
    for a, b in mesh.face_adjacency:
        adj[a].add(b)
        adj[b].add(a)

    clusters = []
    unvisited = set(range(len(normals)))

    while unvisited:
        # Pick a seed face
        seed = next(iter(unvisited))
        seed_normal = normals[seed]

        # BFS expansion
        cluster_faces = []
        queue = [seed]
        unvisited.remove(seed)

        while queue:
            face = queue.pop()
            cluster_faces.append(face)

            for neighbor in adj[face]:
                #if neighbor in unvisited and np.dot(normals[neighbor], seed_normal) > threshold:
                if neighbor in unvisited and np.dot(normals[neighbor], normals[face]) > threshold:
                    unvisited.remove(neighbor)
                    queue.append(neighbor)

        clusters.append({
            "normal": seed_normal,
            "faces": cluster_faces
        })

    return clusters

def fit_plane_svd(points):

    centroid = points.mean(axis=0)
    
    centered = points - centroid
    
    _, _, Vt = np.linalg.svd(centered)
    
    normal = Vt[-1]

    if np.dot(normal, centroid) < 0:
        normal = -normal
    
    return centroid, normal

def get_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n  # evenly spaced from 0.0 to 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # full saturation and brightness
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.array(colors, dtype=np.uint8)

def color_mesh_by_surfaces(mesh, surfaces):
    # Unmerge vertices so each face owns its vertices independently
    # This prevents color interpolation bleeding between adjacent faces
    mesh_colored = mesh.copy()
    mesh_colored.unmerge_vertices()

    # Remove textures
    mesh_colored.visual = trimesh.visual.ColorVisuals(mesh_colored)

    face_colors = np.zeros((len(mesh_colored.faces), 4), dtype=np.uint8)
    colors = get_distinct_colors(len(surfaces))

    for i, surface in enumerate(surfaces):
        for face_idx in surface["faces"]:
            face_colors[face_idx] = [*colors[i], 255]

    mesh_colored.visual.face_colors = face_colors

    return mesh_colored, colors

def create_normal_line(origin, normal, length=0.2):
    line = np.array([origin, origin + normal * length])
    return trimesh.load_path(line)

def compute_viewpoint_centroid(mesh, fx, resolution_target):
    """
    Computes a single viewpoint from the centroid of the mesh

    Parameters:
    mesh (trimesh object): Mesh to be inspected
    fx (float): Focal length of the camera
    resolution_target (float): Desired resolution to centroid

    Returns:

    """
    # Centroid of the mesh
    centroid = mesh.centroid

    # Approximate normal (average of face normals)
    normal = mesh.face_normals.mean(axis=0)
    normal = normal / np.linalg.norm(normal)

    # Compute distance
    Z = resolution_target * fx

    # Camera position
    camera_position = centroid + Z * normal

    return centroid, normal, camera_position

def compute_viewpoint(mesh, fx, resolution_target, surface):
    """
    Computes a single viewpoint for a given surface

    Parameters:
    mesh (trimesh object): Mesh to be inspected
    fx (float): Focal length of the camera
    resolution_target (float): Desired resolution to centroid
    surface (dictionary): Dictionary with triangles forming a surface with common normal

    Returns:

    """

    faces = surface["faces"]
    vertices = mesh.vertices[mesh.faces[faces]].reshape(-1, 3)
    vertices = np.unique(vertices, axis=0) 
    center,normal=fit_plane_svd(vertices)
    normal = normal / np.linalg.norm(normal)#Normalize to unit vector

    # Compute distance
    Z = resolution_target * fx

    # Camera position
    camera_position = center + Z * normal

    viewpoint={
        "position": camera_position,
        "target": center,
        "normal": normal
    }

    return viewpoint

def compute_all_viewpoints(mesh,fx,resolution_target,surfaces):

    viewpoints = []

    for i, surface in enumerate(surfaces):
        viewpoint = compute_viewpoint(mesh, fx, resolution_target, surface)
        
        viewpoints.append(viewpoint)

    return viewpoints

def transform_viewpoints_to_ned(viewpoints, translation_object, translation_origin, enu_to_ned=False):
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
    t = np.array(translation_origin) - np.array(translation_object)  # e.g. [east_offset, north_offset, up_offset]

    def transform_point(p):
        # Translate to NED origin, then rotate
        return R @ (p - t)

    def transform_direction(d):
        # Directions (normal) only rotate, no translation
        return R @ d

    transformed = []
    for vp in viewpoints:
        transformed.append({
            "position": R @ (vp["position"] - t),
            "target":   R @ (vp["target"] - t),
            "normal":   R @ vp["normal"]
        })

    return transformed

def visualize(mesh, surfaces, viewpoints):
    scene = trimesh.Scene()
    
    scene.add_geometry(mesh)
    colored_mesh, colors = color_mesh_by_surfaces(mesh, surfaces)
    scene.add_geometry(colored_mesh)

    #Visualize axis
    origin = np.array([0.0,0.0,0.0])
    x_axis = create_normal_line(origin,np.array([1.0,0.0,0.0]),length=1.5)
    y_axis = create_normal_line(origin,np.array([0.0,1.0,0.0]),length=1.5)
    z_axis = create_normal_line(origin,np.array([0.0,0.0,1.0]),length=1.5)
    x_axis.colors=np.array([[255,0,0,255]])
    y_axis.colors=np.array([[0,255,0,255]])
    z_axis.colors=np.array([[0,0,255,255]])

    scene.add_geometry(x_axis)
    scene.add_geometry(y_axis)
    scene.add_geometry(z_axis)
    

    for i, vp in enumerate(viewpoints):
        sphere = trimesh.creation.icosphere(radius=0.05)
        
        color = colors[i % len(colors)]
        sphere.visual.face_colors = [*color, 255]
        
        sphere.apply_translation(vp["position"])
        scene.add_geometry(sphere)

        line = create_normal_line(vp["target"], vp["normal"])
        scene.add_geometry(line)

    scene.show()

def save_viewpoints(output_path, viewpoints):
    data = {
        "viewpoints": [
            {
                "position": vp["position"].tolist(),
                "target": vp["target"].tolist()
            }
            for vp in viewpoints
        ]
    }

    with open(output_path, 'w') as f:
        yaml.dump(data, f)


def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../config/viewpoint_config.yaml")
    config = load_config(config_path)

    mesh_path = os.path.join(script_dir,config["mesh_path"])
    camera_config_path = os.path.join(script_dir,config["camera_config"])
    resolution_target = config["resolution_target"]
    output_path = os.path.join(script_dir,config["output_path"])

    # Load camera
    fx, fy, width, height = load_camera(camera_config_path)

    print(f"Loaded camera: fx={fx}, fy={fy}, resolution={width}x{height}")

    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')#, skip_materials=True
    mesh.apply_scale(config["mesh_scale"])
    if config["weld_mesh"]:
        mesh = weld_vertices(mesh, tolerance=config["weld_tolerance"])
    diagnose_adjacency(mesh)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    print(f"Mesh loaded. Centroid: {mesh.centroid}")
    #print("Bounding box:", mesh.bounds)
    #print("Size:", mesh.extents)

    #Cluster mesh into surfaces:
    surfaces= cluster_surfaces(mesh,config["surface_threshold"])

    print(f"Found {len(surfaces)} surfaces!")


    # Compute viewpoint
    viewpoints = compute_all_viewpoints(mesh, fx, resolution_target, surfaces)

    print("Viewpoints computed")
    i=7 #TODO Remove this, only used to select specific waypoint
    # Visualize
    if config["visualize"]:
        #visualize(mesh,surfaces,viewpoints)
        
        # for j in range(len(surfaces)):
        #     if j>300:
        #         visualize(mesh,[surfaces[j]],[viewpoints[j]])


        visualize(mesh,[surfaces[i]],[viewpoints[i]])
    
    transformed = transform_viewpoints_to_ned(viewpoints,config["object_offset"],config["origin_offset"],config["enu_to_ned"])
    # Save
    save_viewpoints(output_path, [transformed[i]])

    print(f"Viewpoints saved to {output_path}")


if __name__ == "__main__":
    main()
