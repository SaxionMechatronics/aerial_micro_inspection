import yaml
import numpy as np
import trimesh
import os
import colorsys
import math
import time
from collections import defaultdict
from sklearn.cluster import KMeans


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

def cluster_surfaces(mesh, config):
    """
    Clusters triangles into surfaces by checking normal similarity
    AND adjacency between faces, using a region growing approach.
    """
    threshold = config["surface_threshold"]
    normals = mesh.face_normals

    # Build adjacency lookup
    adj = defaultdict(set)
    for a, b in mesh.face_adjacency:
        adj[a].add(b)
        adj[b].add(a)

    clusters = []
    unvisited = set(range(len(normals)))
    id=0

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
                if neighbor in unvisited and np.dot(normals[neighbor], seed_normal) > threshold:
                #if neighbor in unvisited and np.dot(normals[neighbor], normals[face]) > threshold:
                    unvisited.remove(neighbor)
                    queue.append(neighbor)

        clusters.append({
            "normal": seed_normal,
            "faces": cluster_faces,
            "ID": id
        })

        id+=1

    if config["filter_small_surfaces"]["enabled"]:
        face_areas = mesh.area_faces

        areas = np.array([face_areas[c["faces"]].sum() for c in clusters])

        median_area = np.median(areas)
        iqr = np.percentile(areas, 75) - np.percentile(areas, 25)
        k = config["filter_small_surfaces"].get("iqr_k", 1.5)

        # Avoid zero-IQR edge case (e.g. all surfaces same size)
        if iqr == 0:
            threshold = median_area * config["filter_small_surfaces"].get("fallback_ratio", 0.01)
        else:
            threshold = median_area - k * iqr

        filtered = [c for c, a in zip(clusters, areas) if a >= threshold]
        return filtered

    return clusters

def build_face_to_surface_map(mesh, surfaces):
    face_to_surface = np.full(len(mesh.faces), -1)

    for surface in surfaces:
        for f in surface["faces"]:
            face_to_surface[f] = surface["ID"]

    return face_to_surface

def subdivide_surfaces(mesh, surfaces, max_edge=0.1):
    """
    Subdivides the mesh and propagates surface labels to new faces.

    Parameters:
        mesh (trimesh.Trimesh)
        surfaces (list of dict): original surfaces with "faces"
        max_edge (float, optional): if provided, uses subdivide_to_size

    Returns:
        new_mesh
        new_surfaces
    """

    new_mesh, face_index = mesh.subdivide_to_size(max_edge=max_edge,return_index=True)
        
        # IMPORTANT: subdivide_to_size DOES NOT return mapping
        # → fallback: use nearest face mapping
        #_, face_index = mesh.nearest.on_surface(new_mesh.triangles_center)


    # Prepare new surfaces
    new_surfaces = []
    
    # Convert face_index to numpy for speed
    face_index = np.array(face_index)

    for surface in surfaces:
        original_faces = set(surface["faces"])

        # Find all new faces that come from these original ones
        mask = np.isin(face_index, list(original_faces))
        new_faces = np.where(mask)[0]

        new_surface = {
            "normal": surface["normal"],
            "faces": new_faces.tolist(),
            "ID": surface["ID"]
        }

        new_surfaces.append(new_surface)

    return new_mesh, new_surfaces

def fit_plane_svd(points, reference_normal=None):

    centroid = points.mean(axis=0)
    
    centered = points - centroid
    
    _, _, Vt = np.linalg.svd(centered)
    
    normal = Vt[-1]

    # Flip to agree with the mesh face normals if provided
    if reference_normal is not None:
        if np.dot(normal, reference_normal) < 0:
            normal = -normal
    
    return centroid, normal

def project_points_to_plane(points, normal):
    normal = normal / np.linalg.norm(normal)

    world_up = np.array([0, 0, 1])

    # Avoid parallel case
    if abs(np.dot(normal, world_up)) > 0.95:
        world_up = np.array([0, 1, 0])

    u1 = np.cross(world_up, normal)
    u1 /= np.linalg.norm(u1)

    u2 = np.cross(normal, u1)

    coords_2d = np.stack([
        points @ u1,
        points @ u2
    ], axis=1)

    return coords_2d

def compute_bbox_size(points_2d):
    xmin, ymin = points_2d.min(axis=0)
    xmax, ymax = points_2d.max(axis=0)
    return xmax - xmin, ymax - ymin

def split_surfaces_with_kmeans(mesh, surfaces, width, height, resolution_target, alpha=0.9, circular_fov=False):
    """
    Splits surfaces using iterative KMeans until clusters fit inside camera FOV.
    """

    new_surfaces = []

    # --- FOV (simplified as rectangle in meters)
    fov_w = resolution_target * width * alpha
    fov_h = resolution_target * height * alpha

    for surface in surfaces:

        faces = surface["faces"]

        # --- Get vertex indices
        face_vertex_indices = mesh.faces[faces]
        unique_vertex_indices = np.unique(face_vertex_indices)

        vertices = mesh.vertices[unique_vertex_indices]

        # --- Fit plane
        center, normal = fit_plane_svd(vertices, reference_normal=surface["normal"])

        # --- Project to 2D
        pts_2d = project_points_to_plane(vertices - center, normal)

        # --- Map vertex index → local index
        idx_map = {v: i for i, v in enumerate(unique_vertex_indices)}

        # --- Iterative K search

        area_surface = np.prod(compute_bbox_size(pts_2d))
        area_fov = fov_w * fov_h

        k = max(1, int(np.ceil(area_surface / area_fov)))

        #print(f'K initialized to {k}')

        while True:

            k = min(k, len(pts_2d))  # safety

            kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
            labels = kmeans.fit_predict(pts_2d)

            valid = True

            # --- Check each cluster fits FOV
            for cluster_id in range(k):
                cluster_pts = pts_2d[labels == cluster_id]

                if len(cluster_pts) == 0:
                    continue


                if circular_fov:
                    fov_radius = min(fov_w, fov_h) / 2
                    center = cluster_pts.mean(axis=0)
                    dist = np.linalg.norm(cluster_pts - center, axis=1)
                    if dist.max() > fov_radius:
                        valid = False
                        break
                else:
                    w, h = compute_bbox_size(cluster_pts)
                    if w > fov_w or h > fov_h:
                        valid = False
                        break

            if valid:
                break
            else:
                k += 1

        # --- Assign faces to clusters
        for cluster_id in range(k):

            cluster_faces = []

            for f_idx, face in zip(faces, face_vertex_indices):

                local_ids = [idx_map[v] for v in face]
                face_labels = [labels[i] for i in local_ids]

                # Majority voting
                if face_labels.count(cluster_id) >= 2:
                    cluster_faces.append(f_idx)

            if len(cluster_faces) == 0:
                continue

            new_surfaces.append({
                "normal": normal,
                "faces": cluster_faces,
                "ID": surface['ID']#f"{surface['ID']}_{cluster_id}"
            })

    return new_surfaces

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

def create_plane(point, normal, size=5.0):
    normal = normal / np.linalg.norm(normal)

    v = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    
    u1 = np.cross(normal, v)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(normal, u1)

    corners = [
        point + size*( u1 + u2),
        point + size*( u1 - u2),
        point + size*(-u1 - u2),
        point + size*(-u1 + u2),
    ]

    faces = [[0,1,2], [0,2,3]]

    plane = trimesh.Trimesh(vertices=corners, faces=faces)

    flipped_faces = [[0,2,1], [0,3,2]]
    plane_double = trimesh.Trimesh(vertices=corners, faces=faces + flipped_faces)

    plane_double.visual.face_colors = [100, 100, 255, 80]

    return plane_double

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
    center, normal = fit_plane_svd(vertices, reference_normal=surface["normal"])
    normal = normal / np.linalg.norm(normal)#Normalize to unit vector

    # Compute distance
    Z = resolution_target * fx

    # Camera position
    camera_position = center + Z * normal

    # Drone heading
    yaw=0
    dx=-normal[0]
    dy=-normal[1]
    dz=-normal[2]

    if dx != 0.0 or dy != 0.0:
        yaw = math.atan2(dy, dx)

    # --- PITCH ---
    horizontal_norm = math.sqrt(dx**2 + dy**2)
    pitch = math.atan2(-dz, horizontal_norm)



    viewpoint={
        "position": camera_position,
        "yaw": yaw,
        "pitch": pitch,
        "target": center,
        "normal": normal,
        "surface_id": surface["ID"]
    }

    return viewpoint

def compute_all_viewpoints(mesh,fx,config,surfaces):

    resolution_target = config["resolution_target"]

    viewpoints = []

    for surface in surfaces:
        viewpoint = compute_viewpoint(mesh, fx, resolution_target, surface)
        viewpoints.append(viewpoint)

    filtered, surfaces = apply_filters(viewpoints,config, mesh, surfaces)

    return filtered

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
            "target":   R @ (vp["target"] - t),
            "normal":   R @ vp["normal"]
        })

    return transformed

def visualize(mesh, surfaces, viewpoints, config, specific_id=-1):
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

    if specific_id>=0:
        print(f"Showing viewpoint with id {specific_id}")

    if "ground_plane" in config and config["ground_plane"]["enabled"]:
        plane = create_plane(
            np.array(config["ground_plane"]["point"]),
            np.array(config["ground_plane"]["normal"]),
            10.0
        )
        scene.add_geometry(plane)
    
    scene.show()

def save_viewpoints(output_path, viewpoints):
    data = {
        "viewpoints": [
            {
                "position": vp["position"].tolist(),
                "yaw": vp["yaw"],
                "pitch": vp["pitch"],
                "target": vp["target"].tolist()
            }
            for vp in viewpoints
        ]
    }

    with open(output_path, 'w') as f:
        yaml.dump(data, f)

###### Filters ######

def apply_filters(viewpoints, config, mesh, surfaces):

    if config["ground_plane"]["enabled"]:
        viewpoints, removed = filter_ground(viewpoints, config["ground_plane"])
        for index in sorted(removed, reverse=True):
            del surfaces[index]
    if config["occlude_filter"]["enabled"]:
        face_to_surface=build_face_to_surface_map(mesh, surfaces)
        viewpoints,removed = filter_occluded_viewpoints(mesh,viewpoints, face_to_surface)
        for index in sorted(removed, reverse=True):
            del surfaces[index]
    
    return viewpoints, surfaces

def filter_ground(viewpoints, ground_cfg):
    p0 = np.array(ground_cfg["point"])
    n = np.array(ground_cfg["normal"])
    n = n / np.linalg.norm(n)
    min_dist = ground_cfg["min_distance"]

    filtered = []
    removed = []

    for i,vp in enumerate(viewpoints):
        p = vp["position"]

        dist = np.dot(p - p0, n)

        if dist >= min_dist:
            filtered.append(vp)
        else:
            removed.append(i)

    return filtered, removed

def filter_occluded_viewpoints(mesh, viewpoints, face_to_surface):
    filtered = []
    removed = []

    for i,vp in enumerate(viewpoints):
        origin = vp["position"]
        target = vp["target"]

        direction = target - origin
        direction = direction / np.linalg.norm(direction)

        # offset to avoid auto intersection
        origin = origin + direction * 1e-3

        # Ray cast
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=[origin],
            ray_directions=[direction]
        )

        if len(index_tri) == 0:
            removed.append(i)
            continue  # discard?

        # Closest intersection face
        distances = np.linalg.norm(locations - origin, axis=1)
        closest_idx = np.argmin(distances)
        hit_face = index_tri[closest_idx]

        # Check if the face is from the surface
        if face_to_surface[hit_face] == vp["surface_id"]:
            filtered.append(vp)
        else:
            removed.append(i)

    return filtered, removed

def main():
    #Check execution time
    start = time.time()
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../../config/viewpoint_config.yaml")
    config = load_config(config_path)

    mesh_path = os.path.join(script_dir,config["mesh_path"])
    camera_config_path = os.path.join(script_dir,config["camera_config"])
    output_path = os.path.join(script_dir,config["output_path"])

    # Load camera
    fx, fy, width, height = load_camera(camera_config_path)

    print(f"Loaded camera: fx={fx}, fy={fy}, resolution={width}x{height}")

    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')#, skip_materials=True
    mesh.apply_scale(config["mesh_scale"])
    if config["weld_mesh"]:
        mesh = weld_vertices(mesh, tolerance=config["weld_tolerance"])
    #diagnose_adjacency(mesh)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    #print(f"Mesh loaded. Centroid: {mesh.centroid}")
    #print("Bounding box:", mesh.bounds)
    #print("Size:", mesh.extents)

    #Cluster mesh into surfaces:
    surfaces= cluster_surfaces(mesh,config)
    mesh, surfaces= subdivide_surfaces(mesh, surfaces, max_edge=config["subdivide_edge_size"])
    surfaces = split_surfaces_with_kmeans(mesh,surfaces,width,height,config['resolution_target'],config['alpha'],config['circular_fov'])

    print(f"Found {len(surfaces)} surfaces!")


    # Compute viewpoint
    viewpoints = compute_all_viewpoints(mesh, fx, config, surfaces)

    #We only want to see the time to compute 
    end = time.time()
    print(f"Elapsed: {end - start:.4f}s")

    print(f"Viewpoints computed: {len(viewpoints)}")
    i=102 #TODO Remove this, only used to select specific waypoint
    # Visualize
    if config["visualize"]:
        visualize(mesh,surfaces,viewpoints,config)
        
        # for j in range(len(surfaces)):
            
        #     visualize(mesh,[surfaces[j]],[viewpoints[j]],config,specific_id=j)


        visualize(mesh,[surfaces[i]],[viewpoints[i]],config,specific_id=i)
    
    #print(f"Viewpoint used: {viewpoints[i]}")
    #transformed = transform_viewpoints_to_ned(viewpoints,config["object_offset"],config["origin_offset"],config["camera_offset"],enu_to_ned=config["enu_to_ned"])
    # Save
    # save_viewpoints(output_path, [transformed[i]])
    save_viewpoints(output_path, viewpoints)


    print(f"Viewpoints saved to {output_path}")





if __name__ == "__main__":
    main()
