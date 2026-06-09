import yaml
import numpy as np
import networkx as nx
from networkx.algorithms import approximation as approx
import trimesh
import open3d as o3d
import os
import math
import time


def load_config(config_path):
    """
    Loads the config file provided in path
    
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def save_config(config, config_path):
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)

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

    offset = cam["body_frame_offset"]

    return fx, fy, width, height, offset

# ---------------------------
# Load viewpoints 
# ---------------------------
def load_viewpoints(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    viewpoints = []
    for vp in data["viewpoints"]:
        viewpoints.append({
            "position": np.array(vp["position"]),
            "yaw": vp["yaw"],
            "gimbal_yaw": np.array(vp["gimbal_yaw"]),
            "gimbal_pitch": np.array(vp["gimbal_pitch"]),
            "targets":   vp["targets"]
        })

    return viewpoints


# ---------------------------
# Save path (same format, reordered)
# ---------------------------
def save_path(file_path, ordered_viewpoints, ref_gps:list):
    data = {
        "GPS_ref": ref_gps,
        "viewpoints": [
            {
                "position": vp["position"],
                "yaw": vp["yaw"].tolist(),
                "gimbal_yaw": vp["gimbal_yaw"],
                "gimbal_pitch": vp["gimbal_pitch"],
                "targets": [
                    t.tolist() if hasattr(t, 'tolist') else t 
                    for t in vp["targets"]
                ]

            }
            for vp in ordered_viewpoints
        ]
    }
    with open(file_path, 'w') as f:
        yaml.dump(data, f)

def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Builds a rotation matrix from roll, pitch, yaw (in radians).
    Convention: intrinsic Tait-Bryan ZYX (yaw applied first, then pitch, then roll).
    This is standard aerospace / MAVLink / ROS convention.
    R transforms a vector FROM the source frame TO the target frame.
    """
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)

    # Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return np.array([
        [ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [ sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,     cp*sr,             cp*cr            ]
    ])


def wrap_to_pi(angle: float) -> float:
    """Wraps a scalar or array angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def transform_viewpoints(viewpoints, t, translation_camera, roll: float = 0.0, pitch: float = 0.0,yaw: float = 0.0, degrees = True):

    # Angles must be defined intrinsically in this order: first Yaw rotation, then pitch, last roll
    if degrees:
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)
    R = rotation_matrix_from_euler(roll, pitch, yaw) 

    def transform_heading(angle_rad: float) -> float:
        direction = np.array([math.cos(angle_rad), math.sin(angle_rad), 0.0])
        rotated   = R @ direction
        return wrap_to_pi(math.atan2(rotated[1], rotated[0]))

    def transform_gimbal_pitch(pitch_rad: float) -> float:
        # Unit vector pointing in the direction of (0° yaw, pitch_rad elevation)
        direction = np.array([math.cos(pitch_rad), 0.0, math.sin(pitch_rad)])
        rotated   = R @ direction
        # Elevation angle in target frame: atan2(z, horizontal_magnitude)
        horiz = math.sqrt(rotated[0]**2 + rotated[1]**2)
        return wrap_to_pi(math.atan2(rotated[2], horiz))

    transformed = []
    for vp in viewpoints:
        transformed.append({
            # --- position & targets: standard affine transform ---
            "position": (R @ (vp["position"] - t)) - np.array(translation_camera),
            "targets":  [(R @ (np.array(tgt) - t)).tolist() for tgt in vp["targets"]],

            # --- heading angles: rotate as direction vectors ---
            "yaw":        transform_heading(vp["yaw"]),
            "gimbal_yaw": [transform_heading(y).tolist() for y in vp["gimbal_yaw"]],

            # --- gimbal pitch: rotate as elevation vector ---
            "gimbal_pitch": [transform_gimbal_pitch(p).tolist() for p in vp["gimbal_pitch"]],
        })

    return transformed

def enu_to_gps(
    enu_position: np.ndarray,
    anchor_lat_deg: float,
    anchor_lon_deg: float,
    anchor_alt_m: float,
):
    """
    Converts an ENU Cartesian offset [east, north, up] (metres) to
    absolute GPS coordinates (WGS84), given a GPS anchor point.

    The anchor is the point whose GPS coordinates correspond to ENU (0, 0, 0).

    Args:
        enu_position:    np.array [east, north, up] in metres
        anchor_lat_deg:  GPS latitude  of the ENU origin [degrees]
        anchor_lon_deg:  GPS longitude of the ENU origin [degrees]
        anchor_alt_m:    GPS altitude  of the ENU origin [metres, WGS84]

    Returns:
        (latitude_deg, longitude_deg, altitude_m)
    """

    # WGS84 ellipsoid constants
    _WGS84_A  = 6_378_137.0          # semi-major axis [m]
    _WGS84_E2 = 6.6943799901414e-3   # first eccentricity squared

    lat0 = math.radians(anchor_lat_deg)
    lon0 = math.radians(anchor_lon_deg)
    e, n, u = float(enu_position[0]), float(enu_position[1]), float(enu_position[2])

    # Radius of curvature in the prime vertical
    N = _WGS84_A / math.sqrt(1 - _WGS84_E2 * math.sin(lat0)**2)

    # Latitude: 1 metre north ≈ 1/(M) radians, where M is meridional radius
    M = _WGS84_A * (1 - _WGS84_E2) / (1 - _WGS84_E2 * math.sin(lat0)**2)**1.5
    delta_lat = n / M
    delta_lon = e / (N * math.cos(lat0))

    lat = math.degrees(lat0 + delta_lat)
    lon = math.degrees(lon0 + delta_lon)
    alt = anchor_alt_m + u

    return [lat, lon, alt]


# ─────────────────────────────────────────────
#  Full pipeline — Local frame → GPS
# ─────────────────────────────────────────────

def transform_viewpoints_to_gps(
    viewpoints,
    anchor_lat_deg: float,
    anchor_lon_deg: float,
    anchor_alt_m: float,
    translation_frame,
    translation_camera,
    roll: float  = 0.0,
    pitch: float = 0.0,
    yaw: float   = 0.0,
) -> list[dict]:
    """
    Full pipeline: local frame → ENU → GPS.

    Args:
        viewpoints:       list of dicts (position, targets, yaw, gimbal_yaw, gimbal_pitch)
        anchor_lat/lon/alt: GPS coordinates of the ENU frame origin
        translation_*:    same as transform_viewpoints() — offsets in local frame [m]
        roll/pitch/yaw:   rotation from local frame to ENU (intrinsic ZYX, radians)

    Returns:
        list of dicts, same structure, in GPS or NED
    """
    # Step 1: local → ENU  (identity if all zeros)
    enu_viewpoints = transform_viewpoints(
        viewpoints,
        translation_frame,
        translation_camera,
        roll=roll, pitch=pitch, yaw=yaw,
    )

    # Step 2: ENU → GPS for positions and targets and ENU → NED for angles
    for vp in enu_viewpoints:
        vp["position"] = enu_to_gps(
            vp["position"], anchor_lat_deg, anchor_lon_deg, anchor_alt_m
        )
        vp["targets"] = [
            enu_to_gps(np.array(t), anchor_lat_deg, anchor_lon_deg, anchor_alt_m)
            for t in vp["targets"]
        ]

        vp["yaw"]        = wrap_to_pi(math.pi / 2 - vp["yaw"])
        vp["gimbal_yaw"] = [wrap_to_pi(math.pi / 2 - y).tolist() for y in vp["gimbal_yaw"]]
        vp["gimbal_pitch"] = [-p for p in vp["gimbal_pitch"]]

    # Sort each viewpoint's gimbal data by increasing yaw (clockwise sweep)
    for vp in enu_viewpoints:
        vp["gimbal_yaw"], vp["gimbal_pitch"], vp["targets"] = (
            list(x) for x in zip(*sorted(zip(
                vp["gimbal_yaw"],
                vp["gimbal_pitch"],
                vp["targets"]
            )))
        )

    return enu_viewpoints

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

def create_axis_cylinders(origin, length=0.1):
    radius = length * 0.04
    directions = {
        'picked_axis_x': (np.array([1.0, 0.0, 0.0]), [255, 0,   0,   255]),
        'picked_axis_y': (np.array([0.0, 1.0, 0.0]), [0,   255, 0,   255]),
        'picked_axis_z': (np.array([0.0, 0.0, 1.0]), [0,   0,   255, 255]),
    }
    geometries = {}
    for name, (direction, color) in directions.items():
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=8)
        z = np.array([0.0, 0.0, 1.0])
        if not np.allclose(direction, z):
            cross = np.cross(z, direction)
            angle = np.arctan2(np.linalg.norm(cross), np.dot(z, direction))
            rot_matrix = trimesh.transformations.rotation_matrix(angle, cross)
        else:
            rot_matrix = np.eye(4)
        translation = trimesh.transformations.translation_matrix(
            origin + direction * length / 2
        )
        cyl.apply_transform(translation @ rot_matrix)
        cyl.visual.face_colors = color
        geometries[name] = cyl
    return geometries


def pick_vertex_on_click(mesh, axis_length=0.1):
    scene = trimesh.Scene(mesh)
    picked = {}
    axis_names = ['picked_axis_x', 'picked_axis_y', 'picked_axis_z']

    viewer = scene.show(start_loop=False)

    @viewer.event
    def on_mouse_press(x, y, buttons, modifiers):
        width, height = viewer.width, viewer.height

        ndc_x = (2.0 * x / width) - 1.0
        ndc_y = (2.0 * y / height) - 1.0

        camera = viewer.scene.camera
        fov_y = np.radians(camera.fov[1])
        aspect = width / height
        tan_half_fov = np.tan(fov_y / 2.0)

        ray_dir_cam = np.array([
            ndc_x * aspect * tan_half_fov,
            ndc_y * tan_half_fov,
            -1.0
        ])

        cam_to_world = viewer.scene.camera_transform
        rotation = cam_to_world[:3, :3]
        ray_dir_world = rotation @ ray_dir_cam
        ray_dir_world /= np.linalg.norm(ray_dir_world)
        ray_origin_world = cam_to_world[:3, 3]

        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=[ray_origin_world],
            ray_directions=[ray_dir_world]
        )

        if len(locations) > 0:
            hit = locations[0]
            distances = np.linalg.norm(mesh.vertices - hit, axis=1)
            closest_vertex_idx = np.argmin(distances)
            closest_vertex = mesh.vertices[closest_vertex_idx]
            picked['point'] = closest_vertex
            print(f"\n✓ Vertex index: {closest_vertex_idx}")
            print(f"  Coordinates:  {closest_vertex}")

            # Remove old axes from scene
            for name in axis_names:
                if name in viewer.scene.geometry:
                    viewer.scene.delete_geometry(name)

            # Add new axes
            axes = create_axis_cylinders(closest_vertex, length=axis_length)
            for name, geom in axes.items():
                viewer.scene.add_geometry(geom, geom_name=name)

            # Force full buffer rebuild in correct order
            try:
                viewer._update_meshes()       # rebuild vertex buffers from scene
                viewer._update_vertex_list()  # push to OpenGL
                viewer.invalid = True         # mark as needing redraw
                viewer.on_draw()              # redraw
                viewer.flip()                 # swap front/back buffer
            except Exception as e:
                print(f"Redraw error: {e}")

        else:
            print("✗ No surface hit.")

    import pyglet
    pyglet.app.run()
    return picked.get('point')


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
    config_path_GPS = os.path.join(script_dir, "../../config/GPS_reference.yaml")
    config_GPS = load_config(config_path_GPS)

    mesh_file = os.path.join(script_dir,config["mesh_path"])
    viewpoints_file = os.path.join(script_dir,config["output_path"])
    output_file = os.path.join(script_dir,config["planning_path"])
    camera_config_path = os.path.join(script_dir,config["camera_config"])
    fx, fy, width, height, cam_offset = load_camera(camera_config_path)

    print("Loading mesh...")
    mesh = trimesh.load(mesh_file, force='mesh')
    mesh.apply_scale(config["mesh_scale"])
    if config["weld_mesh"]:
        mesh = weld_vertices(mesh, tolerance=config["weld_tolerance"])
    mesh = mesh.subdivide_to_size(max_edge=config["subdivide_edge_size"])

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

    print("Use the current GPS reference? [yes/no]")
    choose = str(input())

    if choose == "yes":
        ref_lat=config_GPS["latitude_reference"]
        ref_long=config_GPS["longitude_reference"]
        ref_alt=config_GPS["altitude_reference"]
        t = np.array(config_GPS["frame_translation"])
        roll_off=config_GPS["roll_difference"]
        pitch_off=config_GPS["pitch_difference"]
        yaw_off=config_GPS["yaw_difference"]
    else:
        print("Select point as GPS reference. Use W to see vertices. Press Q when done")

        point = pick_vertex_on_click(mesh)
        print(f"\nFinal selected point: {point}")
        t = np.array(point)

        print("Put the phone on the object facing in the green axis direction (Local north)")
        print("Introduce latitude: ")
        ref_lat= float(input())
        print("Introduce longitude: ")
        ref_long= float(input())
        print("Introduce altitude: ")
        ref_alt= float(input())
        print("Introduce difference respect earth's north in degrees: ")
        yaw_off= float(input())
        print("Introduce difference in roll in degrees (Normally 0): ")
        roll_off= float(input())
        print("Introduce difference in pitch north in degrees (Normally 0): ")
        pitch_off= float(input())

        config_GPS["latitude_reference"]=ref_lat
        config_GPS["longitude_reference"]=ref_long
        config_GPS["altitude_reference"]=ref_alt
        config_GPS["frame_translation"]=t.tolist()
        config_GPS["roll_difference"]=roll_off
        config_GPS["pitch_difference"]=pitch_off
        config_GPS["yaw_difference"]=yaw_off
        save_config(config_GPS,config_path_GPS)
        



    print("Visualizing path...")

    previous = viewpoints[0]["position"]
    distance = 0
    for vp in viewpoints:
        distance += np.linalg.norm(previous - vp["position"])
        previous = vp["position"]

    print(f"Total length of the path: {distance} meters")
    visualize_path(mesh, ordered_viewpoints)

    
    
    transformed = transform_viewpoints_to_gps(ordered_viewpoints,ref_lat,ref_long,ref_alt,t,cam_offset,roll_off,pitch_off,yaw_off)

    print("Saving path...")
    save_path(output_file, transformed,[ref_lat,ref_long,ref_alt])

    print("Done!")


if __name__ == "__main__":
    main()