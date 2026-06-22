import csv
import math
import yaml


INPUT_YAML = "../../config/path.yaml"
OUTPUT_CSV = "../../config/mission_litchi_converted.csv"

MAX_ACTIONS = 15
POSITION_TOL = 1e-7  # tolerance for grouping same waypoint


LITCHI_HEADER = [
    "latitude", "longitude", "altitude(m)", "heading(deg)", "curvesize(m)",
    "rotationdir", "gimbalmode", "gimbalpitchangle"
]

for i in range(1, 16):
    LITCHI_HEADER.append(f"actiontype{i}")
    LITCHI_HEADER.append(f"actionparam{i}")

LITCHI_HEADER += [
    "altitudemode", "speed(m/s)", "poi_latitude", "poi_longitude",
    "poi_altitude(m)", "poi_altitudemode", "photo_timeinterval",
    "photo_distinterval"
]


def rad_to_deg(rad):
    return math.degrees(rad)


def normalize_heading_deg(deg):
    return deg % 360


def clamp_gimbal_pitch_deg(deg):
    return max(-90, min(30, deg))


def first_or_default(value, default=0.0):
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def almost_equal(a, b, tol=POSITION_TOL):
    return abs(a - b) <= tol


def same_position(p1, p2, tol=POSITION_TOL):
    return (
        almost_equal(p1[0], p2[0], tol) and
        almost_equal(p1[1], p2[1], tol) and
        almost_equal(p1[2], p2[2], tol)
    )


def group_viewpoints_by_position(viewpoints):
    groups = []
    current_group = []

    for vp in viewpoints:
        pos = vp["position"]
        if not current_group:
            current_group.append(vp)
            continue

        current_pos = current_group[0]["position"]
        if same_position(pos, current_pos):
            current_group.append(vp)
        else:
            groups.append(current_group)
            current_group = [vp]

    if current_group:
        groups.append(current_group)

    return groups


def build_actions_for_group(group):
    """
    Assumed action codes from user's sample:
      0 = stay (milliseconds)
      1 = take photo
      4 = rotate aircraft heading (degrees)

    We do NOT add photo per waypoint automatically.
    We add photo only after each orientation setpoint.

    For gimbal pitch:
    Litchi has a waypoint-level gimbalpitchangle field, but not enough
    information from your sample to safely emit a separate gimbal action code.
    So this implementation assumes heading changes are the orientation actions
    expressed in the action list, and the row's base gimbalpitchangle is taken
    from the first item in the group.

    If you also need per-orientation gimbal pitch action, we can add that once
    you confirm the correct Litchi action code for "tilt gimbal".
    """
    actions = []

    first = True
    prev_heading = None

    for vp in group:
        heading_deg = round(normalize_heading_deg(rad_to_deg(vp.get("yaw", 0.0))))

        # Only create a new orientation action if heading changes
        if first or heading_deg != prev_heading:
            actions.append((4, heading_deg))  # rotate aircraft
            actions.append((1, 0))            # take photo
            actions.append((0, 3000))         # stay 3 seconds
            prev_heading = heading_deg
            first = False

    if len(actions) > MAX_ACTIONS:
        raise ValueError(
            f"Waypoint at position {group[0]['position']} requires {len(actions)} actions, "
            f"but Litchi supports only {MAX_ACTIONS} actions per waypoint."
        )

    return actions


def viewpoint_group_to_row(group, speed=2):
    base = group[0]
    lat, lon, alt = base["position"]

    base_heading_deg = round(normalize_heading_deg(rad_to_deg(base.get("yaw", 0.0))))
    base_gimbal_pitch_deg = round(
        clamp_gimbal_pitch_deg(rad_to_deg(first_or_default(base.get("gimbal_pitch"), 0.0)))
    )

    row = [
        round(lat, 8),
        round(lon, 8),
        round(alt, 2),
        base_heading_deg,
        0,   # curvesize(m): stop at waypoint
        0,   # rotationdir
        2,   # gimbalmode
        base_gimbal_pitch_deg,
    ]

    actions = build_actions_for_group(group)

    # fill action slots
    for action_type, action_param in actions:
        row.extend([action_type, action_param])

    # pad remaining action slots
    remaining = MAX_ACTIONS - len(actions)
    for _ in range(remaining):
        row.extend([-1, -1])

    row += [
        0,      # altitudemode
        speed,
        0,      # poi_latitude
        0,      # poi_longitude
        0,      # poi_altitude(m)
        0,      # poi_altitudemode
        -1,     # photo_timeinterval
        -1,     # photo_distinterval
    ]

    return row


def convert_yaml_to_litchi(input_yaml, output_csv, speed=2):
    with open(input_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    viewpoints = data.get("viewpoints", [])
    groups = group_viewpoints_by_position(viewpoints)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(LITCHI_HEADER)

        for group in groups:
            writer.writerow(viewpoint_group_to_row(group, speed=speed))

    print(
        f"Wrote {len(groups)} Litchi waypoints from {len(viewpoints)} viewpoints to {output_csv}"
    )


if __name__ == "__main__":
    convert_yaml_to_litchi(INPUT_YAML, OUTPUT_CSV)