
import yaml
import os
import numpy as np

def load_yaml(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def quaternion_to_rotmat(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])