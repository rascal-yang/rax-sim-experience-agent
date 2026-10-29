import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def slew(previous, desired, dt):
    result = {}
    for s, v in desired.items():
        a = np.asarray(previous[s])
        b = np.asarray(v)
        delta = b[:3] - a[:3]
        n = np.linalg.norm(delta)
        out = b.copy()
        out[:3] = a[:3] + delta * min(1.0, 0.2 * dt / max(n, 1e-12))
        r = Rotation.from_quat(np.roll(np.stack([a[3:], b[3:]]), -1, axis=1))
        angle = (r[0].inv() * r[1]).magnitude()
        out[3:] = np.roll(
            Slerp([0, 1], r)(min(1.0, dt * 0.8 / max(angle, 1e-12))).as_quat(), 1
        )
        result[s] = out
    return result


def navigation_poses(initial, root_start, current_root):
    """Rigidly attach the neutral upper posture to measured root translation.

    The locomotion policy owns root speed. A slower world-space arm/torso slew
    would drag its targets behind the walking root and impede locomotion.
    """
    offset = np.asarray(current_root)[:3] - np.asarray(root_start)[:3]
    poses = {side: np.asarray(pose).copy() for side, pose in initial.items()}
    for pose in poses.values():
        pose[:3] += offset
    return poses
