"""HOT3D device world frames -> the simulation's Z-up world frame."""
import numpy as np
from scipy.spatial.transform import Rotation

VERSION = 'device_world_zup_v1'


def world_rotation(split):
    if split == 'train_aria':
        return np.eye(3)
    if split == 'train_quest3':
        # Proper rotation: +Y -> +Z, +Z -> -Y. Local object meshes stay intact.
        return np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    raise ValueError(f'Unknown HOT3D coordinate convention: {split}')


def points_to_sim(points, split):
    if split == 'train_aria':
        return points
    return np.asarray(points) @ world_rotation(split).T


def poses_to_sim(poses, split):
    if split == 'train_aria':
        return poses
    out = np.array(poses, copy=True)
    out[..., :3] = points_to_sim(out[..., :3], split)
    rotation = Rotation.from_matrix(world_rotation(split))
    source = Rotation.from_quat(out[..., [4, 5, 6, 3]])
    out[..., 3:] = (rotation * source).as_quat()[..., [3, 0, 1, 2]]
    return out
