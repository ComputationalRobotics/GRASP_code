# Copyright (c) 2026 malik-group
# SPDX-License-Identifier: MIT
# Source: Do As I Do, https://github.com/malik-group/do-as-i-do, retargeting/retargeting/utils/tracking_error.py
# at commit 824591b808c342b20079c3b4198a8c2bdf88c74e; see THIRD_PARTY_NOTICES.md.
"""Object pose tracking-error metrics for retargeted trajectories."""

import numpy as np
from scipy.spatial.transform import Rotation as R

def quat_to_vel(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to angular velocity (..., 3)."""
    axis = quat[..., 1:4]
    sin_a_2 = np.linalg.norm(axis, axis=-1, keepdims=True)

    zero_mask = sin_a_2[..., 0] == 0.0
    result = np.zeros_like(axis)

    non_zero_mask = ~zero_mask
    if np.any(non_zero_mask):
        speed = 2.0 * np.arctan2(sin_a_2[non_zero_mask, 0], quat[non_zero_mask, 0])
        # when axis-angle is larger than pi, rotation is in the opposite direction
        speed = np.where(speed > np.pi, speed - 2.0 * np.pi, speed)

        result[non_zero_mask] = (
            axis[non_zero_mask] * speed[..., np.newaxis] / sin_a_2[non_zero_mask]
        )

    return result


def mul_quat(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Quaternion product of two (w, x, y, z) quaternions."""
    result = np.zeros_like(u)
    result[..., 0] = (
        u[..., 0] * v[..., 0]
        - u[..., 1] * v[..., 1]
        - u[..., 2] * v[..., 2]
        - u[..., 3] * v[..., 3]
    )
    result[..., 1] = (
        u[..., 0] * v[..., 1]
        + u[..., 1] * v[..., 0]
        + u[..., 2] * v[..., 3]
        - u[..., 3] * v[..., 2]
    )
    result[..., 2] = (
        u[..., 0] * v[..., 2]
        - u[..., 1] * v[..., 3]
        + u[..., 2] * v[..., 0]
        + u[..., 3] * v[..., 1]
    )
    result[..., 3] = (
        u[..., 0] * v[..., 3]
        + u[..., 1] * v[..., 2]
        - u[..., 2] * v[..., 1]
        + u[..., 3] * v[..., 0]
    )
    return result


def quat_sub(qa, qb):
    """Angular difference (..., 3) between two (w, x, y, z) quaternions."""
    qneg = qb.copy()
    qneg[..., 1:] = -qneg[..., 1:]
    qdif = mul_quat(qneg, qa)
    return quat_to_vel(qdif)


def _euler_to_quat_wxyz(euler: np.ndarray) -> np.ndarray:
    """Convert intrinsic Euler (xyz) angles to quaternion in (w, x, y, z) format."""
    quat_xyzw = R.from_euler("XYZ", euler, degrees=False).as_quat()
    quat_wxyz = np.empty_like(quat_xyzw)
    quat_wxyz[..., 0] = quat_xyzw[..., 3]
    quat_wxyz[..., 1:] = quat_xyzw[..., :3]
    return quat_wxyz


def compute_object_tracking_error(
    qpos_traj: np.ndarray,
    qpos_ref: np.ndarray,
    embodiment_type: str,
    data_type: str,
) -> dict:
    """Compute object position/quaternion tracking error for a trajectory."""
    use_act = data_type.endswith("_act")
    if embodiment_type == "bimanual":
        if use_act:
            qpos_object_right_traj = qpos_traj[:, -12:-6]
            qpos_object_left_traj = qpos_traj[:, -6:]
            qpos_object_right_ref = qpos_ref[:, -12:-6]
            qpos_object_left_ref = qpos_ref[:, -6:]
        else:
            qpos_object_right_traj = qpos_traj[:, -14:-7]
            qpos_object_left_traj = qpos_traj[:, -7:]
            qpos_object_right_ref = qpos_ref[:, -14:-7]
            qpos_object_left_ref = qpos_ref[:, -7:]

        pos_object_right_traj = qpos_object_right_traj[:, :3]
        pos_object_left_traj = qpos_object_left_traj[:, :3]
        pos_object_right_ref = qpos_object_right_ref[:, :3]
        pos_object_left_ref = qpos_object_left_ref[:, :3]

        pos_err_right = np.linalg.norm(
            pos_object_right_traj - pos_object_right_ref, axis=1
        )
        pos_err_left = np.linalg.norm(
            pos_object_left_traj - pos_object_left_ref, axis=1
        )

        if use_act:
            quat_right_traj = _euler_to_quat_wxyz(qpos_object_right_traj[:, 3:])
            quat_right_ref = _euler_to_quat_wxyz(qpos_object_right_ref[:, 3:])
            quat_left_traj = _euler_to_quat_wxyz(qpos_object_left_traj[:, 3:])
            quat_left_ref = _euler_to_quat_wxyz(qpos_object_left_ref[:, 3:])
        else:
            quat_right_traj = qpos_object_right_traj[:, 3:]
            quat_right_ref = qpos_object_right_ref[:, 3:]
            quat_left_traj = qpos_object_left_traj[:, 3:]
            quat_left_ref = qpos_object_left_ref[:, 3:]

        # per-frame error series (shape (T,))
        quat_err_right = np.linalg.norm(
            quat_sub(quat_right_traj, quat_right_ref), axis=1
        )
        quat_err_left = np.linalg.norm(quat_sub(quat_left_traj, quat_left_ref), axis=1)

        left_mask = (
            np.linalg.norm(pos_object_left_ref - pos_object_left_ref[:1], axis=1).mean()
            < 0.001
        )
        right_mask = (
            np.linalg.norm(
                pos_object_right_ref - pos_object_right_ref[:1], axis=1
            ).mean()
            < 0.001
        )
        # combine the two hands into a single per-frame series: use the moving
        # hand if the other is static, else average the two element-wise
        if left_mask:
            pos_err_frames = pos_err_right
            quat_err_frames = quat_err_right
        elif right_mask:
            pos_err_frames = pos_err_left
            quat_err_frames = quat_err_left
        else:
            pos_err_frames = (pos_err_right + pos_err_left) / 2
            quat_err_frames = (quat_err_right + quat_err_left) / 2

        obj_pos_err = pos_err_frames.mean()
        obj_quat_err = quat_err_frames.mean()
        pos_err_right = pos_err_right.mean()
        pos_err_left = pos_err_left.mean()
        quat_err_right = quat_err_right.mean()
        quat_err_left = quat_err_left.mean()
    else:
        if use_act:
            qpos_object_traj = qpos_traj[:, -6:]
            qpos_object_ref = qpos_ref[:, -6:]
        else:
            qpos_object_traj = qpos_traj[:, -7:]
            qpos_object_ref = qpos_ref[:, -7:]

        pos_object_traj = qpos_object_traj[:, :3]
        pos_object_ref = qpos_object_ref[:, :3]

        if use_act:
            quat_traj = _euler_to_quat_wxyz(qpos_object_traj[:, 3:])
            quat_ref = _euler_to_quat_wxyz(qpos_object_ref[:, 3:])
        else:
            quat_traj = qpos_object_traj[:, 3:]
            quat_ref = qpos_object_ref[:, 3:]

        # per-frame error series (shape (T,))
        pos_err_frames = np.linalg.norm(pos_object_traj - pos_object_ref, axis=1)
        quat_err_frames = np.linalg.norm(quat_sub(quat_traj, quat_ref), axis=1)
        obj_pos_err = pos_err_frames.mean()
        obj_quat_err = quat_err_frames.mean()

        pos_err_right = obj_pos_err if embodiment_type == "right" else 0.0
        pos_err_left = obj_pos_err if embodiment_type == "left" else 0.0
        quat_err_right = obj_quat_err if embodiment_type == "right" else 0.0
        quat_err_left = obj_quat_err if embodiment_type == "left" else 0.0

    return {
        "obj_pos_err": obj_pos_err,
        "obj_quat_err": obj_quat_err,
        "pos_err_right": pos_err_right,
        "pos_err_left": pos_err_left,
        "quat_err_right": quat_err_right,
        "quat_err_left": quat_err_left,
        "pos_err_frames": pos_err_frames,
        "quat_err_frames": quat_err_frames,
    }
