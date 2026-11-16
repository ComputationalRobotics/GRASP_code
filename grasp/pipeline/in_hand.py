# Source: Do As I Do, https://github.com/malik-group/do-as-i-do, at commit 824591b808c342b20079c3b4198a8c2bdf88c74e:
# retargeting/retargeting/utils/in_hand.py (compute_in_hand_mask, compute_near_floor_mask, body_has_mesh_geom,
# extract_body_mesh_verts; assertion message shortened) and retargeting/retargeting/utils/mujoco_utils.py
# (quat_wxyz_to_rotmat).
# MIT License; see THIRD_PARTY_NOTICES.md.

"""Reference-side proximity masks: is a hand within reach of an object, and is an object resting near the floor.

Used by ``grasp.pipeline.oakink2_contact_gates`` to build the contact gates of the OakInk2 objective.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

DEFAULT_DISTANCE_THRESH: float = 0.1
_MAX_OBJ_VERTS_FOR_MASK: int = 512


def quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert MuJoCo wxyz quaternion to a 3x3 rotation matrix."""
    q = q_wxyz / (np.linalg.norm(q_wxyz) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def compute_in_hand_mask(
    points_world: np.ndarray,
    qpos_obj: np.ndarray,
    obj_verts: np.ndarray,
    distance_thresh: float = DEFAULT_DISTANCE_THRESH,
    max_obj_verts: int = _MAX_OBJ_VERTS_FOR_MASK,
) -> np.ndarray:
    """Per-frame ``min(dist(points_world[t], obj_world[t])) < thresh`` mask.

    Generic point-set to object-mesh proximity check, used both as an in-hand
    gate (points = hand verts) and an object-near-pedestal gate (points =
    static pedestal centers). Inputs must already share the trajectory timeline.

    ``obj_verts`` is subsampled to ``max_obj_verts`` (seeded): dense
    convex-decomp meshes carry tens of thousands of sub-mm verts but the
    threshold is ~10cm, so coarse coverage suffices. The endpoint check in
    [[in_hand_at_endpoint]] does not subsample (2 frames, exactness matters).
    """
    T = points_world.shape[0]

    if obj_verts.shape[0] > max_obj_verts:
        rng = np.random.default_rng(0)
        idx = rng.choice(obj_verts.shape[0], size=max_obj_verts, replace=False)
        obj_verts = obj_verts[idx]

    obj_pos = qpos_obj[:, :3]                       # (T, 3)
    obj_quat_xyzw = qpos_obj[:, [4, 5, 6, 3]]       # (T, 4)
    R_obj = Rotation.from_quat(obj_quat_xyzw).as_matrix()  # (T, 3, 3)

    # Rotate+translate object verts for all frames in one batched op.
    obj_world_all = np.einsum("tij,vj->tvi", R_obj, obj_verts) + obj_pos[:, None, :]
    # (T, V_obj, 3)

    # Per frame: compare squared distances against squared threshold so we skip
    # the per-pair sqrt. cdist with 'sqeuclidean' uses the
    # ‖a-b‖² = ‖a‖² + ‖b‖² - 2 a·bᵀ identity internally, dispatching the inner
    # product to BLAS and avoiding the (M, V_obj, 3) diff tensor that
    # broadcasted-subtract would allocate every frame.
    thresh_sq = float(distance_thresh) ** 2
    out = np.zeros(T, dtype=np.bool_)
    for t in range(T):
        out[t] = cdist(points_world[t], obj_world_all[t], "sqeuclidean").min() < thresh_sq
    return out


def compute_near_floor_mask(
    qpos_obj: np.ndarray,
    obj_verts: np.ndarray,
    floor_z: float = 0.0,
    distance_thresh: float = DEFAULT_DISTANCE_THRESH,
    max_obj_verts: int = _MAX_OBJ_VERTS_FOR_MASK,
) -> np.ndarray:
    """Per-frame mask: object's lowest world vertex within ``distance_thresh``
    of a horizontal floor plane at ``floor_z``. Floor analog of
    [[compute_in_hand_mask]]'s pedestal gate (a plane has no finite query
    points, so distance is the lowest vertex height above ``floor_z``).
    """
    if obj_verts.shape[0] > max_obj_verts:
        rng = np.random.default_rng(0)
        idx = rng.choice(obj_verts.shape[0], size=max_obj_verts, replace=False)
        obj_verts = obj_verts[idx]

    obj_pos = qpos_obj[:, :3]                       # (T, 3)
    obj_quat_xyzw = qpos_obj[:, [4, 5, 6, 3]]       # (T, 4)
    R_obj = Rotation.from_quat(obj_quat_xyzw).as_matrix()  # (T, 3, 3)
    obj_world_all = np.einsum("tij,vj->tvi", R_obj, obj_verts) + obj_pos[:, None, :]
    min_z = obj_world_all[..., 2].min(axis=1)       # (T,)
    return (min_z - floor_z) < float(distance_thresh)


def body_has_mesh_geom(model, body_id: int) -> bool:
    """Whether ``body_id`` has a mesh geom: tells a real object body from a
    meshless placeholder (e.g. the bimanual ``left_object`` phantom that
    generate_scene.py emits with only a primitive mass sphere)."""
    import mujoco

    if body_id < 0:
        return False
    for gi in range(model.ngeom):
        if int(model.geom_bodyid[gi]) != body_id:
            continue
        if int(model.geom_type[gi]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        if int(model.geom_dataid[gi]) >= 0:
            return True
    return False


def extract_body_mesh_verts(
    model, body_id: int, data=None, apply_geom_xform: bool = False,
) -> np.ndarray:
    """Concatenate vertices of every mesh geom attached to ``body_id``.

    Returns mesh-local verts by default. ``apply_geom_xform=True`` applies each
    geom's pos/quat → body-local frame (needed whenever non-identity: every
    robot link, and objects with per-geom-repositioned convex pieces). Passing
    ``data`` (after ``mj_kinematics``) additionally applies the body world
    transform, returning world-frame verts.
    """
    import mujoco

    parts: list[np.ndarray] = []
    for gi in range(model.ngeom):
        if int(model.geom_bodyid[gi]) != body_id:
            continue
        if int(model.geom_type[gi]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mesh_id = int(model.geom_dataid[gi])
        if mesh_id < 0:
            continue
        adr = int(model.mesh_vertadr[mesh_id])
        n = int(model.mesh_vertnum[mesh_id])
        v = np.asarray(model.mesh_vert[adr : adr + n], dtype=np.float64)
        if apply_geom_xform:
            Rg = quat_wxyz_to_rotmat(model.geom_quat[gi])
            v = v @ Rg.T + model.geom_pos[gi]
        parts.append(v)
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    assert parts, (
        f"Body {body_id!r} ({body_name!r}) has no mesh geoms attached; "
        f"hand-object distance checks need mesh geometry."
    )
    out = np.concatenate(parts, axis=0)
    if data is not None:
        out = out @ data.xmat[body_id].reshape(3, 3).T + data.xpos[body_id]
    return out
