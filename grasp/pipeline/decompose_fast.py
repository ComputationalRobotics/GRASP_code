# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Source: SPIDER, https://github.com/facebookresearch/spider, spider/preprocess/decompose_fast.py
# at commit 44717007de41cbef7565dff7ff9f4453557a2d3d, adapted for GRASP.
# Licensed under CC BY-NC 4.0; see THIRD_PARTY_NOTICES.md.

try:
    import trimesh
except ModuleNotFoundError:
    print("trimesh is required. Please install with `pip install trimesh`")
    raise SystemExit(1)


import json
import os
from pathlib import Path

import numpy as np
import tyro
from loguru import logger
from scipy.spatial.transform import Rotation as _ScipyR

from grasp import ROOT
from grasp.pipeline._decompose_common import MeshPart, flatten_base


def _R_from_quat(q_xyzw: np.ndarray) -> np.ndarray:
    return _ScipyR.from_quat(q_xyzw).as_matrix()


def fast_voxel_convex_decomp_from_pointcloud(
    points: np.ndarray, pitch: float = 0.1, min_points: int = 20
) -> list[MeshPart]:
    """Approximate convex decomposition via voxel clusters and convex hulls."""
    coords = np.floor(points / pitch).astype(int)
    unique_voxels, inverse = np.unique(coords, axis=0, return_inverse=True)

    hulls: list[MeshPart] = []
    for idx, _ in enumerate(unique_voxels):
        cluster_points = points[inverse == idx]
        if len(cluster_points) < min_points:
            continue

        cluster_mesh = trimesh.Trimesh(vertices=cluster_points, faces=[])
        hull = cluster_mesh.convex_hull
        vertices = np.asarray(hull.vertices)
        faces = np.asarray(hull.faces, dtype=int)
        hulls.append((vertices, faces))

    return hulls


def main(
    dataset_dir: str = f"{ROOT}/../data",
    dataset_name: str = "oakink",
    robot_type: str = "allegro",
    embodiment_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    data_id: int = 0,
    add_flat_base: bool = True,
    plate_size: float = 0.03,
    plate_thickness: float = 0.01,
    check_stability: bool = False,
    stability_max_drift: float = 0.005,
) -> None:
    logger.warning(
        "decompose_fast uses voxel-cluster convex hulls and does NOT capture "
        "concavities (handles, mug interiors). For non-convex objects with "
        "handles (teapot, mug, cup, kettle), prefer grasp.pipeline.decompose "
        "(CoACD-based) for better retargeting performance."
    )
    dataset_path = Path(dataset_dir)

    if embodiment_type == "right":
        hands = ["right"]
    elif embodiment_type == "left":
        hands = ["left"]
    elif embodiment_type == "bimanual":
        hands = ["right", "left"]
    else:
        raise ValueError(f"Invalid hand type: {embodiment_type}")

    processed_dir = (
        dataset_path
        / "processed"
        / dataset_name
        / "mano"
        / embodiment_type
        / task
        / str(data_id)
    )
    task_info_path = processed_dir.parent / "task_info.json"

    if not task_info_path.exists():
        logger.error(
            "Missing task_info at {}. Run dataset preprocessing first.",
            task_info_path,
        )
        return

    with task_info_path.open("r", encoding="utf-8") as file:
        task_info = json.load(file)

    # Load first-frame object qpos from the trajectory_keypoints.npz so the
    # support plate can be aligned to the world-frame floor.
    keypoints_path = processed_dir / "trajectory_keypoints.npz"
    first_obj_pose: dict[str, tuple[np.ndarray, np.ndarray] | None] = {
        "right": None,
        "left": None,
    }
    if keypoints_path.exists():
        kp = np.load(keypoints_path)
        for h in ("right", "left"):
            key = f"qpos_obj_{h}"
            if key in kp.files and len(kp[key]) > 0:
                pos = kp[key][0, :3]
                q_wxyz = kp[key][0, 3:]
                # skip placeholder (zero-pos identity-quat) and degenerate
                # (zero-norm quat) entries -- the latter is what bimanual
                # gigahand uses for the inactive hand.
                placeholder = np.allclose(pos, 0) and np.allclose(q_wxyz, [1, 0, 0, 0])
                degenerate = float(np.linalg.norm(q_wxyz)) < 1e-6
                if not (placeholder or degenerate):
                    q_xyzw = np.concatenate([q_wxyz[1:], q_wxyz[:1]])
                    R_obj = np.asarray(_R_from_quat(q_xyzw))
                    first_obj_pose[h] = (pos, R_obj)

    for hand in hands:
        mesh_dir_key = (
            "right_object_mesh_dir" if hand == "right" else "left_object_mesh_dir"
        )
        mesh_dir = task_info.get(mesh_dir_key)
        mesh_dir = f"{dataset_path}/{mesh_dir}"
        if not mesh_dir:
            logger.warning("No mesh_dir for {} hand; skipping.", hand)
            continue

        mesh_path = Path(mesh_dir)
        input_file = mesh_path / "visual.obj"
        output_dir = mesh_path / "convex"

        if not input_file.exists():
            logger.warning(
                "Input mesh {} does not exist. Skipping {} hand.", input_file, hand
            )
            continue

        mesh = trimesh.load(
            str(input_file), force="mesh", process=False, skip_materials=True
        )

        hulls = fast_voxel_convex_decomp_from_pointcloud(np.asarray(mesh.vertices))
        if not hulls:
            logger.warning("No convex parts generated for {}; skipping export.", hand)
            continue

        if add_flat_base:
            pose = first_obj_pose.get(hand)
            if pose is not None:
                obj_pos, R_obj = pose
                hull_verts = np.vstack([v for v, _ in hulls])
                v_world = (R_obj @ hull_verts.T).T + obj_pos
                obj_min_world_z = float(v_world[:, 2].min())
                # xy of the lowest contact point -- the support plate sits
                # directly under this, not under the object root (which can be
                # 20+ cm away from the actual contact surface for objects with
                # off-center geometry like the arctic box).
                near_floor_mask = v_world[:, 2] < obj_min_world_z + 1e-3
                low_xy = v_world[near_floor_mask, :2].mean(axis=0)
                hulls = flatten_base(
                    hulls,
                    thickness=plate_thickness,
                    R_world_local=R_obj,
                    obj_world_pos=obj_pos,
                    floor_z=obj_min_world_z,
                    plate_xy=low_xy,
                    plate_size=plate_size,
                )
                # Plate's bottom is flush with object's lowest world z; top is
                # plate_thickness above. Object's effective lowest z stays
                # unchanged (the plate lives inside the object's lower bbox).
                plate_top_z = obj_min_world_z + plate_thickness
                task_info[f"{hand}_plate_top_world_z"] = float(plate_top_z)
                task_info[f"{hand}_plate_bottom_world_z"] = float(obj_min_world_z)
                task_info[f"{hand}_obj_first_frame_xy"] = [
                    float(low_xy[0]),
                    float(low_xy[1]),
                ]
                task_info["plate_size"] = float(plate_size)
                task_info["plate_thickness"] = float(plate_thickness)
                logger.info(
                    "Added 3x3cm flat base for {} hand: plate bottom at "
                    "world z={:.4f}, top at z={:.4f} (object lowest z "
                    "unchanged).",
                    hand, obj_min_world_z, plate_top_z,
                )
            else:
                hulls = flatten_base(
                    hulls,
                    thickness=plate_thickness,
                    plate_size=plate_size,
                )
                logger.warning(
                    "No first-frame obj pose for {} hand; falling back to "
                    "local-frame plate (object may not rest stably on floor).",
                    hand,
                )
        # clear stale .obj files so a smaller decomposition doesn't leave
        # leftover hulls from a previous run (which generate_xml would still
        # load and over-collide).
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*.obj"):
            stale.unlink()

        for idx, (vertices, faces) in enumerate(hulls):
            mesh_part = trimesh.Trimesh(vertices, faces)
            part_path = output_dir / f"{idx}.obj"
            mesh_part.export(part_path)
            logger.info("Exported mesh part {} to {}", idx, part_path)

        convex_key = (
            "right_object_convex_dir" if hand == "right" else "left_object_convex_dir"
        )
        # get relative path to dataset_dir
        relative_path = os.path.relpath(output_dir, dataset_path)
        task_info[convex_key] = str(relative_path)

    with task_info_path.open("w", encoding="utf-8") as file:
        json.dump(task_info, file, indent=2)

    logger.info("Updated task_info with convex dirs at {}", task_info_path)

    if check_stability:
        for hand in hands:
            pose = first_obj_pose.get(hand)
            if pose is None:
                logger.warning(
                    "Skipping stability check for {} hand: no first-frame obj pose.",
                    hand,
                )
                continue
            convex_key = (
                "right_object_convex_dir"
                if hand == "right"
                else "left_object_convex_dir"
            )
            convex_dir = task_info.get(convex_key)
            if convex_dir is None:
                logger.warning(
                    "Skipping stability check for {} hand: no convex dir.", hand,
                )
                continue
            convex_path = dataset_path / convex_dir
            # World floor sits at the object's lowest world z at frame 0
            # (which the dataset processor offsets to 0). The plate is
            # body-fixed and sits inside the object's lower bbox; the floor
            # catches the object's actual mesh.
            plate_bottom_z = task_info.get(f"{hand}_plate_bottom_world_z")
            if plate_bottom_z is not None:
                floor_z = float(plate_bottom_z)
            else:
                obj_min_z = task_info.get("obj_first_frame_lowest_world_z")
                floor_z = float(obj_min_z) if obj_min_z is not None else 0.0
            ok, drift = _check_initial_stability(
                convex_path, pose[0], pose[1],
                floor_z=floor_z, max_drift=stability_max_drift,
            )
            if not ok:
                raise RuntimeError(
                    f"Stability check failed for {hand} object: it drifted "
                    f"{drift:.4f}m in 0.5s of physics (threshold "
                    f"{stability_max_drift}m). The support plate is likely "
                    "misaligned. Check obj_first_frame_pos in task_info.json."
                )
            logger.info(
                "Stability check passed for {} hand: drift={:.4f}m (limit {:.4f}m)",
                hand, drift, stability_max_drift,
            )


def _check_initial_stability(
    convex_dir: Path,
    obj_pos: np.ndarray,
    R_obj: np.ndarray,
    floor_z: float = 0.0,
    max_drift: float = 0.005,
    sim_dt: float = 0.005,
    sim_seconds: float = 0.5,
) -> tuple[bool, float]:
    """Drop-test the object at its first-frame pose on a floor at ``floor_z``.

    Builds a minimal mujoco scene with a floor + the object's convex hulls as
    a free body, sets the body to (obj_pos, R_obj), and steps physics. Returns
    (passed, max_drift_m).
    """
    import mujoco
    from scipy.spatial.transform import Rotation as R

    spec = mujoco.MjSpec()
    spec.option.timestep = sim_dt
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        pos=[0, 0, floor_z],
    )
    body = spec.worldbody.add_body(name="obj", pos=[0, 0, 0])
    body.add_freejoint(name="obj_free")
    convex_files = sorted(convex_dir.glob("*.obj"))
    for i, fp in enumerate(convex_files):
        spec.add_mesh(name=f"convex_{i}", file=str(fp))
        body.add_geom(
            name=f"convex_{i}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"convex_{i}",
            condim=3,
            contype=1,
            conaffinity=1,
            density=400,
        )
    model = spec.compile()
    data = mujoco.MjData(model)
    # Set free joint to first-frame world pose
    quat_xyzw = R.from_matrix(R_obj).as_quat()
    quat_wxyz = np.concatenate([quat_xyzw[3:], quat_xyzw[:3]])
    data.qpos[:3] = obj_pos
    data.qpos[3:7] = quat_wxyz
    mujoco.mj_forward(model, data)
    initial_pos = data.qpos[:3].copy()
    n_steps = int(sim_seconds / sim_dt)
    max_drift_seen = 0.0
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        drift = float(np.linalg.norm(data.qpos[:3] - initial_pos))
        if drift > max_drift_seen:
            max_drift_seen = drift
    return max_drift_seen <= max_drift, max_drift_seen


if __name__ == "__main__":
    tyro.cli(main)
