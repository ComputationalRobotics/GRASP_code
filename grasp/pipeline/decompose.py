# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Source: SPIDER, https://github.com/facebookresearch/spider, spider/preprocess/decompose.py
# at commit 44717007de41cbef7565dff7ff9f4453557a2d3d, adapted for GRASP.
# Licensed under CC BY-NC 4.0; see THIRD_PARTY_NOTICES.md.

try:
    import trimesh
except ModuleNotFoundError:
    print("trimesh is required. Please install with `pip install trimesh`")
    exit(1)

import json
import os

import coacd
import loguru
import numpy as np
import tyro
from scipy.spatial.transform import Rotation as _ScipyR

from grasp import ROOT
from grasp.pipeline._decompose_common import flatten_base


def _R_from_quat(q_xyzw: np.ndarray) -> np.ndarray:
    return _ScipyR.from_quat(q_xyzw).as_matrix()


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
):
    dataset_dir = os.path.abspath(dataset_dir)
    if embodiment_type == "right":
        hands = ["right"]
    elif embodiment_type == "left":
        hands = ["left"]
    elif embodiment_type == "bimanual":
        hands = ["right", "left"]
    else:
        raise ValueError(f"Invalid hand type: {embodiment_type}")

    # load task info produced during dataset preprocessing
    processed_dir = f"{dataset_dir}/processed/{dataset_name}/mano/{embodiment_type}/{task}/{data_id}"
    task_info_path = f"{processed_dir}/../task_info.json"
    if not os.path.exists(task_info_path):
        loguru.logger.error(
            f"Missing task_info at {task_info_path}. Run dataset preprocessing first."
        )
        return
    with open(task_info_path) as f:
        task_info = json.load(f)

    # Load first-frame object pose so a flat base can be aligned with the
    # object's actual lowest world-frame point.
    keypoints_path = f"{processed_dir}/trajectory_keypoints.npz"
    first_obj_pose: dict[str, tuple[np.ndarray, np.ndarray] | None] = {
        "right": None,
        "left": None,
    }
    if os.path.exists(keypoints_path):
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
        if hand == "right":
            mesh_dir = task_info.get("right_object_mesh_dir")
        else:
            mesh_dir = task_info.get("left_object_mesh_dir")
        mesh_dir = f"{dataset_dir}/{mesh_dir}"
        if mesh_dir is None:
            loguru.logger.warning(f"No mesh_dir for {hand} hand; skipping.")
            continue
        input_file = f"{mesh_dir}/visual.obj"
        output_dir = f"{mesh_dir}/convex"
        if not os.path.exists(input_file):
            loguru.logger.warning(
                f"Input mesh {input_file} does not exist. Skipping {hand} hand."
            )
            continue

        mesh = trimesh.load(
            input_file, force="mesh", process=False, skip_materials=True
        )
        coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        result = coacd.run_coacd(
            coacd_mesh,
            threshold=0.07,
            max_convex_hull=16,
            preprocess_mode="auto",
            preprocess_resolution=50,
            resolution=2000,
            mcts_nodes=50,
            mcts_iterations=200,
            mcts_max_depth=5,
            pca=False,
            merge=True,
            decimate=True,
            max_ch_vertex=32,
            extrude=True,
            extrude_margin=0.1,
            apx_mode="ch",
            seed=1,
        )

        hulls = [
            (np.asarray(vs), np.asarray(fs, dtype=int)) for (vs, fs) in result
        ]

        if add_flat_base:
            pose = first_obj_pose.get(hand)
            if pose is not None:
                obj_pos, R_obj = pose
                hull_verts = np.vstack([v for v, _ in hulls])
                v_world = (R_obj @ hull_verts.T).T + obj_pos
                obj_min_world_z = float(v_world[:, 2].min())
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
                plate_top_z = obj_min_world_z + plate_thickness
                task_info[f"{hand}_plate_top_world_z"] = float(plate_top_z)
                task_info[f"{hand}_plate_bottom_world_z"] = float(obj_min_world_z)
                task_info[f"{hand}_obj_first_frame_xy"] = [
                    float(low_xy[0]),
                    float(low_xy[1]),
                ]
                task_info["plate_size"] = float(plate_size)
                task_info["plate_thickness"] = float(plate_thickness)
                loguru.logger.info(
                    f"Added 3x3cm flat base for {hand} hand: plate bottom at "
                    f"world z={obj_min_world_z:.4f}, top at z={plate_top_z:.4f} "
                    "(object lowest z unchanged)."
                )
            else:
                hulls = flatten_base(
                    hulls,
                    thickness=plate_thickness,
                    plate_size=plate_size,
                )
                loguru.logger.warning(
                    f"No first-frame obj pose for {hand} hand; falling back to "
                    "local-frame plate (object may not rest stably on floor)."
                )

        # ensure output directory exists; clear any prior .obj files so a
        # smaller decomposition doesn't leave stale hulls from a previous run
        # (which generate_xml would still load and over-collide).
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            for stale in os.listdir(output_dir):
                if stale.endswith(".obj"):
                    os.remove(os.path.join(output_dir, stale))
        for i, (vs, fs) in enumerate(hulls):
            mesh_part = trimesh.Trimesh(vs, fs)
            part_filename = f"{output_dir}/{i}.obj"
            mesh_part.export(part_filename)
            loguru.logger.info(f"Exported mesh part {i} to {part_filename}")

        # persist decomposed path back to task_info for future reference
        key = "right_object_convex_dir" if hand == "right" else "left_object_convex_dir"
        relative_path = os.path.relpath(output_dir, dataset_dir)
        task_info[key] = str(relative_path)

    # save updated task_info
    with open(task_info_path, "w") as f:
        json.dump(task_info, f, indent=2)
    loguru.logger.info(f"Updated task_info with convex dirs at {task_info_path}")


if __name__ == "__main__":
    tyro.cli(main)
