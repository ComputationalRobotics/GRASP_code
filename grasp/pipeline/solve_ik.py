# Copyright (c) 2026 malik-group
# SPDX-License-Identifier: MIT
# Source: Do As I Do, https://github.com/malik-group/do-as-i-do, retargeting/retargeting/pipeline/solve_ik.py
# at commit 824591b808c342b20079c3b4198a8c2bdf88c74e, adapted for GRASP; see THIRD_PARTY_NOTICES.md.
"""Simplified mink-based IK: scene xml + hand keypoints + object trajectories -> qpos npz.

Frame 0 is solved in two phases (wrist targets first, then fingertips) from a
zero-qpos seed; later frames solve all targets together. Object qpos is set
directly rather than retargeted.
"""

import os

import loguru
import tyro
import mink
import mujoco
import numpy as np
from loop_rate_limiters import RateLimiter

from grasp.utils.mujoco_utils import get_viewer


def get_processed_data_dir(output_root_dir, dataset_name, robot_type, embodiment_type, task, data_id) -> str:
    return f"{output_root_dir}/{robot_type}/{embodiment_type}/{task}/{data_id}"


def main(
    output_root_dir: str = "data/processed/oakink",
    dataset_name: str = "oakink",
    robot_type: str = "xhand",
    embodiment_type: str = "bimanual",
    task: str = "",
    show_viewer: bool = False,
    save_video: bool = False,
    data_id: int = 0,
    start_idx: int = 0,
    end_idx: int = -1,
    sim_dt: float = 0.005,
    ref_dt: float = 0.02,
    wrist_pos_cost: float = 0.3,
    wrist_ori_cost: float = 3.0,
    finger_pos_cost: float = 10.0,
    posture_cost: float = 1e-2,
    wrist_init_steps: int = 200,
    finger_init_steps: int = 300,
    smoothing: bool = False,
    average_frame_size: int = 3,
    z_offset: float = 0.0,
    force: bool = False,
):
    output_root_dir = os.path.abspath(output_root_dir)

    out_path = os.path.join(
        get_processed_data_dir(output_root_dir, dataset_name, robot_type, embodiment_type, task, data_id),
        "trajectory_kinematic.npz",
    )
    if not force and os.path.exists(out_path):
        loguru.logger.info(f"Skipping solve_ik.py (output exists: {out_path})")
        return

    processed_dir_robot = get_processed_data_dir(
        output_root_dir=output_root_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    processed_dir_mano = get_processed_data_dir(
        output_root_dir=output_root_dir,
        dataset_name=dataset_name,
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(processed_dir_robot, exist_ok=True)
    # IK is purely kinematic (mink); the pedestal-free structural scene
    # (scene_ik.xml) is sufficient. The post-IK pedestal step
    # (``retargeting/pipeline/resolve_pedestal.py``) injects pedestals into the resolved
    # ``scene.xml`` afterwards.
    model_path = f"{processed_dir_robot}/scene_ik.xml"

    file_path = f"{processed_dir_mano}/trajectory_keypoints.npz"
    loaded_data = np.load(file_path)
    qpos_finger_right = loaded_data["qpos_finger_right"][start_idx:end_idx]
    qpos_finger_left = loaded_data["qpos_finger_left"][start_idx:end_idx]
    qpos_wrist_right = loaded_data["qpos_wrist_right"][start_idx:end_idx]
    qpos_wrist_left = loaded_data["qpos_wrist_left"][start_idx:end_idx]
    qpos_obj_right = loaded_data["qpos_obj_right"][start_idx:end_idx]
    qpos_obj_left = loaded_data["qpos_obj_left"][start_idx:end_idx]

    # Build reference array: (H, num_sites, 7) where 7 = [x, y, z, qw, qx, qy, qz]
    qpos_ref = np.concatenate(
        [
            qpos_wrist_right[:, None],
            qpos_finger_right,
            qpos_wrist_left[:, None],
            qpos_finger_left,
            qpos_obj_right[:, None],
            qpos_obj_left[:, None],
        ],
        axis=1,
    )
    qpos_ref[:, :, 2] += z_offset

    num_frames = qpos_finger_right.shape[0]

    # ref index layout: 0 right_palm, 1-5 right fingers, 6 left_palm,
    # 7-11 left fingers, 12 right_object, 13 left_object
    ref_idx = {}
    ref_idx["right_palm"] = 0
    for i, name in enumerate(
        ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    ):
        ref_idx[f"right_{name}"] = i + 1
    ref_idx["left_palm"] = 6
    for i, name in enumerate(
        ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    ):
        ref_idx[f"left_{name}"] = i + 7
    ref_idx["right_object"] = 12
    ref_idx["left_object"] = 13

    model = mujoco.MjModel.from_xml_path(model_path)

    if embodiment_type == "bimanual":
        nq_obj = 14
    elif embodiment_type in ["right", "left"]:
        nq_obj = 7
    else:
        nq_obj = 0

    configuration = mink.Configuration(model)
    data = configuration.data

    finger_names = ["thumb", "index", "middle", "ring", "pinky"]

    sides = []
    if embodiment_type in ["right", "bimanual"]:
        sides.append("right")
    if embodiment_type in ["left", "bimanual"]:
        sides.append("left")

    wrist_sites = [f"{s}_palm" for s in sides]
    finger_sites = [f"{s}_{f}_tip" for s in sides for f in finger_names]

    # cost priority: finger_pos > wrist_pos > wrist_ori
    wrist_tasks = [
        mink.FrameTask(
            frame_name=s,
            frame_type="site",
            position_cost=wrist_pos_cost,
            orientation_cost=wrist_ori_cost,
            lm_damping=1.0,
        )
        for s in wrist_sites
    ]

    finger_tasks = [
        mink.FrameTask(
            frame_name=s,
            frame_type="site",
            position_cost=finger_pos_cost,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        for s in finger_sites
    ]

    posture_task = mink.PostureTask(model, cost=posture_cost)
    posture_task.set_target(configuration.q.copy())

    tasks_wrist = [posture_task, *wrist_tasks]
    tasks_all = [posture_task, *wrist_tasks, *finger_tasks]

    solver = "daqp"
    n_substeps = max(1, int(round(ref_dt / sim_dt)))

    def set_object_qpos(t):
        if embodiment_type == "bimanual":
            data.qpos[-14:-7] = qpos_ref[t, ref_idx["right_object"]]
            data.qpos[-7:] = qpos_ref[t, ref_idx["left_object"]]
        elif embodiment_type == "right":
            data.qpos[-7:] = qpos_ref[t, ref_idx["right_object"]]
        elif embodiment_type == "left":
            data.qpos[-7:] = qpos_ref[t, ref_idx["left_object"]]

    def set_wrist_targets(t):
        for wrist_task, site_name in zip(wrist_tasks, wrist_sites, strict=True):
            pos = qpos_ref[t, ref_idx[site_name], :3]
            quat_wxyz = qpos_ref[t, ref_idx[site_name], 3:]
            wrist_task.set_target(mink.SE3(wxyz_xyz=np.concatenate([quat_wxyz, pos])))

    def set_finger_targets(t):
        for finger_task, site_name in zip(finger_tasks, finger_sites, strict=True):
            pos = qpos_ref[t, ref_idx[site_name], :3]
            finger_task.set_target(
                mink.SE3(wxyz_xyz=np.array([1.0, 0.0, 0.0, 0.0, *pos]))
            )

    # phase 1 settles wrist targets only; phase 2 adds fingertip targets
    def run_phase12(q_start):
        configuration.data.qpos[:] = q_start
        set_object_qpos(0)
        configuration.update()
        set_wrist_targets(0)
        posture_task.set_target(q_start)
        for _ in range(wrist_init_steps):
            configuration.integrate_inplace(
                mink.solve_ik(configuration, tasks_wrist, sim_dt, solver, damping=1e-5), sim_dt)
            set_object_qpos(0)
        configuration.update()
        set_finger_targets(0)
        for _ in range(finger_init_steps):
            configuration.integrate_inplace(
                mink.solve_ik(configuration, tasks_all, sim_dt, solver, damping=1e-5), sim_dt)
            set_object_qpos(0)
        mujoco.mj_forward(model, configuration.data)
        cost = sum(
            w ** 2 * float(np.sum(
                (configuration.data.site(s).xpos - qpos_ref[0, ref_idx[s], :3]) ** 2))
            for w, names in [(wrist_pos_cost, wrist_sites), (finger_pos_cost, finger_sites)]
            for s in names
        )
        return configuration.q.copy(), cost

    best_q, best_cost = run_phase12(np.zeros(model.nq))
    loguru.logger.info(f"Frame-0 init: cost={best_cost:.6f}")
    configuration.data.qpos[:] = best_q
    configuration.update()
    set_object_qpos(0)

    loguru.logger.info(f"Running IK for {num_frames} frames...")
    qpos_list = []
    images = []

    if save_video:
        import imageio

        model.vis.global_.offwidth = 720
        model.vis.global_.offheight = 480
        renderer = mujoco.Renderer(model, height=480, width=720)

    run_viewer = get_viewer(show_viewer, model, data)
    rate_limiter = RateLimiter(1 / ref_dt)

    with run_viewer() as gui:
        for t in range(num_frames):
            set_wrist_targets(t)
            set_finger_targets(t)
            set_object_qpos(t)
            configuration.update()

            for _ in range(n_substeps):
                vel = mink.solve_ik(
                    configuration, tasks_all, sim_dt, solver, damping=1e-5
                )
                configuration.integrate_inplace(vel, sim_dt)
                set_object_qpos(t)

            qpos_list.append(configuration.q.copy())

            if save_video:
                mujoco.mj_forward(model, data)
                renderer.update_scene(data=data, camera="front")
                images.append(renderer.render())

            if show_viewer:
                mujoco.mj_forward(model, data)
                gui.sync()
                rate_limiter.sleep()

    qpos_list = np.array(qpos_list)

    if smoothing:
        def moving_average_filter(signal_data, window_size=5):
            return np.convolve(
                signal_data, np.ones(window_size) / window_size, mode="valid"
            )

        filtered = np.zeros(
            (qpos_list.shape[0] - average_frame_size + 1, qpos_list.shape[1])
        )
        for i in range(qpos_list.shape[1]):
            filtered[:, i] = moving_average_filter(qpos_list[:, i], average_frame_size)
        qpos_list = filtered
    else:
        loguru.logger.info("Skipping IK smoothing.")

    # qvel via finite differences; first frame starts from rest
    n_filtered = qpos_list.shape[0]
    qvel_list = np.zeros((n_filtered, model.nv))
    for i in range(1, n_filtered):
        mujoco.mj_differentiatePos(
            model, qvel_list[i], ref_dt, qpos_list[i - 1], qpos_list[i]
        )
    assert qpos_list.shape[0] == qvel_list.shape[0]

    # forward rollout for validation
    mj_data_rollout = mujoco.MjData(model)
    n_rollout_substeps = max(1, int(round(ref_dt / sim_dt)))
    model.opt.timestep = ref_dt / n_rollout_substeps
    mj_data_rollout.qpos[:] = qpos_list[0]
    mj_data_rollout.qvel[:] = qvel_list[0]
    mj_data_rollout.ctrl[:] = qpos_list[0][: model.nq - nq_obj]
    for _ in range(n_rollout_substeps):
        mujoco.mj_step(model, mj_data_rollout)
    n_final = qpos_list.shape[0]
    qpos_rollout = np.zeros((n_final, model.nq))
    qpos_rollout[0] = qpos_list[0]
    for i in range(1, n_final):
        mj_data_rollout.ctrl[:] = qpos_list[i][: model.nq - nq_obj]
        for _ in range(n_rollout_substeps):
            mujoco.mj_step(model, mj_data_rollout)
        qpos_rollout[i] = mj_data_rollout.qpos.copy()

    file_dir = processed_dir_robot
    if save_video:
        import imageio

        video_path = f"{file_dir}/visualization_ik.mp4"
        imageio.mimsave(video_path, images, fps=int(1 / ref_dt))
        loguru.logger.info(f"Saved video to {video_path}")

    out_npz = f"{file_dir}/trajectory_kinematic.npz"
    np.savez(out_npz, qpos=qpos_list, qvel=qvel_list, frequency=1 / ref_dt)
    loguru.logger.info(f"Saved {out_npz}")

    out_npz = f"{file_dir}/trajectory_ikrollout.npz"
    np.savez(out_npz, qpos=qpos_rollout)
    loguru.logger.info(f"Saved {out_npz}")


if __name__ == "__main__":
    tyro.cli(main)
