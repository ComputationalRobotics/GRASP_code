"""AMASS clips from SPIDER's processed G1 data -> GRASP humanoid tasks.

The source is SPIDER's released ``processed/amass/unitree_g1/humanoid/<task>`` data.  Its joint trajectory is kept
unchanged; scene, contact sites and contact targets are regenerated with ``grasp.pipeline.gmr`` on the given G1 model.
The converter starts from rest (``qvel[0] = 0``); ``--source-initial-velocity`` keeps the source's ``qvel[0]`` instead,
as the paper did for cartwheels, fight_and_sports, run_to_crouch and walk_to_leap_to_walk.

    python -m grasp.pipeline.amass --task martial_arts --source-dir spider_data/processed/amass \\
        --dataset-dir data/amass --robot-asset-dir assets/unitree_g1
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from grasp.pipeline.gmr import main as convert


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-id", type=int, default=0)
    parser.add_argument("--source-dir", type=Path, required=True, help="SPIDER's processed/amass directory")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--robot-asset-dir", type=Path, required=True, help="Unitree G1 model directory")
    parser.add_argument("--source-initial-velocity", action="store_true", help="keep the source's qvel[0]")
    args = parser.parse_args()
    src = args.source_dir.resolve() / "unitree_g1/humanoid" / args.task
    dataset_dir = args.dataset_dir.resolve()
    dst = dataset_dir / "processed/amass/unitree_g1/humanoid" / args.task
    (dst / str(args.data_id)).mkdir(parents=True, exist_ok=True)

    source = np.load(src / str(args.data_id) / "trajectory_kinematic.npz")
    q0 = source["qpos"]
    ref_dt = float(json.loads((src / "task_info.json").read_text())["ref_dt"])
    with (dst / str(args.data_id) / "trajectory_gmr.pkl").open("wb") as f:
        pickle.dump(dict(fps=1.0 / ref_dt, root_pos=q0[:, :3], root_rot=q0[:, [4, 5, 6, 3]], dof_pos=q0[:, 7:]), f)
    convert(robot_asset_dir=str(args.robot_asset_dir.resolve()), dataset_dir=str(dataset_dir), dataset_name="amass",
            robot_type="unitree_g1", embodiment_type="humanoid", task=args.task, data_id=args.data_id,
            show_viewer=False, save_video=False, overwrite=False, enable_rate_limiter=False, start_frame=0,
            end_frame=len(q0), contact_detection_mode="one")
    out = dst / str(args.data_id) / "trajectory_kinematic.npz"
    z = dict(np.load(out))
    assert np.allclose(z["qpos"], q0, atol=1e-6), (args.task, np.abs(z["qpos"] - q0).max())
    if args.source_initial_velocity:
        z["qvel"][0] = source["qvel"][0]
        np.savez(out, **z)
    print(f"[{args.task}/{args.data_id}] {dst}", flush=True)


if __name__ == "__main__":
    main()
