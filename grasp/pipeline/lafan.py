"""LAFAN1 clips retargeted to the Unitree G1 -> GRASP humanoid tasks.

Downloads one clip of the G1 retargeting of LAFAN1 (https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset,
``g1/<task>.csv``: 30 Hz root position, root quaternion xyzw and 29 joint positions), keeps the first ``--frames``
frames (331 = 11 s: 10 s of execution plus the planning lookahead), and converts it with ``grasp.pipeline.gmr`` into
``scene.xml`` and ``0/trajectory_kinematic.npz`` under ``<dataset_dir>/processed/lafan/unitree_g1/humanoid/<task>``.

    python -m grasp.pipeline.lafan --task dance1_subject1 --dataset-dir data/lafan --robot-asset-dir assets/unitree_g1
"""

from __future__ import annotations

import argparse
import pickle
import urllib.request
from pathlib import Path

import numpy as np

from grasp.pipeline.gmr import main as convert

URL = "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1/{task}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="clip name, e.g. dance1_subject1")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--robot-asset-dir", type=Path, required=True, help="Unitree G1 model directory (scene.xml)")
    parser.add_argument("--frames", type=int, default=331)
    parser.add_argument("--csv", type=Path, default=None, help="use a local copy of the clip instead of downloading")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve()

    csv = args.csv or dataset_dir / "raw/lafan" / f"{args.task}.csv"
    if not csv.exists():
        csv.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(task=args.task), csv)
    raw = np.loadtxt(csv, delimiter=",")
    assert raw.ndim == 2 and raw.shape[1] == 36 and raw.shape[0] >= args.frames, (args.task, raw.shape)
    assert np.isfinite(raw).all() and np.max(np.abs(np.linalg.norm(raw[:, 3:7], axis=1) - 1)) < 1e-3
    selected = raw[: args.frames].copy()

    folder = dataset_dir / "processed/lafan/unitree_g1/humanoid" / args.task
    (folder / "0").mkdir(parents=True, exist_ok=True)
    with (folder / "0/trajectory_gmr.pkl").open("wb") as f:
        pickle.dump(dict(fps=30, root_pos=selected[:, :3], root_rot=selected[:, 3:7], dof_pos=selected[:, 7:]), f)
    convert(robot_asset_dir=str(args.robot_asset_dir.resolve()), dataset_dir=str(dataset_dir), dataset_name="lafan",
            robot_type="unitree_g1", embodiment_type="humanoid", task=args.task, data_id=0, show_viewer=False,
            save_video=False, overwrite=False, enable_rate_limiter=False, start_frame=0, end_frame=len(selected),
            contact_detection_mode="one")

    q = np.load(folder / "0/trajectory_kinematic.npz")["qpos"]
    np.testing.assert_allclose(q[:, :3], selected[:, :3])
    np.testing.assert_allclose(q[:, 3:7], selected[:, [6, 3, 4, 5]])
    np.testing.assert_allclose(q[:, 7:], selected[:, 7:])
    print(f"[{args.task}] scene {folder / 'scene.xml'}; kinematic qpos {q.shape}", flush=True)


if __name__ == "__main__":
    main()
