"""Turn one raw bimanual OakInk2 clip into a GRASP task: MANO keypoints -> object convex decomposition -> MuJoCo
scene with the robot -> kinematic reference by IK.

Input: ``<dataset_dir>/raw/oakink/<task>_bimanual.pkl`` from ``grasp.pipeline.oakink2_maniptrans``, and the robot model
under ``<dataset_dir>/processed/oakink/assets/robots/<robot_type>``.  Output: ``scene.xml`` and
``0/trajectory_kinematic.npz`` under ``<dataset_dir>/processed/oakink/<robot_type>/bimanual/<task>``.

    python -m grasp.pipeline.prepare_oakink2 --task oakbi_000_cut --dataset-dir data/oakink2 --robot-type sharpa_bimanual
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from grasp.pipeline import decompose_fast, generate_xml, ik_fast, oakink


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--robot-type", default="sharpa_bimanual")
    args = parser.parse_args()
    dataset_dir = str(args.dataset_dir.resolve())
    common = dict(dataset_dir=dataset_dir, dataset_name="oakink", embodiment_type="bimanual", task=args.task)

    oakink.main(dataset_dir=dataset_dir, embodiment_type="bimanual", task=args.task, show_viewer=False, save_video=False)
    decompose_fast.main(**common)
    generate_xml.main(**common, robot_type=args.robot_type, show_viewer=False)
    ik_fast.main(**common, robot_type=args.robot_type, show_viewer=False, save_video=False)

    out = Path(dataset_dir) / "processed/oakink" / args.robot_type / "bimanual" / args.task
    qpos = np.load(out / "0/trajectory_kinematic.npz")["qpos"]
    if not np.isfinite(qpos).all():
        raise RuntimeError(f"non-finite kinematic reference for {args.task}")
    print(f"[{args.task}] scene {out / 'scene.xml'}; kinematic qpos {qpos.shape}", flush=True)


if __name__ == "__main__":
    main()
