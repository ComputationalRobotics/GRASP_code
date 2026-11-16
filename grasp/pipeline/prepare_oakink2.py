"""Turn one raw bimanual OakInk2 clip into a GRASP task: MANO keypoints -> object collision hulls (thicken, CoACD,
dilate; support plate) -> MuJoCo scene with the robot -> kinematic reference by IK -> contact gates.

Input: ``<dataset_dir>/raw/oakink/<task>_bimanual.pkl`` from ``grasp.pipeline.oakink2_maniptrans``, the robot model
under ``<dataset_dir>/processed/oakink/assets/robots/<robot_type>``, the OakInk2 release (``anno_preview/``) and the
MANO model files.  Output: ``scene.xml``, ``0/trajectory_kinematic.npz`` and ``0/contact_gates.npz`` under
``<dataset_dir>/processed/oakink/<robot_type>/bimanual/<task>``.

    python -m grasp.pipeline.prepare_oakink2 --task oakbi_000_cut --dataset-dir data/oakink2 --robot-type sharpa_bimanual \\
        --oakink2-root /path/to/OakInk-v2 --mano-dir /path/to/mano_v1_2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from grasp.pipeline import decompose_fast, generate_xml, ik_fast, oakink, oakink2_contact_gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--robot-type", default="sharpa_bimanual")
    parser.add_argument("--oakink2-root", type=Path, required=True, help="OakInk2 release directory containing anno_preview/")
    parser.add_argument("--mano-dir", type=Path, required=True, help="MANO directory containing models/MANO_{RIGHT,LEFT}.pkl")
    args = parser.parse_args()
    dataset_dir = str(args.dataset_dir.resolve())
    common = dict(dataset_dir=dataset_dir, dataset_name="oakink", embodiment_type="bimanual", task=args.task)

    oakink.main(dataset_dir=dataset_dir, embodiment_type="bimanual", task=args.task, show_viewer=False, save_video=False)
    decompose_fast.main(**common, method="coacd")
    generate_xml.main(**common, robot_type=args.robot_type, show_viewer=False)
    ik_fast.main(**common, robot_type=args.robot_type, show_viewer=False, save_video=False)

    out = Path(dataset_dir) / "processed/oakink" / args.robot_type / "bimanual" / args.task
    qpos = np.load(out / "0/trajectory_kinematic.npz")["qpos"]
    if not np.isfinite(qpos).all():
        raise RuntimeError(f"non-finite kinematic reference for {args.task}")
    oakink2_contact_gates.main(args.task, dataset_dir, str(args.oakink2_root.resolve()), str(args.mano_dir.resolve()),
                               robot_type=args.robot_type)
    print(f"[{args.task}] scene {out / 'scene.xml'}; kinematic qpos {qpos.shape}", flush=True)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only, as grasp.pipeline.oakink
    main()
