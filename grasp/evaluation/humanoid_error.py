"""Humanoid tracking errors (paper humanoid table): joint, pelvis position and pelvis orientation errors, and for
humanoid-object tasks the object position and orientation errors.

Executed row ``i`` is the state after simulation step ``i + 1`` and is compared with the kinematic reference linearly
interpolated at ``t = (i + 1) * sim_dt`` (quaternions renormalised).  Each error is the per-step norm averaged over the
executed trajectory: joint positions (L2 over the actuated joints), pelvis position (m), pelvis orientation and object
orientation (rotation-vector norm, rad), object position (m).

    python -m grasp.evaluation.humanoid_error runs/lafan_dance1/lafan_unitree_g1_humanoid_dance1_subject1_0_seed0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from grasp.utils.math import quat_sub


def _orientation_error(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    diff = quat_sub(torch.from_numpy(np.ascontiguousarray(qa)), torch.from_numpy(np.ascontiguousarray(qb)))
    return np.linalg.norm(diff.numpy(), axis=1)


def interpolate_reference(reference_qpos: np.ndarray, ref_dt: float, times: np.ndarray, object_quat: bool) -> np.ndarray:
    frame_times = np.arange(len(reference_qpos)) * ref_dt
    ref = np.stack([np.interp(times, frame_times, reference_qpos[:, j]) for j in range(reference_qpos.shape[1])], 1)
    ref[:, 3:7] /= np.linalg.norm(ref[:, 3:7], axis=1, keepdims=True)
    if object_quat:
        ref[:, -4:] /= np.linalg.norm(ref[:, -4:], axis=1, keepdims=True)
    return ref


def humanoid_errors(executed_qpos: np.ndarray, reference_qpos: np.ndarray, ref_dt: float, sim_dt: float,
                    with_object: bool = False) -> dict:
    """Errors of an executed humanoid trajectory; ``with_object``: the last 7 qpos entries are a free object."""
    q = np.asarray(executed_qpos, dtype=np.float64)
    ref = interpolate_reference(np.asarray(reference_qpos, dtype=np.float64), ref_dt,
                                np.arange(1, len(q) + 1) * sim_dt, with_object)
    joints = slice(7, -7) if with_object else slice(7, None)
    pos = np.linalg.norm(q[:, :3] - ref[:, :3], axis=1)
    out = dict(joint=float(np.linalg.norm(q[:, joints] - ref[:, joints], axis=1).mean()),
               pelvis_pos_m=float(pos.mean()),
               pelvis_ori_rad=float(_orientation_error(q[:, 3:7], ref[:, 3:7]).mean()),
               pelvis_pos_max_m=float(pos.max()),
               seconds=len(q) * sim_dt)
    if with_object:
        out.update(object_pos_m=float(np.linalg.norm(q[:, -7:-4] - ref[:, -7:-4], axis=1).mean()),
                   object_ori_rad=float(_orientation_error(q[:, -4:], ref[:, -4:]).mean()))
    return out


def evaluate_run(result_json: Path) -> dict:
    """Errors of a ``grasp.mpc.run`` result, with the reference, ref_dt and sim_dt the run itself resolved."""
    from grasp.config import load_config

    result_json = Path(result_json)
    args = json.loads(result_json.read_text())["args"]
    config = load_config(args["config"], args["override"], horizon=args["horizon"], num_samples=1, seed=args["seed"])
    executed = np.load(result_json.with_suffix(".npz"))["executed_qpos"]
    reference = np.load(config.data_path)["qpos"]
    return humanoid_errors(executed, reference, float(config.ref_dt), float(config.sim_dt),
                           with_object=config.embodiment_type == "humanoid_object")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+", help="result .json files written by grasp.mpc.run")
    args = parser.parse_args()
    rows = {}
    for path in args.results:
        rows[str(path)] = evaluate_run(path)
        print(json.dumps(dict(result=str(path), **rows[str(path)])), flush=True)
    keys = [k for k in next(iter(rows.values())) if k != "seconds"]
    mean = {k: float(np.mean([r[k] for r in rows.values() if k in r])) for k in keys}
    print(json.dumps(dict(runs=len(rows), mean=mean)), flush=True)


if __name__ == "__main__":
    main()
