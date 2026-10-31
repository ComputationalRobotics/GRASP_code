"""Surface ADD-AUC10 of object tracking (paper HOT3D table).

512 area-uniform points on the object's visual mesh (sampled with a seed derived from the task name) are moved by the
executed and the reference object pose; ADD is their mean distance.  AUC integrates the fraction of evaluated frames
with ADD below a threshold over 100 thresholds in [0, 0.1] m (trapezoidal, in percent).  Frames are evaluated at
12.5 Hz (every 0.08 s) and 50 Hz; executed row ``s`` is the state after step ``s + 1`` and is compared with the
reference the optimizer tracked at that step; the terminal padding beyond the reference is excluded.

    python -m grasp.evaluation.add_auc runs/hot3d_001849/hot3d_sharpa_bimanual_hot3d_clip-001849_obj28_0_seed0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def auc(distance) -> float:
    thresholds = np.linspace(0.0, 0.1, 100)
    fraction = (np.asarray(distance)[:, None] <= thresholds).mean(axis=0)
    return float(np.trapezoid(fraction, thresholds / 0.1) * 100)


def surface_points(mesh_path: Path, task: str) -> np.ndarray:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    triangles, areas = np.asarray(mesh.triangles), np.asarray(mesh.area_faces)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(task.encode()).digest()[:8], "little"))
    faces = rng.choice(len(areas), 512, p=areas / areas.sum())
    uv = rng.random((512, 2))
    flip = uv.sum(axis=1) > 1
    uv[flip] = 1 - uv[flip]
    tri = triangles[faces]
    return tri[:, 0] + uv[:, :1] * (tri[:, 1] - tri[:, 0]) + uv[:, 1:] * (tri[:, 2] - tri[:, 0])


def object_reference(kinematic_qpos: np.ndarray, steps_per_frame: int) -> np.ndarray:
    """Object reference at every simulation step, as the optimizer interpolates it (float32, align-corners)."""
    source = kinematic_qpos.astype(np.float32)
    steps = (len(source) - 1) * steps_per_frame
    grid = np.arange(1, steps + 1, dtype=np.float32) * np.float32((len(source) - 1) / (len(source) * steps_per_frame - 1))
    lo = grid.astype(int)
    hi = np.minimum(lo + 1, len(source) - 1)
    weight = (grid - lo).astype(np.float32)[:, None]
    # float32 interpolation with fused multiply-add rounding, as torch.nn.functional.interpolate
    return (((np.float32(1) - weight).astype(np.float64) * source[lo])
            + (weight * source[hi]).astype(np.float64)).astype(np.float32).astype(np.float64)


def add_auc_scores(executed_qpos: np.ndarray, kinematic_qpos: np.ndarray, embodiment: str, steps_per_frame: int,
                   points: np.ndarray) -> dict:
    """ADD-AUC10 at 12.5 Hz and 50 Hz; the object is the right-hand object of a bimanual scene."""
    block = slice(-14, -7) if embodiment == "bimanual" else slice(-7, None)
    reference = object_reference(kinematic_qpos[:, block], steps_per_frame)
    steps = len(reference)
    achieved = np.asarray(executed_qpos)[:steps, block].astype(np.float64)
    assert len(achieved) == steps, (len(achieved), steps)
    finite = np.isfinite(achieved).all(axis=1) & (np.linalg.norm(achieved[:, 3:], axis=1) > 0)
    scores = {}
    for hz, stride in [(12.5, 4 * steps_per_frame), (50, steps_per_frame)]:
        indices = np.arange(stride - 1, steps, stride)
        mask = finite[indices]
        ids = indices[mask]
        distance = np.full(len(indices), np.inf)
        ra = Rotation.from_quat(achieved[ids][:, [4, 5, 6, 3]]).as_matrix()
        rb = Rotation.from_quat(reference[ids][:, [4, 5, 6, 3]]).as_matrix()
        difference = np.einsum("tij,pj->tpi", ra - rb, points)
        difference += (achieved[ids, :3] - reference[ids, :3])[:, None, :]
        distance[mask] = np.linalg.norm(difference, axis=-1).mean(axis=1)
        scores[str(hz)] = dict(add_auc_percent=auc(distance), evaluated_frames=len(indices),
                               invalid_frames=int((~mask).sum()),
                               mean_add_m=float(distance.mean()) if mask.all() else None)
    return scores


def evaluate_run(result_json: Path) -> dict:
    """ADD-AUC of a ``grasp.mpc.run`` HOT3D result (object mesh ``assets/objects/obj_<id>/visual.obj``)."""
    from grasp.config import load_config

    result_json = Path(result_json)
    args = json.loads(result_json.read_text())["args"]
    config = load_config(args["config"], args["override"], horizon=args["horizon"], num_samples=1, seed=args["seed"])
    object_id = int(config.task.rsplit("obj", 1)[1])
    mesh = Path(config.dataset_dir) / "processed" / config.dataset_name / "assets/objects" / f"obj_{object_id:06d}/visual.obj"
    executed = np.load(result_json.with_suffix(".npz"))["executed_qpos"]
    kinematic = np.load(config.data_path)["qpos"]
    steps_per_frame = int(round(config.ref_dt / config.sim_dt))
    scores = add_auc_scores(executed, kinematic, config.embodiment_type, steps_per_frame,
                            surface_points(mesh, config.task))
    return dict(task=config.task, embodiment=config.embodiment_type, scores=scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+", help="result .json files written by grasp.mpc.run")
    args = parser.parse_args()
    rows = [evaluate_run(path) for path in args.results]
    for row in rows:
        print(json.dumps(row), flush=True)
    print(json.dumps(dict(runs=len(rows), **{hz: float(np.mean([r["scores"][hz]["add_auc_percent"] for r in rows]))
                                              for hz in ("12.5", "50")})), flush=True)


if __name__ == "__main__":
    main()
