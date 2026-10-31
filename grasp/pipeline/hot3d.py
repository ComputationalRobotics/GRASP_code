"""HOT3D clips -> GRASP tasks for the Sharpa hand.

``export`` downloads one HOT3D training clip's annotations (https://huggingface.co/datasets/bop-benchmark/hot3d),
evaluates MANO for every observed hand, rotates the device world to Z-up, and writes one task per object a hand
interacts with (at least 5 frames within 3 cm of the object surface): wrist, fingertip and object trajectories
resampled to 50 Hz, with the object mesh.  Hand participation is estimated from proximity; it is not a HOT3D label.

``prepare`` turns each exported task into a scene (CoACD convex decomposition, MuJoCo scene with the Sharpa hand),
a kinematic reference (Do As I Do IK, no smoothing) and, for objects resting away from the floor at an endpoint, a
static support surface under them.

    python -m grasp.pipeline.hot3d export --split train_aria --clip clip-001849 --dataset-dir data/hot3d \\
        --mano-dir ~/mano_v1_2/models
    python -m grasp.pipeline.hot3d prepare --split train_aria --clip clip-001849 --dataset-dir data/hot3d
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from grasp import ROOT

HF = "https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/main/"
OBJECT_MODEL_FILES = Path(ROOT) / "assets/hot3d_object_model_files.json"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _download(relative: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(HF + relative, timeout=180) as response, temp.open("wb") as out:
        shutil.copyfileobj(response, out)
    temp.replace(path)


def read_clip(dataset_dir: Path, split: str, clip: str) -> dict:
    """Hand, object and frame annotations of one clip (image streams are not kept)."""
    folder = dataset_dir / "raw" / split / clip
    ready = folder / "annotations.json"
    if ready.exists():
        return json.loads(ready.read_text())
    archive = folder.with_suffix(".tar")
    _download(f"{split}/{clip}.tar", archive)
    result = {}
    with tarfile.open(archive) as tar:
        for member in tar:
            name = Path(member.name).name
            if member.isfile() and (name.endswith((".hands.json", ".objects.json", ".info.json"))
                                    or name == "__hand_shapes.json__"):
                result[name] = json.load(tar.extractfile(member))
    assert len([n for n in result if n.endswith(".hands.json")]) == 150
    _write(ready, result)
    archive.unlink()
    return result


def export(dataset_dir: Path, split: str, clip: str, mano_dir: Path) -> list[dict]:
    import numpy as np
    import torch
    import trimesh
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation, Slerp

    from grasp.pipeline.hot3d_coordinates import VERSION as FRAME_VERSION, points_to_sim, poses_to_sim, world_rotation
    from grasp.pipeline.hot3d_mano import MANOHandModel

    torch.set_num_threads(1)
    proc = dataset_dir / "processed/hot3d"
    row = {"split": split, "clip": clip}
    raw = read_clip(dataset_dir, split, clip)
    frames = sorted(n.split(".")[0] for n in raw if n.endswith(".hands.json"))
    times = np.array([raw[f + ".info.json"]["ref_timestamp_ns"] for f in frames], dtype=np.int64)
    times = (times - times[0]) * 1e-9
    assert np.all(np.diff(times) > 0)
    target_times = np.arange(int(np.floor(times[-1] / .02)) + 1) * .02
    mano = MANOHandModel(str(mano_dir))
    beta = torch.tensor(raw["__hand_shapes.json__"]["mano"]).float().repeat(len(frames), 1)
    hands = {}
    # smplx MANO joints plus fingertips -> OpenPose ordering
    order = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    for side in ["right", "left"]:
        entries = [raw[f + ".hands.json"].get(side, {}).get("mano_pose") for f in frames]
        if not all(e is not None for e in entries):
            # never interpolate an unobserved hand or silently shorten a clip
            continue
        theta = torch.tensor([e["thetas"] for e in entries]).float()
        wrist = torch.tensor([e["wrist_xform"] for e in entries]).float()
        with torch.no_grad():
            verts, joints = mano(beta, theta, wrist, torch.full((len(frames),), side == "right"))
        hands[side] = (points_to_sim(verts.numpy(), split), points_to_sim(joints.numpy()[:, order], split))

    def interp_pose(q):
        p = np.column_stack([np.interp(target_times, times, q[:, j]) for j in range(3)])
        quat = Slerp(times, Rotation.from_quat(q[:, [4, 5, 6, 3]]))(target_times).as_quat()[:, [3, 0, 1, 2]]
        return np.column_stack([p, quat])

    model_names = json.loads(OBJECT_MODEL_FILES.read_text())
    task_rows, audit = [], []
    object_ids = sorted(set().union(*(raw[f + ".objects.json"].keys() for f in frames)), key=int)
    for oid in object_ids:
        entries = [raw[f + ".objects.json"].get(oid, []) for f in frames]
        if not all(len(e) == 1 for e in entries):
            audit.append({"object_id": oid, "state": "incomplete_object_pose"})
            continue
        entries = [e[0] for e in entries]
        uid = entries[0]["object_uid"]
        mesh_name = model_names.get(str(uid), model_names.get(f"obj_{int(oid):06d}"))
        if mesh_name is None:
            raise ValueError(f"missing object model mapping: {oid}, {uid}")
        glb = dataset_dir / "raw/object_models_eval" / Path(mesh_name).name
        _download(mesh_name, glb)
        mesh = trimesh.load(glb, force="mesh", process=False, skip_materials=True)
        vertices = np.asarray(mesh.vertices)
        poses = np.array([e["T_world_from_object"]["translation_xyz"] + e["T_world_from_object"]["quaternion_wxyz"]
                          for e in entries])
        poses = poses_to_sim(poses, split)
        matrices = Rotation.from_quat(poses[:, [4, 5, 6, 3]]).as_matrix()
        tree = cKDTree(vertices)
        participation = {}
        for side, (hand_vertices, _) in hands.items():
            local = np.einsum("tji,tkj->tki", matrices, hand_vertices - poses[:, None, :3])
            distance = tree.query(local.reshape(-1, 3), workers=1)[0].reshape(len(frames), -1).min(axis=1)
            participation[side] = {"frames_within_3cm": int(np.sum(distance < .03)),
                                   "min_distance_m": float(distance.min())}
        active = [s for s in hands if participation[s]["frames_within_3cm"] >= 5]
        record = {"object_id": oid, "object_uid": uid, "participation": participation, "active_hands": active}
        if not active:
            record["state"] = "no_hand_interaction_detected"
            audit.append(record)
            continue
        embodiment = "bimanual" if len(active) == 2 else active[0]
        task = f"hot3d_{clip}_obj{int(oid):02d}"
        target_side = "right" if embodiment == "bimanual" else embodiment
        # translate in the Z-up frame so the object's lowest point over the clip is at the floor
        world_vertices = np.einsum("tij,vj->tvi", matrices, vertices) + poses[:, None, :3]
        offset = np.array([poses[0, 0], poses[0, 1], world_vertices[:, :, 2].min()])
        arrays = {}
        for side in ["right", "left"]:
            w = np.zeros((len(target_times), 7)); w[:, 3] = 1
            f = np.zeros((len(target_times), 5, 7)); f[:, :, 3] = 1
            obj = w.copy()
            if side in active:
                j = hands[side][1]
                z = j[:, 9] - j[:, 0]
                z /= np.linalg.norm(z, axis=1, keepdims=True)
                y_aux = j[:, 5] - j[:, 13]
                x = np.cross(y_aux, z); x /= np.linalg.norm(x, axis=1, keepdims=True)
                y = np.cross(z, x)
                if side == "left":
                    x, y = -x, -y
                quat = Rotation.from_matrix(np.stack([x, y, z], axis=-1)).as_quat()[:, [3, 0, 1, 2]]
                w = interp_pose(np.column_stack([j[:, 9] - offset, quat]))
                tips = j[:, [4, 8, 12, 16, 20]] - offset
                for k in range(5):
                    for axis in range(3):
                        f[:, k, axis] = np.interp(target_times, times, tips[:, k, axis])
            if side == target_side:
                q = poses.copy(); q[:, :3] -= offset
                obj = interp_pose(q)
            arrays.update({f"qpos_wrist_{side}": w, f"qpos_finger_{side}": f, f"qpos_obj_{side}": obj})
        mesh_dir = proc / "assets/objects" / f"obj_{int(oid):06d}"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        if not (mesh_dir / "visual.obj").exists():
            mesh.export(mesh_dir / "visual.obj")
        out = proc / "mano" / embodiment / task
        (out / "0").mkdir(parents=True, exist_ok=True)
        np.savez(out / "0/trajectory_keypoints.npz", **arrays, timestamps=target_times)
        info = dict(task=task, dataset_name="hot3d", dataset_dir=str(dataset_dir), robot_type="mano",
                    embodiment_type=embodiment, data_id=0, ref_dt=.02,
                    right_object_mesh_dir=None, left_object_mesh_dir=None,
                    right_object_convex_dir=None, left_object_convex_dir=None,
                    source_clip=row, source_frames=len(frames), source_duration_s=float(times[-1]),
                    reference_frames=len(target_times), world_translation=offset.tolist(),
                    hand_selection=record, object_uid=uid, object_bop_id=oid)
        info.update(coordinate_version=FRAME_VERSION,
                    source_world_up_axis="Y" if split == "train_quest3" else "Z",
                    simulation_world_up_axis="Z",
                    source_world_to_sim_rotation=world_rotation(split).tolist())
        info[target_side + "_object_mesh_dir"] = str(mesh_dir.relative_to(dataset_dir))
        _write(out / "task_info.json", info)
        task_rows.append({"task": task, "embodiment": embodiment, "object_side": target_side,
                          "mesh_dir": str(mesh_dir), "frames": len(target_times)})
        record.update(state="exported", task=task, embodiment=embodiment)
        audit.append(record)
    folder = dataset_dir / "raw" / split / clip
    _write(folder / "selection.json", audit)
    _write(folder / "tasks.json", task_rows)
    print("[exported]", clip, len(task_rows), flush=True)
    return task_rows


def prepare(dataset_dir: Path, task_row: dict, robot_type: str = "sharpa") -> None:
    import mujoco
    import numpy as np

    from grasp.pipeline.decompose import main as decompose
    from grasp.pipeline.generate_xml import main as generate
    from grasp.pipeline.hot3d_scene_support import resolve
    from grasp.pipeline.solve_ik import main as solve

    proc = dataset_dir / "processed/hot3d"
    task, embodiment = task_row["task"], task_row["embodiment"]
    out = proc / robot_type / embodiment / task
    info_path = proc / "mano" / embodiment / task / "task_info.json"
    mesh = Path(task_row["mesh_dir"])
    if not (mesh / "convex/.done").exists():
        decompose(dataset_dir=str(dataset_dir), dataset_name="hot3d", embodiment_type=embodiment, task=task,
                  add_flat_base=False)
        (mesh / "convex/.done").touch()
    info = json.loads(info_path.read_text())
    info[task_row["object_side"] + "_object_convex_dir"] = str((mesh / "convex").relative_to(dataset_dir))
    _write(info_path, info)
    generate(dataset_dir=str(dataset_dir), dataset_name="hot3d", robot_type=robot_type, embodiment_type=embodiment,
             task=task, show_viewer=False)
    # the IK scene lives one directory deeper: make its mesh paths absolute
    tree = ET.parse(out / "scene.xml")
    tree.getroot().find("compiler").set("meshdir", str(proc / "assets"))
    tree.write(out / "0/scene_ik.xml")

    solve(output_root_dir=str(proc), dataset_name="hot3d", robot_type=robot_type, embodiment_type=embodiment,
          task=task, end_idx=task_row["frames"], smoothing=False, show_viewer=False, save_video=False, force=False)
    z = np.load(out / "0/trajectory_kinematic.npz")
    m = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
    assert z["qpos"].shape == (task_row["frames"], m.nq) and z["qvel"].shape == (task_row["frames"], m.nv)
    assert np.isfinite(z["qpos"]).all() and np.isfinite(z["qvel"]).all()

    audit = resolve(out)
    print(f"[{task}] scene {out / 'scene.xml'}, {len(audit['placements'])} support surfaces", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["export", "prepare"])
    parser.add_argument("--split", required=True, choices=["train_aria", "train_quest3"])
    parser.add_argument("--clip", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--mano-dir", type=Path, help="MANO models directory (export)")
    parser.add_argument("--robot-type", default="sharpa")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if args.stage == "export":
        if args.mano_dir is None:
            sys.exit("--mano-dir is required for export")
        export(dataset_dir, args.split, args.clip, args.mano_dir.resolve())
    else:
        for row in json.loads((dataset_dir / "raw" / args.split / args.clip / "tasks.json").read_text()):
            prepare(dataset_dir, row, args.robot_type)


if __name__ == "__main__":
    main()
