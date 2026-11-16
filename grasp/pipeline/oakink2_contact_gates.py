"""Contact gates of an OakInk2 task: for every kinematic-reference row and object side, whether the demonstrating hand
holds the object (``gate_inhand``) or the object rests on the floor (``gate_rest``).

1. MANO hand surface for every keypoint row, from OakInk2's own MANO annotation (``anno_preview/<sequence>.pkl``,
   ``raw_mano``; manotorch ``ManoLayer`` with quaternion pose coefficients, full pose, flat hand mean, root-centred,
   plus the translation), rotated into the processed world frame (``rot_x(pi/2)``, as ``grasp.pipeline.oakink``).
   Keypoint row -> mocap frame: the frame whose object positions match the row exactly (< 1e-5 m); rows of static
   objects (several exact matches) take the candidate nearest the affine fit of the unambiguous rows, or, when objects
   are static nearly throughout, the ``linspace(first, last).round()`` sampling that best matches the stored fingertips.
2. Gates on the robot task's scene and reference: ``in_hand`` = some MANO vertex within 0.1 m of the object's
   collision-mesh vertices (512 sampled); ``near_floor`` = the object's lowest sampled vertex within 0.01 m of the floor
   geom; ``gate_inhand = in_hand & ~near_floor``, ``gate_rest = near_floor & ~in_hand``.  A hand whose own side has no
   object is measured against the other side's object (masks of one object OR'ed over hands).  Keypoint rows map to
   reference rows by the offset that minimises the object-position gap (IK trims boundary rows).

Writes ``mano_verts.npz`` beside the task's ``trajectory_keypoints.npz`` and ``contact_gates.npz`` beside its
``trajectory_kinematic.npz``.  Needs the MANO model files (``<mano_dir>/models/MANO_{RIGHT,LEFT}.pkl``, from
https://mano.is.tue.mpg.de) and manotorch.

    python -m grasp.pipeline.oakink2_contact_gates --task oakbi_443_take_outside --dataset-dir data/oakink2 \\
        --oakink2-root /path/to/OakInk-v2 --mano-dir /path/to/mano_v1_2
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from grasp.pipeline.in_hand import (
    body_has_mesh_geom,
    compute_in_hand_mask,
    compute_near_floor_mask,
    extract_body_mesh_verts,
)

SIDES = (("right", "rh"), ("left", "lh"))
TIP_JOINTS = [4, 8, 12, 16, 20]
R_GLOBAL = R.from_euler("xyz", [np.pi / 2, 0, 0])


class _CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def _mano(anno, frames, layer, prefix):
    pose = torch.cat([anno["raw_mano"][f][f"{prefix}__pose_coeffs"] for f in frames]).float()
    betas = torch.cat([anno["raw_mano"][f][f"{prefix}__betas"] for f in frames]).float()
    tsl = torch.cat([anno["raw_mano"][f][f"{prefix}__tsl"] for f in frames]).float()
    with torch.no_grad():
        out = layer(pose_coeffs=pose, betas=betas)
    verts = (out.verts + tsl[:, None]).numpy().astype(np.float64)
    joints = (out.joints + tsl[:, None]).numpy().astype(np.float64)
    return (R_GLOBAL.apply(verts.reshape(-1, 3)).reshape(verts.shape),
            R_GLOBAL.apply(joints.reshape(-1, 3)).reshape(joints.shape))


def frame_map(kp, anno, raw, layers) -> tuple[np.ndarray, str]:
    """Mocap frame id of every keypoint row (see module docstring)."""
    fids = np.asarray(anno["mocap_frame_id_list"])
    n = len(kp["qpos_obj_right"])
    cost = np.zeros((n, len(fids)))
    used = []
    for s, _ in SIDES:
        k = kp[f"qpos_obj_{s}"]
        if np.allclose(k[:, :3], 0) or s not in raw or "obj_id" not in raw[s]:
            continue
        o = raw[s]["obj_id"][0]
        P = np.stack([R_GLOBAL.apply(np.asarray(anno["obj_transf"][o][f])[:3, 3]) for f in fids])
        cost += np.linalg.norm(k[:, None, :3] - P[None], axis=-1)
        used.append(s)
    if not used:
        raise RuntimeError("no object side to match keypoint rows against the annotation")
    exact = cost < 1e-5
    if not exact.any(1).all():
        raise RuntimeError(f"{int((~exact.any(1)).sum())} keypoint rows have no exact object match in the annotation")
    lo = exact.argmax(1)
    hi = exact.shape[1] - 1 - exact[:, ::-1].argmax(1)
    anchors = np.flatnonzero(hi - lo <= 2)
    if len(anchors) >= min(10, max(3, n // 2)) and anchors.max() - anchors.min() >= 0.3 * n:
        fit = np.polyfit(anchors, (lo[anchors] + hi[anchors]) / 2.0, 1)
        how = "object anchors"
    else:
        cand = np.arange(lo.min(), hi.max() + 1)
        sub = np.arange(0, n, 10)
        gap = 0
        for s, prefix in SIDES:
            _, joints = _mano(anno, fids[cand], layers[s], prefix)
            tips = joints[:, TIP_JOINTS]
            gap = gap + np.linalg.norm(kp[f"qpos_finger_{s}"][sub][:, None, :, :3] - tips[None], axis=-1).mean(-1)
        best = (np.inf, None)
        for a0 in np.flatnonzero(exact[0]):
            for e0 in np.flatnonzero(exact[n - 1]):
                stride = (e0 - a0) / max(n - 1, 1)
                if not 1.9 <= stride <= 3.0:       # 2.4 = 50 Hz, 2.0 = 60 Hz; mocap index gaps can stretch it
                    continue
                ii = np.round(np.linspace(a0, e0, n)).astype(int)
                if not exact[np.arange(n), ii].all():
                    continue
                c = gap[np.arange(len(sub)), ii[sub] - cand[0]].mean()
                if c < best[0]:
                    best = (c, (a0, e0))
        if best[1] is None:
            raise RuntimeError("no linspace frame map fits the exact object matches")
        a0, e0 = best[1]
        fit = np.array([(e0 - a0) / max(n - 1, 1), a0])
        how = f"fingertips (mean gap {500 * best[0]:.1f} mm)"
    pred = np.polyval(fit, np.arange(n))
    idx = np.array([np.flatnonzero(exact[i])[np.abs(np.flatnonzero(exact[i]) - pred[i]).argmin()] for i in range(n)])
    if np.abs(idx - pred).max() > 1.0 or not np.all(np.diff(idx) > 0):
        raise RuntimeError(f"frame map is not a monotone sampling of the annotation ({how})")
    return fids[idx], how


def mano_vertices(kp_path: Path, raw_path: Path, oakink2_root: Path, mano_dir: Path) -> dict:
    """MANO vertices ``mano_verts_{side}`` (rows, 778, 3) for every keypoint row, with the frame map and a fingertip
    check against the stored keypoints."""
    from manotorch.manolayer import ManoLayer

    kp = np.load(kp_path)
    with open(raw_path, "rb") as f:
        raw = _CPUUnpickler(f).load()
    anno_name = Path(raw["right"]["data_path"][0]).name
    with open(Path(oakink2_root) / "anno_preview" / anno_name, "rb") as f:
        anno = pickle.load(f)
    layers = {s: ManoLayer(mano_assets_root=str(mano_dir), rot_mode="quat", side=s, center_idx=0, use_pca=False,
                           flat_hand_mean=True) for s, _ in SIDES}
    frames, how = frame_map(kp, anno, raw, layers)
    out = dict(mocap_frames=frames, anno=anno_name, frame_map=how)
    for s, prefix in SIDES:
        verts, joints = _mano(anno, frames, layers[s], prefix)
        gap = np.linalg.norm(joints[:, TIP_JOINTS] - kp[f"qpos_finger_{s}"][:, :, :3], axis=-1) * 1000
        out[f"mano_verts_{s}"] = verts.astype(np.float32)
        out[f"tip_gap_mm_{s}"] = np.array([np.median(gap), np.percentile(gap, 95), gap.max()])
    return out


def contact_gates(task_dir: Path, data_id: int, kp, mano: dict, hand_thresh: float = 0.1,
                  floor_thresh: float = 0.01) -> dict:
    """Gates per kinematic-reference row of the robot task in ``task_dir`` (see module docstring)."""
    m = mujoco.MjModel.from_xml_path(str(task_dir / "scene.xml"))
    info = json.loads((task_dir / "task_info.json").read_text())
    real = [s for s, _ in SIDES if f"{s}_plate_bottom_world_z" in info]
    if not real:
        raise RuntimeError(f"{task_dir}: task_info.json names no object side ({{side}}_plate_bottom_world_z)")
    kin = np.load(task_dir / str(data_id) / "trajectory_kinematic.npz")["qpos"]
    nk, nkp = len(kin), len(kp["qpos_obj_right"])
    sl = {"right": slice(-14, -11), "left": slice(-7, -4)}
    gaps = [np.mean([np.linalg.norm(kin[:, sl[s]] - kp[f"qpos_obj_{s}"][o:o + nk, :3], axis=1).mean() for s in real])
            for o in range(nkp - nk + 1)]
    o = int(np.argmin(gaps))
    if gaps[o] > 5e-3:       # IK smoothing alone leaves < 1 mm (all 1,352 paper tasks)
        raise RuntimeError(f"reference rows do not match keypoint rows (object gap {1000 * gaps[o]:.2f} mm)")
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_z = float(m.geom_pos[floor][2])
    body_in, body_q, body_verts = {}, {}, {}
    for hs, _ in SIDES:
        own = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{hs}_object")
        other = "left" if hs == "right" else "right"
        if own >= 0 and body_has_mesh_geom(m, own) and hs in real:
            bs = hs
        elif other in real:
            bs = other
        else:
            continue
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{bs}_object")
        if bs not in body_verts:
            body_verts[bs] = extract_body_mesh_verts(m, b, apply_geom_xform=True)
        body_q[bs] = kp[f"qpos_obj_{bs}"]
        mask = compute_in_hand_mask(points_world=mano[f"mano_verts_{hs}"].astype(np.float64), qpos_obj=body_q[bs],
                                    obj_verts=body_verts[bs], distance_thresh=hand_thresh)
        body_in[bs] = mask if bs not in body_in else (body_in[bs] | mask)
    gin = np.zeros((nk, 2), bool)
    grest = np.zeros((nk, 2), bool)
    for i, (s, _) in enumerate(SIDES):
        if s not in body_in:
            continue
        near = compute_near_floor_mask(qpos_obj=body_q[s], obj_verts=body_verts[s], floor_z=floor_z,
                                       distance_thresh=floor_thresh)
        gin[:, i] = (body_in[s] & ~near)[o:o + nk]
        grest[:, i] = (near & ~body_in[s])[o:o + nk]
    return dict(gate_inhand=gin, gate_rest=grest, keypoint_row_offset=o, object_gap_m=gaps[o])


def main(task: str, dataset_dir: str, oakink2_root: str, mano_dir: str, robot_type: str = "sharpa_bimanual",
         data_id: int = 0) -> Path:
    root = Path(dataset_dir).resolve() / "processed" / "oakink"
    kp_dir = root / "mano" / "bimanual" / task / str(data_id)
    task_dir = root / robot_type / "bimanual" / task
    kp = np.load(kp_dir / "trajectory_keypoints.npz")
    verts_path = kp_dir / "mano_verts.npz"
    if verts_path.exists():
        mano = dict(np.load(verts_path))
        if len(mano["mano_verts_right"]) != len(kp["qpos_obj_right"]):
            raise RuntimeError(f"{verts_path} does not match the keypoint rows; delete it to rebuild")
    else:
        raw_path = Path(dataset_dir).resolve() / "raw" / "oakink" / f"{task}_bimanual.pkl"
        mano = mano_vertices(kp_dir / "trajectory_keypoints.npz", raw_path, Path(oakink2_root), Path(mano_dir))
        np.savez_compressed(verts_path, **mano)
        print(f"[{task}] MANO vertices: frame map by {mano['frame_map']}; fingertip gap vs keypoints (median, p95, max "
              f"mm) right {np.round(mano['tip_gap_mm_right'], 1).tolist()} left "
              f"{np.round(mano['tip_gap_mm_left'], 1).tolist()}", flush=True)
    gates = contact_gates(task_dir, data_id, kp, mano)
    out = task_dir / str(data_id) / "contact_gates.npz"
    np.savez_compressed(out, **gates)
    print(f"[{task}] {robot_type} contact gates {out}: in hand {gates['gate_inhand'].mean(0).round(3).tolist()}, "
          f"resting {gates['gate_rest'].mean(0).round(3).tolist()} of reference rows (right, left)", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--oakink2-root", required=True, help="OakInk2 release directory containing anno_preview/")
    parser.add_argument("--mano-dir", required=True, help="MANO directory containing models/MANO_{RIGHT,LEFT}.pkl")
    parser.add_argument("--robot-type", default="sharpa_bimanual")
    parser.add_argument("--data-id", type=int, default=0)
    a = parser.parse_args()
    main(a.task, a.dataset_dir, a.oakink2_root, a.mano_dir, a.robot_type, a.data_id)
