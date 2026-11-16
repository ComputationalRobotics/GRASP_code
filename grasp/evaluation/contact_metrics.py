"""Hand-object contact metrics of an executed trajectory, measured on the task's collision geometry.

* penetration: every ``stride`` simulation steps, the deepest hand-object contact [mm] (``collision_hand_*`` against
  ``{right,left}_object_*`` collision geoms, MuJoCo CPU collision on the executed ``qpos``).  Reported per run: depth at
  the first executed state, median over the remaining samples, and the fraction of samples deeper than 5 mm / 10 mm.
* in-hand miss: among reference rows (every ``stride_rows``-th) whose ``gate_inhand`` is on for an object side, the
  fraction with no touching (``dist <= 0``) hand contact on that side's object in the executed state.

    python -m grasp.evaluation.contact_metrics runs/x/oakink_sharpa_bimanual_bimanual_oakbi_443_take_outside_0_seed0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def _geoms(model):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
    hand = {i for i, n in enumerate(names) if n.startswith("collision_hand_")}
    obj = {i for i, n in enumerate(names) if n.startswith(("right_object_", "left_object_")) and not n.endswith("visual")}
    return names, hand, obj


def penetration(model, qpos: np.ndarray, stride: int = 20) -> dict:
    """Deepest hand-object contact [mm] every ``stride`` executed steps and its summary."""
    data = mujoco.MjData(model)
    _, hand, obj = _geoms(model)
    depth = []
    for k in range(0, len(qpos), stride):
        data.qpos[:] = qpos[k]
        mujoco.mj_fwdPosition(model, data)
        deepest = 0.0
        for c in data.contact[:data.ncon]:
            a, b = int(c.geom[0]), int(c.geom[1])
            if ((a in hand and b in obj) or (b in hand and a in obj)) and c.dist < deepest:
                deepest = float(c.dist)
        depth.append(-deepest * 1000 + 0.0)   # + 0.0: no "-0.0" for a contact-free sample
    depth = np.asarray(depth)
    rest = depth[1:] if len(depth) > 1 else depth
    return dict(first_mm=float(depth[0]), median_mm=float(np.median(rest)), frac_over_5mm=float((depth > 5).mean()),
                frac_over_10mm=float((depth > 10).mean()), samples=int(len(depth)))


def inhand_miss(model, qpos: np.ndarray, gate_inhand: np.ndarray, sim_dt: float, ref_dt: float,
                stride_rows: int = 5) -> float:
    """Fraction of in-hand-gated (row, side) samples with no touching hand contact on that side's object."""
    data = mujoco.MjData(model)
    names, _, _ = _geoms(model)
    total = missed = 0
    for row in range(0, len(gate_inhand), stride_rows):
        k = min(len(qpos) - 1, int(round(row * ref_dt / sim_dt)) - 1)
        if k < 0 or not gate_inhand[row].any():
            continue
        data.qpos[:] = qpos[k]
        mujoco.mj_fwdPosition(model, data)
        for i, side in enumerate(("right", "left")):
            if not gate_inhand[row, i]:
                continue
            total += 1
            hit = False
            for c in data.contact[:data.ncon]:
                a, b = names[c.geom[0]], names[c.geom[1]]
                obj = lambda n: n.startswith(f"{side}_object_") and not n.endswith("visual")
                if c.dist <= 0 and ((a.startswith("collision_hand_") and obj(b)) or (b.startswith("collision_hand_") and obj(a))):
                    hit = True
                    break
            missed += not hit
    return missed / total if total else float("nan")


def evaluate(summary_path: Path, stride: int = 20) -> dict:
    """Contact metrics of one ``grasp.mpc.run`` result (the summary JSON next to its NPZ)."""
    summary = json.loads(Path(summary_path).read_text())
    overrides = dict(o.split("=", 1) for o in summary["args"]["override"])
    cfg = summary["config"]
    task_dir = (Path(overrides["dataset_dir"]) / "processed" / cfg["dataset_name"] / cfg["robot_type"]
                / cfg["embodiment_type"] / cfg["task"])
    data_id = cfg["data_id"]
    model = mujoco.MjModel.from_xml_path(str(task_dir / "scene.xml"))
    sim_dt = float(overrides.get("sim_dt", model.opt.timestep))
    info = json.loads((task_dir / "task_info.json").read_text())
    ref_dt = float(info.get("ref_dt", 0.02))
    qpos = np.load(Path(summary_path).with_suffix(".npz"))["executed_qpos"]
    out = dict(task=cfg["task"], penetration=penetration(model, qpos, stride))
    gates = task_dir / str(data_id) / "contact_gates.npz"
    if gates.exists():
        with np.load(gates) as g:
            out["inhand_miss"] = inhand_miss(model, qpos, np.asarray(g["gate_inhand"], bool), sim_dt, ref_dt)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", nargs="+", type=Path, help="grasp.mpc.run summary JSON(s)")
    parser.add_argument("--stride", type=int, default=20, help="simulation steps between penetration samples")
    a = parser.parse_args()
    for p in a.summary:
        print(json.dumps(evaluate(p, a.stride)), flush=True)
