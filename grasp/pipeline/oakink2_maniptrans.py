"""Export one bimanual OakInk2 clip as the raw pickle read by ``grasp.pipeline.oakink`` (``<dataset_dir>/raw/oakink/<task>_bimanual.pkl``).

The clip is loaded with ManipTrans's OakInk2 loaders (``main/dataset/oakink2_dataset_dexhand_{rh,lh}.py``), which are
not part of GRASP: install ManipTrans (https://github.com/ManipTrans/ManipTrans) with its OakInk2 data under
``<maniptrans_root>/data/OakInk-v2`` and pass ``--maniptrans-root``.  ManipTrans's CUDA chamfer extension is replaced by
the same squared nearest-neighbour distance in torch (it only feeds ``tips_distance``, which the processor does not read).

The clip is loaded at OakInk2's 120 Hz and resampled to ``--target-fps`` with ``idx = linspace(0, n - 1, int(n / 120 *
fps)).round()``; the object meshes are linked into ``<dataset_dir>/raw/oakink/meshes`` from ``--oakink2-root``.

    python -m grasp.pipeline.oakink2_maniptrans DATA_IDX --task oakbi_000_cut --dataset-dir data/oakink2 \\
        --maniptrans-root ~/ManipTrans --oakink2-root ~/OakInk-v2 --target-fps 50 --max-seconds 20
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch

TIPS = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
PER_FRAME = ("obj_trajectory", "wrist_pos", "wrist_rot", "tips_distance", "obj_velocity", "obj_angular_velocity",
             "wrist_velocity", "wrist_angular_velocity", "mano_joints")


class _ChamferDistance:
    def __call__(self, x, y):
        d = torch.cdist(x, y) ** 2
        d1, i1 = d.min(-1)
        d2, i2 = d.min(-2)
        return d1, d2, i1, i2


def _import_loaders(root: Path):
    """Import ManipTrans's OakInk2 loaders without its RL packages (isaacgym, bps_torch)."""
    chamfer = types.ModuleType("chamfer_distance")
    chamfer.ChamferDistance = _ChamferDistance
    sys.modules["chamfer_distance"] = chamfer
    for name, sub in (("maniptrans_envs", "maniptrans_envs"), ("maniptrans_envs.lib", "maniptrans_envs/lib"),
                      ("maniptrans_envs.lib.envs", "maniptrans_envs/lib/envs"), ("main", "main"),
                      ("main.dataset", "main/dataset")):
        module = types.ModuleType(name)
        module.__path__ = [str(root / sub)]
        sys.modules[name] = module
    sys.path.insert(0, str(root))
    from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
    DexHandFactory.auto_register_hands(str(root / "maniptrans_envs/lib/envs/dexhands"),
                                       "maniptrans_envs.lib.envs.dexhands")
    from main.dataset.oakink2_dataset_dexhand_lh import OakInk2DatasetDexHandLH
    from main.dataset.oakink2_dataset_dexhand_rh import OakInk2DatasetDexHandRH
    return DexHandFactory, OakInk2DatasetDexHandRH, OakInk2DatasetDexHandLH


def _to_cpu(value):
    return value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(np.asarray(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_idx", help="ManipTrans data index '<first 5 sequence-hash digits>@<primitive index>'")
    parser.add_argument("--task", required=True, help="task name used for the processed dataset")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--maniptrans-root", type=Path, required=True)
    parser.add_argument("--oakink2-root", type=Path, required=True, help="OakInk2 release (contains object_raw/align_ds)")
    parser.add_argument("--dexhand", default="inspire", help="ManipTrans hand model used only to construct the loaders")
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--max-seconds", type=float, default=None, help="keep only the first this-many seconds")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    out_path = dataset_dir / "raw/oakink" / f"{args.task}_bimanual.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root = args.maniptrans_root.resolve()
    os.chdir(root)    # the loaders read data/OakInk-v2 relative to the ManipTrans checkout
    factory, loader_rh, loader_lh = _import_loaders(root)

    out = {"env_type": "bimanual", "data_idx": args.data_idx}
    for side, loader in (("right", loader_rh), ("left", loader_lh)):
        dataset = loader(data_dir="data/OakInk-v2", device=args.device,
                         mujoco2gym_transf=torch.eye(4, device=args.device),
                         dexhand=factory.create_hand(args.dexhand, side), skip=1)
        d = dataset[args.data_idx]
        hand = {k: _to_cpu(d[k]) for k in ("obj_verts", "obj_trajectory", "wrist_pos", "wrist_rot", "tips_distance",
                                           "obj_velocity", "obj_angular_velocity", "wrist_velocity",
                                           "wrist_angular_velocity")}
        hand["mano_joints"] = torch.stack([d["mano_joints"][t].detach().cpu() for t in TIPS], dim=1)
        velocity = {k: _to_cpu(v) for k, v in d["mano_joints_velocity"].items()}

        n = len(hand["wrist_pos"])
        keep = torch.as_tensor(np.linspace(0, n - 1, num=int(n / 120.0 * args.target_fps)).round().astype(np.int64))
        if args.max_seconds:
            keep = keep[: int(round(args.max_seconds * args.target_fps))]
        for k in PER_FRAME:
            assert len(hand[k]) == n, (k, len(hand[k]), n)
            hand[k] = hand[k][keep]
        hand["mano_joints_velocity"] = [{k: v[keep] for k, v in velocity.items()}]
        for k in ("data_path", "obj_id", "obj_mesh_path", "obj_urdf_path"):
            hand[k] = [d[k]]
        hand["scene_objs"] = [d["scene_objs"]]
        out[side] = hand

        # the processor reads the object mesh from raw/oakink/meshes/<object>/<file>
        relative = d["obj_mesh_path"].split("align_ds/")[1]
        source = args.oakink2_root.resolve() / "object_raw/align_ds" / relative
        if not source.exists():
            raise FileNotFoundError(source)
        link = dataset_dir / "raw/oakink/meshes" / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(source)
        print(f"[{side}] {args.data_idx}: {n} frames @ 120 Hz -> {len(keep)} @ {args.target_fps:g} Hz, "
              f"object {d['obj_id']}", flush=True)

    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print("[saved]", out_path, flush=True)


if __name__ == "__main__":
    main()
