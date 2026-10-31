"""Paper table numbers from a directory of ``grasp.mpc.run`` results (the ``*.json`` files, searched recursively).

    python experiments/tables.py oakink2 runs/oakink2      # success rate, object position / rotation errors
    python experiments/tables.py hot3d runs/hot3d         # ADD-AUC10 at 12.5 and 50 Hz per embodiment
    python experiments/tables.py humanoid runs/lafan      # joint, pelvis position / orientation (and object) errors

OakInk2: every task of ``experiments/oakink2_references.csv`` counts; a task without a finite result counts as a
failure.  Errors are averaged over the tasks whose position and rotation errors are both finite.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def oakink2_summary(metrics: dict[str, dict | None], tasks: list[str]) -> dict:
    """``metrics``: task -> tracking metrics (``obj_pos_err`` m, ``obj_quat_err`` rad, ``success``) or None."""
    found = [metrics.get(t) for t in tasks]
    pos = np.array([m["obj_pos_err"] if m else np.nan for m in found], dtype=float)
    rot = np.array([m["obj_quat_err"] if m else np.nan for m in found], dtype=float)
    finite = np.isfinite(pos) & np.isfinite(rot)
    success = sum(bool(m and m["success"]) for m in found)
    return dict(tasks=len(tasks), results=sum(m is not None for m in found), success=success,
                success_rate_percent=100 * success / len(tasks), errors_over=int(finite.sum()),
                pos_mean_cm=100 * pos[finite].mean(), pos_median_cm=100 * np.median(pos[finite]),
                rot_mean_rad=rot[finite].mean(), rot_median_rad=np.median(rot[finite]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=("oakink2", "hot3d", "humanoid"))
    parser.add_argument("results", type=Path, help="directory searched recursively for result .json files")
    args = parser.parse_args()
    paths = sorted(p for p in args.results.rglob("*.json") if p.with_suffix(".npz").exists())
    if args.dataset == "oakink2":
        with (REPO / "experiments/oakink2_references.csv").open() as f:
            tasks = [row["task"] for row in csv.DictReader(f)]
        metrics = {}
        for p in paths:
            result = json.loads(p.read_text())
            metrics[result["config"]["task"]] = result["tracking_metrics"]
        out = oakink2_summary(metrics, tasks)
    elif args.dataset == "hot3d":
        from grasp.evaluation.add_auc import evaluate_run

        rows = [evaluate_run(p) for p in paths]
        out = {}
        for group in ("all", "right", "left", "bimanual"):
            g = [r for r in rows if group == "all" or r["embodiment"] == group]
            if g:
                out[group] = dict(tasks=len(g), **{f"add_auc_{hz}hz": float(np.mean([r["scores"][hz]["add_auc_percent"] for r in g]))
                                                   for hz in ("12.5", "50")})
    else:
        from grasp.evaluation.humanoid_error import evaluate_run

        rows = [evaluate_run(p) for p in paths]
        keys = [k for k in rows[0] if k not in ("seconds", "pelvis_pos_max_m")]
        out = dict(runs=len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in keys})
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
