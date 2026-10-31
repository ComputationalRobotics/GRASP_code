"""GRASP-MPC with the settings used in the paper, per dataset.

    python experiments/paper.py oakink2 --robot sharpa_bimanual --task oakbi_000_cut
    python experiments/paper.py oakink2 --robot xhand --task oakbi_000_cut
    python experiments/paper.py hot3d --embodiment right --task hot3d_clip-001849_obj28
    python experiments/paper.py amass --task martial_arts
    python experiments/paper.py lafan --task dance1_subject1
    python experiments/paper.py omomo --task move_largebox

The horizon is the task's whole reference, ``(frames - 1) * ref_dt`` rounded to the knot spacing: down for OakInk2,
AMASS and OMOMO, up for HOT3D (the reference's last frame is held for the remainder); LAFAN1 uses the first 10 s.  An
OakInk2 reference shorter than one 0.2 s knot uses 0.1 s knots.  ``--horizon-rounding`` overrides the dataset's rule
(the paper's OakInk2 runs of the full-length, missing-object and object-connection task sets rounded up).  Hand tasks
use ``tau = 0.006 / horizon``; humanoid tasks use a fixed tau.  Arguments after ``--`` are passed to ``grasp.mpc.run``
unchanged, e.g. ``-- --output-dir runs/x`` or ``-- --override nconmax_per_env=512`` for a scene whose contacts exceed
the default buffer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]

HAND = [
    "--config", str(REPO / "config/override/oakink.yaml"),
    "--override", "num_dr=0", "--override", "sim_dt=0.005", "--override", "knot_dt=0.2",
    "--num-particles", "2000", "--window-knots", "15", "--commit-knots", "2",
    "--warm-start-plan", "--joint-start", "--pin-first-knot",
    "--prior-rho", "0.9", "--prior-sigma", "0.0044", "--finger-channel-scale", "10", "--prior-window-ramp", "4",
    "--library-reuse-period", "2", "--library-local-improve", "--robust-select-k", "50",
    "--mala-preconditioner", "prior_full", "--surrogate-rank", "2", "--step-size", "0.0026",
    "--adaptive-step-target", "0.05", "--adaptive-step-rate", "3", "--adaptive-step-min-scale", "0.1",
]
HUMANOID = [
    "--override", "num_dr=0", "--override", f"sim_dt={1 / 60}", "--override", "knot_dt=0.1",
    "--num-particles", "1000", "--window-knots", "8", "--commit-knots", "1",
    "--warm-start-plan", "--pin-first-knot",
    "--prior-rho", "0", "--prior-sigma", "0.065", "--arm-channel-scale", "0.5", "--prior-window-ramp", "4",
    "--library-reuse-period", "2", "--library-local-improve", "--robust-select-k", "50",
    "--mala-preconditioner", "prior_full", "--surrogate-rank", "2", "--step-size", "0.02",
    "--adaptive-step-target", "0.05", "--adaptive-step-rate", "3", "--adaptive-step-min-scale", "0.1",
    "--max-stages-per-window", "0",
]
# dataset: (dataset_name, settings, config, knot spacing, default robot, default embodiment,
#           tau (None: 0.006 / horizon), horizon (None: the whole reference), rounding of the whole reference)
DATASETS = {
    "oakink2": ("oakink", HAND, "oakink.yaml", 0.2, "sharpa_bimanual", "bimanual", None, None, "down"),
    "hot3d": ("hot3d", HAND, "oakink.yaml", 0.2, "sharpa", "bimanual", None, None, "up"),
    "amass": ("amass", HUMANOID, "humanoid.yaml", 0.1, "unitree_g1", "humanoid", 0.0000455, None, "down"),
    "lafan": ("lafan", HUMANOID, "humanoid.yaml", 0.1, "unitree_g1", "humanoid", 0.000182, 10.0, None),
    "omomo": ("omomo", HUMANOID, "humanoid_object.yaml", 0.1, "unitree_g1", "humanoid_object", 0.000182, None, "down"),
}


def reference_horizon(task_dir: Path, data_id: int, knot_dt: float, sim_dt: float, config: Path,
                      rounding: str) -> float:
    """Whole reference, rounded to the knot spacing.  ref_dt as grasp.config resolves it: the task's task_info.json,
    else the config file."""
    frames = len(np.load(task_dir / str(data_id) / "trajectory_kinematic.npz")["qpos"])
    info = task_dir / "task_info.json"
    info = json.loads(info.read_text()) if info.exists() else {}
    ref_dt = float(info["ref_dt"] if "ref_dt" in info else OmegaConf.load(config)["ref_dt"])
    if rounding == "down":  # 1e-4: config files give 1/30 s as 0.0333333
        return round(math.floor((frames - 1) * ref_dt / knot_dt + 1e-4) * knot_dt, 1)
    # in simulation steps, as the paper's runs computed it (tau's last digit depends on the float)
    steps, knot_steps = round((frames - 1) * ref_dt / sim_dt), round(knot_dt / sim_dt)
    return -(-steps // knot_steps) * knot_steps * sim_dt


def build(dataset: str, task: str, robot: str | None = None, embodiment: str | None = None, data_id: int = 0,
          dataset_dir: Path = REPO / "data", horizon_rounding: str | None = None) -> list[str]:
    """``grasp.mpc.run`` arguments for one task."""
    name, settings, config, knot_dt, default_robot, default_embodiment, tau, horizon, rounding = DATASETS[dataset]
    robot, embodiment = robot or default_robot, embodiment or default_embodiment
    rounding = horizon_rounding or rounding
    task_dir = Path(dataset_dir).resolve() / "processed" / name / robot / embodiment / task
    config = REPO / "config/override" / config
    args = list(settings)
    sim_dt = float(next(a for a in args if a.startswith("sim_dt=")).split("=")[1])
    if horizon is None:
        horizon = reference_horizon(task_dir, data_id, knot_dt, sim_dt, config, rounding)
        if horizon == 0 and settings is HAND:  # reference shorter than one 0.2 s knot
            horizon = reference_horizon(task_dir, data_id, 0.1, sim_dt, config, rounding)
            args = [("knot_dt=0.1" if a == "knot_dt=0.2" else a) for a in args]
    assert horizon > 0, (task_dir, horizon)
    if tau is None:
        tau = round(0.006 / horizon, 6)
    if str(config) not in args:
        args += ["--config", str(config)]
    for key, value in [("dataset_dir", Path(dataset_dir).resolve()), ("dataset_name", name), ("robot_type", robot),
                       ("embodiment_type", embodiment), ("task", task), ("data_id", data_id)]:
        args += ["--override", f"{key}={value}"]
    return args + ["--horizon", str(horizon), "--tau", str(tau)]


def main() -> None:
    argv = sys.argv[1:]
    passthrough = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--task", required=True)
    parser.add_argument("--robot", help="oakink2: sharpa_bimanual, xhand or wuji_topo")
    parser.add_argument("--embodiment", help="hot3d: right, left or bimanual")
    parser.add_argument("--data-id", type=int, default=0)
    parser.add_argument("--dataset-dir", type=Path, default=REPO / "data")
    parser.add_argument("--horizon-rounding", choices=("down", "up"), help="rounding of the whole reference to knots")
    parser.add_argument("--print", action="store_true", help="print the command instead of running it")
    args = parser.parse_args(argv)
    run_args = build(args.dataset, args.task, args.robot, args.embodiment, args.data_id, args.dataset_dir,
                     args.horizon_rounding) + passthrough
    if args.print:
        print(" ".join(["python", "-m", "grasp.mpc.run", *run_args]))
        return
    from grasp.mpc.run import main as run

    sys.argv = ["grasp.mpc.run", *run_args]
    run()


if __name__ == "__main__":
    main()
