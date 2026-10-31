#!/usr/bin/env python3
"""Receding-horizon GRASP MPC on a retargeting task (MuJoCo Warp).

``J = -(mean tracking reward over the full horizon)`` for delta-knot controls
around the kinematic reference controls, nominal world (num_dr=0).

Windows of ``--window-knots`` knots are solved from the common current state;
``--commit-knots`` knots are executed per window.
The host tempered SMC sampler solves each window.
The committed step-level control trajectory is re-evaluated at the end with
the original full-horizon evaluator and compared with the sum of the
executed window rewards.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np



def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receding-horizon GRASP MPC (MuJoCo Warp)"
    )
    parser.add_argument("--config", type=Path, action="append", default=[],
                        help="config file merged over config/default.yaml (repeatable)")
    parser.add_argument("--override", action="append", default=[], help="key=value config override (repeatable)")
    parser.add_argument("--warm-start-plan", action="store_true",
                        help="centre each window's AR(1) prior on the previous window's unexecuted plan, shifted by "
                             "the commit and padded with its last knot")
    parser.add_argument("--horizon", type=float, default=4.5)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-particles", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--window-knots", type=int, default=10)
    parser.add_argument("--commit-knots", type=int, default=2)
    parser.add_argument(
        "--intermediate-terminal-weight",
        type=float,
        default=1.0,
        help="weight of the terminal tracking reward at each window end (0 = none)",
    )
    parser.add_argument("--tau", type=float, default=0.0015)
    parser.add_argument("--arm-channel-scale", type=float, default=1.0,
        help="prior width of the shoulder/elbow/wrist actuators relative to the other channels "
             "(prior and MALA metric; 1 = uniform)")
    parser.add_argument("--joint-start", action="store_true",
                        help="first window: each candidate's knot 0 is also its initial configuration of the actuated joints "
                             "(at rest at its own position targets); initial state and first-window controls are optimised "
                             "jointly under the ordinary window cost and prior.  Objects start at the reference, free, gravity on")
    parser.add_argument("--finger-channel-scale", type=float, default=1.0, help="prior and MALA metric scale for articulated finger actuators")
    parser.add_argument("--prior-window-ramp", type=float, default=1.0,
                        help="widen the window prior toward the window end: knot k of K gets width x ramp**(k/(K-1)) "
                             "(1 = constant width); warm-start prior only")
    parser.add_argument("--prior-rho", type=float, default=0.9)
    parser.add_argument("--prior-sigma", type=float, default=0.01)

    parser.add_argument(
        "--kernel", default="surrogate_mala",
        choices=["surrogate_mala", "surrogate_nuts", "rwm"],
    )
    parser.add_argument("--library-reuse-period", type=int, default=0,
                        help="2: alternate heavy windows (TSMC, keep all stage controls) and light windows "
                             "(no TSMC; shift the library by D, append the AR(1)-mean tail, re-score from the actual state, execute argmin); "
                             "3: one heavy then two light windows, the light window's candidates carry on as the next library")
    parser.add_argument("--robust-select-k", type=int, default=0,
                        help="execute the lowest MEAN window cost among the top-k distinct candidates, each re-simulated in "
                             "N//k world slots (one extra batch per window); 0 = argmin of single rollouts")
    parser.add_argument("--library-local-improve", action="store_true",
                        help="light windows score first-generation, one-shot comparison, and second-generation "
                             "perturbations of the top library plans; select the best plan from their union")
    parser.add_argument("--improve-scale", type=float, default=0.3, help="perturbation innovation std as a fraction of prior sigma")
    parser.add_argument("--improve-parents", type=int, default=50)
    parser.add_argument("--pin-first-knot", action="store_true",
                        help="knot 0 of every window = the control offset executed just before it (in candidate rollouts and "
                             "in execution), so replans cannot jump the command")
    parser.add_argument("--target-ess", type=float, default=0.9)
    parser.add_argument("--num-mcmc-steps", type=int, default=5)
    parser.add_argument("--step-size", type=float, default=3e-3)
    parser.add_argument("--mala-preconditioner", choices=("identity", "prior_full"),
                        default="identity", help="fixed MALA metric; prior_full uses the window prior's full covariance "
                        "(AR(1) across knots x per-channel widths)")
    parser.add_argument("--no-adaptive-step-size", action="store_true")
    parser.add_argument("--adaptive-step-target", type=float, default=0.25)
    parser.add_argument("--adaptive-step-rate", type=float, default=1.0)
    parser.add_argument("--adaptive-step-min-scale", type=float, default=0.02)
    parser.add_argument("--surrogate-rank", type=int, default=4)
    parser.add_argument("--surrogate-ridge", type=float, default=1e-3)
    parser.add_argument("--surrogate-window", type=float, default=3.0)
    parser.add_argument("--surrogate-crossfit-shrink", action=argparse.BooleanOptionalAction, default=True,
                        help="scale each stage's surrogate by mean(gA.gB)/mean|g|^2 from two half-data fits "
                        "(removes the estimation-noise inflation of the slope along its own gradient); proposal only, "
                        "skipped where it does not apply (needs single-quadratic surrogate_mala fitted on a torch device)")
    parser.add_argument("--nuts-max-doublings", type=int, default=8)
    parser.add_argument("--max-stages-per-window", type=int, default=64, help="maximum tempering stages per window; 0 means no limit")
    parser.add_argument(
        "--surrogate-fit-device", default="cuda",
        help="device for the per-stage quadratic fit ('cpu' = original numpy path)",
    )
    parser.add_argument(
        "--fit-max-points", type=int, default=60000,
        help="archive points used per surrogate refit (CPU cost ~ points x features^2)",
    )

    parser.add_argument(
        "--output-dir", type=Path, default=Path("media/grasp_mpc")
    )
    return parser.parse_args(argv)




def _to_numpy(x, dtype=np.float32):
    return (x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)).astype(dtype)

def main() -> None:
    args = parse_args()
    prior_channel_scale = None

    import jax
    import torch
    import warp as wp
    from grasp.config import load_config
    from grasp.utils.interp import get_slice
    from grasp.utils.io import load_data
    from grasp.objective.tracking_cost import FullHorizonCost
    from grasp.sim.rollout import make_rollout_fn
    from grasp.sim import mjwp as mjwp_module
    from grasp.sim.mjwp import (
        copy_sample_state,
        get_reward,
        get_terminal_reward,
        get_terminate,
        get_trace,
        load_env_params,
        load_state,
        save_env_params,
        save_state,
        setup_env,
        step_env,
    )

    from grasp.mpc.grasp_mpc import (
        GraspMPC,
        GraspWindowEvaluator,
    )

    if jax.default_backend() != "cpu":
        raise RuntimeError("JAX must run on CPU beside MJWarp")

    config = load_config(args.config, args.override, horizon=args.horizon,
                         num_samples=args.num_particles, seed=args.seed)
    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(
        config, config.data_path
    )
    start_sim_step = int(
        round(int(args.start_frame) * float(config.ref_dt) / float(config.sim_dt))
    )
    if start_sim_step + int(config.horizon_steps) + 1 > qpos_ref.shape[0]:
        raise ValueError("reference window is outside the available trajectory")
    qpos_ref = qpos_ref[start_sim_step:]
    qvel_ref = qvel_ref[start_sim_step:]
    ctrl_ref = ctrl_ref[start_sim_step:]
    contact = contact[start_sim_step:]
    contact_pos = contact_pos[start_sim_step:]
    if ctrl_ref.shape[1] != config.nu and qpos_ref.shape[1] >= config.nu:
        ctrl_ref = qpos_ref[:, : config.nu]
    ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)
    env = setup_env(config, ref_data)
    full_reference = ctrl_ref[:int(config.horizon_steps)].clone()
    T = int(config.horizon_steps)
    ref_slice = get_slice(ref_data, 1, T + 1)
    if int(config.num_dr) != 0:
        raise ValueError("receding-horizon MPC supports num_dr=0 only")
    env_params = {"xy_offset": 0.0, "pair_margin": 0.0}
    n = int(round(float(config.knot_dt) / float(config.sim_dt)))
    if n <= 0 or T % n != 0:
        raise ValueError("horizon_steps must be a multiple of knot steps")
    total_knots = T // n

    rollout = make_rollout_fn(
        step_env, save_state, load_state, get_reward, get_terminal_reward,
        get_terminate, get_trace, save_env_params, load_env_params,
        copy_sample_state,
    )
    full_horizon_cost = FullHorizonCost(config, env, rollout, ref_slice, env_params)
    real_ids = list(range(int(config.nu)))
    m = len(real_ids)
    if float(args.arm_channel_scale) != 1.0:
        import mujoco as _mj
        _names = [_mj.mj_id2name(env.model_cpu, _mj.mjtObj.mjOBJ_ACTUATOR, int(a_)) or "" for a_ in real_ids]
        _arm = ("shoulder", "elbow", "wrist")
        _sel = np.asarray([any(k in n for k in _arm) for n in _names])
        if not _sel.any():
            raise ValueError("--arm-channel-scale matched no shoulder/elbow/wrist actuator; "
                             f"names are {_names}")
        prior_channel_scale = np.where(_sel, float(args.arm_channel_scale), 1.0).astype(np.float32)
    if float(args.finger_channel_scale) != 1.0:
        import mujoco as _mj
        _an = [(_mj.mj_id2name(env.model_cpu, _mj.mjtObj.mjOBJ_ACTUATOR, int(a)) or "") for a in real_ids]
        _finger = np.asarray([("_hand_" in n_) or ("finger" in n_) or any(f in n_ for f in ("thumb", "index", "middle", "ring", "pinky"))
                              for n_ in _an])
        if not _finger.any() or float(args.finger_channel_scale) <= 0:
            raise ValueError("finger-channel-scale requires finger actuators and a positive scale")
        if prior_channel_scale is None:
            prior_channel_scale = np.ones(m, np.float32)
        prior_channel_scale = prior_channel_scale * np.where(_finger, float(args.finger_channel_scale), 1.0)
        print(f"[finger-scale] {_finger.sum()} joints x{args.finger_channel_scale} (prior + metric): "
              + ", ".join(n_ for n_, k in zip(_an, _finger) if k), flush=True)
    N_worlds = int(config.num_samples)                    # MJWarp worlds
    N = int(args.num_particles)                           # particles seen by the sampler / MPC
    if N != N_worlds:
        raise ValueError("config.num_samples must equal num_particles")


    # Snapshot of the initial state for the final full-horizon re-evaluation.
    with wp.ScopedDevice(env.device):
        snapshot = mjwp_module.mjwarp.put_data(
            env.model_cpu, env.data_cpu, nworld=N_worlds,
            nconmax=int(config.nconmax_per_env), njmax=int(config.njmax_per_env),
        )
    mjwp_module._copy_state(env.data_wp, snapshot)

    # Reference replica check without aborting on non-finite worlds: report
    # the distribution instead (some contact-rich references are not
    # reproducible across MJWarp worlds).
    _ref_values = full_horizon_cost(
        torch.as_tensor(full_reference, dtype=torch.float32, device=config.device)
        .unsqueeze(0).repeat(N_worlds, 1, 1))
    _finite = np.isfinite(_ref_values)
    print(
        "[replica-check] full reference: "
        f"finite={int(_finite.sum())}/{N_worlds} min={np.nanmin(_ref_values):.6g} "
        f"median={np.nanmedian(_ref_values):.6g} max={np.nanmax(_ref_values[_finite]) if _finite.any() else float('nan'):.6g}",
        flush=True,
    )
    reference_cost = float(np.nanmedian(_ref_values))
    print(
        f"[problem] task={config.task} robot={config.robot_type} "
        f"T={T} sim steps, knots={total_knots} x {n} steps, robot_nu={m}, "
        f"N={N}, window={args.window_knots} knots, commit={args.commit_knots} knots, "
        f"beta={1.0 / args.tau:.1f}, J0(reference)={reference_cost:.6f}",
        flush=True,
    )

    evaluator_kwargs = dict(
        config=config, env=env, mjwp=mjwp_module, ref_data=ref_data,
        full_reference=full_reference, real_actuator_ids=real_ids,
        env_params=env_params, steps_per_knot=n, total_steps=T,
    )
    evaluator = GraspWindowEvaluator(**evaluator_kwargs)
    evaluator.pin_first_knot = bool(args.pin_first_knot)
    if args.pin_first_knot:
        print("[pin-first-knot] knot 0 of each window pinned to the last executed control offset", flush=True)

    # Window-level cost purity: repeat the same batch twice from the same state.
    rng = np.random.default_rng(int(args.seed) + 777)
    # Short clips have fewer knots than the configured planning window.
    # Match the MPC loop's window clipping for this startup diagnostic too.
    probe_knots = min(int(args.window_knots), total_knots)
    probe = (float(args.prior_sigma) * rng.standard_normal(
        (N, probe_knots, m)
    )).astype(np.float32)
    c1 = evaluator.window_cost(probe, 0, float(args.intermediate_terminal_weight))
    c2 = evaluator.window_cost(probe, 0, float(args.intermediate_terminal_weight))
    dev = np.abs(c1 - c2)
    same = np.repeat(probe[:1], N, axis=0)
    c3 = evaluator.window_cost(same, 0, float(args.intermediate_terminal_weight))
    print(
        "[purity] window batch repeated: "
        f"q50={np.quantile(dev, 0.5):.3g} q90={np.quantile(dev, 0.9):.3g} "
        f"max={dev.max():.3g} (cost units, beta*max={dev.max() / args.tau:.2f}); "
        f"identical control across slots: spread={np.ptp(c3):.3g} std={c3.std():.3g}",
        flush=True,
    )
    evaluator.sim_steps = 0
    common_kwargs = dict(
        target_ess=float(args.target_ess),
        num_mcmc_steps=int(args.num_mcmc_steps),
        step_size=float(args.step_size),
        adapt_step_size=not bool(args.no_adaptive_step_size),
        adapt_step_target=float(args.adaptive_step_target),
        adapt_step_rate=float(args.adaptive_step_rate),
        adapt_step_min_scale=float(args.adaptive_step_min_scale),
        max_iterations=int(args.max_stages_per_window),
    )
    sampler_kwargs = dict(
        common_kwargs,
        kernel=str(args.kernel),
        mala_preconditioner=str(args.mala_preconditioner),
        surrogate_ridge=float(args.surrogate_ridge),
        surrogate_window=float(args.surrogate_window),
        surrogate_rank=int(args.surrogate_rank),
        nuts_max_doublings=int(args.nuts_max_doublings),
        fit_max_points=int(args.fit_max_points),
        surrogate_fit_device=str(args.surrogate_fit_device),
        surrogate_crossfit_shrink=bool(args.surrogate_crossfit_shrink),
    )
    mpc = GraspMPC(
        evaluator,
        total_knots=total_knots,
        window_knots=int(args.window_knots),
        commit_knots=int(args.commit_knots),
        control_dim=m,
        prior_rho=float(args.prior_rho),
        prior_sigma=float(args.prior_sigma),
        beta=1.0 / float(args.tau),
        sampler_kwargs=sampler_kwargs,
        phi_weight=float(args.intermediate_terminal_weight),
        seed=int(args.seed),
        trace_dir=None,
        prior_channel_scale=prior_channel_scale,
        library_reuse_period=int(args.library_reuse_period),
        library_local_improve=bool(args.library_local_improve),
        improve_scale=float(args.improve_scale), improve_parents=int(args.improve_parents),
    )

    mpc.robust_select_k = int(args.robust_select_k)
    mpc.warm_start_plan = bool(args.warm_start_plan)
    mpc.prior_window_ramp = float(args.prior_window_ramp)
    if mpc.prior_window_ramp != 1.0:
        if not mpc.warm_start_plan or mpc.prior_window_ramp <= 0:
            raise ValueError("--prior-window-ramp needs --warm-start-plan and a positive ramp")
        print(f"[prior-ramp] window prior width x{mpc.prior_window_ramp}**(k/(K-1)) over each window's knots", flush=True)
    if mpc.warm_start_plan:
        print("[warm-start] AR(1) prior centred on the previous window's unexecuted plan (shifted by the commit, "
              "padded with its last knot); the first particle starts exactly there", flush=True)
    if args.joint_start:
        evaluator.enable_joint_start(env.model_cpu)
        print("[joint-start] first window optimises the initial configuration of the actuated joints jointly with its controls "
              "(knot 0 = initial position targets = initial joint positions, at rest)", flush=True)
    result = mpc.run(verbose=True)

    # Final validation: exact re-evaluation of the executed step-level controls
    # from the initial state with the original full-horizon evaluator.
    mjwp_module._copy_state(snapshot, env.data_wp)
    if args.joint_start:
        if evaluator.joint_start_q is None:
            raise RuntimeError("--joint-start: the first window was never committed")
        evaluator._set_start(torch.as_tensor(evaluator.joint_start_q, dtype=torch.float32, device=config.device))
    full_controls = full_reference.clone()
    full_controls[:, real_ids] += torch.as_tensor(
        result["committed_steps"], dtype=torch.float32, device=config.device
    )
    j_full = full_horizon_cost(full_controls.unsqueeze(0).repeat(N_worlds, 1, 1))
    committed = result["committed_cost_world"]
    mismatch0 = float(abs(j_full[0] - committed[0]))
    print(
        f"[validation] J_full slot0={j_full[0]:.6f} (slots: min={j_full.min():.6f} "
        f"median={np.median(j_full):.6f} max={j_full.max():.6f}); "
        f"committed slot0={committed[0]:.6f} (slots: min={committed.min():.6f} "
        f"max={committed.max():.6f}); |diff| slot0={mismatch0:.3e}, "
        f"median over slots={np.median(np.abs(j_full - committed)):.3e}",
        flush=True,
    )
    total_sim_steps = int(result["sim_steps"])
    print(
        f"[mpc] closed-loop J={committed[0]:.6f} (executed trajectory; states saved) "
        f"replay J={j_full[0]:.6f} reference J0={reference_cost:.6f} "
        f"sampler_rollouts={result['sampler_rollouts']} sim_steps={total_sim_steps} "
        f"(= {total_sim_steps / T:.0f} full-length rollouts) "
        f"runtime={result['runtime_seconds']:.0f}s",
        flush=True,
    )

    # Object tracking metrics (Epos m, Erot rad, success = Epos < 0.1 and Erot < 0.5) on the executed world-0 states.
    # The state after sim step s is compared with reference row s of ref_full (= ref[s+1]), the frame the reward at
    # step s uses.  The object block layout (free joint quat vs 6-dof euler) follows from nq_obj.
    from grasp.evaluation.tracking_error import compute_object_tracking_error
    qpos_exec = np.asarray(result["executed_qpos"], dtype=np.float64)
    qpos_ref_np = evaluator.ref_full[0].detach().cpu().numpy().astype(np.float64)
    if qpos_exec.shape != qpos_ref_np.shape:      # never expected; trim instead of losing the run at its very end
        print(f"[metrics] WARNING executed {qpos_exec.shape} vs reference {qpos_ref_np.shape}; trimming", flush=True)
        L_min = min(len(qpos_exec), len(qpos_ref_np)); qpos_exec, qpos_ref_np = qpos_exec[:L_min], qpos_ref_np[:L_min]
    act_layout = {("bimanual", 12): True, ("bimanual", 14): False, ("right", 6): True, ("right", 7): False,
                  ("left", 6): True, ("left", 7): False}.get((str(config.embodiment_type), int(config.nq_obj)))
    if act_layout is not None:
        tracking_err = compute_object_tracking_error(
            qpos_exec, qpos_ref_np, str(config.embodiment_type), "mjwp_act" if act_layout else "mjwp")
        tracking_metrics = {k: float(tracking_err[k]) for k in
                        ("obj_pos_err", "obj_quat_err", "pos_err_right", "pos_err_left", "quat_err_right", "quat_err_left")}
    else:
        # humanoid_object: one free-joint object = the last 7 qpos (pos, wxyz quat); per-frame position distance and
        # rotation angle, time-averaged.  Embodiments without an object get no metrics.
        from scipy.spatial.transform import Rotation as _R
        if str(config.embodiment_type) == "humanoid_object":
            pe = np.linalg.norm(qpos_exec[:, -7:-4] - qpos_ref_np[:, -7:-4], axis=1)
            qe, qr = qpos_exec[:, -4:][:, [1, 2, 3, 0]], qpos_ref_np[:, -4:][:, [1, 2, 3, 0]]
            re = (_R.from_quat(qe) * _R.from_quat(qr).inv()).magnitude()
        else:
            pe = re = np.full(len(qpos_exec), np.nan)
        tracking_err = {"pos_err_frames": pe, "quat_err_frames": re}
        tracking_metrics = {"obj_pos_err": float(np.mean(pe)), "obj_quat_err": float(np.mean(re))}
        act_layout = False
    tracking_metrics["success"] = bool(tracking_metrics["obj_pos_err"] < 0.1 and tracking_metrics["obj_quat_err"] < 0.5)
    print(f"[metrics] Epos={tracking_metrics['obj_pos_err']:.4f} m Erot={tracking_metrics['obj_quat_err']:.4f} rad "
          f"success={tracking_metrics['success']} (object layout {'6-dof euler' if act_layout else 'free quat'})", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{config.dataset_name}_{config.robot_type}_{config.embodiment_type}_{config.task}_{config.data_id}_seed{args.seed}"
    np.savez_compressed(
        args.output_dir / f"{tag}.npz",
        committed_steps=result["committed_steps"],
        committed_knots=result["committed_knots"],
        committed_cost_world=committed,
        j_full_slots=np.asarray(j_full),
        executed_qpos=result["executed_qpos"],
        executed_qvel=result["executed_qvel"],
        pos_err_frames=np.asarray(tracking_err["pos_err_frames"]),
        quat_err_frames=np.asarray(tracking_err["quat_err_frames"]),
    )
    summary = {
        "config": {
            "dataset_name": config.dataset_name, "robot_type": config.robot_type,
            "embodiment_type": config.embodiment_type, "task": config.task,
            "data_id": int(config.data_id),
        },
        "args": {
            k: (
                str(v) if isinstance(v, Path)
                else [str(x) for x in v] if isinstance(v, list)
                else v
            )
            for k, v in vars(args).items()
        },
        "reference_J0": reference_cost,
        "tracking_metrics": tracking_metrics,
        "J_full_slot0": float(j_full[0]),
        "J_full_slots": {"min": float(j_full.min()), "median": float(np.median(j_full)), "max": float(j_full.max())},
        "committed_cost_slot0": float(committed[0]),
        "mismatch_slot0": mismatch0,
        "sampler_rollouts": int(result["sampler_rollouts"]),
        "sim_steps": total_sim_steps,
        "runtime_seconds": result["runtime_seconds"],
        "windows": result["windows"],
        "joint_start": None if not args.joint_start else {
            "initial_joint_q": [float(v) for v in evaluator.joint_start_q],
            "offset_from_reference": [float(v) for v in (evaluator.joint_start_q - _to_numpy(full_reference, np.float64)[0, real_ids])],
        },
        "purity": {
            "batch_repeat_q50": float(np.quantile(dev, 0.5)),
            "batch_repeat_q90": float(np.quantile(dev, 0.9)),
            "batch_repeat_max": float(dev.max()),
            "identical_slot_spread": float(np.ptp(c3)),
        },
    }
    with (args.output_dir / f"{tag}.json").open("w") as handle:
        json.dump(
            summary, handle, indent=2,
            default=lambda o: str(o) if isinstance(o, Path) else float(o),
        )
    print(f"[saved] {args.output_dir / tag}.json")


if __name__ == "__main__":
    main()
