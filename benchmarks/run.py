#!/usr/bin/env python3
"""Trajectory-optimization benchmarks: acrobot swing-up and contact-mode pusher-slider (horizon 200).

Methods (Table "Smooth and nonsmooth trajectory optimization benchmarks"):
    grasp      GRASP (GraspSampler): surrogate MALA on the acrobot, surrogate NUTS on the pusher-slider.
    rwm        random-walk Metropolis tempered SMC (GraspSampler, kernel "rwm"), same move budget per stage.
    tsmc_grad  gradient-based tempered SMC with simulator derivatives (NUTS); on the pusher-slider it differentiates
               a straight-through Gumbel-Softmax relaxation of the contact-edge choice.
    mppi       independent MPPI refinement of every initial particle.

All methods start from the same AR(1) prior draws per seed, and every reported cost is a rollout of the final
controls under the true (hard-contact) dynamics.

    python -m benchmarks.run --task acrobot --method grasp --seeds 0-6
    python -m benchmarks.run --task pusher --method grasp --seeds 0-10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from grasp.optim.prior_ar1 import AR1Prior, WindowAR1Prior
from grasp.optim.sampler import GraspSampler
from benchmarks.systems import AcrobotRK4System
from benchmarks.systems.pusher_slider import (
    PusherSliderQSContactModeCostWeights,
    PusherSliderQSContactModeParams,
    PusherSliderQSContactModeSystem,
)

NUM_PARTICLES = 100
TARGET_ESS = 0.8
MAX_STAGES = 200
MOVES_PER_STAGE = 99           # rollout-only samplers: 3 moves x 33 rollouts, the zeroth-order MALA budget per stage
PUSHER_SUCCESS = (0.10, 0.5)   # a pusher run is solved when its best final box pose is within 10 cm and 0.5 rad of the goal
TASKS = {
    "acrobot": dict(
        horizon=200, beta=10.0, rho=0.9, sigma=2.0,
        grasp=dict(kernel="surrogate_mala", step_size=0.2, adapt_step_min_scale=0.01, surrogate_rank=32,
                   surrogate_window=3.0, mala_preconditioner="prior_full"),
        rwm=dict(step_size=0.3),
        tsmc_grad=dict(step_size=0.02),
        mppi=dict(noise_sigma=0.5),
    ),
    "pusher": dict(
        horizon=200, beta=5.0, rho=0.95, sigma=0.8,
        grasp=dict(kernel="surrogate_nuts", step_size=0.0031, adapt_step_min_scale=0.1, surrogate_rank=16,
                   surrogate_window=1.0, nuts_mass="prior_full", nuts_max_doublings=8),
        rwm=dict(step_size=0.1),
        tsmc_grad=dict(step_size=0.01),
        mppi=dict(noise_sigma=0.4),
    ),
}


def build_system(task: str, edge_selection: str = "st_argmax"):
    if task == "acrobot":
        return AcrobotRK4System(dt=0.025)
    return PusherSliderQSContactModeSystem(
        params=PusherSliderQSContactModeParams(
            dt=0.01, u_max=1.0, pusher_radius=0.04, box_w=0.4, box_h=0.2, s_max_frac=0.8, max_push_angle_deg=60.0,
            edge_selection=edge_selection, edge_temp=0.15, gumbel_tau=0.2, gumbel_seed=0),
        cost_weights=PusherSliderQSContactModeCostWeights(
            goal_x=0.0, goal_y=0.3, goal_theta=float(np.pi), w_pos=2.0, w_theta=0.05, w_v=0.05, w_pos_T=100.0,
            w_theta_T=50.0, w_mode_switch=0.05, w_s_switch=0.05, w_v_switch=0.05),
        box_init=(0.0, -0.5, 0.5 * np.pi),
    )


def trajectory_cost(system, horizon: int):
    """Scalar rollout cost of a flat control vector, from the system's default initial state."""
    x0 = system.default_initial_state
    m = int(getattr(system, "control_dim", 1))

    def cost(position):
        return system.trajectory_cost_from_raw(x0, position if m == 1 else jnp.reshape(position, (horizon, m)))

    return cost


def run_mppi(*, key, cost_fn, initial_particles, beta, num_iterations, num_samples, noise_sigma):
    """Independent MPPI refinement of every initial particle; returns the final particles."""
    particles = jnp.asarray(initial_particles)
    count, dimension = particles.shape

    def one_iteration(current, rng):
        noise = float(noise_sigma) * jax.random.normal(rng, (count, int(num_samples), dimension), dtype=current.dtype)
        costs = jax.vmap(cost_fn)((current[:, None, :] + noise).reshape((-1, dimension))).reshape(
            (count, int(num_samples)))
        weights = jax.nn.softmax(-float(beta) * costs, axis=1)
        return current + jnp.sum(weights[:, :, None] * noise, axis=1), None

    final, _ = jax.jit(lambda start, keys: jax.lax.scan(one_iteration, start, keys))(
        particles, jax.random.split(key, int(num_iterations)))
    return final


def run_seed(task: str, method: str, seed: int, args) -> tuple[dict, np.ndarray, np.ndarray]:
    cfg = TASKS[task]
    H, beta = cfg["horizon"], cfg["beta"]
    system = build_system(task)
    m = int(getattr(system, "control_dim", 1))
    cost_fn = trajectory_cost(system, H)
    cost_batch = jax.jit(jax.vmap(cost_fn))

    def batch_cost(flat: np.ndarray) -> np.ndarray:
        return np.asarray(cost_batch(jnp.asarray(flat, dtype=jnp.float32)), dtype=np.float64)

    prior = AR1Prior(rho=cfg["rho"], sigma=cfg["sigma"])
    master_key = jax.random.PRNGKey(seed)
    init_keys = jax.random.split(jax.random.fold_in(master_key, 1), NUM_PARTICLES)
    initial = np.asarray(jax.vmap(lambda k: prior.sample(k, H, control_dim=m))(init_keys),
                         dtype=np.float64).reshape(NUM_PARTICLES, H * m)

    start = time.perf_counter()
    info = {}
    if method in ("grasp", "rwm"):
        window_prior = WindowAR1Prior(control_dim=m, horizon=H, rho=cfg["rho"], sigma=cfg["sigma"])
        sampler = GraspSampler(batch_cost, window_prior, horizon=H, control_dim=m, beta=beta)
        common = dict(seed=seed, num_particles=NUM_PARTICLES, target_ess=TARGET_ESS, num_mcmc_steps=MOVES_PER_STAGE,
                      max_iterations=MAX_STAGES, initial_particles=initial, verbose=args.verbose)
        if method == "grasp":
            result = sampler.run(**common, **cfg["grasp"], adapt_step_size=True, adapt_step_target=0.05,
                                 adapt_step_rate=3.0, surrogate_ridge=1e-3, fit_max_points=60_000,
                                 surrogate_fit_device=args.fit_device, surrogate_crossfit_shrink=True)
        else:
            result = sampler.run(**common, kernel="rwm", **cfg["rwm"])
        samples = np.asarray(result.samples, dtype=np.float64)
        info = dict(completed=bool(result.info["completed"]), stages=int(result.info["num_iterations"]),
                    rollouts=int(result.info["total_rollout_calls"]),
                    final_accept=float(result.info["acceptance_rates"][-1]))
    elif method == "tsmc_grad":
        from benchmarks.baselines.tempered_smc import TemperedSMCSampler
        relaxed = build_system(task, "st_gumbel") if task == "pusher" else system
        result = TemperedSMCSampler(relaxed, prior, beta=beta).run(
            key=jax.random.fold_in(master_key, 10), H=H, kernel="nuts", target_ess=TARGET_ESS, num_mcmc_steps=3,
            step_size=cfg["tsmc_grad"]["step_size"], max_num_doublings=8, max_iterations=MAX_STAGES,
            initial_particles=jnp.asarray(initial if m > 1 else initial.reshape(NUM_PARTICLES, H)))
        samples = np.asarray(result.samples, dtype=np.float64).reshape(NUM_PARTICLES, H * m)
    else:
        samples = np.asarray(run_mppi(key=jax.random.fold_in(master_key, 40), cost_fn=cost_fn,
                                      initial_particles=initial, beta=beta, num_iterations=args.mppi_iterations,
                                      num_samples=64, noise_sigma=cfg["mppi"]["noise_sigma"]), dtype=np.float64)
    seconds = time.perf_counter() - start

    costs = batch_cost(samples)                                  # rollout re-evaluation under the true dynamics
    best = int(np.argmin(costs))
    row = dict(seed=seed, best=float(costs[best]), median=float(np.median(costs)), seconds=seconds, **info)
    if task == "pusher":
        U = jnp.asarray(samples[best], dtype=jnp.float32).reshape(H, m)
        xT = np.asarray(system.rollout(system.default_initial_state, system.clamp_control(U)))[-1]
        W = system._W
        row.update(final_pos_err_m=float(np.hypot(xT[0] - W.goal_x, xT[1] - W.goal_y)),
                   final_theta_err_rad=float(abs((xT[2] - W.goal_theta + np.pi) % (2 * np.pi) - np.pi)))
        row["success"] = row["final_pos_err_m"] < PUSHER_SUCCESS[0] and row["final_theta_err_rad"] < PUSHER_SUCCESS[1]
    return row, samples, costs


def parse_seeds(text: str) -> list[int]:
    seeds = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seeds += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--method", choices=("grasp", "rwm", "tsmc_grad", "mppi"), required=True)
    parser.add_argument("--seeds", default="0-6", help="e.g. 0-6 (acrobot) or 0-10 (pusher)")
    parser.add_argument("--mppi-iterations", type=int, default=15)
    parser.add_argument("--fit-device", default="cuda", help="torch device for the GRASP surrogate fit")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    out = args.output_dir / args.task / args.method
    out.mkdir(parents=True, exist_ok=True)
    print(f"[benchmark] task={args.task} method={args.method} backend={jax.default_backend()}", flush=True)
    for seed in parse_seeds(args.seeds):
        row, samples, costs = run_seed(args.task, args.method, seed, args)
        print(json.dumps(row), flush=True)
        (out / f"seed{seed}.json").write_text(json.dumps(row, indent=2))
        np.savez_compressed(out / f"seed{seed}.npz", samples=samples.astype(np.float32), costs=costs)
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("seed*.json"))]
    bests = np.array([r["best"] for r in rows])
    summary = dict(task=args.task, method=args.method, seeds=len(rows), best_mean=float(bests.mean()),
                   best_std=float(bests.std()), seconds_mean=float(np.mean([r["seconds"] for r in rows])))
    if args.task == "pusher":
        summary["success"] = sum(r["final_pos_err_m"] < PUSHER_SUCCESS[0] and r["final_theta_err_rad"] < PUSHER_SUCCESS[1]
                                 for r in rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[summary]", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
