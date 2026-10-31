"""Small deterministic rollout for MPC orchestration regression (not physics)."""
from types import SimpleNamespace

import numpy as np
import torch

from grasp.mpc.grasp_mpc import local_interp


class DeterministicEvaluator:
    N, n, T, m = 16, 2, 16, 3

    def __init__(self, *, pin=False, joint_start=False):
        self.sim_steps = 0
        self.state = np.zeros(self.m, np.float32)
        self.full_reference = torch.zeros((self.T, self.m))
        self.env = SimpleNamespace(data_wp=SimpleNamespace())
        self.pin_first_knot = pin
        self.last_exec_delta = None
        self.joint_start_pending = joint_start
        self.joint_start_q = None

    def simulate(self, knots, k0):
        knots = np.asarray(knots, np.float32).copy()
        if self.pin_first_knot and self.last_exec_delta is not None:
            knots[:, 0] = self.last_exec_delta
        controls = local_interp(torch.from_numpy(knots), self.n).numpy()
        state = np.broadcast_to(self.state, (len(knots), self.m)).copy()
        if self.joint_start_pending and k0 == 0:
            state = np.clip(knots[:, 0], -.2, .2)
        states, rewards = [], []
        for j in range(controls.shape[1]):
            state = .8 * state + .2 * controls[:, j]
            target = np.array([.04, -.08, .15]) + .01 * np.sin(k0 * self.n + j)
            rewards.append(-np.sum((state - target) ** 2, axis=1))
            states.append(state.copy())
        states = np.stack(states, axis=1)
        return controls, states, np.stack(rewards, axis=1), rewards[-1]

    def rollout_window(self, knots, k0):
        _, _, rewards, terminal = self.simulate(knots, k0)
        self.sim_steps += rewards.size
        return rewards, terminal

    def window_cost(self, knots, k0, phi):
        rewards, terminal = self.rollout_window(knots, k0)
        return -(rewards.sum(1) + phi * terminal) / self.T

    def commit(self, best, k0, count):
        controls, states, rewards, _ = self.simulate(
            np.repeat(best[None], self.N, axis=0), k0,
        )
        steps = count * self.n
        previous = np.concatenate([self.state[None], states[0, :steps - 1]])
        self.last_executed_states = (
            states[0, :steps].copy(), states[0, :steps] - previous,
        )
        self.state = states[0, steps - 1].copy()
        self.last_exec_delta = controls[0, steps - 1].copy()
        if self.joint_start_pending:
            self.joint_start_q = np.clip(best[0], -.2, .2)
            self.joint_start_pending = False
        self.sim_steps += self.N * steps
        return controls[0, :steps].copy(), rewards[:, :steps].copy()


def make_case(mpc_class, *, legacy=False, pin=False, trace_dir=None):
    evaluator = DeterministicEvaluator(pin=pin, joint_start=pin)
    kwargs = dict(
        total_knots=8, window_knots=4, commit_knots=2, control_dim=3,
        prior_rho=.9, prior_sigma=.0044, beta=1 / .06, seed=7,
        prior_channel_scale=np.array([1., 10., 10.], np.float32),
        library_reuse_period=2, library_local_improve=True,
        improve_scale=.3, improve_parents=4, trace_dir=trace_dir,
        sampler_kwargs=dict(
            kernel='surrogate_mala', mala_preconditioner='prior_full',
            surrogate_fit_device='cpu', surrogate_rank=2,
            surrogate_crossfit_shrink=True, step_size=.0026,
            num_mcmc_steps=2, max_iterations=16, target_ess=.9,
            adapt_step_target=.05, adapt_step_rate=3., adapt_step_min_scale=.1,
        ),
    )
    if legacy:
        kwargs['improve_mode'] = 'twogen'
    mpc = mpc_class(evaluator, **kwargs)
    mpc.warm_start_plan = True
    mpc.prior_window_ramp = 4.
    mpc.robust_select_k = 4
    return mpc


def summarize(result):
    return {
        name: np.asarray(result[name]).tolist()
        for name in ['committed_steps', 'committed_knots',
                     'committed_cost_world', 'executed_qpos', 'executed_qvel']
    } | {
        'windows': [{
            'k0': w['k0'], 'K': w['K'], 'D': w['D'],
            'best_cost': w['best_cost'], 'best_plan': w['best_plan'],
            'chosen_rank': w['robust_select']['chosen_rank'],
            'library_mode': (w['library_reuse'] or {}).get('mode'),
            'executed_set': (w['library_reuse'] or {}).get('executed_set'),
        } for w in result['windows']],
        'sampler_rollouts': result['sampler_rollouts'],
    }
