"""Non-finite simulator results must never enter SMC weights or execution selection."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from grasp.optim.sampler import GraspSampler
from grasp.mpc.grasp_mpc import GraspMPC, GraspWindowEvaluator, _finite_order, _replica_cost_stats
from grasp.optim.prior_ar1 import AR1Prior


class NonfiniteCandidatesTest(unittest.TestCase):
    def sampler(self, cost, beta=1.):
        return GraspSampler(cost, AR1Prior(rho=.5, sigma=.3), 2, beta=beta)

    def test_initial_rejection_sampling_preserves_batch_and_frozen_coordinate(self):
        batches = []
        def cost(x):
            batches.append(x.copy())
            return np.where(x[:, 1] > 0, np.sum(x*x, axis=1), np.nan)
        initial = np.zeros((16, 2)); initial[:8, 1] = .2; initial[8:, 1] = -.2
        sampler = self.sampler(cost)
        sampler.frozen_mask = np.array([True, False])
        result = sampler.run(seed=0, initial_particles=initial, max_iterations=1,
                             num_mcmc_steps=1, verbose=False)
        self.assertEqual(result.info['initial_invalid_particles'], 8)
        self.assertGreater(result.info['initial_resample_rounds'], 0)
        self.assertTrue(all(b.shape == (16, 2) for b in batches))
        self.assertTrue(np.isfinite(result.costs).all())
        self.assertTrue((result.samples[:, 1] > 0).all())
        np.testing.assert_array_equal(result.samples[:, 0], 0.)
        self.assertTrue(np.isfinite(result.info['ess_values']).all())

    def test_invalid_proposals_rejected_even_at_zero_beta_and_excluded_from_fit(self):
        calls = 0; moves = []
        def cost(x):
            nonlocal calls
            calls += 1
            values = np.sum(x*x, axis=1)
            if calls > 1:
                values[:3] = [np.nan, np.inf, -np.inf]
            return values
        def trace(kind, **data):
            if kind == 'move':
                moves.append(data)
        sampler = self.sampler(cost, beta=0.)
        result = sampler.run(seed=2, num_particles=16, max_iterations=1,
            num_mcmc_steps=2, trace_callback=trace, verbose=False)
        self.assertEqual(result.info['invalid_evaluations'], 6)
        for move in moves:
            self.assertFalse(move['accept'][:3].any())
            self.assertTrue(np.isneginf(move['log_alpha'][:3]).all())
        np.testing.assert_array_equal(result.samples[:3], moves[-1]['particles'][:3].astype(np.float32))
        self.assertTrue(np.isfinite(result.costs).all())
        self.assertTrue(np.isfinite(sampler._last_fit_data[1]).all())

    def test_all_invalid_initial_population_stops(self):
        with self.assertRaisesRegex(RuntimeError, 'no finite candidates'):
            self.sampler(lambda x: np.full(len(x), np.nan)).run(seed=0, num_particles=8, verbose=False)

    def test_initial_resampling_is_bounded(self):
        def cost(x):
            c = np.sum(x*x, axis=1); c[0] = np.nan; return c
        with self.assertRaisesRegex(RuntimeError, '16 resampling rounds'):
            self.sampler(cost).run(seed=0, num_particles=8, verbose=False)

    def test_replica_statistics_drop_nan_and_both_infinities(self):
        mean, std = _replica_cost_stats([
            [0., np.nan, np.inf, -np.inf], [2., 3., np.nan, np.inf],
            [np.inf, -1., np.nan, -np.inf], [np.nan, np.inf, -np.inf, np.nan],
        ])
        np.testing.assert_array_equal(_finite_order(mean), [2, 0, 1])
        np.testing.assert_array_equal(mean, [0., 2.5, -1., np.inf])
        np.testing.assert_array_equal(std, [0., .5, 0., np.inf])

    def test_all_failed_replicas_remain_ineligible(self):
        mean, std = _replica_cost_stats([[np.nan, np.inf], [-np.inf, np.nan]])
        self.assertTrue(np.isposinf(mean).all())
        self.assertTrue(np.isposinf(std).all())
        with self.assertRaisesRegex(RuntimeError, 'no finite candidate'):
            _finite_order(mean)

    def test_nonfinite_executed_state_stops_before_broadcast(self):
        ev = object.__new__(GraspWindowEvaluator)
        ev.N = 2; ev.config = SimpleNamespace(device='cpu'); ev.env_params = {}
        ev.env = SimpleNamespace(data_wp=SimpleNamespace(
            qpos=torch.tensor([[float('nan')], [0.]]), qvel=torch.zeros((2, 1))))
        ev._reset_dead = lambda: None
        ev._step_and_reward = lambda ctrl, step: (torch.zeros(2), None)
        ev.mjwp = SimpleNamespace(save_env_params=lambda *a: {}, load_env_params=lambda *a: None)
        with patch('warp.to_torch', side_effect=lambda value: value):
            with self.assertRaisesRegex(RuntimeError, 'non-finite executed state/reward at simulation step 7'):
                ev._execute(torch.zeros((2, 1, 1)), 7, np.zeros((1, 1)))

    def test_light_library_and_robust_selection_survive_partial_failures(self):
        class Evaluator:
            N, n, T = 8, 1, 4
            sim_steps = 0
            full_reference = torch.zeros((4, 2))
            env = SimpleNamespace(data_wp=SimpleNamespace())

            def rollout_window(self, knots, k0):
                rewards = -np.sum((knots - .1)**2, axis=2)
                if k0:
                    rewards[-1, 0] = np.nan
                return rewards, np.zeros(self.N)

            def window_cost(self, knots, k0, phi):
                rewards, _ = self.rollout_window(knots, k0)
                return -rewards.sum(1) / self.T

            def commit(self, best, k0, count):
                assert np.isfinite(best).all()
                self.last_executed_states = (np.zeros((count, 2)), np.zeros((count, 2)))
                rewards = -np.sum((best[:count] - .1)**2, axis=1)
                return best[:count].copy(), np.repeat(rewards[None], self.N, axis=0)

        mpc = GraspMPC(Evaluator(), total_knots=4, window_knots=3,
            commit_knots=2, control_dim=2, prior_rho=.8, prior_sigma=.075,
            beta=0., sampler_kwargs=dict(num_mcmc_steps=1, max_iterations=1),
            library_reuse_period=2, seed=0)
        mpc.robust_select_k = 4
        result = mpc.run(verbose=False)
        self.assertTrue(np.isfinite(result['committed_steps']).all())
        light = result['windows'][1]
        self.assertEqual(light['library_reuse']['mode'], 'light')
        self.assertGreater(light['invalid_candidate_evaluations'], 0)
        self.assertTrue(np.isfinite(light['best_cost']))
        self.assertEqual(light['reeval_invalid_replicas'], 1)


if __name__ == '__main__':
    unittest.main()
