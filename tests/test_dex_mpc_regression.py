import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from grasp.mpc.grasp_mpc import GraspMPC
from tests.dex_case import make_case, summarize


class DexMPCRegressionTest(unittest.TestCase):
    def test_cleaned_mpc_matches_frozen_migration(self):
        golden = json.loads((Path(__file__).parent / 'fixtures/dex_mpc.json').read_text())
        for name, pin in [('warm_library', False), ('pinned_joint_start', True)]:
            with self.subTest(case=name):
                actual = summarize(make_case(GraspMPC, pin=pin).run(verbose=False))
                expected = golden['cases'][name]
                for key in ['committed_steps', 'committed_knots', 'committed_cost_world',
                            'executed_qpos', 'executed_qvel']:
                    np.testing.assert_allclose(actual[key], expected[key], rtol=2e-5, atol=2e-7)
                self.assertEqual(actual['sampler_rollouts'], expected['sampler_rollouts'])
                self.assertEqual(len(actual['windows']), len(expected['windows']))
                for new, old in zip(actual['windows'], expected['windows'], strict=True):
                    for key in ['k0', 'K', 'D', 'chosen_rank', 'library_mode', 'executed_set']:
                        self.assertEqual(new[key], old[key], key)
                    np.testing.assert_allclose(new['best_plan'], old['best_plan'], rtol=2e-5, atol=2e-7)
                    self.assertAlmostEqual(new['best_cost'], old['best_cost'], places=7)

    def test_trace_works_after_package_move_and_does_not_change_execution(self):
        ordinary = make_case(GraspMPC, pin=True).run(verbose=False)
        with tempfile.TemporaryDirectory() as folder:
            traced = make_case(GraspMPC, pin=True, trace_dir=folder).run(verbose=False)
            np.testing.assert_array_equal(traced['committed_steps'], ordinary['committed_steps'])
            context = next(Path(folder).glob('window_001/*run_context.npz'))
            with np.load(context, allow_pickle=False) as data:
                for source in ['mpc/grasp_mpc.py', 'optim/sampler.py', 'optim/trace.py', 'mpc/run.py']:
                    self.assertIn('sources/' + source, data.files)


if __name__ == '__main__':
    unittest.main()
