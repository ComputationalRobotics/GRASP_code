import ast
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest

from grasp.mpc.run import parse_args


class DexEntrypointTest(unittest.TestCase):
    def test_paper_options_remain_supported(self):
        args = parse_args([
            '--warm-start-plan', '--joint-start', '--pin-first-knot',
            '--kernel', 'surrogate_mala', '--num-particles', '2000',
            '--window-knots', '15', '--commit-knots', '2', '--tau', '0.06',
            '--prior-rho', '0.9', '--prior-sigma', '0.0044',
            '--finger-channel-scale', '10', '--prior-window-ramp', '4',
            '--library-reuse-period', '2', '--library-local-improve',
            '--improve-scale', '0.3', '--improve-parents', '50',
            '--robust-select-k', '50', '--mala-preconditioner', 'prior_full',
            '--surrogate-rank', '2', '--surrogate-ridge', '0.001',
            '--surrogate-crossfit-shrink', '--step-size', '0.0026',
            '--adaptive-step-target', '0.05', '--adaptive-step-rate', '3',
            '--adaptive-step-min-scale', '0.1',
        ])
        self.assertTrue(args.joint_start and args.warm_start_plan and args.library_local_improve)
        self.assertEqual(args.robust_select_k, 50)
        self.assertEqual(args.finger_channel_scale, 10)

    def test_removed_experiment_flags_fail_instead_of_silently_changing_behavior(self):
        for argv in [
            ['--kernel', 'mixture_field'], ['--particle-bridge'],
            ['--daid-objective'], ['--split-knots', '3'],
            ['--execute-mean', '.1'], ['--improve-mode', 'guided'],
        ]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    parse_args(argv)
                self.assertEqual(caught.exception.code, 2)

    def test_every_runtime_argument_is_defined(self):
        source = Path(__file__).parents[1] / 'grasp/mpc/run.py'
        tree = ast.parse(source.read_text())
        attrs = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                 and isinstance(n.value, ast.Name) and n.value.id == 'args']
        loaded = {n.attr for n in attrs if isinstance(n.ctx, ast.Load)}
        assigned = {n.attr for n in attrs if isinstance(n.ctx, ast.Store)}
        self.assertEqual(loaded - set(vars(parse_args([]))) - assigned, set())

    def test_help_does_not_require_cuda_or_simulator_imports(self):
        code = '''
import sys
from grasp.mpc.run import parse_args
try:
    parse_args(['--help'])
except SystemExit as e:
    assert e.code == 0
assert 'torch' not in sys.modules and 'warp' not in sys.modules
'''
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
