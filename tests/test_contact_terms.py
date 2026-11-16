import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from grasp.mpc.capacity_retry import next_capacity

REPO = Path(__file__).parents[1]


def _paper():
    spec = importlib.util.spec_from_file_location("paper", REPO / "experiments/paper.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CapacityRetryTest(unittest.TestCase):
    def test_runtime_overflow_grows_the_overflowing_buffer(self):
        msg = ("RuntimeError: MJWarp capacity exceeded: broadphase pairs 207872 > naconmax 122880. "
               "Physics results are invalid.\n")
        self.assertEqual(next_capacity(msg, 1024, {}), {"nconmax_per_env": 256})
        msg = "RuntimeError: MJWarp capacity exceeded: constraints 1038 > njmax 1024. Physics results are invalid.\n"
        self.assertEqual(next_capacity(msg, 1024, {"nconmax_per_env": 256}), {"njmax_per_env": 2048})

    def test_initial_state_overflow_is_recognised(self):
        msg = "Traceback ...\nValueError: nconmax overflow (nconmax must be >= 148)\n"
        self.assertEqual(next_capacity(msg, 1024, {}), {"nconmax_per_env": 256})
        self.assertEqual(next_capacity(msg, 1024, {"nconmax_per_env": 256}), {"nconmax_per_env": 512})
        msg = "ValueError: njmax overflow (njmax must be >= 1088)\n"
        self.assertEqual(next_capacity(msg, 1024, {}), {"njmax_per_env": 2048})

    def test_other_failures_are_not_retried(self):
        msg = "RuntimeError: could not initialize a finite population after 16 resampling rounds\n"
        self.assertEqual(next_capacity(msg, 1024, {}), {})


class GateTest(unittest.TestCase):
    def test_gate_rows_follow_the_reference_rate(self):
        from grasp.objective.contact_terms import load_gates

        gin = np.zeros((5, 2), bool)
        gin[2:, 0] = True
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "contact_gates.npz"
            np.savez(path, gate_inhand=gin, gate_rest=np.zeros((5, 2), bool))
            steps_in, steps_rest = load_gates(path, 5, 16, 0.005, 0.02)
            # step s uses reference row round(s / 4)
            self.assertEqual(steps_in[:, 0].tolist(), [False] * 6 + [True] * 10)
            self.assertFalse(steps_rest.any())
            with self.assertRaises(ValueError):
                load_gates(path, 6, 16, 0.005, 0.02)

    def test_overlapping_gates_are_rejected(self):
        from grasp.objective.contact_terms import load_gates

        g = np.ones((3, 2), bool)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "contact_gates.npz"
            np.savez(path, gate_inhand=g, gate_rest=g)
            with self.assertRaises(ValueError):
                load_gates(path, 3, 4, 0.005, 0.02)


class StandardSettingsTest(unittest.TestCase):
    def test_oakink2_uses_the_contact_terms_and_hot3d_does_not(self):
        paper = _paper()
        with tempfile.TemporaryDirectory() as d:
            for name, robot, emb, task in [("oakink", "sharpa_bimanual", "bimanual", "oakbi_t"),
                                           ("hot3d", "sharpa", "bimanual", "hot3d_t")]:
                task_dir = Path(d) / "processed" / name / robot / emb / task
                (task_dir / "0").mkdir(parents=True)
                np.savez(task_dir / "0/trajectory_kinematic.npz", qpos=np.zeros((51, 3)))
                (task_dir / "task_info.json").write_text('{"ref_dt": 0.02}')
            oak = paper.build("oakink2", "oakbi_t", dataset_dir=Path(d))
            hot = paper.build("hot3d", "hot3d_t", dataset_dir=Path(d))
        value = lambda args, flag: args[args.index(flag) + 1]
        self.assertEqual(value(oak, "--num-particles"), "1024")
        self.assertEqual(value(oak, "--hand-penetration-penalty"), "3000")
        self.assertEqual(value(oak, "--penetration-margin"), "0.005")
        self.assertEqual(value(oak, "--contact-mismatch-penalty"), "0.5")
        self.assertEqual(value(hot, "--num-particles"), "2000")
        self.assertNotIn("--hand-penetration-penalty", hot)
        self.assertNotIn("--contact-mismatch-penalty", hot)


if __name__ == "__main__":
    unittest.main()
