from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from grasp.mpc.grasp_mpc import GraspWindowEvaluator


class JointStartTest(unittest.TestCase):
    def test_actuator_order_joint_limits_and_object_state(self):
        model = mujoco.MjModel.from_xml_string('''
        <mujoco>
          <worldbody>
            <body name="object"><freejoint/><geom type="sphere" size=".02" mass="1"/></body>
            <body name="wrist"><joint name="wrist" type="slide" range="-.1 .1"/>
              <geom type="sphere" size=".02" mass="1"/>
              <body name="finger"><joint name="finger" range="-30 30"/>
                <geom type="sphere" size=".01" mass=".1"/>
              </body>
            </body>
          </worldbody>
          <actuator><position joint="finger"/><position joint="wrist"/></actuator>
        </mujoco>''')
        ev = object.__new__(GraspWindowEvaluator)
        ev.config = SimpleNamespace(device='cpu')
        ev.real_ids = [1, 0]
        ev.full_reference = torch.tensor([[.2, .02], [.2, .02]])
        ev.enable_joint_start(model)
        initial = ev.joint_start_config(np.array([[.3, -.9], [-.4, .9]]))
        np.testing.assert_allclose(initial, [[.1, -np.pi/6], [-.1, np.pi/6]], atol=1e-7)
        qpos = torch.arange(2 * model.nq, dtype=torch.float32).reshape(2, model.nq)
        qvel = torch.ones((2, model.nv))
        object_positions = qpos[:, :7].clone()
        ev.env = SimpleNamespace(device='cpu', data_wp=SimpleNamespace(qpos=qpos, qvel=qvel))
        with patch('warp.to_torch', side_effect=lambda x: x), patch('warp.ScopedDevice', return_value=nullcontext()):
            ev._set_start(initial)
        np.testing.assert_array_equal(qpos[:, :7], object_positions)
        np.testing.assert_allclose(qpos[:, 7:], initial)
        np.testing.assert_array_equal(qvel[:, :6], 1.)
        np.testing.assert_array_equal(qvel[:, 6:], 0.)

    def test_pinned_controls_keep_virtual_actuator_reference(self):
        ev = object.__new__(GraspWindowEvaluator)
        ev.config = SimpleNamespace(device='cpu')
        ev.N, ev.n, ev.T = 2, 2, 4
        ev.real_ids = [1, 0]
        ev.full_reference = torch.tensor([[.2, .02, .9]]).repeat(4, 1)
        ev.pin_first_knot = True
        ev.last_exec_delta = np.array([.01, -.03], np.float32)
        candidates = np.zeros((2, 2, 2), np.float32)
        candidates[:, 0] = 10
        before = candidates.copy()
        controls = ev.controls_for_window(candidates, 0).numpy()
        np.testing.assert_array_equal(candidates, before)
        np.testing.assert_array_equal(controls[:, :, 2], np.float32(.9))
        np.testing.assert_allclose(controls[:, 0, :2], [[.17, .03], [.17, .03]], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
