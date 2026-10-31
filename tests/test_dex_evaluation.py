import json
from pathlib import Path
import unittest

import numpy as np

from grasp.evaluation.tracking_error import compute_object_tracking_error


class DexEvaluationTest(unittest.TestCase):
    def test_saved_short_oakink2_result_and_frame_alignment(self):
        folder = Path(__file__).parent / 'fixtures'
        meta = json.loads((folder / 'oakink2_sharpa_short.json').read_text())
        with np.load(folder / 'oakink2_sharpa_short.npz', allow_pickle=False) as fixture:
            result = compute_object_tracking_error(
                fixture['executed_object_qpos'], fixture['reference_object_qpos'],
                meta['embodiment'], meta['data_type'],
            )
            for key in ['pos_err_frames', 'quat_err_frames']:
                np.testing.assert_allclose(result[key], fixture['expected_' + key], atol=1e-12)
        for key, expected in meta['metrics'].items():
            if key != 'success':
                self.assertAlmostEqual(result[key], expected, places=12)
        self.assertEqual(result['obj_pos_err'] < .1 and result['obj_quat_err'] < .5,
                         meta['metrics']['success'])

    def test_bimanual_static_object_is_excluded_from_combined_error(self):
        ref = np.zeros((3, 14)); ref[:, [3, 10]] = 1
        ref[:, 0] = [0., .01, .02]
        executed = ref.copy(); executed[:, 0] += .03; executed[:, 7] += .5
        result = compute_object_tracking_error(executed, ref, 'bimanual', 'mjwp')
        self.assertAlmostEqual(result['obj_pos_err'], .03)
        self.assertAlmostEqual(result['pos_err_left'], .5)
        # With both objects moving, the combined series averages their errors.
        ref[:, 7] = [0., .02, .04]
        executed = ref.copy(); executed[:, 0] += .03; executed[:, 7] += .5
        result = compute_object_tracking_error(executed, ref, 'bimanual', 'mjwp')
        self.assertAlmostEqual(result['obj_pos_err'], .265)

    def test_single_hand_euler_and_free_joint_layouts_agree(self):
        angle = .3
        euler = np.zeros((3, 6)); euler[:, 0] = .02; euler[:, 5] = angle
        ref_euler = np.zeros_like(euler)
        quat = np.zeros((3, 7)); quat[:, 0] = .02
        quat[:, 3] = np.cos(angle / 2); quat[:, 6] = np.sin(angle / 2)
        ref_quat = np.zeros_like(quat); ref_quat[:, 3] = 1
        for hand in ['left', 'right']:
            with self.subTest(hand=hand):
                a = compute_object_tracking_error(euler, ref_euler, hand, 'mjwp_act')
                b = compute_object_tracking_error(quat, ref_quat, hand, 'mjwp')
                self.assertAlmostEqual(a['obj_pos_err'], .02)
                self.assertAlmostEqual(a['obj_quat_err'], angle)
                for key in a:
                    np.testing.assert_allclose(a[key], b[key], atol=1e-12)
                quat_sign_flip = quat.copy(); quat_sign_flip[:, 3:] *= -1
                c = compute_object_tracking_error(quat_sign_flip, ref_quat, hand, 'mjwp')
                self.assertAlmostEqual(c['obj_quat_err'], angle)


if __name__ == '__main__':
    unittest.main()
