"""Focused geometric checks; run with the Panda environment's Python unittest."""

import unittest

import numpy as np
from yourdfpy import URDF

from panda_replay_kinematics import PandaReplayKinematics, _rotation_vector


class PandaReplayKinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinematics = PandaReplayKinematics()

    def test_forward_matches_independent_urdf_reader(self):
        kinematics = self.kinematics
        urdf = URDF.load(str(kinematics.urdf_path), load_meshes=False, build_scene_graph=True)
        generator = np.random.default_rng(318)
        for q in generator.uniform(kinematics.joint_limits[:, 0], kinematics.joint_limits[:, 1], (20, 7)):
            urdf.update_cfg(dict(zip(kinematics.joint_names, q)))
            expected = urdf.get_transform("panda_hand", "panda_link0")
            position, rotation, tip = kinematics.forward(q)
            np.testing.assert_allclose(position, expected[:3, 3], atol=1e-12)
            np.testing.assert_allclose(rotation, expected[:3, :3], atol=1e-12)
            np.testing.assert_allclose(tip, expected[:3, 3] + 0.225 * expected[:3, 2], atol=1e-12)

    def test_analytic_jacobian_matches_pose_finite_differences(self):
        kinematics = self.kinematics
        q = np.array([0.4, -0.3, 0.2, -1.9, 0.5, 2.0, -0.3])
        _, rotation, tip = kinematics.forward(q)
        jacobian = np.empty((6, 7))
        epsilon = 1e-7
        for index in range(7):
            changed = q.copy()
            changed[index] += epsilon
            _, changed_rotation, changed_tip = kinematics.forward(changed)
            jacobian[:3, index] = (changed_tip - tip) / epsilon
            jacobian[3:, index] = _rotation_vector(changed_rotation @ rotation.T) / epsilon
        np.testing.assert_allclose(kinematics.jacobian(q), jacobian, atol=1e-7)

    def test_paper_grid_is_reachable_with_downward_brush(self):
        kinematics = self.kinematics
        targets = np.array([[x, y, z] for z in (-0.01, 0.002, 0.045)
                            for x in np.linspace(0.46, 0.64, 7)
                            for y in np.linspace(-0.09, 0.09, 7)])
        result = kinematics.solve_path(targets)
        self.assertTrue(result.success)
        self.assertLessEqual(float(result.position_error_m.max()), 1e-8)
        self.assertLessEqual(float(result.orientation_error_rad.max()), 1e-7)
        self.assertTrue(np.all(result.q > kinematics.joint_limits[:, 0]))
        self.assertTrue(np.all(result.q < kinematics.joint_limits[:, 1]))
        self.assertFalse(result.q.flags.writeable)
        self.assertFalse(result.converged.flags.writeable)

    def test_unreachable_target_reports_failure(self):
        solution = self.kinematics.solve_pose(np.array([4.0, 0.0, 0.0]), max_iters=30)
        self.assertFalse(solution.converged)
        self.assertGreater(solution.position_error_m, 1.0)
        self.assertTrue(np.all(np.isfinite(solution.q)))

    def test_invalid_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):
            self.kinematics.forward(np.zeros(8))
        with self.assertRaises(ValueError):
            self.kinematics.solve_pose(np.array([0.55, np.nan, 0.0]))
        with self.assertRaises(ValueError):
            self.kinematics.solve_pose(np.array([0.55, 0.0, 0.0]), target_rotation=np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()
