"""Tests for image orientation, native ink parity, and Panda replay checks.

Run with the Panda environment for geometric tests. Run again with calli_rl2
to enable comparisons against the released Writing_Brush implementation.
"""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

import callirewrite_replay as replay


def source_example(native=True):
    source = {
        "xy_uv": np.array([[.2311, .2723], [.2311, .3823], [.4311, .3823],
                           [.6511, .6823], [.7611, .6823]]),
        "radius_norm": np.full(5, .0137),
        "stroke_id": np.array([0, 0, 0, 1, 1]),
        "metadata": {"source_kind": "baseline" if native else "coarse"},
        "native": native,
    }
    if native:
        source.update(length_norm=np.full(5, .0341), heading_deg=np.array([90., 90., 0., 0., 0.]))
    return source


def _source_file(directory, source):
    path = Path(directory) / "source.npz"
    values = {key: value for key, value in source.items() if key not in ("metadata", "native")}
    np.savez(path, **values, metadata=json.dumps(source["metadata"]))
    return path


class CoordinateTests(unittest.TestCase):
    def test_image_axes_and_asymmetric_shape(self):
        # An L shaped stroke: image down, then right. Panda motion is +X, then +Y.
        uv = np.array([[.2, .2], [.2, .7], [.6, .7]])
        xy = replay.uv_to_base(uv, .18)
        np.testing.assert_allclose(np.diff(xy, axis=0), [[.09, 0.], [0., .072]], atol=1e-15)
        np.testing.assert_allclose(replay.base_to_uv(xy, .18), uv, atol=1e-15)
        np.testing.assert_allclose(replay.uv_to_base([.5, .5]), [.55, 0.])
        np.testing.assert_allclose(replay.uv_to_base([1., 0.]), [.46, .09])

    def test_image_basis_is_upright_when_viewed_from_above(self):
        center = replay.uv_to_base([.5, .5])
        image_right = np.r_[replay.uv_to_base([.6, .5])-center, 0.]
        image_up = np.r_[replay.uv_to_base([.5, .4])-center, 0.]
        image_right /= np.linalg.norm(image_right)
        image_up /= np.linalg.norm(image_up)
        np.testing.assert_allclose(image_right, [0., 1., 0.])
        np.testing.assert_allclose(image_up, [-1., 0., 0.])
        np.testing.assert_allclose(np.cross(image_right, image_up), [0., 0., 1.])

    def test_viser_internal_image_flip_is_compensated(self):
        # Viser's image mesh applies Rx(pi), confirmed in ThreeAssets.tsx.
        # Its planeGeometry texture-right is +local X and texture-up +local Y.
        w, x, y, z = replay.PAPER_WXYZ
        outer_rotation = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])
        image_rotation = outer_rotation @ np.diag([1., -1., -1.])
        np.testing.assert_allclose(image_rotation[:, 0], [0., 1., 0.], atol=1e-15)
        np.testing.assert_allclose(image_rotation[:, 1], [-1., 0., 0.], atol=1e-15)
        np.testing.assert_allclose(image_rotation[:, 2], [0., 0., 1.], atol=1e-15)

    def test_brush_heading_follows_image_to_base_mapping(self):
        uv = np.array([.5, .5])
        points = replay.footprint_points(uv, .01, .03, 90.)
        tip_direction = replay.uv_to_base(points[2])-replay.uv_to_base(uv)
        np.testing.assert_allclose(tip_direction, [-.0054, 0.], atol=1e-15)
        vertices, _, _ = replay.tuft_geometry(
            np.array([.55, 0., .22]), np.diag([1., -1., -1.]),
            np.array([.55, 0., -.005]), .003, 90., .006, True)
        # At 90 degrees the tip points toward -X; the illustrated drag spine
        # bends behind it toward +X before returning to the contact endpoint.
        centers = vertices.reshape(12, 20, 3).mean(axis=1)
        self.assertGreater(float(centers[6, 0]), .55)

    def test_coordinate_roundtrip_various_shapes_and_scales(self):
        generator = np.random.default_rng(819)
        uv = generator.uniform(-.1, 1.1, (4, 7, 2))
        for side in (.04, .18, .26):
            np.testing.assert_allclose(replay.base_to_uv(replay.uv_to_base(uv, side), side), uv, atol=2e-15)


class RasterTests(unittest.TestCase):
    def test_native_disk_preserves_released_normalized_radius(self):
        mask = np.zeros((64, 64), bool)
        replay.stamp(mask, [.5, .5], .03, native=True)
        self.assertEqual(int(mask.sum()), 1)
        self.assertTrue(mask[32, 32])
        replay.stamp(mask, [.5, .5], .03, native=False)
        self.assertGreater(int(mask.sum()), 1)

    def test_degenerate_polygon_renders_only_vertices(self):
        for size in (32, 64, 300):
            for triangle in (
                [[.2, .2], [.4, .4], [.6, .6]],
                [[.2, .4], [.5, .4], [.8, .4]],
                [[.3, .3], [.3, .3], [.3, .3]],
                [[-1., .2], [-1., .4], [-1., .6]],
            ):
                points = np.array([[.1, .1]] + triangle)
                expected = np.zeros((size, size), bool)
                vertices = np.clip((np.array(triangle) * size).astype(int), 0, size-1)
                expected[vertices[:, 1], vertices[:, 0]] = True
                actual = np.zeros_like(expected)
                replay.stamp(actual, points[0], 0., points, native=True)
                np.testing.assert_array_equal(actual, expected)

    def test_boundary_triangle_contains_its_edges(self):
        size = 32
        points = np.array([[0., 0.], [0., 0.], [1., 0.], [0., 1.]])
        actual = np.zeros((size, size), bool)
        replay.stamp(actual, points[0], 0., points, native=True)
        y, x = np.mgrid[:size, :size]
        np.testing.assert_array_equal(actual, x+y <= size-1)


_HAVE_OFFICIAL_DEPS = all(importlib.util.find_spec(module) is not None
                          for module in ("skimage", "gym", "pygame", "cv2"))


@unittest.skipUnless(_HAVE_OFFICIAL_DEPS, "Run with calli_rl2 for the released brush renderer oracle")
class OfficialBrushParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "rl_finetune/Callienv/envs/tools.py"
        spec = importlib.util.spec_from_file_location("callirewrite_official_tools_oracle", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.brush = module.Writing_Brush(0., .064, 0., .17875, 0., 360., 10.)

    def test_native_footprints_match_official_across_resolutions_and_boundaries(self):
        generator = np.random.default_rng(314)
        for size in (32, 64, 100, 256, 300):
            for _ in range(200):
                uv = generator.uniform(-.1, 1.1, 2)
                radius = generator.uniform(.00001, .064)
                length = .003 + 2.021 * radius
                heading = generator.uniform(-360., 720.)
                actual_points = replay.footprint_points(uv, radius, length, heading)
                expected_points, _ = self.brush.calc_four_points(uv[::-1], radius, length, np.deg2rad(heading))
                np.testing.assert_allclose(actual_points, expected_points[:, ::-1], atol=1e-15)
                rows, columns = self.brush.draw_canvas(size, expected_points, radius)
                expected = np.zeros((size, size), bool)
                expected[rows, columns] = True
                actual = np.zeros_like(expected)
                replay.stamp(actual, uv, radius, actual_points, native=True)
                np.testing.assert_array_equal(actual, expected)

    def test_exact_degenerate_and_boundary_cases_match_official(self):
        for size in (32, 64, 300):
            for points in (
                [[.2, .2], [.2, .2], [.4, .4], [.6, .6]],
                [[.3, .4], [.2, .4], [.5, .4], [.8, .4]],
                [[0., 0.], [0., 0.], [1., 0.], [0., 1.]],
                [[1., 1.], [1.2, 1.2], [1.4, 1.4], [1.6, 1.6]],
            ):
                points = np.asarray(points)
                expected = np.zeros((size, size), bool)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    rows, columns = self.brush.draw_canvas(size, points[:, ::-1], 0.)
                expected[rows, columns] = True
                actual = np.zeros_like(expected)
                replay.stamp(actual, points[0], 0., points, native=True)
                np.testing.assert_array_equal(actual, expected)


class SourceValidationTests(unittest.TestCase):
    def test_load_preserves_order_and_provenance(self):
        original = source_example()
        with tempfile.TemporaryDirectory() as temporary:
            path = _source_file(temporary, original)
            loaded = replay.read_source(path)
        for field in ("xy_uv", "radius_norm", "stroke_id", "length_norm", "heading_deg"):
            np.testing.assert_array_equal(loaded[field], original[field])
        self.assertEqual(len(loaded["metadata"]["source_sha256"]), 64)
        self.assertTrue(loaded["native"])

    def test_invalid_source_arrays_rejected(self):
        invalid = [
            ("xy_uv", np.full((5, 2), np.nan)),
            ("xy_uv", np.zeros((5, 3))),
            ("radius_norm", np.full(5, -.1)),
            ("radius_norm", np.full(5, np.inf)),
            ("radius_norm", np.ones(4)),
            ("stroke_id", np.array([0, 0, 1, 0, 1])),
            ("stroke_id", np.array([0, 0, .5, 1, 1])),
            ("length_norm", np.full(5, -.01)),
            ("length_norm", np.full(5, np.nan)),
            ("heading_deg", np.full(5, np.inf)),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            for field, value in invalid:
                with self.subTest(field=field, value=value):
                    source = source_example()
                    source[field] = value
                    with self.assertRaises(ValueError):
                        replay.read_source(_source_file(temporary, source))


class ReplayVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = replay.prepare(source_example(), resolution=100)

    def test_contact_events_exactly_once_and_no_ink_on_travel(self):
        prepared = self.prepared
        np.testing.assert_array_equal(prepared.contact_index[prepared.contact_index >= 0], np.arange(5))
        travel_mask = np.zeros((100, 100), bool)
        for frame in np.flatnonzero(prepared.contact_index < 0):
            prepared.deposit(travel_mask, frame)
        self.assertFalse(travel_mask.any())
        expected, actual = np.zeros((100, 100), bool), np.zeros((100, 100), bool)
        source = prepared.source
        for i in range(5):
            points = replay.footprint_points(source["xy_uv"][i], source["radius_norm"][i],
                                             source["length_norm"][i], source["heading_deg"][i])
            replay.stamp(expected, source["xy_uv"][i], source["radius_norm"][i], points, native=True)
        for frame in range(len(prepared.time_s)):
            prepared.deposit(actual, frame, achieved=False)
        np.testing.assert_array_equal(actual, expected)

    def test_timing_joint_limits_and_pose_accuracy(self):
        prepared = self.prepared
        self.assertTrue(prepared.solution.success)
        intervals = np.diff(prepared.time_s)
        self.assertTrue(np.all(intervals > 0))
        velocities = np.abs(np.diff(prepared.solution.q, axis=0)) / intervals[:, None]
        self.assertTrue(np.all(velocities <= prepared.kinematics.velocity_limits))
        self.assertTrue(np.all(prepared.solution.q > prepared.kinematics.joint_limits[:, 0]))
        self.assertTrue(np.all(prepared.solution.q < prepared.kinematics.joint_limits[:, 1]))
        self.assertLessEqual(prepared.solution.position_error_m.max(), 1e-8)
        self.assertLessEqual(prepared.solution.orientation_error_rad.max(), 1e-7)

    def test_verification_passes_good_replay_and_saves_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            report = replay.verify(self.prepared, output)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["contact_samples"], 5)
            self.assertEqual(report["stroke_groups"], 2)
            self.assertEqual(report["source_to_achieved"]["different_pixels"], 0)
            for name in ("source_reconstruction.png", "panda_achieved.png", "comparison.png",
                         "panda_trajectory.npz", "verification.json"):
                self.assertTrue((output / name).is_file(), name)

    def test_verification_does_not_pass_when_contact_ink_is_lost(self):
        achieved = self.prepared.solution.achieved_tip.copy()
        achieved[:, 2] = replay.PAPER_Z + .001
        changed_solution = replace(self.prepared.solution, achieved_tip=achieved)
        changed = replace(self.prepared, solution=changed_solution)
        with tempfile.TemporaryDirectory() as temporary:
            report = replay.verify(changed, Path(temporary))
        self.assertNotEqual(report["status"], "passed")
        self.assertEqual(report["source_to_achieved"]["iou"], 0.)

    def test_native_manifest_raster_must_match_exported_footprints(self):
        native_mask = np.zeros((100, 100), bool)
        for frame in range(len(self.prepared.time_s)):
            self.prepared.deposit(native_mask, frame, achieved=False)
        source = dict(self.prepared.source, native_ink_mask=native_mask)
        changed = replace(self.prepared, source=source)
        with tempfile.TemporaryDirectory() as temporary:
            report = replay.verify(changed, Path(temporary))
            self.assertEqual(report["status"], "passed")
            self.assertTrue(report["checks"]["export_matches_official_native_raster"])
            self.assertEqual(report["native_to_source"]["different_pixels"], 0)
            source["native_ink_mask"] = native_mask.copy()
            source["native_ink_mask"][0, 0] = ~source["native_ink_mask"][0, 0]
            report = replay.verify(changed, Path(temporary))
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["checks"]["export_matches_official_native_raster"])
        self.assertEqual(report["native_to_source"]["different_pixels"], 1)

    def test_nonfinite_speed_and_oversized_footprints_rejected(self):
        for speed in (np.nan, np.inf, 0., -1.):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                replay.prepare(source_example(), writing_speed=speed)
        source = source_example()
        source["length_norm"][:] = 1.2
        with self.assertRaises(ValueError):
            replay.prepare(source)


class InterpolationCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = source_example()
        source["xy_uv"][:3] = source["xy_uv"][0] + .1*(source["xy_uv"][:3]-source["xy_uv"][0])
        source["xy_uv"][3:] = source["xy_uv"][3] + .1*(source["xy_uv"][3:]-source["xy_uv"][3])
        cls.prepared = replay.prepare(source, resolution=100)

    def save_trajectory(self, directory, time_scale=1.):
        path = Path(directory) / "trajectory.npz"
        prepared = self.prepared
        np.savez(path, q=prepared.solution.q, target_tip=prepared.target_tip,
                 time_s=prepared.time_s*time_scale, contact_index=prepared.contact_index,
                 metadata=json.dumps({"urdf_sha256": prepared.kinematics.urdf_sha256}))
        return path

    def test_sampled_interpolation_records_bounds_and_provenance(self):
        from verify_replay_interpolation import check_trajectory
        with tempfile.TemporaryDirectory() as temporary:
            result = check_trajectory(self.save_trajectory(temporary))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["interior_samples"], 3*(len(self.prepared.time_s)-1))
        self.assertEqual(len(result["trajectory_sha256"]), 64)
        self.assertGreater(result["minimum_ferrule_clearance_m"], .03)
        self.assertGreater(result["approach_samples_in_geometric_contact"], 0)
        self.assertIn("no continuous collision proof", result["scope"])

    def test_interpolation_report_rejects_excessive_velocity_and_strict_deviation(self):
        from verify_replay_interpolation import check_trajectory
        with tempfile.TemporaryDirectory() as temporary:
            result = check_trajectory(self.save_trajectory(temporary, time_scale=1e-6))
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["checks"]["linear_joint_velocity_within_limits"])
            result = check_trajectory(self.save_trajectory(temporary), max_tip_deviation_m=1e-12)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["checks"]["interior_tip_deviation_within_limit"])

    def test_interpolation_rejects_invalid_sampling_fractions(self):
        from verify_replay_interpolation import check_trajectory
        for fractions in ([], [0.], [1.], [np.nan], [[.5]]):
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                check_trajectory("unused.npz", fractions=fractions)


if __name__ == "__main__":
    unittest.main()
