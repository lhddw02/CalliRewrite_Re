"""Focused schema/geometry regressions; no inference, GPU, or simulator launch."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panda_calligraphy.callirewrite_io import load_coarse


class CoarseIOTests(unittest.TestCase):
    def write(self, directory, rows, *, rounds=None, cursors=None):
        rows = np.array(rows, dtype=np.float64)
        path = Path(directory)/"input.npz"
        np.savez(path, strokes_data=rows,
                 init_cursors=np.array(cursors if cursors is not None else [[.5, .5]]),
                 image_size=256, round_length=np.array(rounds if rounds is not None else [len(rows)]),
                 init_width=.25)
        return path

    def test_all_rounds_and_pen_up_state(self):
        rows = [[0, .5, .5, .3, .1, .25, .5],
                [1, .5, .5, .2, -.2, .5, 1],
                [0, .5, .5, .2, .1, .25, 1],
                [0, .5, .5, -.1, -.1, .25, 1]]
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, rows, rounds=[3, 1], cursors=[[.5, .5], [.75, .75]])
            decoded = load_coarse(path)
            self.assertEqual(decoded["curves_px"].shape, (4, 7))
            self.assertEqual(len(decoded["strokes"]), 3)
            self.assertEqual(len(decoded["moves"]), 1)
            self.assertEqual([s["curve_indices"] for s in decoded["strokes"]], [[0], [2], [3]])
            self.assertEqual([s["window_size_px"] for s in decoded["segments"]], [128, 64, 64, 128])
            # A move changes the following position and width state, not just the ink flag.
            np.testing.assert_allclose(decoded["segments"][2]["width_start"], .5)
            self.assertGreater(decoded["curves_px"][3, 1], decoded["curves_px"][2, 1])
            np.testing.assert_array_equal(decoded["raw_strokes_data"], rows)
            with np.load(path) as original:
                np.testing.assert_array_equal(original["strokes_data"], rows)

    def test_known_pixel_centers_and_radius(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, [[0, .5, .5, 0, 0, .25, 1]])
            decoded = load_coarse(path)
            # For an integer 128-pixel window: highres center 128 -> resized
            # coordinate 63.75 -> crop output 63.25 -> global pixel 127.25.
            np.testing.assert_allclose(decoded["curves_px"][0, 1:], 127.25)
            stroke = decoded["strokes"][0]
            self.assertEqual(stroke["xy_uv"].shape, (2, 2))
            np.testing.assert_allclose(stroke["radius_norm"], 8.5/256)

    def test_endpoints_axis_order_and_spatial_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, [[0, .1, .9, .7, -.4, .4, 1]])
            decoded = load_coarse(path, sample_spacing_px=.7)
            xy = decoded["strokes"][0]["xy_uv"]
            curve = decoded["curves_px"][0]
            np.testing.assert_array_equal(xy[0], curve[[2, 1]]/256)
            np.testing.assert_array_equal(xy[-1], curve[[6, 5]]/256)
            self.assertLess(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).max()*256), .700001)
            self.assertLess(xy[-1, 0], xy[0, 0])  # d_col=-.4 -> left
            self.assertGreater(xy[-1, 1], xy[0, 1])  # d_row=.7 -> down

    def test_tiny_segments_and_moves_are_not_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, [[1, .5, .5, 0, 0, .25, 1],
                                          [0, .5, .5, 1e-10, -1e-10, .25, 1]])
            decoded = load_coarse(path)
            self.assertEqual(len(decoded["moves"]), 1)
            self.assertEqual(len(decoded["strokes"]), 1)
            self.assertEqual(decoded["strokes"][0]["xy_uv"].shape, (2, 2))
            np.testing.assert_array_equal(decoded["moves"][0]["radius_norm"], 0)

    def test_malformed_round_counts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, [[0, .5, .5, .1, .1, .25, 1]], rounds=[2])
            with self.assertRaisesRegex(ValueError, "summing"):
                load_coarse(path)


if __name__ == "__main__":
    unittest.main()
