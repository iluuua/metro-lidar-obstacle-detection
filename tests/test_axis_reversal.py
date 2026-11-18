import math
import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'eval')]
import numpy as np

from axis_reversal import frame_reversal, reversal, summarize

X = np.arange(0., 331., 2.)


class AxisReversalTests(unittest.TestCase):
    def test_straight_and_one_way_bends_do_not_reverse(self):
        self.assertEqual(reversal(X, 0.01 * X), 0.0)
        self.assertEqual(reversal(X, X ** 2 / 600.), 0.0)
        self.assertEqual(reversal(X, -X ** 2 / 600.), 0.0)

    def test_curve_that_straightens_is_not_a_reversal(self):
        y = np.where(X < 150, X ** 2 / 600., 150 ** 2 / 600. + (X - 150) * 150 / 300.)
        self.assertEqual(reversal(X, y), 0.0)

    def test_s_bend_and_hook_are_reversals(self):
        s_bend = np.where(X < 150, X ** 2 / 600., 150 ** 2 / 600. + (X - 150) * 0.5 - (X - 150) ** 2 / 600.)
        self.assertGreater(math.degrees(reversal(X, s_bend)), 20.)
        hook = np.where(X < 100, X ** 2 / 400., 25. + (X - 100) * 0.5 - (X - 100) ** 2 / 200.)
        self.assertGreater(math.degrees(reversal(X, hook)), 30.)

    def test_noise_below_straight_curvature_is_ignored(self):
        self.assertEqual(reversal(X, 0.5 * np.sin(X / 1000.) * 1e-3), 0.0)

    def test_reversal_is_split_by_measured_support(self):
        y = np.where(X < 150, X ** 2 / 600., 150 ** 2 / 600. + (X - 150) * 0.5 - (X - 150) ** 2 / 300.)
        frame = dict(status='clear', path=[[x, v, 0., x] for x, v in zip(X, y)], path_info=dict(support_m=100.))
        inside, beyond, whole = frame_reversal(frame)
        self.assertEqual(inside, 0.0)
        self.assertGreater(math.degrees(beyond), 10.)
        self.assertGreater(whole, 0.)
        frame['path_info']['support_m'] = 280.
        inside, beyond, _ = frame_reversal(frame)
        self.assertGreater(math.degrees(inside), 10.)
        self.assertEqual(beyond, 0.0)

    def test_frames_without_axis_are_skipped(self):
        report = summarize([dict(status='unknown', path=None), dict(status='clear', path=[[0, 0, 0, 0]] * 3)])
        self.assertIsNone(report)


if __name__ == '__main__':
    unittest.main()
