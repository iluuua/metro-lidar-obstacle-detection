"""Smoothing of the drawn envelope in the core (display_path): it changes only the drawn axis, never
a decision, damps the frame-to-frame jitter of a still scene, starts afresh after a gap, and the
drawn curve has no visible kinks."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from metro import native
from test_native import make
from test_tunnel import tunnel_frame


def lateral_at(result, arcs):
    p = np.asarray(result['path'], float)  # rows [grid index, lateral, vertical, arc]
    return np.interp(arcs, p[:, 3], p[:, 1])


def decisions(result):
    return (result['status'], result.get('distance_m'),
            [(round(c['distance_m'], 3), bool(c.get('confirmed'))) for c in result.get('candidates', [])])


@unittest.skipUnless(native.available(), 'libmetro_core is not built')
class NativeDisplaySmoothingTests(unittest.TestCase):
    def test_decisions_do_not_depend_on_smoothing(self):
        runs = {}
        for smooth in (False, True):
            det = make(display_smoothing=smooth, spatial_path=False)
            runs[smooth] = [det.process(tunnel_frame(1. * k, 1 / 600., (80., .3, .5, .5), seed=k, n=40000), None, .1 * k)
                            for k in range(30)]
        self.assertTrue(any(r['obstacle'] for r in runs[True]))
        for plain, smooth in zip(runs[False], runs[True]):
            self.assertEqual(decisions(plain), decisions(smooth))
        drawn = max(np.abs(np.asarray(a['path'])[:, 1] - np.asarray(b['path'])[:, 1]).max()
                    for a, b in zip(runs[False], runs[True]))
        self.assertGreater(drawn, .01)  # the filter did act on the drawn axis

    def test_still_scene_jitters_less(self):
        arcs = np.array([30., 60.])
        jitter = {}
        for smooth in (False, True):
            det = make(display_smoothing=smooth, spatial_path=False)
            res = [det.process(tunnel_frame(0., 1 / 800., seed=k, n=40000), None, .1 * k) for k in range(30)]
            lat = np.array([lateral_at(r, arcs) for r in res[5:]])
            jitter[smooth] = np.abs(np.diff(lat, axis=0)).mean(0)
        # about half on Windows (g++ -O3), 0.52 in the Linux image: the bound leaves room for the compiler's rounding
        self.assertTrue(np.all(jitter[True] < .6 * jitter[False]), jitter)

    def test_drawn_curve_has_no_kinks(self):
        det = make()
        for k in range(12):
            r = det.process(tunnel_frame(1. * k, 1 / 400., seed=k, n=40000), None, .1 * k)
            p = np.asarray(r['path'], float)
            # neighbouring 2 m segments of the drawn axis turn by less than 2 degrees (the visible kinks were 65-136)
            heading = np.degrees(np.arctan2(np.diff(p[:, 1]), np.diff(p[:, 0])))
            self.assertLess(np.max(np.abs(np.diff(heading))), 2.)

    def test_axis_is_drawn_afresh_after_a_gap(self):
        # a filter without memory draws each frame's own (joined) axis
        smooth, plain = make(display_smoothing=True), make(display_smoothing=True, display_tau_s=1e-9)
        for k in range(10):
            frame = tunnel_frame(0., 1 / 800., seed=k, n=40000)
            smooth.process(frame, None, .1 * k), plain.process(frame, None, .1 * k)
        frame = tunnel_frame(0., 1 / 800., seed=99, n=40000)
        a, b = smooth.process(frame, None, 3.), plain.process(frame, None, 3.)
        np.testing.assert_allclose(np.asarray(a['path']), np.asarray(b['path']))


def noisy_frame(S0, bend, seed, sigma):
    """The synthetic tunnel whose returns beyond 90 m scatter sideways: the far axis of a lidar with noisier points."""
    q = tunnel_frame(S0, bend, None, seed=seed, n=40000, xmax=200.).astype(np.float64)
    far = q[:, 0] > 90.
    q[far, 1] += np.random.default_rng(10000 + seed).normal(0., sigma, int(far.sum()))
    return q.astype(np.float32)


@unittest.skipUnless(native.available(), 'libmetro_core is not built')
class NativeFarDisplayTests(unittest.TestCase):
    """The drawn axis far from the train is smoothed more (display_far_smoothing): small changes are damped, a real change
    passes, and no decision depends on it."""

    def test_no_decision_depends_on_it(self):
        runs = {}
        for on in (False, True):
            det = make(display_far_smoothing=on)
            runs[on] = [det.process(tunnel_frame(1. * k, 1 / 600., (80., .3, .5, .5), seed=k, n=40000), None, .1 * k)
                        for k in range(30)]
        self.assertTrue(any(r['obstacle'] for r in runs[True]))
        for plain, smooth in zip(runs[False], runs[True]):
            self.assertEqual(decisions(plain), decisions(smooth))

    def test_far_axis_of_a_noisy_scene_jitters_less(self):
        arcs = np.array([130., 160.])
        jitter = {}
        for on in (False, True):
            det = make(display_far_smoothing=on)
            res = [det.process(noisy_frame(0., 1 / 800., k, .25), None, .1 * k) for k in range(60)]
            lat = np.array([lateral_at(r, arcs) for r in res[15:]])
            jitter[on] = np.abs(np.diff(lat, axis=0)).mean(0)
        self.assertTrue(np.all(jitter[True] < .85 * jitter[False]), jitter)

    def test_a_real_change_of_the_track_passes(self):
        # a bend starts at frame 20: ten frames later the far axis is where the unsmoothed one is, to within the scatter
        at = {}
        for on in (False, True):
            det = make(display_far_smoothing=on)
            out = [lateral_at(det.process(tunnel_frame(1. * k, 0. if k < 20 else 1 / 300., seed=k, n=40000), None, .1 * k), [120.])[0]
                   for k in range(40)]
            at[on] = out[30]
        self.assertLess(abs(at[True] - at[False]), 1.5, at)
        self.assertGreater(at[True], 5.)  # and the bend is there at all


if __name__ == '__main__':
    unittest.main()
