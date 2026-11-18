"""Tunnel-profile detector on a synthetic lidar-like tunnel (no datasets needed)."""
import json
import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]
import numpy as np

from metro.tunnel import TunnelDetector, TunnelConfig, PathModel, _group_max, _penalized_fit

SENSOR_OVER_RAIL = 1.12


def tunnel_frame(S0=0., bend=0., obstacle=None, seed=0, n=90000, xmax=150.):
    """Points of a round single-track tunnel seen from a sensor at world arc S0.

    Track coordinates: s along the track, u lateral, h above rail heads.
    `bend` is the lateral curvature 1/R (constant); `obstacle` is
    (s_world, u, h, size) of a box. Density decays with range like a lidar.
    """
    rng = np.random.default_rng(seed)
    parts = []

    def along(k):
        r = rng.random(k)
        return 3. + (xmax - 3.) * r ** 2      # dense near, sparse far

    # track bed and trough
    s = along(n // 3); u = rng.uniform(-1.45, 1.45, len(s))
    h = np.where(np.abs(u) < .2, -.5, -.25) + rng.normal(0, .005, len(s))
    parts.append((s, u, h))
    # rail heads (7 cm wide ridges) and rail sides
    for side in (-1, 1):
        s = along(n // 8); u = side * .795 + rng.uniform(-.035, .035, len(s))
        parts.append((s, u, rng.normal(0, .004, len(s))))
    # round tunnel wall (radius 2.6 m, centre 1.6 m above rails), with brackets every 1.7 m
    s = along(n // 2); a = rng.uniform(-1.9, 1.9, len(s))
    R = 2.6 - .12 * (((s + S0) % 1.7) < .15)
    parts.append((s, R * np.sin(a), 1.6 + R * np.cos(a)))
    if obstacle is not None:
        so, uo, ho, size = obstacle
        k = 2500
        parts.append((so - S0 + rng.uniform(0, size, k) * .3, uo + rng.uniform(-size / 2, size / 2, k),
                      ho + rng.uniform(0, size, k)))
    s = np.concatenate([p[0] for p in parts]); u = np.concatenate([p[1] for p in parts])
    h = np.concatenate([p[2] for p in parts])
    keep = (s > 1.) & (s < xmax)
    s, u, h = s[keep], u[keep], h[keep]
    # constant-curvature track: centre y = bend*x^2/2, normal offset u
    x = s
    slope = bend * x
    norm = np.sqrt(1 + slope * slope)
    X = x - u * slope / norm
    Y = bend * x * x / 2 + u / norm
    return np.column_stack([X, Y, h - SENSOR_OVER_RAIL]).astype(np.float32)


def run(frames, det=None, speed=10., bend=0., obstacle=None, t0=0.):
    det = det or TunnelDetector(TunnelConfig())
    out = []
    for k in range(frames):
        S0 = speed * .1 * k
        out.append(det.process(tunnel_frame(S0, bend, obstacle, seed=k), None, t0 + .1 * k))
    return det, out


class TunnelDetectorTests(unittest.TestCase):
    def test_clear_straight_tunnel(self):
        _, res = run(12)
        self.assertTrue(all(r['status'] in ('clear', 'candidate') for r in res[1:]))
        self.assertFalse(any(r['obstacle'] for r in res))
        self.assertGreater(res[-1]['analysis_range_m'], 60.)
        json.dumps(res[-1], allow_nan=False)

    def test_detects_central_obstacle_with_along_track_distance(self):
        det, res = run(14, obstacle=(60., 0., .6, .5))
        alarms = [r for r in res if r['obstacle']]
        self.assertTrue(alarms, 'central 0.5 m box must be confirmed')
        last = res[-1]
        expected = 60. - 10. * .1 * 13
        self.assertAlmostEqual(last['distance_m'], expected, delta=1.)

    def test_object_outside_envelope_is_ignored(self):
        _, res = run(14, obstacle=(45., 1.45, .6, .4))
        self.assertFalse(any(r['obstacle'] for r in res))

    def test_confirmed_obstacle_is_held_beyond_verified_range(self):
        # confirmed while approaching, then the far tunnel stops returning points:
        # the obstacle's place is no longer verified, so it must not turn "clear"
        det = TunnelDetector(TunnelConfig())
        res = []
        for k in range(26):
            xmax = 150. if k < 20 else 40.
            res.append(det.process(tunnel_frame(k * 1., obstacle=(70., 0., .6, .5), seed=k, xmax=xmax), None, .1 * k))
        self.assertTrue(res[19]['obstacle'])
        late = res[-1]
        self.assertLess(late['analysis_range_m'], 45.)
        self.assertTrue(late['obstacle'])
        self.assertAlmostEqual(late['distance_m'], 70. - 25., delta=1.5)

    def test_crawling_train_does_not_hold_unverified_obstacle(self):
        # confirmations from positions only ~4 m apart give no parallax:
        # "static in the world" proves little, so no long hold
        det = TunnelDetector(TunnelConfig())
        res = []
        for k in range(24):
            xmax = 150. if k < 14 else 40.
            res.append(det.process(tunnel_frame(.3 * k, obstacle=(50., 0., .6, .5), seed=k, xmax=xmax), None, .1 * k))
        self.assertTrue(res[13]['obstacle'])
        self.assertFalse(res[-1]['obstacle'])

    def test_object_appearing_at_standstill_is_new_and_background_is_not(self):
        # sensor-frame occupancy history while the train stands still
        det = TunnelDetector(TunnelConfig())
        rng = np.random.default_rng(3)
        wall = np.column_stack([rng.uniform(5., 80., 4000), rng.choice([-2.5, 2.5], 4000),
                                rng.uniform(-1., 2., 4000)]).astype(np.float32)
        obj = np.column_stack([np.full(12, 56.4), rng.uniform(-.9, -.65, 12),
                               rng.uniform(-.99, -.9, 12)]).astype(np.float32)   # low, next to a rail
        for f in range(40):
            det.frame, det.now = f, .1 * f
            det._update_change(np.vstack([wall, obj]) if f >= 28 else wall, 0.)
        self.assertGreaterEqual(det._appeared_frac(obj), .9)
        self.assertEqual(det._appeared_frac(wall[:500]), 0.)
        det.frame, det.now = 40, 4.
        det._update_change(wall, .5)                  # the train moves: history is dropped
        self.assertIsNone(det.still_f0)
        self.assertEqual(det._appeared_frac(obj), 0.)

    def test_standstill_reference_is_measured_in_seconds(self):
        # a busy computer processes every 5th frame; the object appears 3 s into the stop
        det = TunnelDetector(TunnelConfig())
        rng = np.random.default_rng(4)
        wall = np.column_stack([rng.uniform(5., 80., 4000), rng.choice([-2.5, 2.5], 4000),
                                rng.uniform(-1., 2., 4000)]).astype(np.float32)
        obj = np.column_stack([np.full(12, 56.4), rng.uniform(-.9, -.65, 12),
                               rng.uniform(-.99, -.9, 12)]).astype(np.float32)
        for k in range(12):
            det.frame, det.now = k, .5 * k
            det._update_change(np.vstack([wall, obj]) if det.now >= 3. else wall, 0.)
        self.assertGreaterEqual(det._appeared_frac(obj), .9)
        self.assertEqual(det._appeared_frac(wall[:500]), 0.)

    def _moving_run(self, train_v, obj_s0, obj_v, size, h, frames=20):
        det, res = TunnelDetector(TunnelConfig()), []
        for k in range(frames):
            t = .1 * k
            res.append((obj_s0 + obj_v * t - train_v * t,
                        det.process(tunnel_frame(train_v * t, obstacle=(obj_s0 + obj_v * t, 0., h, size), seed=k), None, t)))
        return res

    def test_train_ahead_moving_away_is_an_obstacle(self):
        # our train at 10 m/s, a 2 m wide body 60 m ahead moving away at 8 m/s
        res = self._moving_run(10., 60., 8., 2., .3)
        late = [r for _, r in res[5:]]
        self.assertTrue(all(r['obstacle'] for r in late))
        moving = [c for r in late for c in r['candidates'] if c.get('moving')]
        self.assertTrue(moving)
        self.assertAlmostEqual(np.median([c['speed_mps'] for c in moving]), 8., delta=.5)
        for true_d, r in res[5:]:
            self.assertAlmostEqual(r['distance_m'], true_d, delta=1.)

    def test_person_running_towards_the_train_is_an_obstacle(self):
        # our train at 5 m/s, a 0.5 m object runs towards it at 6 m/s
        res = self._moving_run(5., 50., -6., .5, 0.)
        for true_d, r in res[2:]:
            self.assertTrue(r['obstacle'])
            self.assertAlmostEqual(r['distance_m'], true_d, delta=1.)

    def test_long_object_on_track_centre_is_not_taken_for_a_wall(self):
        # long, wide and tall like an oblique wall, but it blocks the track centre
        det = TunnelDetector(TunnelConfig())
        res = []
        for k in range(14):
            S0 = k * 1.
            rng = np.random.default_rng(50 + k)
            s = 40. - S0 + rng.uniform(0., 12., 4000)
            box = np.column_stack([s, rng.uniform(-.8, .8, len(s)), rng.uniform(.2, 2.2, len(s)) - SENSOR_OVER_RAIL])
            res.append(det.process(np.vstack([tunnel_frame(S0, seed=k), box.astype(np.float32)]), None, .1 * k))
        self.assertTrue(res[-1]['obstacle'])
        self.assertAlmostEqual(res[-1]['distance_m'], 40. - 13., delta=1.)

    def test_follows_curve_and_stays_clear(self):
        bend = 1 / 500.
        det, res = run(12, bend=bend)
        self.assertFalse(any(r['obstacle'] for r in res))
        path = det.path
        x = np.array([20., 40., 60.])
        err = np.abs(path.c(x) - bend * x * x / 2)
        self.assertLess(err.max(), .3, err)

    def test_odometry_matches_motion(self):
        det, res = run(10, speed=12.)
        ds = [r['path_info']['odometry_ds'] for r in res[3:]]
        self.assertLess(np.median(np.abs(np.asarray(ds) - 1.2)), .1, ds)

    def test_empty_cloud_is_unknown(self):
        r = TunnelDetector().process(np.empty((0, 3), np.float32), None, 0.)
        self.assertEqual(r['status'], 'unknown')
        self.assertFalse(r['obstacle'])

    def test_time_gap_resets_confirmation(self):
        det, res = run(8, obstacle=(50., 0., .6, .5))
        self.assertTrue(res[-1]['obstacle'])
        r = det.process(tunnel_frame(8., obstacle=(50., 0., .6, .5)), None, 5.)
        self.assertFalse(r['obstacle'], 'a 4 s gap must invalidate temporal evidence')


class HelperTests(unittest.TestCase):
    def test_group_max(self):
        rng = np.random.default_rng(1)
        key = rng.integers(0, 50, 3000)
        val = rng.normal(size=3000).astype(np.float32)
        ref = np.full(50, -np.inf, np.float32)
        np.maximum.at(ref, key, val)
        np.testing.assert_allclose(_group_max(key, val, 50), ref, atol=1.1e-3)

    def test_penalized_fit_reproduces_line(self):
        g = PathModel.GRID
        x = np.linspace(0, 300, 40)
        f = _penalized_fit(g, x, 2 + .01 * x, np.ones_like(x), 1e3)
        inside = g <= 300
        np.testing.assert_allclose(f[inside], 2 + .01 * g[inside], atol=1e-3)

    def test_shifted_path_keeps_sensor_pose(self):
        g = PathModel.GRID
        lat = .05 + .002 * g + g * g / 1000.
        p = PathModel(lat, np.full(len(g), -1.1), 0., 100.)
        q = p.shifted(2.)
        self.assertAlmostEqual(q.lat[0], p.lat[0], places=6)
        self.assertAlmostEqual(q.dlat[0], p.dlat[0], delta=2e-3)  # finite-difference boundary
        # curvature is transported
        self.assertAlmostEqual(np.gradient(np.gradient(q.lat, g), g)[50], 1 / 500., places=5)


if __name__ == '__main__':
    unittest.main()
