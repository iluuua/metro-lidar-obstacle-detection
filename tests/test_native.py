"""The C++ core (libmetro_core) on the synthetic tunnel of test_tunnel: the same behaviour as the
Python reference, plus what only the core does (stable envelope length, dust). Skipped when the
library is not built: cmake -S core -B core/build -DCMAKE_BUILD_TYPE=Release && cmake --build core/build
"""
import json
import sys
import unittest
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import numpy as np

from metro import native
from metro.tunnel import TunnelConfig
from test_tunnel import SENSOR_OVER_RAIL, tunnel_frame


def make(**overrides):
    # a null handle (refused config) is reported by the callers
    det = native.NativeTunnelDetector.__new__(native.NativeTunnelDetector)
    det.cfg = TunnelConfig()
    cfg = det.cfg.as_dict()
    cfg.update(overrides)
    det._lib, cfg['max_range'] = native.library_for(cfg['max_range'])
    det._h = det._lib.metro_create(json.dumps(cfg).encode())
    return det


def section_frame(S0, seed, lo, hi, wide, whole=False, obstacle=None):
    """The synthetic tunnel with its section widened between lo and hi metres: only the walls (a platform beside the track)
    or, with `whole`, everything (another tunnel)."""
    p = tunnel_frame(S0, 0., obstacle, seed=seed, xmax=200.).astype(np.float64)
    above = p[:, 2] + SENSOR_OVER_RAIL
    chosen = (p[:, 0] > lo) & (p[:, 0] < hi) & (np.ones(len(p), bool) if whole else (above > .5) & (np.abs(p[:, 1]) > 1.3))
    p[chosen, 1] *= wide
    return p.astype(np.float32)


def run_sections(frames, speed=5., **section):
    det = make(**section.pop('config', {}))
    return [det.process(section_frame(speed * .1 * k, k, **section), None, .1 * k) for k in range(frames)]


NO_BRIDGE = dict(confirm_explained=0., break_bridge_m=0., extension_stops_at_break=0)


def run(frames, det=None, speed=10., bend=0., obstacle=None, t0=0.):
    det = det or make()
    return det, [det.process(tunnel_frame(speed * .1 * k, bend, obstacle, seed=k), None, t0 + .1 * k)
                 for k in range(frames)]


@unittest.skipUnless(native.available(), 'libmetro_core is not built')
class NativeDetectorTests(unittest.TestCase):
    def test_clear_straight_tunnel(self):
        _, res = run(12)
        self.assertFalse(any(r['obstacle'] for r in res))
        self.assertGreater(res[-1]['analysis_range_m'], 60.)
        self.assertEqual(res[-1]['model_revision'], 'tunnel-profile-v2-cpp')

    def test_detects_central_obstacle_with_along_track_distance(self):
        _, res = run(14, obstacle=(60., 0., .6, .5))
        self.assertTrue(any(r['obstacle'] for r in res))
        self.assertAlmostEqual(res[-1]['distance_m'], 60. - 10. * .1 * 13, delta=1.)

    def test_new_confirmation_waits_for_a_range_that_holds(self):
        # an object beyond a verified range that comes and goes is attention, not an alarm; the same object in a
        # steady range is confirmed at once, and the gate can be switched off
        def alarms(frames_to_run, **overrides):
            det, flags = make(**overrides), []
            for k in range(frames_to_run):
                xmax = 150. if k % 3 else 70.
                r = det.process(tunnel_frame(k * 1., obstacle=(95., 0., .6, .5), seed=k, xmax=xmax), None, .1 * k)
                flags.append(bool(r['obstacle']))
            return flags
        self.assertFalse(any(alarms(30)))
        self.assertTrue(any(alarms(30, confirm_range_frames=0)))
        _, steady = run(14, obstacle=(95., 0., .6, .5))
        self.assertTrue(any(r['obstacle'] for r in steady))

    def test_object_outside_envelope_is_ignored(self):
        _, res = run(14, obstacle=(45., 1.45, .6, .4))
        self.assertFalse(any(r['obstacle'] for r in res))

    def test_confirmed_obstacle_is_held_beyond_verified_range(self):
        det, res = make(), []
        for k in range(26):
            xmax = 150. if k < 20 else 40.
            res.append(det.process(tunnel_frame(k * 1., obstacle=(70., 0., .6, .5), seed=k, xmax=xmax), None, .1 * k))
        self.assertTrue(res[19]['obstacle'])
        self.assertTrue(res[-1]['obstacle'])
        self.assertAlmostEqual(res[-1]['distance_m'], 70. - 25., delta=1.5)

    def test_person_running_towards_the_train_is_an_obstacle(self):
        det = make()
        for k in range(20):
            t = .1 * k
            s_obj = 50. - 6. * t
            r = det.process(tunnel_frame(5. * t, obstacle=(s_obj, 0., 0., .5), seed=k), None, t)
            if k >= 2:
                self.assertTrue(r['obstacle'])
                self.assertAlmostEqual(r['distance_m'], s_obj - 5. * t, delta=1.)

    def test_follows_curve_and_stays_clear(self):
        bend = 1 / 500.
        _, res = run(12, bend=bend)
        self.assertFalse(any(r['obstacle'] for r in res))
        path = np.asarray(res[-1]['path'])
        x = np.array([20., 40., 60.])
        err = np.abs(np.interp(x, path[:, 0], path[:, 1]) - bend * x * x / 2)
        self.assertLess(err.max(), .3, err)

    def test_axis_goes_on_along_a_bend_beyond_the_measured_range(self):
        # the data end near 100 m: farther the axis continues the bend of its last 80 m instead of straight ahead
        bend = 1 / 400.
        det = make()
        for k in range(16):
            res = det.process(tunnel_frame(1. * k, bend, None, seed=k, xmax=100.), None, .1 * k)
        path = np.asarray(res['path'])
        for x in (130., 160., 200.):
            self.assertLess(abs(np.interp(x, path[:, 0], path[:, 1]) - bend * x * x / 2), 2.5, x)
        plain = make(tail_window_m=0.)
        for k in range(16):
            res = plain.process(tunnel_frame(1. * k, bend, None, seed=k, xmax=100.), None, .1 * k)
        old = np.asarray(res['path'])
        self.assertGreater(abs(np.interp(200., old[:, 0], old[:, 1]) - bend * 200. ** 2 / 2),
                           abs(np.interp(200., path[:, 0], path[:, 1]) - bend * 200. ** 2 / 2))

    def test_axis_stays_straight_beyond_the_measured_range_of_a_straight_track(self):
        det = make()
        for k in range(16):
            res = det.process(tunnel_frame(1. * k, 0., None, seed=k, xmax=100.), None, .1 * k)
        path = np.asarray(res['path'])
        self.assertLess(np.abs(np.interp([130., 200., 300.], path[:, 0], path[:, 1])).max(), .3)

    def test_a_platform_beside_the_track_does_not_end_the_envelope(self):
        # walls farther out between 45 and 95 m: the profile explains only part of those slabs, but they sit on the
        # axis and nothing is in the envelope; that is the same tunnel, and the verified range goes on through it
        scene = dict(lo=45., hi=95., wide=1.6)
        self.assertGreater(run_sections(24, **scene)[-1]['analysis_range_m'], 150.)
        self.assertLess(run_sections(24, config=NO_BRIDGE, **scene)[-1]['analysis_range_m'], 60.)

    def test_an_obstacle_beyond_a_platform_entrance_is_found(self):
        scene = dict(lo=45., hi=95., wide=1.6, obstacle=(75., 0., .6, .5))
        self.assertTrue(run_sections(30, **scene)[-1]['obstacle'])
        self.assertFalse(run_sections(30, config=NO_BRIDGE, **scene)[-1]['obstacle'])

    def test_another_tunnel_still_ends_the_envelope(self):
        # the whole section changes at 120 m: the profile explains none of it, so nothing beyond counts
        res = run_sections(30, lo=120., hi=200., wide=2.2, whole=True)[-1]
        self.assertLess(res['analysis_range_m'], 130.)
        self.assertLess(res['path_info']['break_m'], 135.)

    def test_explain_shows_the_axis_after_every_stage(self):
        for options in (dict(explain=True), dict(explain=True, continuous_path=True)):
            res = make(**options).process(tunnel_frame(0., 1 / 500., None, seed=1), None, 0.)
            stages = np.asarray(res['path_info']['explain']['stages'])
            self.assertEqual(stages.shape[1], 5)  # node, after registration, after the filter over frames, after the joint fit, shown
            self.assertGreater(len(stages), 40)
            self.assertTrue(np.all(np.isfinite(stages)))

    def test_odometry_matches_motion(self):
        _, res = run(10, speed=12.)
        ds = [r['path_info']['odometry_ds'] for r in res[3:]]
        self.assertLess(np.median(np.abs(np.asarray(ds) - 1.2)), .1, ds)

    def test_empty_cloud_is_unknown(self):
        r = make().process(np.empty((0, 3), np.float32), None, 0.)
        self.assertEqual(r['status'], 'unknown')

    def test_time_gap_resets_confirmation(self):
        det, res = run(8, obstacle=(50., 0., .6, .5))
        self.assertTrue(res[-1]['obstacle'])
        r = det.process(tunnel_frame(8., obstacle=(50., 0., .6, .5)), None, 5.)
        self.assertFalse(r['obstacle'])

    def test_unknown_config_key_is_refused(self):
        with self.assertRaises(ValueError):
            cfg = TunnelConfig().as_dict()
            cfg['no_such_parameter'] = 1
            lib = native.library()
            h = lib.metro_create(json.dumps(cfg).encode())
            if not h:
                raise ValueError(lib.metro_create_error().decode())
            lib.metro_destroy(h)

    def test_envelope_length_survives_a_short_dropout(self):
        # one frame without the far tunnel must not shorten the drawn envelope; a lasting change does.
        # The verified range is honest all the time: it follows what the frame itself sees.
        det, drawn, verified = make(), [], []
        for k in range(44):
            xmax = 40. if k == 12 or k >= 18 else 150.
            r = det.process(tunnel_frame(k * 1., seed=k, xmax=xmax), None, .1 * k)
            drawn.append(r['envelope_range_m'])
            verified.append(r['analysis_range_m'])
        self.assertGreater(drawn[12], drawn[11] - 6., drawn)
        self.assertLess(verified[12], 45., verified)
        # the drawn length follows a lasting change: a drop is shown once it persists (envelope_hold_frames), then in steps
        self.assertLess(drawn[-1], drawn[17] - 8., drawn)
        self.assertTrue(all(b <= a + 1e-6 for a, b in zip(drawn[20:], drawn[21:])), drawn)
        self.assertLess(verified[-1], 45., verified)

    def test_envelope_does_not_grow_on_a_single_far_frame(self):
        # a far tunnel seen in one frame only is not drawn; seen in the following frames it is, in steps
        det, drawn = make(), []
        for k in range(30):
            xmax = 150. if k == 12 or k >= 20 else 40.
            r = det.process(tunnel_frame(k * 1., seed=k, xmax=xmax), None, .1 * k)
            drawn.append(r['envelope_range_m'])
        self.assertLess(drawn[12], drawn[11] + 3., drawn)
        self.assertLess(drawn[13], drawn[11] + 3., drawn)
        self.assertGreater(drawn[-1], drawn[11] + 40., drawn)
        self.assertTrue(all(b <= a + 31. for a, b in zip(drawn, drawn[1:])), drawn)

    def test_dust_speck_is_not_an_obstacle(self):
        # a handful of returns into one spot 4 m ahead, the same spot in the world for three frames
        det, res = make(), []
        for k in range(12):
            cloud = tunnel_frame(k * .3, seed=k)
            if 8 <= k <= 10:
                x = 6. - k * .3 + 3.
                speck = np.array([[x, .1, .4 - SENSOR_OVER_RAIL + 1.]] * 6, np.float32)
                cloud = np.vstack([cloud, speck + np.float32([0, 0, -1.])])
            res.append(det.process(cloud, None, .1 * k))
        self.assertFalse(any(r['obstacle'] for r in res))

    def test_matches_python_reference_on_central_obstacle(self):
        from metro.tunnel import TunnelDetector
        py, cpp = TunnelDetector(TunnelConfig()), make()
        for k in range(14):
            cloud = tunnel_frame(k * 1., obstacle=(60., 0., .6, .5), seed=k)
            a, b = py.process(cloud, None, .1 * k), cpp.process(cloud, None, .1 * k)
            self.assertLessEqual(set(a), set(b))  # the core reports everything the reference does
            if a['obstacle'] and b['obstacle']:
                self.assertAlmostEqual(a['distance_m'], b['distance_m'], delta=.3)
        self.assertTrue(b['obstacle'])


if __name__ == '__main__':
    unittest.main()
