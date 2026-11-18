"""Geometry and scoring contracts of controlled boundary insertions; no bags needed."""
import sys
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'eval')]
from inject import WorldTrack, boundary_cases, boundary_placement, boundary_suite, box_hits, inject, _matches_injected, _boundary_reference, geometry_breakdown


class BoundaryInjectionTests(unittest.TestCase):
    def test_lateral_nearest_face_and_center_are_distinct(self):
        near = boundary_cases(sizes=(.3,), ranges=(60.,), offsets=(.95, 1.05, 1.15),
                              axes=('side',), sides=(1,))
        self.assertAlmostEqual(near[0]['center_u_m'], 1.10)
        np.testing.assert_allclose(near[0]['u_bounds_m'], [.95, 1.25])
        self.assertAlmostEqual(near[0]['envelope_overlap_u_m'], .10)
        self.assertEqual([c['geometry'] for c in near], ['inside', 'touching', 'outside'])
        centered = boundary_cases(sizes=(.5,), ranges=(60.,), offsets=(1.15,),
                                  axes=('side',), sides=(-1,), lateral_reference='center')[0]
        self.assertTrue(centered['positive'])
        np.testing.assert_allclose(centered['u_bounds_m'], [-1.4, -.9])

    def test_top_offset_is_bottom_face_and_touching_is_outside(self):
        cases = boundary_cases(sizes=(.3,), ranges=(30.,), axes=('top',),
                               top_offsets=(2.95, 3., 3.05))
        self.assertAlmostEqual(cases[0]['center_h_m'], 3.10)
        self.assertAlmostEqual(cases[0]['envelope_overlap_h_m'], .05)
        self.assertEqual([c['geometry'] for c in cases], ['inside', 'touching', 'outside'])

    def test_box_is_normal_to_track_with_grade_and_roll(self):
        world = WorldTrack({k: [0.] for k in range(170)}, {k: [0.] for k in range(170)})
        frame = dict(S=10., c0=.2, c1=.03, r0=-1.1, r1=.02, phi=.04)
        case = boundary_cases(sizes=(.5,), ranges=(60.,), offsets=(.95,), axes=('side',), sides=(-1,))[0]
        center, size, basis = boundary_placement(case, frame, world, 70.25)
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(basis), 1.)
        x = 60.25 / np.sqrt(1. + .03 ** 2 + .02 ** 2)
        rail = np.array([x, .2 + .03 * x, -1.1 + .02 * x])
        local = (center - rail) @ basis
        np.testing.assert_allclose(local, [0., case['center_u_m'], case['center_h_m']], atol=1e-12)
        self.assertEqual(size, (.5, .5, .5))

    def test_rotated_box_hits_lie_on_surface(self):
        a = .2
        basis = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
        center, size = np.array([40., .5, 0.]), np.array([1., 2., 2.])
        hits, _ = box_hits(center, size, axes=basis)
        self.assertGreater(len(hits), 0)
        local = (hits - center) @ basis
        self.assertTrue(np.all(np.abs(local) <= size / 2 + 1e-9))
        self.assertTrue(np.all(np.isclose(np.abs(local), size / 2, atol=1e-9).any(axis=1)))

    def test_world_curvature_is_reanchored_to_clean_near_observations(self):
        world = WorldTrack({k: [.0003] for k in range(170)}, {k: [.0001] for k in range(170)})
        x = np.arange(5., 31.)
        dl, dv, _ = world.offsets(0., x)
        frame = dict(S=0., c0=-.2, c1=-.02, r0=-1., r1=-.03, phi=0., reference_x=x.tolist(),
                     reference_y=(.04 + .002 * x + dl).tolist(), reference_z=(-1.1 + .003 * x + dv).tolist())
        anchored = world.anchored_frame(frame)
        self.assertAlmostEqual(anchored['c0'], .04)
        self.assertAlmostEqual(anchored['c1'], .002)
        self.assertAlmostEqual(anchored['r0'], -1.1)
        self.assertAlmostEqual(anchored['r1'], .003)
        self.assertLess(anchored['reference_near_max_u_m'], 1e-12)
        self.assertLess(anchored['reference_near_max_h_m'], 1e-12)

    def test_reference_cache_preserves_measurements_and_rejects_other_window(self):
        measurements = ({7: {'S': 3.}}, {10: [.0002]}, {10: [0.]})
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / 'reference.json'
            with patch('inject.track_pass', return_value=measurements) as run:
                self.assertEqual(_boundary_reference('clean', 0, 600, cache), (*measurements, False))
                run.assert_called_once_with('clean', 0, 600, observed_only=True)
            with patch('inject.track_pass', side_effect=AssertionError('cache should avoid a new pass')):
                self.assertEqual(_boundary_reference('clean', 0, 600, cache), (*measurements, True))
                with self.assertRaisesRegex(ValueError, 'source/window/version differs'):
                    _boundary_reference('clean', 0, 30, cache)

    def test_report_keeps_touching_separate_from_exterior(self):
        def row(geometry, alarms):
            return dict(geometry=geometry, alarm_frames=alarms, attention_frames=alarms,
                        unmatched_alarm_frames=0, reference_extrapolated=False)
        result = geometry_breakdown([row('inside', 2), row('touching', 4), row('outside', 0)])
        self.assertEqual(result['inside']['alarm_frames'], 2)
        self.assertEqual(result['touching']['alarm_frames'], 4)
        self.assertEqual(result['outside']['alarm_frames'], 0)
        self.assertEqual(sum(r['cases'] for r in result.values()), 3)

    def test_far_attention_without_confident_points_is_reported(self):
        x = list(range(5, 31))
        track = {0: dict(S=0., c0=0., c1=0., r0=-1.1, r1=0., phi=0., reference_x=x,
                         reference_y=[0.] * len(x), reference_z=[-1.1] * len(x))}
        kappa = {k: [0.] for k in range(171)}
        empty = np.empty((0, 3), np.float32)
        candidate = dict(distance_m=150., bbox=[[149., -2., -2.], [151., 2., 2.]],
                         confirmed=False, confident_points=0, far_attention=True)
        with patch('inject._boundary_reference', return_value=(track, kappa, kappa, False)), \
                patch('inject.iter_frames', return_value=[(0, empty, 0.)]), \
                patch('inject.inject', return_value=(empty, [np.array([[150., 1., 0.]])])), \
                patch('inject.TunnelDetector') as detector, contextlib.redirect_stdout(io.StringIO()):
            detector.return_value.process.return_value = dict(analysis_range_m=300., candidates=[candidate])
            cases = boundary_cases(sizes=(.3,), ranges=(150.,), offsets=(.95,), axes=('side',), sides=(1,))
            result = boundary_suite('mock', 0, 1, cases)
        self.assertEqual(result['cases'][0]['attention_frames'], 1)
        self.assertEqual(result['cases'][0]['first_attention']['frame'], 0)
        self.assertTrue(result['cases'][0]['frames'][0]['candidates'][0]['far_attention'])

    def test_added_returns_and_scoring_do_not_match_unrelated_wall(self):
        empty = np.empty((0, 3), np.float32)
        cloud, added = inject(empty, [([30., 0., 0.], [.5, .5, .5])], return_added=True)
        self.assertGreater(len(cloud), 0)
        np.testing.assert_array_equal(cloud, added[0])
        self.assertTrue(_matches_injected({'bbox': [[29.7, -.3, -.3], [30.3, .3, .3]]}, added[0]))
        self.assertFalse(_matches_injected({'bbox': [[29.7, 2., -.3], [30.3, 3., .3]]}, added[0]))
        np.testing.assert_array_equal(inject(empty, [([30., 0., 0.], [.5, .5, .5])]), cloud)
        unchanged, no_added = inject(cloud, [], return_added=True)
        np.testing.assert_array_equal(unchanged, cloud)
        self.assertEqual(no_added, [])

    def test_injection_preserves_shared_read_only_input(self):
        hits, _ = box_hits([30., 0., 0.], [.5, .5, .5])
        points = np.vstack([hits * 1.5, hits[:1] * .5, [[20., 3., 0.]]]).astype(np.float32)
        before = points.copy()
        points.flags.writeable = False
        cloud, added = inject(points, [([30., 0., 0.], [.5, .5, .5])], return_added=True)
        self.assertGreater(len(added[0]), 0)
        np.testing.assert_array_equal(points, before)
        self.assertFalse(points.flags.writeable)
        self.assertFalse(np.shares_memory(cloud, points))


if __name__ == '__main__':
    unittest.main()
