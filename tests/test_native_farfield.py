"""Contracts for sparse far returns and the end of the verified C++ envelope of a long-range lidar
(max_range 400 m, as the pipeline sets for the Seyond Falcon).

The background is identical between frames: a standing train must not acquire
synthetic odometry from independently resampled tunnel walls. Only the injected
returns change. No recorded datasets or viewer files are needed.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from metro import native
from test_native import make
from test_tunnel import SENSOR_OVER_RAIL, tunnel_frame


def returns(distance=254., lateral=0.):
    """Three distinct neighbouring returns, too low for ordinary far confidence."""
    return np.array([[distance + .10, lateral - .06, .40 - SENSOR_OVER_RAIL],
                     [distance + .35, lateral + .07, .55 - SENSOR_OVER_RAIL],
                     [distance + .60, lateral, .70 - SENSOR_OVER_RAIL]], np.float32)


def configured(**overrides):
    det = make(**{'max_range': 400., **overrides})
    # An older library rejects the new keys. Report the missing contract as a
    # normal test error instead of calling the C API with a null handle.
    if not det._h:
        raise ValueError(native.library().metro_create_error().decode())
    return det


@unittest.skipUnless(native.available(), 'libmetro_core is not built')
class NativeFarfieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # End at the last supported slab centre. A half-slab tail beyond the
        # support would itself enter the straight-extension structure check.
        cls.short = tunnel_frame(seed=23, n=24000, xmax=175.)
        cls.long = tunnel_frame(seed=23, n=24000, xmax=295.)

    def far_detector(self):
        return configured(farfield=True, far_window=15, far_attention_hits=6,
                          far_confirm_hits=9, far_min_points=8,
                          far_min_range=120., far_max_sigma_u=.6,
                          sigma_vert0=.5)

    def feed(self, det, background, targets, start=0.):
        results = []
        for k, target in enumerate(targets):
            cloud = background if target is None else np.vstack([background, target])
            results.append(det.process(cloud, None, start + .1 * k))
        return results

    def warm(self, det, background):
        result = self.feed(det, background, [None] * 3)
        self.assertFalse(any(r['obstacle'] for r in result))
        return .3

    @staticmethod
    def target_candidates(result, distance=254.):
        return [c for c in result.get('candidates', [])
                if abs(c['distance_m'] - distance) < 2.]

    def test_sparse_central_target_and_isolated_side_returns_do_not_cut_extension(self):
        det = configured()
        start = self.warm(det, self.short)
        isolated = np.array([[252., -1.7, .6 - SENSOR_OVER_RAIL],
                             [256., 1.7, 1.2 - SENSOR_OVER_RAIL]], np.float32)
        cloud = np.vstack([returns(), isolated])
        results = self.feed(det, self.short, [cloud] * 3, start)
        for r in results:
            self.assertTrue(r['path_info']['extension_ok'])
            self.assertGreater(r['path_info']['support_m'], 160.)
            self.assertGreater(r['analysis_range_m'], 280., r['path_info'])

    def test_dense_transverse_wall_still_cuts_extension(self):
        det = configured()
        start = self.warm(det, self.short)
        u, h = np.meshgrid(np.linspace(-1.9, 1.9, 32), np.linspace(.3, 3.3, 16))
        wall = np.column_stack([np.full(u.size, 250.), u.ravel(), h.ravel() - SENSOR_OVER_RAIL]).astype(np.float32)
        results = self.feed(det, self.short, [wall] * 3, start)
        for r in results:
            self.assertGreater(r['analysis_range_m'], 190.)
            self.assertLess(r['analysis_range_m'], 250.)

    def test_new_cross_section_overrides_a_recent_far_geometry_certificate(self):
        # Individual far frames have too few wall returns. Recent aligned frames
        # certify the straight tunnel; a newly visible wall must revoke that
        # certificate before the corridor starts searching only beyond its end.
        det = configured(explain=True)
        near = self.long[self.long[:, 0] < 175.]
        far = self.long[self.long[:, 0] >= 175.]
        for k in range(18):
            r = det.process(np.vstack([near, far[k::120]]), None, .1 * k)
        self.assertGreater(r['path_info']['support_m'], 280.)
        self.assertGreaterEqual(r['path_info']['explain']['geometry_frames'], 3)

        u, h = np.meshgrid(np.linspace(-1.9, 1.9, 32), np.linspace(.3, 3.3, 16))
        wall = np.column_stack([np.full(u.size, 210.), u.ravel(),
                                h.ravel() - SENSOR_OVER_RAIL]).astype(np.float32)
        r = det.process(np.vstack([near, wall]), None, 1.8)
        self.assertLess(r['path_info'].get('break_m', np.inf), 220.)
        self.assertLess(r['analysis_range_m'], 210., r['path_info'])
        # the decision range ends at the wall at once; the drawn envelope follows once the wall is seen again and again
        for k in range(4):
            r = det.process(np.vstack([near, wall]), None, 1.9 + .1 * k)
        self.assertLess(r['envelope_range_m'], 210., r['path_info'])

    def test_persistent_sparse_far_returns_give_attention_then_obstacle(self):
        det = self.far_detector()
        start = self.warm(det, self.long)
        results = self.feed(det, self.long, [returns()] * 15, start)
        self.assertTrue(any(r['status'] == 'candidate' for r in results[:8]))
        self.assertTrue(any(r['obstacle'] for r in results[8:]))
        self.assertTrue(results[-1]['obstacle'])
        for r in results:
            self.assertGreater(r['analysis_range_m'], 260.)
            self.assertTrue(all(c['confident_points'] == 0 for c in self.target_candidates(r)))

    def test_one_sparse_frame_does_not_confirm(self):
        det = self.far_detector()
        start = self.warm(det, self.long)
        results = self.feed(det, self.long, [returns()] + [None] * 14, start)
        self.assertFalse(any(r['obstacle'] for r in results))

    def test_duplicated_single_ray_does_not_gain_independent_evidence(self):
        det = configured(explain=True)
        start = self.warm(det, self.long)
        spot = np.array([[90., .1, .8 - SENSOR_OVER_RAIL]] * 4, np.float32)
        results = self.feed(det, self.long, [spot] * 15, start)
        candidates = self.target_candidates(results[-1], distance=90.)
        self.assertTrue(candidates)
        self.assertTrue(any(c['confident_points'] >= 2 for c in candidates))
        self.assertTrue(all(c['evidence_obs'] <= 1 for c in candidates))
        self.assertFalse(any(r['obstacle'] for r in results))

    def test_multiray_body_retains_evidence_when_only_one_ray_is_confident(self):
        # the mechanism is tested with the evidence thresholds before the long-ride calibration, which asks for more
        det = configured(explain=True, evidence_near=4., evidence_per_m=.08)
        start = self.warm(det, self.long)
        # A second physical return supports the body, even though it is too
        # close to the lower margin to be confident. This is not a lone glint.
        body = np.array([[90., .1, .4 - SENSOR_OVER_RAIL],
                         [90., .1, .4 - SENSOR_OVER_RAIL],
                         [90.1, .15, .1 - SENSOR_OVER_RAIL]], np.float32)
        results = self.feed(det, self.long, [body] * 15, start)
        candidates = self.target_candidates(results[-1], distance=90.)
        self.assertTrue(any(c['unique_point_count'] >= 2 for c in candidates))
        self.assertTrue(any(c['evidence_obs'] > 1 for c in candidates))
        self.assertTrue(any(r['obstacle'] for r in results))

    def test_far_uncertainty_is_bounded_by_actual_lateral_clearance(self):
        # A centre target has more lateral clearance than the same target near the edge.
        # Persistence must not let an uncertain edge bypass the spatial requirement.
        results = []
        for lateral in (0., .6):
            det = configured(farfield=True, far_max_sigma_u=.9, sigma_lat0=.75,
                             sigma_lat_per_m=0., sigma_lat_per_m_rail=0.,
                             sigma_lat_per_m_far=0., sigma_vert0=.5)
            start = self.warm(det, self.long)
            results.append(self.feed(det, self.long, [returns(lateral=lateral)] * 15, start))
        self.assertTrue(any(r['obstacle'] for r in results[0]))
        self.assertFalse(any(r['obstacle'] for r in results[1]))

    def test_wandering_depths_do_not_confirm(self):
        det = self.far_detector()
        start = self.warm(det, self.long)
        targets = [returns(s) for s in (180., 205., 230., 255., 280.) * 3]
        results = self.feed(det, self.long, targets, start)
        self.assertFalse(any(r['obstacle'] for r in results))

    def test_repeated_returns_outside_fixed_gauge_do_not_confirm(self):
        det = self.far_detector()
        start = self.warm(det, self.long)
        results = self.feed(det, self.long, [returns(lateral=1.35)] * 15, start)
        self.assertFalse(any(r['obstacle'] for r in results))

    def test_low_return_does_not_widen_a_clipped_far_vertical_facet(self):
        det = self.far_detector()
        start = self.warm(det, self.long)
        # The far returns lie on a laterally clipped face. A nearby low return
        # joins its cluster but lies outside the far evidence height band. It
        # cannot turn that face into a body with independently measured width.
        face = returns(lateral=-.65)
        face[:, 1] = -.65
        low = np.array([[254.3, -.4, .10 - SENSOR_OVER_RAIL]], np.float32)
        results = self.feed(det, self.long, [np.vstack([face, low])] * 15, start)
        candidates = self.target_candidates(results[-1])
        self.assertTrue(candidates)
        self.assertGreater(candidates[0]['u_range'][1] - candidates[0]['u_range'][0], .05)
        self.assertTrue(all(c['confident_points'] == 0 for c in candidates))
        self.assertFalse(any(r['obstacle'] for r in results))

    def test_timestamp_gap_resets_sparse_far_evidence(self):
        det = self.far_detector()
        start = self.warm(det, self.long)
        before = self.feed(det, self.long, [returns()] * 8, start)
        self.assertTrue(any(r['status'] == 'candidate' for r in before))
        self.assertFalse(any(r['obstacle'] for r in before))
        after = self.feed(det, self.long, [returns()] * 3, start=6.)
        self.assertFalse(any(r['obstacle'] for r in after))

    def test_explain_is_opt_in(self):
        default = configured()
        enabled = configured(explain=True)
        for det, expected in ((default, False), (enabled, True)):
            start = self.warm(det, self.long)
            r = self.feed(det, self.long, [returns()], start)[0]
            candidates = self.target_candidates(r)
            self.assertTrue(candidates)
            self.assertTrue(all(('explain' in c) == expected for c in candidates))
            json.dumps(r, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
