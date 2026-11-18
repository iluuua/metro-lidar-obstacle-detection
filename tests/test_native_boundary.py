"""Sparse returns at the fixed gauge edge warrant attention, never an alarm.

Keep the tunnel and train stationary and disable far-field and geometry pooling
so the tests isolate accumulation of boundary observations.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from metro import native
from test_native import make
from test_tunnel import SENSOR_OVER_RAIL, tunnel_frame


def edge_returns(kind, outside=False):
    if kind == 'side':
        lateral = (1.08, 1.10, 1.12) if outside else (.99, 1.01, 1.03)
        height = (.70, .85, 1.)
    else:
        lateral = (-.10, 0., .10)
        height = (3.04, 3.06, 3.08) if outside else (2.94, 2.96, 2.98)
    return np.column_stack([[60.1, 60.2, 60.3], lateral,
                            np.asarray(height) - SENSOR_OVER_RAIL]).astype(np.float32)


@unittest.skipUnless(native.available(), 'libmetro_core is not built')
class NativeBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.background = tunnel_frame(seed=23, n=24000, xmax=145.)

    def detector(self):
        det = make(explain=True, farfield=False, geometry_frames=1)
        if not det._h:
            raise ValueError(native.library().metro_create_error().decode())
        for k in range(3):
            r = det.process(self.background, None, .1 * k)
            self.assertFalse(r['obstacle'])
        return det

    def feed(self, det, targets):
        results = []
        for k, target in enumerate(targets):
            cloud = self.background if target is None else np.vstack([self.background, target])
            results.append(det.process(cloud, None, .3 + .1 * k))
        return results

    @staticmethod
    def target_candidates(result):
        return [c for c in result.get('candidates', []) if 58. < c['distance_m'] < 62.]

    def assert_sparse_attention(self, kind):
        results = self.feed(self.detector(), [edge_returns(kind)] * 6)
        self.assertFalse(any(r['obstacle'] for r in results))
        for r in results:
            candidates = self.target_candidates(r)
            self.assertTrue(candidates)
            self.assertGreater(r['analysis_range_m'], 65.)
            self.assertTrue(all(c['unique_point_count'] == 3 for c in candidates))
            self.assertTrue(all(c['confident_points'] == 0 for c in candidates))
        self.assertFalse(any(c.get('boundary') for r in results[:2] for c in self.target_candidates(r)))
        self.assertTrue(any(r['status'] == 'candidate' and any(c.get('boundary') for c in self.target_candidates(r))
                            for r in results[2:]))

    def test_confident_independent_edge_returns_confirm_on_third_frame(self):
        for kind, points in (
            ('side', [[60.1,.72,.7-SENSOR_OVER_RAIL],[60.2,.76,.9-SENSOR_OVER_RAIL]]),
            ('top', [[60.1,-.1,2.72-SENSOR_OVER_RAIL],[60.2,.1,2.76-SENSOR_OVER_RAIL]])):
            with self.subTest(kind=kind):
                target = np.repeat(np.array(points,np.float32),2,axis=0)
                fast = self.feed(self.detector(), [target] * 4)
                baseline = make(edge_confirmation=False, farfield=False, geometry_frames=1)
                for k in range(3): baseline.process(self.background, None, .1*k)
                slow = self.feed(baseline, [target] * 4)
                self.assertFalse(any(r['obstacle'] for r in fast[:2]))
                self.assertTrue(fast[2]['obstacle'])
                self.assertTrue(any(c.get('edge_confirmed') for c in self.target_candidates(fast[2])))
                self.assertFalse(slow[2]['obstacle'])
                self.assertTrue(slow[3]['obstacle'])

    def test_duplicated_single_return_does_not_count_as_independent_edge_evidence(self):
        target = np.repeat(edge_returns('side')[:1], 20, axis=0)
        results = self.feed(self.detector(), [target] * 6)
        self.assertFalse(any(r['obstacle'] for r in results))
        self.assertFalse(any(c.get('boundary') for r in results for c in self.target_candidates(r)))

    def test_sparse_lateral_edge_gives_only_attention(self):
        self.assert_sparse_attention('side')

    def test_sparse_upper_edge_gives_only_attention(self):
        self.assert_sparse_attention('top')

    def test_one_edge_frame_does_not_give_boundary_attention(self):
        for kind in ('side', 'top'):
            with self.subTest(edge=kind):
                results = self.feed(self.detector(), [edge_returns(kind)] + [None] * 5)
                self.assertFalse(any(r['obstacle'] for r in results))
                self.assertFalse(any(c.get('boundary') for r in results for c in self.target_candidates(r)))
                self.assertTrue(all(c['confident_points'] == 0 for c in self.target_candidates(results[0])))

    def test_returns_outside_gauge_do_not_give_alarm_or_boundary_attention(self):
        for kind in ('side', 'top'):
            with self.subTest(edge=kind):
                results = self.feed(self.detector(), [edge_returns(kind, outside=True)] * 6)
                self.assertFalse(any(r['obstacle'] for r in results))
                self.assertFalse(any(c.get('boundary') for r in results for c in self.target_candidates(r)))

    def test_attention_that_keeps_its_place_gets_stronger_but_is_never_an_alarm(self):
        results = self.feed(self.detector(), [edge_returns('side')] * 24)
        self.assertFalse(any(r['obstacle'] for r in results))
        levels = [r.get('attention_level', 0) for r in results]
        self.assertEqual(max(levels[:8]), 1)  # first plain ATTENTION
        self.assertEqual(levels[-1], 2)  # then the stronger level while the object stays
        self.assertTrue(any(c.get('attention_strong') for c in results[-1]['candidates']))
        flat = make(explain=True, farfield=False, geometry_frames=1, attention_strong_frames=0)
        for k in range(3):
            flat.process(self.background, None, .1 * k)
        for k in range(24):
            r = flat.process(np.vstack([self.background, edge_returns('side')]), None, .3 + .1 * k)
        self.assertEqual(r.get('attention_level', 0), 1)


if __name__ == '__main__':
    unittest.main()
