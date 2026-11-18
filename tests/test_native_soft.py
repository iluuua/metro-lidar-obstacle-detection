"""Continuous accumulation of evidence at the gauge edge in the core (soft_evidence): a small body held
inside the edge is confirmed after a couple of seconds; the same body right at the edge or outside it,
or the core with the accumulation off, gives at most attention.

The train and tunnel are still, far-field and geometry pooling are off, as in test_native_boundary,
so only the accumulation decides.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from metro import native
from test_native import make
from test_tunnel import SENSOR_OVER_RAIL, tunnel_frame


def body(lateral):
    """Distinct returns of a small body 60 m ahead, 0.7-1.0 m above the rail heads."""
    lateral = np.asarray(lateral, float)
    return np.column_stack([60.1 + .1 * np.arange(len(lateral)), lateral,
                            np.linspace(.7, 1., len(lateral)) - SENSOR_OVER_RAIL]).astype(np.float32)


@unittest.skipUnless(native.available(), 'libmetro_core is not built')
class NativeSoftConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.background = tunnel_frame(seed=23, n=24000, xmax=145.)

    def feed(self, target, frames=40, **cfg):
        det = make(explain=True, farfield=False, geometry_frames=1, **cfg)
        for k in range(3):
            det.process(self.background, None, .1 * k)
        cloud = np.vstack([self.background, target])
        return [det.process(cloud, None, .3 + .1 * k) for k in range(frames)]

    @staticmethod
    def rules(results):
        return {c['explain']['rule'] for r in results for c in r.get('candidates', []) if c.get('confirmed')}

    def test_body_held_inside_the_edge_is_confirmed_by_accumulation(self):
        for side in (1, -1):
            with self.subTest(side=side):
                # 11-17 cm inside the 1.05 m edge: less than the uncertainty margin, no confident point
                results = self.feed(body(side * np.array([.88, .90, .92, .94])))
                self.assertFalse(any(r['obstacle'] for r in results[:10]))
                self.assertTrue(all(r['obstacle'] for r in results[30:]))
                self.assertEqual(self.rules(results), {'confirmed_soft'})
                self.assertTrue(any(c.get('soft_confirmed') for c in results[-1]['candidates']))

    def test_without_accumulation_the_same_body_gives_attention_only(self):
        results = self.feed(body([.88, .90, .92, .94]), soft_confirmation=False)
        self.assertFalse(any(r['obstacle'] for r in results))
        self.assertTrue(all(r['status'] == 'candidate' for r in results[3:]))

    def test_body_at_the_edge_or_outside_is_never_confirmed(self):
        for lateral in ([.98, 1., 1.02, 1.04], [1.07, 1.09, 1.11, 1.13]):
            with self.subTest(lateral=lateral):
                results = self.feed(body(lateral))
                self.assertFalse(any(r['obstacle'] for r in results))


if __name__ == '__main__':
    unittest.main()
