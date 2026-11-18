import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'eval'))
from soft_threat import Accumulator,observation


class SoftThreatTests(unittest.TestCase):
    def first(self,z):
        acc=Accumulator(tau=3,reference=.3,threshold=.3)
        for i in range(300):
            if acc.observe(i*.1,z,1):return i*.1
        return None

    def test_deep_inside_faster_than_edge_and_slight_exterior(self):
        self.assertLess(self.first(2),self.first(0))
        self.assertLess(self.first(0),self.first(-.25))

    def test_weak_exterior_does_not_grow_without_bound(self):
        self.assertIsNone(self.first(-.5))
        self.assertIsNone(self.first(-2))

    def test_clear_exterior_reduces_score_and_gap_resets(self):
        a=Accumulator()
        a.observe(0,2,1);a.observe(.1,2,1)
        previous=a.value
        a.observe(.2,-2,1)
        self.assertLess(a.value,previous)
        a.observe(2,-2,1)
        self.assertEqual(a.value,0)
        self.assertEqual(a.hits,1)

    def test_duplicate_ray_and_held_candidate_supply_no_evidence(self):
        self.assertIsNone(observation(dict(unique_point_count=1,point_count=100)))
        self.assertIsNone(observation(dict(unique_point_count=10,held=True)))
