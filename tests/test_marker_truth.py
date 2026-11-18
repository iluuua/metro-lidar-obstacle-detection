from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from import_marker_truth import marker_box


class MarkerTruthTests(unittest.TestCase):
    def marker(self,action=0):
        q=np.sqrt(.5)
        return NS(action=action,id=23,pose=NS(position=NS(x=2.,y=-3.,z=1.),orientation=NS(x=0.,y=0.,z=q,w=q)),
                  points=[NS(x=-1.,y=0.,z=0.),NS(x=1.,y=0.,z=0.)])

    def test_pose_rotation_and_sensor_axis_are_applied_before_bounds(self):
        box=marker_box(self.marker())
        self.assertEqual(box['marker_id'],23)
        np.testing.assert_allclose(box['bbox'],[[2,2,1],[4,2,1]],atol=1e-6)

    def test_deleted_marker_is_not_an_object(self):
        self.assertIsNone(marker_box(self.marker(action=2)))
