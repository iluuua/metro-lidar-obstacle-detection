import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'eval'))
from projection_audit import closest_coordinates


class ProjectionReferenceTests(unittest.TestCase):
    def test_straight_graded_banked_section_recovers_known_offsets(self):
        x=np.arange(61,dtype=float)
        path=np.column_stack((x,.2*x,.05*x,np.zeros(len(x))))
        tangent=np.array([1.,.2,.05]);tangent/=np.linalg.norm(tangent)
        lateral=np.array([-.2,1.,0.]);lateral/=np.linalg.norm(lateral)
        phi=.08
        up=np.cross(tangent,lateral)
        lateral=(lateral+phi*up)/np.sqrt(1+phi**2)
        up=np.cross(tangent,lateral)
        center=np.array([20.,4.,1.])
        expected=np.array([[1.05,.5],[-1.2,2.],[.1,3.]])
        points=center+expected[:,0,None]*lateral+expected[:,1,None]*up
        u,h,t=closest_coordinates(points,path,phi,spatial=True)
        np.testing.assert_allclose(u,expected[:,0],atol=1e-9)
        np.testing.assert_allclose(h,expected[:,1],atol=1e-9)
        np.testing.assert_allclose(t,20.,atol=1e-9)

    def test_curved_path_recovers_perpendicular_foot_between_grid_nodes(self):
        x=np.arange(61,dtype=float)
        path=np.column_stack((x,.005*x*x,np.zeros(len(x)),np.zeros(len(x))))
        t0=20.35;d=.01*t0
        normal=np.array([-d,1.,0.])/np.sqrt(1+d*d)
        center=np.array([t0,.005*t0*t0,0.])
        points=np.array([center+normal*offset+[0,0,.5] for offset in [1.05,-1.4]])
        u,h,t=closest_coordinates(points,path)
        np.testing.assert_allclose(u,[1.05,-1.4],atol=1e-9)
        np.testing.assert_allclose(t,t0,atol=1e-9)
        np.testing.assert_allclose(h,.5,atol=1e-9)
