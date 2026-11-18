import math
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'eval'),str(Path(__file__).resolve().parent)]
from metro import native
from axis_reversal import reversal
from test_native import make
from test_tunnel import tunnel_frame


@unittest.skipUnless(native.available(),'native core unavailable')
class ContinuousPathTests(unittest.TestCase):
    def test_curved_track_is_preserved_and_central_obstacle_detected(self):
        detector=make(continuous_path=True);baseline=make()
        results=[]
        for i in range(12):
            points=tunnel_frame(i,bend=1/300,obstacle=(65,0,.6,.5),seed=i)
            results.append(detector.process(points,None,.1*i))
            original=baseline.process(points,None,.1*i)
        self.assertTrue(any(r['obstacle'] for r in results))
        path=np.asarray(results[-1]['path'])
        at80=np.interp(80,path[:,0],path[:,1])
        base_path=np.asarray(original['path']);baseline_error=abs(np.interp(80,base_path[:,0],base_path[:,1])-80**2/600)
        # Existing profile registration is already biased on this noisy scene;
        # the new prior must preserve the turn without adding >15 cm error.
        self.assertLessEqual(abs(at80-80**2/600),baseline_error+.15)
        self.assertLess(np.max(np.abs(np.diff(path[:,1],2)/4)),1/100+1e-4)

    def test_time_gap_discards_path_history(self):
        detector=make(continuous_path=True)
        for i in range(4):detector.process(tunnel_frame(i,bend=1/400,seed=i,n=24000),None,.1*i)
        points=tunnel_frame(0,bend=0,seed=77,n=24000)
        after=detector.process(points,None,10.)
        fresh=make(continuous_path=True).process(points,None,10.)
        np.testing.assert_allclose(after['path'],fresh['path'],atol=1e-7)

    def test_clear_curve_does_not_become_obstacle(self):
        detector=make(continuous_path=True)
        for i in range(8):
            result=detector.process(tunnel_frame(i,bend=1/300,seed=i),None,.1*i)
            self.assertFalse(result['obstacle'])

    def test_spatial_strategy_filters_near_geometry_in_actual_kernel(self):
        # legacy without the per-node display filter: compares the computed axis, which is what decisions use
        smooth=make(spatial_path=True,path_tau=.35);legacy=make(spatial_path=False,display_smoothing=0)
        points=tunnel_frame(0,seed=12,n=24000);new=[];old=[]
        for i in range(32):
            p=points.copy();p[:,1]+=.02*np.sin(i*1.7)
            a=smooth.process(p,None,.1*i);b=legacy.process(p,None,.1*i)
            new.append(a['path'][10][1]);old.append(b['path'][10][1])
            path=np.asarray(a['path'])
            self.assertLess(np.max(np.abs(np.diff(path[:,1],2)/4)),.0101)
            self.assertFalse(a['obstacle'])
        self.assertLess(np.std(new[8:]),.8*np.std(old[8:]))

    def test_spatial_strategy_keeps_curve_and_detects_central_obstacle(self):
        detector=make(spatial_path=True,path_tau=.35)
        results=[detector.process(tunnel_frame(i,bend=1/300,obstacle=(65,0,.6,.5),seed=i),None,.1*i) for i in range(12)]
        self.assertTrue(any(r['obstacle'] for r in results))
        p=np.asarray(results[-1]['path']);self.assertGreater(np.interp(80,p[:,0],p[:,1]),8.)
        self.assertLess(np.max(abs(np.diff(p[:,1],2)/4)),.0101)

    def test_axis_never_bends_back_beyond_measured_support(self):
        for options in (dict(spatial_path=True,path_tau=.35),dict(continuous_path=True)):
            for bend in (1/300,-1/300,0.):
                detector=make(**options)
                for i in range(10):
                    result=detector.process(tunnel_frame(i,bend=bend,seed=i),None,.1*i)
                    path=np.asarray(result['path']);support=max(result['path_info']['support_m'],30.)
                    beyond=path[:,0]>=support
                    self.assertTrue(result['path_info']['no_reversal'])
                    self.assertLess(reversal(path[beyond,0],path[beyond,1]),math.radians(1))
