import unittest
from types import SimpleNamespace as NS
import numpy as np
from metro.pipeline import PointCloudPipeline,DEFAULT_TRANSFORM
from metro.io import decode


def message():
    points=np.array([[2,3,4],[0,0,0],[float('nan'),1,2]],dtype='<f4')
    return NS(fields=[NS(name=n,datatype=7,offset=i*4) for i,n in enumerate(['x','y','z'])],
              data=points.tobytes(),height=1,width=3,point_step=12,row_step=36,is_bigendian=False,
              header=NS(frame_id='source_sensor',stamp=NS(sec=11,nanosec=123456789)))


class PipelineTests(unittest.TestCase):
    def test_organizer_point_layouts_with_and_without_ring(self):
        for step, with_ring in ((16,False),(26,True)):
            dtype=np.dtype({'names':['x','y','z','intensity']+(['ring','timestamp'] if with_ring else []),
                'formats':['<f4']*4+(['<u2','<f8'] if with_ring else []),
                'offsets':[0,4,8,12]+([16,18] if with_ring else []),'itemsize':step})
            data=np.zeros(2,dtype=dtype)
            data['x']=[1,2];data['y']=[-10,-20];data['z']=[.5,1]
            fields=[NS(name=name,offset=dtype.fields[name][1],datatype=4 if name=='ring' else 8 if name=='timestamp' else 7)
                    for name in dtype.names]
            msg=NS(fields=fields,data=data.tobytes(),height=1,width=2,point_step=step,row_step=2*step,is_bigendian=False)
            xyz,rings=decode(msg)
            np.testing.assert_allclose(xyz,[[10,1,.5],[20,2,1]])
            self.assertEqual(rings is not None,with_ring)

    def test_configurable_rigid_transform_and_explicit_output_frame(self):
        transform=np.eye(4);transform[:3,3]=[1,2,3]
        points,_=decode(message(),source_to_analysis=transform)
        np.testing.assert_allclose(points,[[3,5,7]])
        pipeline=PointCloudPipeline(source_to_analysis=transform)
        result=pipeline.process(message())
        self.assertEqual(result['source_frame_id'],'source_sensor')
        self.assertEqual(result['frame_id'],'metro_lidar_analysis')
        self.assertEqual(result['stamp_nanosec'],123456789)
        self.assertFalse(result['valid'])

    def test_bad_transform_rejected(self):
        with self.assertRaises(ValueError):PointCloudPipeline(source_to_analysis=np.zeros((4,4)))
        self.assertEqual(PointCloudPipeline().config.max_range,300.)

    def test_reach_follows_the_lidar(self):
        # organizers: full marks for the envelope up to 200 m for the Pandar128 and 400 m for the Falcon
        falcon=message();falcon.fields=falcon.fields+[NS(name='scan_id',datatype=2,offset=8)]
        for msg,lidar,reach in ((message(),'Hesai Pandar128',300.),(falcon,'Seyond Falcon Prime',400.)):
            pipeline=PointCloudPipeline(backend='python')
            result=pipeline.process(msg)
            self.assertEqual(result['lidar'],lidar)
            self.assertEqual(min(reach,330.),pipeline.detector.cfg.max_range)
        explicit=PointCloudPipeline(config=PointCloudPipeline().config.__class__(max_range=250.),backend='python')
        explicit.process(falcon)
        self.assertEqual(explicit.detector.cfg.max_range,250.)

    def test_decode_failure_resets_temporal_state(self):
        pipeline=PointCloudPipeline();previous=pipeline.detector
        broken=message();broken.data=b''
        with self.assertRaises(ValueError):pipeline.process(broken)
        self.assertIsNot(previous,pipeline.detector)


if __name__=='__main__':unittest.main()
