"""Run INSIDE the ROS image: real DDS delivery of recorded PointCloud2 messages.

Serial delivery verifies integration, not 10 Hz throughput. Input DB is read-only.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0,'/app')
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from metro_interfaces.msg import ClearanceStatus
from visualization_msgs.msg import MarkerArray
from tf2_msgs.msg import TFMessage
from rclpy.qos import QoSProfile, DurabilityPolicy
from metro.ros_node import MetroNode
from metro.pipeline import PointCloudPipeline
from metro.tunnel import TunnelConfig


def main():
    ap=argparse.ArgumentParser();ap.add_argument('bag',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--start-frame',type=int,default=0,help='First source frame in the single SQLite source bag')
    ap.add_argument('--input-timeout',type=float,default=2.,help='Smoke watchdog threshold; not a throughput requirement')
    args=ap.parse_args()
    if args.start_frame<0:ap.error('--start-frame must be nonnegative')
    if args.input_timeout<=0:ap.error('--input-timeout must be positive')
    files=sorted(args.bag.rglob('*.db3'))
    if len(files)!=1:raise ValueError('Smoke expects exactly one SQLite file in '+str(args.bag))
    with sqlite3.connect(files[0].resolve().as_uri()+'?mode=ro',uri=True) as db:
        topics=db.execute("SELECT id FROM topics WHERE type='sensor_msgs/msg/PointCloud2'").fetchall()
        if len(topics)!=1:raise ValueError('Smoke expects exactly one PointCloud2 topic')
        rows=db.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp,id LIMIT 3 OFFSET ?',
            (topics[0][0],args.start_frame)).fetchall()
    assert len(rows)==3,'Need three source frames'
    config=TunnelConfig()
    reference=PointCloudPipeline(config)
    messages=[deserialize_message(blob,PointCloud2) for blob, in rows]
    expected_frames=[reference.process(msg) for msg in messages]
    ros_args=['--ros-args','-p','input_timeout_s:='+str(args.input_timeout),
              '-p','max_result_age_s:='+str(args.input_timeout),
              '-p','worker_timeout_s:='+str(args.input_timeout)]
    rclpy.init(args=ros_args)
    detector=MetroNode();probe=rclpy.create_node('metro_smoke_probe')
    pub=probe.create_publisher(PointCloud2,'/lidar_points',qos_profile_sensor_data)
    received=[];typed=[];markers=[];transforms=[]
    sub=probe.create_subscription(String,'/metro/obstacle',lambda msg:received.append(json.loads(msg.data)),10)
    typed_sub=probe.create_subscription(ClearanceStatus,'/metro/clearance',typed.append,10)
    marker_sub=probe.create_subscription(MarkerArray,'/metro/debug/markers',markers.append,10)
    tf_sub=probe.create_subscription(TFMessage,'/tf_static',transforms.append,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    executor=SingleThreadedExecutor();executor.add_node(detector);executor.add_node(probe)
    report={'transport':'actual ROS2 Humble DDS PointCloud2 -> MetroNode -> String JSON',
            'mode':'serial integration smoke; not throughput benchmark',
            'config':asdict(config),'timestamp_basis':'header.stamp',
            'source_start_frame':args.start_frame,'input_timeout_s':args.input_timeout,'frames':[]}
    try:
        until=time.monotonic()+10
        while pub.get_subscription_count()==0 and time.monotonic()<until:executor.spin_once(timeout_sec=.1)
        assert pub.get_subscription_count()>0,'DDS discovery failed'
        until=time.monotonic()+20
        while not detector.runtime.ready and time.monotonic()<until:executor.spin_once(timeout_sec=.02)
        assert detector.runtime.ready,'Worker startup failed'
        for msg,expected in zip(messages,expected_frames):
            start=len(received);pub.publish(msg)
            until=time.monotonic()+15;actual=None
            while time.monotonic()<until:
                executor.spin_once(timeout_sec=.1)
                matches=[r for r in received[start:] if r.get('stamp_nanosec')==msg.header.stamp.nanosec and r.get('stamp_sec')==msg.header.stamp.sec]
                if matches:actual=matches[-1];break
            assert actual is not None,'No detection message'
            assert actual['status']==expected['status'],(actual['status'],expected['status'])
            assert actual['distance_m']==expected['distance_m']
            assert actual['frame_id']=='metro_lidar_analysis'
            assert actual['source_frame_id']==msg.header.frame_id
            assert actual['stamp_sec']==msg.header.stamp.sec
            assert actual['stamp_nanosec']==msg.header.stamp.nanosec
            assert actual['stamp_s']==msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
            for field in ['model_revision','obstacle','valid','source_to_analysis','candidates',
                          'analysis_range_m','geometry_contract','floor','path','section_frames']:
                assert actual.get(field)==expected.get(field),'Mismatch in '+field
            if 'path' in actual:
                assert actual['geometry_contract']['height_datum']=='rail_heads'
                assert actual['geometry_contract']['curve_margin_m']==0.
                assert actual['vehicle']['envelope_mode']=='section'
            report['frames'].append({k:actual.get(k) for k in ['model_id','model_revision','status','obstacle','valid',
                'distance_m','analysis_range_m','geometry_contract','frame_id','source_frame_id',
                'stamp_s','stamp_sec','stamp_nanosec','decode_and_detect_ms','runtime']})
        start=len(received);until=time.monotonic()+args.input_timeout+3
        while time.monotonic()<until and not any(r.get('reason')=='input_timeout' for r in received[start:]):
            executor.spin_once(timeout_sec=.1)
        assert any(r.get('reason')=='input_timeout' and r['status']=='unknown' and r['valid'] is False for r in received[start:])
        assert transforms and markers and typed,'Missing typed/TF/debug messages'
        for msg, expected in zip(messages, expected_frames):
            matches=[m for m in typed if m.has_source_stamp and m.header.stamp==msg.header.stamp]
            assert matches,'No typed result for source stamp'
            actual=matches[-1]
            assert actual.obstacle==expected['obstacle'] and actual.valid==expected['valid']
            assert len(actual.candidates)==len(expected.get('candidates',[]))
            assert actual.header.frame_id=='metro_lidar_analysis'
        import numpy as np
        from scipy.spatial.transform import Rotation
        tf=transforms[-1].transforms[0]
        assert tf.header.frame_id==messages[0].header.frame_id and tf.child_frame_id=='metro_lidar_analysis'
        q=tf.transform.rotation
        matrix=np.eye(4);matrix[:3,:3]=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
        matrix[:3,3]=[tf.transform.translation.x,tf.transform.translation.y,tf.transform.translation.z]
        np.testing.assert_allclose(matrix@reference.transform,np.eye(4),atol=1e-6)
        assert markers[-1].markers[0].action==3,'Unknown must clear stale markers'
        report.update(status='PASS',watchdog='PASS',typed='PASS',tf_roundtrip='PASS',markers='PASS')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report))
    finally:
        executor.shutdown();probe.destroy_node();detector.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
