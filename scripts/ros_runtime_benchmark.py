"""Actual DDS fault-injection benchmark. Artificial processor, NOT accuracy data.

Run in the built ROS image to check the process-based runtime under load and stalls.
Separate publisher process keeps the 10 Hz source independent of detector stalls.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import struct
import time
import sys
sys.path.insert(0, '/app')
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String


class ArtificialProcessor:
    def __init__(self, options=None):
        self.analysis_frame = 'metro_lidar_analysis'
        self.reset()

    def reset(self):
        self.frames = 0

    def process(self, msg):
        time.sleep(struct.unpack('<f', bytes(msg.data[:4]))[0])
        self.frames += 1
        return dict(status='clear', valid=True, obstacle=False, distance_m=None,
            frame_id=self.analysis_frame, source_frame_id=msg.header.frame_id,
            stamp_sec=msg.header.stamp.sec, stamp_nanosec=msg.header.stamp.nanosec,
            candidates=[], artificial=True)


def artificial_factory(options):
    return ArtificialProcessor()


def publish_sequence(delay, count, output, bag=None, start_frame=60):
    rclpy.init(args=[])
    node = rclpy.create_node('fault_source')
    pub = node.create_publisher(PointCloud2, '/lidar_points', qos_profile_sensor_data)
    until = time.monotonic()+10
    while pub.get_subscription_count()==0 and time.monotonic()<until:
        rclpy.spin_once(node, timeout_sec=.02)
    if pub.get_subscription_count()==0:
        output.put(dict(error='source DDS discovery failed'))
        return
    messages=[]
    if bag:
        import sqlite3
        from rclpy.serialization import deserialize_message
        files=sorted(Path(bag).rglob('*.db3'))
        assert len(files)==1
        with sqlite3.connect(files[0].resolve().as_uri()+'?mode=ro',uri=True) as db:
            topic=db.execute("SELECT id FROM topics WHERE type='sensor_msgs/msg/PointCloud2'").fetchone()[0]
            blobs=db.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp,id LIMIT ? OFFSET ?',
                             (topic,count,start_frame)).fetchall()
        messages=[deserialize_message(blob,PointCloud2) for blob, in blobs]
        assert len(messages)==count
    start = time.monotonic()
    stamps = []; header_stamps=[]
    for i in range(count):
        deadline = start+i*.1
        time.sleep(max(0, deadline-time.monotonic()))
        stamp = time.monotonic_ns()
        msg = PointCloud2(height=1, width=1, point_step=12, row_step=12, is_dense=True,
            fields=[PointField(name=n, offset=j*4, datatype=7, count=1) for j, n in enumerate('xyz')],
            data=struct.pack('<fff', delay, 0., 0.))
        msg.header.frame_id='fault_lidar'
        msg.header.stamp.sec, msg.header.stamp.nanosec=divmod(stamp, 1000000000)
        if messages:msg=messages[i]
        pub.publish(msg)
        stamps.append(stamp)
        header_stamps.append(msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec)
    # Keep participant alive long enough to flush the final sample.
    time.sleep(.2)
    output.put(dict(stamps=stamps,header_stamps=header_stamps,count=count,delay_s=delay))
    node.destroy_node()
    rclpy.shutdown()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=['normal','overload','stall'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--bag', type=Path)
    ap.add_argument('--start-frame', type=int, default=60)
    ap.add_argument('--paused-clock', action='store_true')
    args=ap.parse_args()
    delay, count={'normal':(.02, 10), 'overload':(.45, 10), 'stall':(1., 1)}[args.scenario]
    if args.bag:count=12
    timeout=.6 if args.bag else .25
    budget=.3 if args.bag else .2
    ros_args=['--ros-args','-p','input_timeout_s:='+str(timeout)]
    ros_args+=['-p','max_result_age_s:='+str(budget),'-p','worker_timeout_s:='+str(5. if args.bag else .75)]
    if args.paused_clock:ros_args+=['-p','use_sim_time:=true']
    rclpy.init(args=ros_args)
    from metro.ros_node import MetroNode
    detector=MetroNode() if args.bag else MetroNode(factory=artificial_factory)
    probe=rclpy.create_node('fault_probe')
    received=[]
    sub=probe.create_subscription(String, '/metro/obstacle',
        lambda msg: received.append((time.monotonic_ns(),json.loads(msg.data))), 100)
    executor=SingleThreadedExecutor();executor.add_node(detector);executor.add_node(probe)
    context=mp.get_context('spawn');queue=context.Queue()
    publisher=context.Process(target=publish_sequence,args=(delay,count,queue,args.bag,args.start_frame))
    try:
        deadline=time.monotonic()+20
        while not detector.runtime.ready and time.monotonic()<deadline:
            executor.spin_once(timeout_sec=.02)
        assert detector.runtime.ready
        # Allow result-topic discovery before the source starts.
        until=time.monotonic()+.3
        while time.monotonic()<until:executor.spin_once(timeout_sec=.02)
        publisher.start()
        source=None;stop=None
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            executor.spin_once(timeout_sec=.01)
            if source is None and not queue.empty():
                source=queue.get(timeout=1);stop=time.monotonic()+2
            if stop and time.monotonic()>stop:break
        assert source and 'error' not in source, source
        first,last=source['stamps'][0],source['stamps'][-1]
        ages=[];unknown_after=[];compute=[]
        sent_by_stamp=dict(zip(source['header_stamps'],source['stamps']))
        for arrived,result in received:
            if arrived<first:continue
            if result['valid']:
                stamp=result['stamp_sec']*1000000000+result['stamp_nanosec']
                ages.append((arrived-sent_by_stamp[stamp])/1e6)
                if 'decode_and_detect_ms' in result:compute.append(result['decode_and_detect_ms'])
            if result.get('reason')=='input_timeout' and arrived>=last:
                unknown_after.append((arrived-last)/1e6)
        counters=detector.runtime.counters
        report=dict(scenario=args.scenario,transport='actual DDS',paused_clock=args.paused_clock,
            artificial_processor=not bool(args.bag),source_rate_hz=10.,input_timeout_ms=timeout*1000,freshness_budget_ms=budget*1000,
            source_header_stamps=source['header_stamps'],source_send_intervals_ms=[(b-a)/1e6 for a,b in zip(source['stamps'],source['stamps'][1:])],
            decode_and_detect_ms=compute,
            source_frames=count,valid_results=len(ages),valid_older_than_budget=sum(a>budget*1000 for a in ages),
            delivery_age_ms=ages,input_timeout_after_last_source_ms=min(unknown_after) if unknown_after else None,
            unknown_reasons=[r.get('reason') for t,r in received if t>=first and not r['valid']],
            counters=counters)
        assert not report['valid_older_than_budget'],report
        assert report['input_timeout_after_last_source_ms'] is not None,report
        assert report['input_timeout_after_last_source_ms']<timeout*1000+200,report
        if args.scenario=='normal' and not args.bag:assert report['valid_results']>=8,report
        if args.scenario=='stall' and not args.bag:assert counters['restarts']>=1,report
        report['status']='PASS'
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
    finally:
        if publisher.pid:
            publisher.join(timeout=1)
            if publisher.is_alive():publisher.terminate();publisher.join(timeout=1)
        executor.shutdown();probe.destroy_node();detector.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
