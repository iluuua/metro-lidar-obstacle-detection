"""ROS 2 Humble adapter with a supervised, single-owner geometry process."""
import json
import time
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from visualization_msgs.msg import MarkerArray
from metro_interfaces.msg import ClearanceStatus
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from .pipeline import PointCloudPipeline, DEFAULT_TRANSFORM
from .tunnel import TunnelConfig
from .runtime import LatestFrameRuntime, pipeline_factory
from .ros_messages import status_message, analysis_transform, debug_markers


# Topics tried first when input_topic is 'auto'; any other PointCloud2 topic is
# picked up by discovery. The first topic that delivers a cloud is used.
KNOWN_INPUTS = ('/lidar_points', '/sensing/lidar/hesai128/pointcloud', '/hesai/pandar', '/points_raw',
                '/pandar_points', '/lidar/points', '/iv_points')
OWN_TOPICS = ('/metro/',)


class MetroNode(Node):
    def __init__(self, *, factory=pipeline_factory):
        super().__init__('metro_obstacle_detector')
        descriptor = ParameterDescriptor(read_only=True, description='Startup parameter; restart node to change')
        defaults = dict(input_topic='auto', analysis_frame='metro_lidar_analysis', source_to_analysis=DEFAULT_TRANSFORM,
            sensor_axes='auto', backend='auto', rediscover_after_s=2.,
            input_timeout_s=.6, max_result_age_s=.5, worker_timeout_s=2., startup_timeout_s=20.,
            max_worker_restarts=3, max_cloud_bytes=67108864,
            publish_tf=True, publish_markers=True)
        for key, value in defaults.items():
            self.declare_parameter(key, value, descriptor)
        value = lambda key: self.get_parameter(key).value
        config = TunnelConfig()
        for key, default in vars(config).items():
            if isinstance(default, tuple):
                self.declare_parameter(key, [float(v) for v in default], descriptor)
                setattr(config, key, tuple(float(v) for v in value(key)))
            else:
                self.declare_parameter(key, default, descriptor)
                setattr(config, key, value(key))
        # An explicit source_to_analysis wins; otherwise the forward axis is named or found from the cloud.
        explicit = list(value('source_to_analysis')) != list(DEFAULT_TRANSFORM)
        # with found axes another sensor may follow (a new recording played into the same node)
        self.sensor_switch = not explicit and value('sensor_axes') == 'auto'
        options = dict(config=config, source_to_analysis=value('source_to_analysis') if explicit else value('sensor_axes'),
            analysis_frame=value('analysis_frame'), backend=value('backend'))
        # Validate configuration synchronously, before advertising a working runtime.
        PointCloudPipeline(**options)
        self.analysis_frame = options['analysis_frame']
        self.source_frame, self.published_transform = None, None
        self.pub = self.create_publisher(String, '/metro/obstacle', 1)
        self.typed_pub = self.create_publisher(ClearanceStatus, '/metro/clearance', 1)
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 1)
        self.marker_pub = self.create_publisher(MarkerArray, '/metro/debug/markers', 1) if value('publish_markers') else None
        # the cloud the detector works on, whatever the input topic: RViz always finds it here
        self.cloud_pub = self.create_publisher(PointCloud2, '/metro/debug/cloud', QoSProfile(
            depth=1, durability=DurabilityPolicy.VOLATILE, reliability=ReliabilityPolicy.BEST_EFFORT)) \
            if value('publish_markers') else None
        # the simplest answer from the task: is there an obstacle, and how far along the track
        self.flag_pub = self.create_publisher(Bool, '/metro/obstacle_detected', 1)
        self.distance_pub = self.create_publisher(Float32, '/metro/obstacle_distance', 1)
        self.tf = StaticTransformBroadcaster(self) if value('publish_tf') else None
        self.runtime = LatestFrameRuntime(options, factory=factory,
            input_timeout_s=value('input_timeout_s'), max_result_age_s=value('max_result_age_s'),
            worker_timeout_s=value('worker_timeout_s'), startup_timeout_s=value('startup_timeout_s'),
            max_restarts=value('max_worker_restarts'), shared_cloud=True,
            max_cloud_bytes=value('max_cloud_bytes'))
        self.subs, self.input_topic = {}, None
        requested = value('input_topic')
        # an automatically found input is given up after a silence, so the next recording may
        # come on another topic (the Falcon publishes /iv_points)
        self.auto_input = not requested or requested == 'auto'
        self.rediscover_after_s, self.last_cloud = value('rediscover_after_s'), None
        if requested and requested != 'auto':
            self._subscribe(requested)
            self.input_topic = requested
        else:
            for topic in KNOWN_INPUTS:
                self._subscribe(topic)
        # finds the input topic and follows the publishers' reliability
        self.discovery = self.create_timer(.5, self._discover)
        # A paused /clock must not freeze input/worker supervision.
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(.005, self.check_input, clock=self.steady_clock)
        self.reset_service = self.create_service(Trigger, '~/reset', self.on_reset)
        self.last_diagnostic = 0.
        self.last_result = dict(status='unknown', valid=False, reason='starting')
        self.get_logger().info('Worker starting')

    def _qos(self, topic):
        """Reliable when every publisher is reliable: DDS then resends lost fragments
        of a large cloud instead of dropping the whole cloud under load. Best effort
        otherwise, because it connects to any publisher (a sensor driver included)."""
        infos = self.get_publishers_info_by_topic(topic)
        reliable = bool(infos) and all(i.qos_profile.reliability == ReliabilityPolicy.RELIABLE for i in infos)
        return QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE,
                          reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT)

    def _subscribe(self, topic):
        qos = self._qos(topic)
        old = self.subs.get(topic)
        if old is not None:
            if old.qos_profile.reliability == qos.reliability:
                return
            self.destroy_subscription(self.subs.pop(topic))
        self.subs[topic] = self.create_subscription(
            PointCloud2, topic, lambda msg, t=topic: self.on_cloud(msg, t), qos)

    def _discover(self):
        if self.input_topic is not None:
            self._subscribe(self.input_topic)
            return
        for topic, types in self.get_topic_names_and_types():
            if 'sensor_msgs/msg/PointCloud2' in types and not topic.startswith(OWN_TOPICS):
                self._subscribe(topic)

    def on_cloud(self, msg, topic=None):
        self.last_cloud = time.monotonic()
        if topic is not None and self.input_topic is None:
            self.input_topic = topic
            self.get_logger().info('Input point cloud topic: ' + topic)
            for other in [t for t in self.subs if t != topic]:
                self.destroy_subscription(self.subs.pop(other))
        elif topic is not None and topic != self.input_topic:
            return
        source = msg.header.frame_id
        if not source or source == self.analysis_frame:
            self.runtime.reset('invalid_source_frame')
            return
        if self.source_frame is not None and source != self.source_frame:
            # Do not apply a calibration given for one sensor to another frame.
            if not self.sensor_switch:
                self.runtime.reset('invalid_source_frame')
                return
            # another sensor: its axes are found again, history starts anew, TF gets the new edge
            self.get_logger().info(f'New sensor frame {source} (was {self.source_frame}): detector reset')
            self.runtime.reset('new_source_frame')
            self.published_transform = None
        self.source_frame = source
        stamp = int(msg.header.stamp.sec)*1000000000 + int(msg.header.stamp.nanosec)
        self.runtime.submit(msg, stamp_ns=stamp, source_frame=source)
        if self.cloud_pub is not None and self.cloud_pub.get_subscription_count():
            self.cloud_pub.publish(msg)

    def on_reset(self, request, response):
        self.runtime.reset()
        response.success, response.message = True, 'Temporal evidence invalidated; worker resets before next frame'
        return response

    def check_input(self):
        for result in self.runtime.poll():
            result.setdefault('frame_id', self.analysis_frame)
            result.setdefault('source_frame_id', self.source_frame or '')
            self.publish_result(result)
        now = time.monotonic()
        if (self.auto_input and self.input_topic is not None and self.last_cloud is not None
                and now - self.last_cloud > self.rediscover_after_s):
            # paused or finished: listen to every cloud topic again; the same topic resumes as before
            self.input_topic, self.last_cloud = None, None
        if now-self.last_diagnostic >= 1.:
            self.last_diagnostic = now
            detail = self.last_result.get('reason') or self.last_result['status']
            diag = DiagnosticStatus(name='metro/perception', hardware_id='Pandar128E3X',
                level=DiagnosticStatus.OK if self.last_result['valid'] else DiagnosticStatus.WARN,
                message=detail)
            diag.values = [KeyValue(key=key, value=str(value)) for key, value in self.runtime.counters.items()]
            self.diag_pub.publish(DiagnosticArray(header=self._now_header(), status=[diag]))

    def _now_header(self):
        from std_msgs.msg import Header
        return Header(stamp=self.get_clock().now().to_msg(), frame_id=self.analysis_frame)

    def publish_tf(self, result):
        # the transform is known once the worker has seen a cloud (the forward axis may be found from it)
        matrix = result.get('source_to_analysis')
        if not self.tf or not matrix or not self.source_frame or matrix == self.published_transform:
            return
        transform = analysis_transform(self.source_frame, self.analysis_frame, matrix)
        transform.header.stamp = self.get_clock().now().to_msg()
        self.tf.sendTransform(transform)
        self.published_transform = matrix

    def publish_result(self, result):
        self.publish_tf(result)
        typed = status_message(result)
        text = json.dumps(result, ensure_ascii=False, allow_nan=False)
        self.typed_pub.publish(typed)
        self.pub.publish(String(data=text))
        obstacle = bool(result.get('obstacle')) and result.get('distance_m') is not None
        self.flag_pub.publish(Bool(data=obstacle))
        self.distance_pub.publish(Float32(data=float(result['distance_m']) if obstacle else float('nan')))
        if self.marker_pub:
            self.marker_pub.publish(debug_markers(result, typed.header))
        self.last_result = result

    def destroy_node(self):
        if hasattr(self, 'runtime'):
            self.runtime.close()
        return super().destroy_node()


def main():
    rclpy.init()
    node = MetroNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
