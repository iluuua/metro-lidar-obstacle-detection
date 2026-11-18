"""ROS serialization and debug geometry; no detector decisions are made here."""
import numpy as np
from geometry_msgs.msg import Point, TransformStamped
from metro_interfaces.msg import ClearanceStatus, ObstacleCandidate
from visualization_msgs.msg import Marker, MarkerArray


def status_message(result):
    msg = ClearanceStatus()
    msg.header.frame_id = result['frame_id']
    msg.has_source_stamp = 'stamp_sec' in result
    if msg.has_source_stamp:
        msg.header.stamp.sec = result['stamp_sec']
        msg.header.stamp.nanosec = result['stamp_nanosec']
    msg.source_frame_id = result.get('source_frame_id', '')
    msg.state = {'clear': msg.CLEAR, 'candidate': msg.CANDIDATE,
                 'obstacle': msg.OBSTACLE}.get(result['status'], msg.UNKNOWN)
    msg.valid = result['valid']
    msg.reason = result.get('reason', '')
    msg.obstacle = result['obstacle']
    msg.attention_level = int(result.get('attention_level', 0))
    msg.has_distance = result.get('distance_m') is not None
    msg.distance_along_path_m = float(result['distance_m']) if msg.has_distance else 0.
    msg.analysis_range_m = float(result.get('analysis_range_m', 0.))
    msg.envelope_range_m = float(max(result.get('envelope_range_m') or 0., msg.analysis_range_m))
    msg.model_revision = str(result.get('model_revision', ''))
    runtime = result.get('runtime', {})
    for key in ('generation', 'sequence', 'receive_to_publish_ms', 'compute_ms', 'max_result_age_s'):
        if key in runtime:
            setattr(msg, key, runtime[key])
    counts = runtime.get('counters', {})
    for target, key in [('received_frames', 'received'), ('superseded_frames', 'superseded'),
                        ('stale_results', 'stale_results'), ('worker_restarts', 'restarts')]:
        setattr(msg, target, counts.get(key, 0))
    for index, candidate in enumerate(result.get('candidates', [])):
        item = ObstacleCandidate(candidate_index=index)
        item.bbox_min, item.bbox_max = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in candidate['bbox']]
        item.distance_along_path_m = float(candidate['distance_m'])
        item.point_count = candidate['point_count']
        item.unique_point_count = candidate.get('unique_point_count', 0)
        item.ring_count = candidate.get('rings', 0)
        item.confirmation_hits = candidate.get('hits', 0)
        item.confirmed = candidate.get('confirmed', False)
        msg.candidates.append(item)
    return msg


def analysis_transform(source_frame, analysis_frame, source_to_analysis):
    # TF describes pose of CHILD in PARENT. Our decode matrix maps source -> analysis.
    from scipy.spatial.transform import Rotation
    inverse = np.linalg.inv(np.asarray(source_to_analysis).reshape(4, 4))
    msg = TransformStamped()
    msg.header.frame_id, msg.child_frame_id = source_frame, analysis_frame
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, inverse[:3, 3])
    q = Rotation.from_matrix(inverse[:3, :3]).as_quat()
    msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w = map(float, q)
    return msg


# RViz fonts have no Cyrillic glyphs, so the on-screen state is in English
STATE_TEXT = {'obstacle': 'OBSTACLE', 'candidate': 'WARNING', 'clear': 'CLEAR',
              'unknown': 'INSUFFICIENT DATA'}
STATE_COLOR = {'obstacle': (1., .25, .25), 'candidate': (1., .65, .1), 'clear': (.65, .3, 1.), 'unknown': (.6, .6, .6)}


def debug_markers(result, header):
    markers = [Marker(action=Marker.DELETEALL)]
    status = 'obstacle' if result.get('obstacle') else result.get('status', 'unknown')
    text = Marker(header=header, ns='state', id=0, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
    text.pose.position.x, text.pose.position.z = 4., 5.
    text.pose.orientation.w = 1.
    text.scale.z = .6
    text.text = STATE_TEXT.get(status, status.upper())
    if status == 'candidate' and result.get('attention_level', 0) >= 2:
        text.text = 'STRONG WARNING'
    if status == 'obstacle' and result.get('distance_m') is not None:
        text.text += f" {result['distance_m']:.1f} m"
    envelope = max(result.get('envelope_range_m') or 0., result.get('analysis_range_m') or 0.)
    if envelope:
        text.text += f" | envelope to {envelope:.0f} m"
    text.color.r, text.color.g, text.color.b = STATE_COLOR.get(status, STATE_COLOR['unknown'])
    text.color.a = 1.
    text.lifetime.nanosec = 500000000
    markers.append(text)
    for index, candidate in enumerate(result.get('candidates', [])):
        low, high = np.asarray(candidate['bbox'])
        marker = Marker(header=header, ns='candidates', id=index, type=Marker.CUBE, action=Marker.ADD)
        marker.pose.position = Point(**dict(zip(('x', 'y', 'z'), map(float, (low+high)/2))))
        marker.pose.orientation.w = 1.
        marker.scale.x, marker.scale.y, marker.scale.z = map(float, np.maximum(high-low, .03))
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (1., .15 if candidate.get('confirmed') else .65, .1, .6)
        marker.lifetime.nanosec = 500000000
        markers.append(marker)
    sections = result.get('section_frames', [])
    if sections:
        contract = result.get('geometry_contract', {})
        width = contract.get('section_width_m', 2.1)
        height = contract.get('section_height_m', 3.)
        marker = Marker(header=header, ns='clearance', id=0, type=Marker.LINE_LIST, action=Marker.ADD)
        marker.pose.orientation.w = 1.
        marker.scale.x = .025
        # the envelope takes the colour of the state: red with an obstacle, orange while a candidate waits
        marker.color.r, marker.color.g, marker.color.b = STATE_COLOR.get(status, STATE_COLOR['clear'])
        marker.color.a = 1.
        marker.lifetime.nanosec = 500000000
        previous = None
        for row in sections:
            center, lateral, up = np.asarray(row[1:4]), np.asarray(row[4:7]), np.asarray(row[7:10])
            corners = [center+lateral*x+up*z for x, z in [(-width/2, 0), (width/2, 0), (width/2, height), (-width/2, height)]]
            edges = [(corners[i], corners[(i+1)%4]) for i in range(4)]
            if previous is not None:
                edges += list(zip(previous, corners))
            for edge in edges:
                marker.points.extend(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in edge)
            previous = corners
        markers.append(marker)
    return MarkerArray(markers=markers)
