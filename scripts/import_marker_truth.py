"""Import locally supplied ROS MarkerArray annotations for an exported viewer run.

Example:
    python scripts/import_marker_truth.py /path/to/your/part.db3 --sequence my_run
"""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from rosbags.typesys import Stores, get_typestore
from metro.io import AXES


def marker_box(marker, forward_axis='-y'):
    if marker.action != 0 or not marker.points:
        return None
    position = marker.pose.position
    rotation = marker.pose.orientation
    local = np.array([[point.x, point.y, point.z] for point in marker.points])
    world = Rotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).apply(local)
    world += [position.x, position.y, position.z]
    transform = np.asarray(AXES[forward_axis]).reshape(4, 4)
    analysis = world @ transform[:3, :3].T + transform[:3, 3]
    return dict(marker_id=marker.id,
                bbox=[analysis.min(0).round(6).tolist(), analysis.max(0).round(6).tolist()])


def import_db(path, sequence, forward_axis='-y'):
    if not sequence.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Sequence id must contain only letters, digits, _ or -')
    folder = ROOT / 'viewer' / 'data' / sequence
    model = json.loads((folder / 'model.tunnel.json').read_text(encoding='utf-8'))
    by_stamp = {frame['header_stamp_ns']: frame for frame in model['frames']}
    store = get_typestore(Stores.ROS2_HUMBLE)
    frames = []
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
        topic = db.execute(
            "SELECT id FROM topics WHERE type='visualization_msgs/msg/MarkerArray'"
        ).fetchall()
        if len(topic) != 1:
            raise ValueError('Expected exactly one MarkerArray topic')
        for raw, in db.execute(
                'SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp,id', topic[0]):
            msg = store.deserialize_cdr(raw, 'visualization_msgs/msg/MarkerArray')
            stamps = {marker.header.stamp.sec * 10**9 + marker.header.stamp.nanosec
                      for marker in msg.markers}
            if not stamps:
                continue
            if len(stamps) != 1:
                raise ValueError('Mixed marker timestamps')
            stamp = stamps.pop()
            if stamp not in by_stamp:
                continue
            base = by_stamp[stamp]
            boxes = [box for marker in msg.markers
                     if (box := marker_box(marker, forward_axis)) is not None]
            frames.append(dict(frame=base['frame'], time_s=base['time_s'],
                               header_stamp_ns=stamp, markers=boxes))
    payload = dict(source='Local MarkerArray', sequence=sequence,
                   alignment='header_stamp_ns', frames=sorted(frames, key=lambda frame: frame['frame']))
    target = folder / 'ground-truth.markers.json'
    target.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(f'Wrote {len(frames)} annotated frames to {target}')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path, help='local ROS 2 SQLite .db3 file')
    parser.add_argument('--sequence', required=True, help='existing viewer run id')
    parser.add_argument('--forward-axis', choices=tuple(AXES), default='-y')
    args = parser.parse_args()
    import_db(args.database, args.sequence, args.forward_axis)


if __name__ == '__main__':
    main()
