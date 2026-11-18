"""Random access to original SQLite ROS bags; no cloud files are exported."""
from contextlib import closing
from pathlib import Path
import sqlite3
import threading

import numpy as np
from rosbags.typesys import Stores, get_typestore
from ruamel.yaml import YAML

from metro.io import AXES, decode


class SourceClouds:
    def __init__(self):
        self.lock = threading.Lock()
        self.indices = {}
        self.typestore = get_typestore(Stores.ROS2_HUMBLE)
        self.decode_slots = threading.Semaphore(2)

    def index(self, summary):
        bag = Path(summary['source_bag']).resolve()
        metadata = bag / 'metadata.yaml'
        key = (str(bag), metadata.stat().st_mtime_ns, summary['source_topic'],
               summary.get('source_start_s', 0), summary.get('source_duration_s'),
               summary.get('source_every', 1), summary.get('source_max_frames'), summary['frames'])
        with self.lock:
            if key in self.indices:
                return self.indices[key]
            info = YAML(typ='safe').load(metadata.read_text(encoding='utf-8'))['rosbag2_bagfile_information']
            start = info['starting_time']['nanoseconds_since_epoch'] + round(summary.get('source_start_s', 0) * 1e9)
            duration = summary.get('source_duration_s')
            stop = None if duration is None else start + round(duration * 1e9)
            rows = []
            for name in info['relative_file_paths']:
                path = (bag / name).resolve()
                if not path.is_relative_to(bag) or path.suffix != '.db3':
                    raise ValueError('Invalid SQLite bag part')
                with closing(sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
                    for mid, stamp in db.execute(
                        'SELECT m.id,m.timestamp FROM messages m JOIN topics t ON t.id=m.topic_id '
                        'WHERE t.name=? AND t.type=? AND m.timestamp>=? ORDER BY m.timestamp,m.id',
                        (summary['source_topic'], 'sensor_msgs/msg/PointCloud2', start)):
                        if stop is None or stamp < stop:
                            rows.append((stamp, path, mid))
            rows.sort(key=lambda r: (r[0], str(r[1]), r[2]))
            rows = rows[::summary.get('source_every', 1)]
            limit = summary.get('source_max_frames')
            if limit is not None:
                rows = rows[:limit]
            if len(rows) != summary['frames']:
                raise ValueError('Source frame count differs from viewer manifest')
            self.indices[key] = rows
            return rows

    def read(self, summary, frame):
        rows = self.index(summary)
        if frame < 0 or frame >= len(rows):
            raise ValueError('Unknown frame')
        _, path, mid = rows[frame]
        with self.decode_slots:
            with closing(sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
                row = db.execute('SELECT data FROM messages WHERE id=?', (mid,)).fetchone()
            if row is None:
                raise ValueError('Source message missing')
            msg = self.typestore.deserialize_cdr(row[0], 'sensor_msgs/msg/PointCloud2')
            axis = summary.get('sensor_forward_axis', '-y')
            transform = None if axis == '-y' else np.asarray(AXES[axis]).reshape(4, 4)
            points, _ = decode(msg, min_distance=0., source_to_analysis=transform)
            return points.astype('<f4', copy=False).tobytes()
