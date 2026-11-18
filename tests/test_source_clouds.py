import json
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from source_clouds import SourceClouds


class SourceCloudTests(unittest.TestCase):
    def test_split_bag_window_stride_and_frame_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for part, stamps in [('later.db3', [400, 500, 600]), ('first.db3', [100, 200, 300])]:
                with closing(sqlite3.connect(root / part)) as db:
                    db.execute('CREATE TABLE topics(id INTEGER,name TEXT,type TEXT)')
                    db.execute("INSERT INTO topics VALUES(1,'/points','sensor_msgs/msg/PointCloud2')")
                    db.execute('CREATE TABLE messages(id INTEGER PRIMARY KEY,topic_id INTEGER,timestamp INTEGER,data BLOB)')
                    db.executemany('INSERT INTO messages(topic_id,timestamp,data) VALUES(1,?,?)',
                                   [(t,b'payload') for t in stamps])
                    db.commit()
            (root / 'metadata.yaml').write_text(json.dumps({'rosbag2_bagfile_information': {
                'starting_time': {'nanoseconds_since_epoch':100},
                'relative_file_paths':['later.db3','first.db3']}}))
            summary = dict(source_bag=str(root), source_topic='/points', source_start_s=1e-7,
                           source_duration_s=4e-7, source_every=2, source_max_frames=2, frames=2)
            clouds = SourceClouds()
            rows = clouds.index(summary)
            self.assertEqual([r[0] for r in rows], [200,400])
            self.assertEqual([r[1].name for r in rows], ['first.db3','later.db3'])
            with self.assertRaises(ValueError): clouds.read(summary,-1)
            with self.assertRaises(ValueError): clouds.read(summary,2)
            bad = dict(summary, source_max_frames=1, frames=2)
            with self.assertRaises(ValueError): clouds.index(bad)
