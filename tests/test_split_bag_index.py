import json
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from index_split_bag import index_bag


class SplitBagIndexTests(unittest.TestCase):
    def part(self, root, name, timestamps):
        with closing(sqlite3.connect(root/name)) as db:
            db.execute('CREATE TABLE topics(id INTEGER,name TEXT,type TEXT,serialization_format TEXT,offered_qos_profiles TEXT)')
            db.execute("INSERT INTO topics VALUES(1,'/points','sensor_msgs/msg/PointCloud2','cdr','')")
            db.execute('CREATE TABLE messages(topic_id INTEGER,timestamp INTEGER,data BLOB)')
            db.executemany('INSERT INTO messages VALUES(1,?,?)',[(t,b'payload') for t in timestamps])
            db.commit()

    def test_parts_are_ordered_by_time_and_all_messages_are_counted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.part(root,'part_10.db3',[100,110])
            self.part(root,'part_2.db3',[120,130,140])
            info=index_bag(root)
            self.assertEqual(info['relative_file_paths'],['part_10.db3','part_2.db3'])
            self.assertEqual(info['message_count'],5)
            self.assertEqual(info['topics_with_message_count'][0]['message_count'],5)
            self.assertEqual(info['duration']['nanoseconds'],40)
            self.assertEqual(json.loads((root/'metadata.yaml').read_text())['rosbag2_bagfile_information'],info)
            with self.assertRaises(FileExistsError): index_bag(root)

    def test_overlapping_parts_are_rejected_without_writing_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.part(root,'part_0.db3',[100,120])
            self.part(root,'part_1.db3',[110,130])
            with self.assertRaises(ValueError): index_bag(root)
            self.assertFalse((root/'metadata.yaml').exists())
