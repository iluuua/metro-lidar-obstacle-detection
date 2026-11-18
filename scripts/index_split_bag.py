"""Restore ROS 2 metadata for consecutive SQLite parts without copying their payloads."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3


def index_bag(folder):
    folder = Path(folder).resolve()
    parts, topics = [], {}
    for file in folder.glob('*.db3'):
        with closing(sqlite3.connect(file.as_uri() + '?mode=ro&immutable=1', uri=True)) as db:
            count, start, end = db.execute('SELECT count(*), min(timestamp), max(timestamp) FROM messages').fetchone()
            if not count:
                continue
            for topic_id, name, msgtype, serialization, qos in db.execute(
                    'SELECT id,name,type,serialization_format,offered_qos_profiles FROM topics'):
                entry = topics.setdefault(name, dict(topic_metadata=dict(name=name, type=msgtype,
                                          serialization_format=serialization, offered_qos_profiles=qos), message_count=0))
                if (entry['topic_metadata']['type'], entry['topic_metadata']['serialization_format']) != (msgtype, serialization):
                    raise ValueError(f'Inconsistent topic {name} in {file.name}')
                entry['message_count'] += db.execute('SELECT count(*) FROM messages WHERE topic_id=?', (topic_id,)).fetchone()[0]
        parts.append(dict(path=file.name, start=start, end=end, count=count))
    if not parts:
        raise ValueError('No nonempty SQLite bag parts found')
    parts.sort(key=lambda p: p['start'])
    if any(b['start'] <= a['end'] for a, b in zip(parts, parts[1:])):
        raise ValueError('Overlapping parts: inspect them before joining into one recording')
    files = [dict(path=p['path'], starting_time=dict(nanoseconds_since_epoch=p['start']),
                  duration=dict(nanoseconds=p['end']-p['start']), message_count=p['count']) for p in parts]
    metadata = dict(rosbag2_bagfile_information=dict(version=5, storage_identifier='sqlite3',
                    duration=dict(nanoseconds=parts[-1]['end']-parts[0]['start']),
                    starting_time=dict(nanoseconds_since_epoch=parts[0]['start']),
                    message_count=sum(p['count'] for p in parts), topics_with_message_count=list(topics.values()),
                    compression_format='', compression_mode='', relative_file_paths=[p['path'] for p in parts], files=files))
    target = folder / 'metadata.yaml'
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    # JSON is also valid YAML, including the nanosecond integers without precision loss.
    target.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata['rosbag2_bagfile_information']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    info = index_bag(parser.parse_args().folder)
    print(json.dumps(dict(parts=len(info['files']), messages=info['message_count'], duration=info['duration'])))
