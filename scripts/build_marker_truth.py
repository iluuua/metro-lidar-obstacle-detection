"""Group locally supplied marker boxes using a user-provided object map.

The labels JSON is a list of objects with id, name, label and marker_ids.
No recording annotations or predefined marker IDs are included.

    python scripts/build_marker_truth.py --markers /path/to/markers.json \
        --labels /path/to/groups.json --output viewer/data/my_run/ground-truth.json
"""
import argparse
import json
from pathlib import Path


def build(markers, groups):
    mapping = {}
    for group in groups:
        if group['label'] not in ('positive', 'negative', 'ambiguous'):
            raise ValueError('Unknown object label')
        for marker_id in group['marker_ids']:
            if marker_id in mapping:
                raise ValueError('A marker belongs to more than one group')
            mapping[marker_id] = group
    frames = []
    for frame in markers['frames']:
        objects = {}
        for marker in frame['markers']:
            group = mapping.get(marker['marker_id'])
            if group is None:
                raise ValueError('Marker ID is missing from the supplied labels')
            obj = objects.setdefault(group['id'], dict(
                id=group['id'], name=group['name'], label=group['label'], parts=[]))
            obj['parts'].append(marker['bbox'])
        frames.append(dict(frame=frame['frame'], time_s=frame['time_s'],
                           objects=list(objects.values())))
    return dict(source=markers.get('source', 'Local annotations'),
                sequence=markers.get('sequence', ''), frames=frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--markers', required=True, type=Path)
    parser.add_argument('--labels', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = build(json.loads(args.markers.read_text(encoding='utf-8')),
                   json.loads(args.labels.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')),
                           encoding='utf-8')


if __name__ == '__main__':
    main()
