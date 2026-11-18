"""Final label of every object in datasets/labels/<run>.json: the review by eye where there is one
(datasets/labels/<run>.review.json), otherwise the organizers' marker, otherwise the code's verdict at close range.

    final:        enters / outside / boundary / dust / structure / unclear
    should_alarm: true (an obstacle the train would hit), false, or null (the disputed edge cubes: neither way)
    source:       review / marker / code

Rules of the code verdict: an object of fewer than 5 returns or seen in fewer than 3 frames is a speck (dust);
on the recordings without obstacles an object inside the envelope is track equipment (structure).

The verdict is written to the object and to each of its rows, so a row (one candidate in one frame) carries the
label of the whole track: far frames get the answer that is only known near.

    .venv/bin/python eval/label_merge.py            # all runs with labels
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / 'datasets' / 'labels'
# recordings without obstacles: whatever the code saw inside the envelope there is track equipment or noise
CLEAN = set()
SPECK_POINTS, SPECK_FRAMES = 5, 3  # an object never larger than this or seen in fewer frames is a speck


def from_marker(marker):
    if not marker:
        return None
    if marker['label'] == 'positive':
        return 'enters', True
    if marker['label'] == 'ambiguous':
        return 'boundary', None
    return ('dust' if 'dust' in marker['id'] else 'outside'), False


def from_code(obj, run):
    v = obj['verdict']
    if v == 'enters' and (obj['max_points'] < SPECK_POINTS or obj['n'] < SPECK_FRAMES):
        return 'dust', False
    if v == 'enters' and run in CLEAN:
        return 'structure', False
    if v == 'enters':
        return 'enters', True
    if v == 'outside':
        return 'outside', False
    if v == 'boundary':
        return 'boundary', False
    return 'unclear', False


def merge(run):
    path = LABELS / f'{run}.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    review_path = LABELS / f'{run}.review.json'
    review = {}
    if review_path.is_file():
        review = {r['object']: r for r in json.loads(review_path.read_text(encoding='utf-8'))['objects']}
    final = {}
    for obj in data['tracks']:
        r = review.get(obj['object'])
        if r:
            label, alarm, source = r['verdict'], r['should_alarm'], 'review'
            if r['verdict'] == 'boundary' and (obj.get('marker') or {}).get('label') == 'ambiguous':
                alarm = None
        elif from_marker(obj.get('marker')):
            (label, alarm), source = from_marker(obj['marker']), 'marker'
        else:
            (label, alarm), source = from_code(obj, run), 'code'
        obj.update(final=label, should_alarm=alarm, source=source)
        final[obj['object']] = (label, alarm)
    for row in data['rows']:
        if row.get('object') in final:
            row['final'], row['should_alarm'] = final[row['object']]
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    counts = {}
    for label, alarm in final.values():
        counts[(label, alarm)] = counts.get((label, alarm), 0) + 1
    return counts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--clean-runs', nargs='*', default=[])
    args = parser.parse_args()
    CLEAN.update(args.clean_runs)
    runs = args.runs
    for run in runs:
        counts = merge(run)
        print(run, ', '.join(f'{k[0]}/{k[1]}: {v}' for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))


if __name__ == '__main__':
    main()
