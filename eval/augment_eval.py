"""What the core did on the injected objects (eval/augment_all.py): by depth into the envelope, by shape, and the false alarms.

    python eval/augment_eval.py experiment      # one run
    python eval/augment_eval.py experiment baseline         # side by side (the same objects if v50 used --ref-dll of v57)
    python eval/augment_eval.py experiment --write docs/AUGMENTATION_RESULTS.md

Per object: it is found when a candidate that holds its injected returns is confirmed at least once; the first alarm
range is the distance to the object at the first such frame. Inside objects are grouped by how deep they are in the
envelope (`depth_m`), outside objects by how far outside. A false alarm is a confirmed candidate that holds the
returns of an outside object, or (recordings without obstacles) a confirmed candidate that holds no injected return.
Objects within 5 cm of the envelope are in a dead band and are not counted either way.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INSIDE_BINS = [(.05, .1), (.1, .2), (.2, .4), (.4, 9.)]
OUTSIDE_BINS = [(.05, .1), (.1, .2), (.2, .5), (.5, 9.)]


def load(tag):
    """objects: (run, seed, id) -> dict; frames: list; rows/miss grouped per object."""
    objects, frames, rows, misses = {}, [], defaultdict(list), defaultdict(list)
    for path in sorted((ROOT / 'output' / 'augment' / tag).glob('*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r['run'], r['seed'], r.get('object') if r['kind'] != 'object' else r['id'])
            if r['kind'] == 'object':
                objects[key] = r
            elif r['kind'] == 'row' and r.get('object') is not None:
                rows[key].append(r)
            elif r['kind'] == 'miss':
                misses[key].append(r)
            elif r['kind'] == 'frame':
                frames.append(r)
    return objects, frames, rows, misses


def first_alarm(rs):
    confirmed = [r['front_m'] for r in rs if r.get('confirmed') and r.get('front_m') is not None]
    return max(confirmed) if confirmed else None


def per_object(objects, rows, misses):
    out = {}
    for key, o in objects.items():
        rs = rows.get(key, [])
        seen_frames = len(rs) + sum(m.get('returns', 0) > 0 for m in misses.get(key, []))
        out[key] = dict(o, first=first_alarm(rs), alarm_frames=sum(bool(r.get('confirmed')) for r in rs),
                        candidate_frames=len(rs), seen_frames=seen_frames,
                        reachable=max([r['front_m'] for r in rs if r.get('front_m')] + [m['front_m'] for m in misses.get(key, []) if m.get('returns', 0) > 0], default=None))
    return out


def stats(items):
    n = len(items)
    firsts = [i['first'] for i in items if i['first'] is not None]
    return dict(n=n, found=len(firsts), mean=float(np.mean(firsts)) if firsts else None,
                median=float(np.median(firsts)) if firsts else None,
                p10=float(np.percentile(firsts, 10)) if firsts else None,
                at60=sum(f >= 60. for f in firsts), at100=sum(f >= 100. for f in firsts))


def summarize(tag):
    objects, frames, rows, misses = load(tag)
    objs = per_object(objects, rows, misses)
    # an object whose returns never reached the lidar (hidden behind the tunnel or another object) cannot be found
    visible = {k: o for k, o in objs.items() if o['seen_frames'] >= 3}
    res = dict(tag=tag, objects=len(objs), visible=len(visible), inside={}, outside={}, shapes={}, clean={}, frames=len(frames))
    for lo, hi in INSIDE_BINS:
        res['inside'][(lo, hi)] = stats([o for o in visible.values() if o['truth'] == 'inside' and lo < o['depth_m'] <= hi])
    for lo, hi in OUTSIDE_BINS:
        sel = [o for o in visible.values() if o['truth'] == 'outside' and lo < -o['depth_m'] <= hi]
        res['outside'][(lo, hi)] = dict(n=len(sel), alarmed=sum(o['alarm_frames'] > 0 for o in sel), frames=sum(o['alarm_frames'] for o in sel))
    for shape in ('cube', 'person', 'low'):
        res['shapes'][shape] = stats([o for o in visible.values() if o['truth'] == 'inside' and o['depth_m'] > .2 and o['shape'] == shape])
    res['runs'] = {}
    for run in sorted({k[0].split('@')[0] for k in visible}):
        res['runs'][run] = stats([o for k, o in visible.items() if k[0].split('@')[0] == run and o['truth'] == 'inside' and o['depth_m'] > .2])
    by_run = defaultdict(lambda: dict(frames=0, foreign=0, alarm=0))
    for f in frames:
        if f['clean']:
            r = by_run[f['run'].split('@')[0]]
            r['frames'] += 1
            r['foreign'] += f['foreign'] > 0
            r['alarm'] += f['status'] == 'obstacle' and not f['owners'] and not f['live']
    res['clean'] = dict(by_run)
    return res


def position_check(tag):
    """Where the core sees the inner face of an injected object against where it was put, by range (non-held candidates with
    at least 5 returns). Near the train it is the check of the placement; far out it is the placement drift and the core's
    own axis error together, which the two cannot be told apart: it bounds how far the labels of the far objects can be trusted."""
    objects, errs = {}, defaultdict(list)
    for path in sorted((ROOT / 'output' / 'augment' / tag).glob('*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r['kind'] == 'object':
                objects[(r['run'], r['seed'], r['id'])] = r
            elif (r['kind'] == 'row' and r.get('object') is not None and not r.get('held') and (r.get('point_count') or 0) >= 5
                  and r.get('front_m') and r.get('u_range')):
                o = objects[(r['run'], r['seed'], r['object'])]
                seen = min(r['u_range']) if o['side'] > 0 else max(r['u_range'])
                truth = o['u_range'][0] if o['side'] > 0 else o['u_range'][1]
                errs[int(r['front_m'] // 25) * 25].append((truth - seen) * o['side'])  # > 0: seen nearer the centre than put
    return {b: np.array(e) for b, e in sorted(errs.items())}


def fmt(v, digits=0):
    return '-' if v is None else f'{v:.{digits}f}'


def report(results):
    lines = []
    tags = [r['tag'] for r in results]
    lines.append(f"objects: " + ', '.join(f"{r['tag']} {r['objects']} placed, {r['visible']} with returns" for r in results))
    lines.append('')
    lines.append('Inside the envelope (found / objects; first alarm range, m: median, 10th percentile; found from >= 60 m / >= 100 m)')
    lines.append('| depth, m | ' + ' | '.join(tags) + ' |')
    lines.append('|---|' + '---|' * len(tags))
    for b in INSIDE_BINS:
        cells = []
        for r in results:
            s = r['inside'][b]
            cells.append(f"{s['found']}/{s['n']}; {fmt(s['median'])}, {fmt(s['p10'])}; {s['at60']} / {s['at100']}")
        lines.append(f"| {b[0]}-{b[1] if b[1] < 9 else '...'} | " + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append('Outside the envelope (objects with a false alarm / objects; alarm frames)')
    lines.append('| distance outside, m | ' + ' | '.join(tags) + ' |')
    lines.append('|---|' + '---|' * len(tags))
    for b in OUTSIDE_BINS:
        lines.append(f"| {b[0]}-{b[1] if b[1] < 9 else '...'} | " + ' | '.join(
            f"{r['outside'][b]['alarmed']}/{r['outside'][b]['n']}; {r['outside'][b]['frames']}" for r in results) + ' |')
    lines.append('')
    lines.append('Inside by more than 0.2 m, by shape (found / objects; median first alarm range, m)')
    lines.append('| shape | ' + ' | '.join(tags) + ' |')
    lines.append('|---|' + '---|' * len(tags))
    for shape in ('cube', 'person', 'low'):
        lines.append(f'| {shape} | ' + ' | '.join(f"{r['shapes'][shape]['found']}/{r['shapes'][shape]['n']}; {fmt(r['shapes'][shape]['median'])}" for r in results) + ' |')
    lines.append('')
    lines.append('Inside by more than 0.2 m, by recording (found / objects; median first alarm range, m)')
    lines.append('| recording | ' + ' | '.join(tags) + ' |')
    lines.append('|---|' + '---|' * len(tags))
    for run in sorted({k for r in results for k in r['runs']}):
        lines.append(f'| {run} | ' + ' | '.join(f"{r['runs'].get(run, {}).get('found', 0)}/{r['runs'].get(run, {}).get('n', 0)}; {fmt(r['runs'].get(run, {}).get('median'))}" for r in results) + ' |')
    lines.append('')
    lines.append('Recordings without obstacles: frames with a confirmed candidate that holds no injected return / frames')
    runs = sorted({k for r in results for k in r['clean']})
    lines.append('| recording | ' + ' | '.join(tags) + ' |')
    lines.append('|---|' + '---|' * len(tags))
    for run in runs:
        lines.append(f'| {run} | ' + ' | '.join(
            f"{r['clean'].get(run, {}).get('foreign', 0)} / {r['clean'].get(run, {}).get('frames', 0)}" for r in results) + ' |')
    lines.append('')
    lines.append('Where the core sees the inner face of an object minus where it was put (m; + = nearer the centre line), by range')
    lines.append('| range, m | n | median | 10th percentile | 90th percentile |')
    lines.append('|---|---|---|---|---|')
    for b, e in position_check(tags[0]).items():
        if len(e) >= 20:
            lines.append(f'| {b}-{b + 25} | {len(e)} | {np.median(e):+.2f} | {np.percentile(e, 10):+.2f} | {np.percentile(e, 90):+.2f} |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--write')
    a = ap.parse_args()
    text = report([summarize(t) for t in a.tags])
    print(text)
    if a.write:
        Path(a.write).write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
