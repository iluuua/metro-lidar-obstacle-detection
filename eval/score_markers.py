"""Score detections against user-provided object boxes in viewer/data/<run>/ground-truth.json.

Targets count within the configured range. Confirmed boxes match targets by
bounding-box overlap with one-to-one maximum matching. Unmatched boxes count as
false alarms. Objects labelled ambiguous are excluded from the totals.

python eval/score_markers.py --runs sample --pad .2
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]

PAD = .2
RANGE = (1., 300.)

def overlap(box, part, pad=PAD):
    (lo, hi), (glo, ghi) = box, part
    return all(hi[k] >= glo[k] - pad and lo[k] <= ghi[k] + pad for k in range(3))


def in_range(obj):
    return any(p[1][0] >= RANGE[0] and p[0][0] <= RANGE[1] for p in obj['parts'])


def match(edges, n_objects):
    """Maximum matching of boxes to targets (augmenting paths); edges[i] = targets box i touches."""
    owner = {}

    def assign(i, seen):
        for o in edges[i]:
            if o in seen:
                continue
            seen.add(o)
            if o not in owner or assign(owner[o], seen):
                owner[o] = i
                return True
        return False

    for i in range(len(edges)):
        assign(i, set())
    return owner


def score_run(frames, truth, pad=PAD):
    gt = {f['frame']: f['objects'] for f in truth['frames']}
    totals = dict(tp=0, fp=0, fn=0, exterior_fp=0)
    objects, errors = {}, []
    for f in frames:
        objs = gt.get(f['frame'], [])
        targets = [o for o in objs if o['label'] == 'positive' and in_range(o)]
        exterior = [o for o in objs if o['label'] == 'negative']
        disputed = [o for o in objs if o['label'] == 'ambiguous']
        boxes = [c for c in f.get('candidates', []) if c.get('confirmed') and c.get('bbox')]
        edges = [[k for k, o in enumerate(targets) if any(overlap(c['bbox'], p, pad) for p in o['parts'])] for c in boxes]
        owner = match(edges, len(targets))
        matched = set(owner.values())
        for k, o in enumerate(targets):
            s = objects.setdefault(o['id'], dict(name=o['name'], frames=0, tp=0, first_range_m=None))
            s['frames'] += 1
            if k in owner:
                s['tp'] += 1
                if s['first_range_m'] is None:
                    s['first_range_m'] = round(min(p[0][0] for p in o['parts']), 1)
        for i, c in enumerate(boxes):
            if i in matched or any(overlap(c['bbox'], p, pad) for o in disputed for p in o['parts']):
                continue
            totals['fp'] += 1
            ext = any(overlap(c['bbox'], p, pad) for o in exterior for p in o['parts'])
            totals['exterior_fp'] += ext
            errors.append((f['frame'], round(c['distance_m'], 1), 'exterior' if ext else 'free'))
        totals['tp'] += len(owner)
        totals['fn'] += len(targets) - len(owner)
    return dict(totals=totals, objects=objects, errors=errors[:20])


def frames_live(run, overrides):
    from metro.native import NativeTunnelDetector
    from stability import frames_of, run_config
    det = NativeTunnelDetector(run_config(run), overrides)
    for i, pts, t in frames_of(run):
        r = det.process(pts, None, t)
        r['frame'] = i
        yield r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='user-provided viewer runs with ground-truth.json')
    ap.add_argument('--config', default='{}')
    ap.add_argument('--pad', type=float, default=PAD, help='box overlap tolerance, m')
    ap.add_argument('--save')
    a = ap.parse_args()
    overrides = json.loads(a.config)
    out, total = {}, dict(tp=0, fp=0, fn=0, exterior_fp=0)
    for run in a.runs:
        path = ROOT / 'viewer' / 'data' / run / 'ground-truth.json'
        if not path.is_file():
            print(f'{run}: no ground-truth.json (scripts/import_marker_truth.py, scripts/build_marker_truth.py)')
            continue
        res = score_run(frames_live(run, overrides), json.loads(path.read_text(encoding='utf-8')), a.pad)
        out[run] = res
        t = res['totals']
        print(f"== {run}: TP {t['tp']}  FP {t['fp']} (на наружных {t['exterior_fp']})  FN {t['fn']}")
        for oid, s in res['objects'].items():
            print(f"   {oid:24s} {s['name'][:28]:28s} тревога в {s['tp']:4d} из {s['frames']:4d} кадров, "
                  f"первая на {s['first_range_m']} м")
        if res['errors']:
            print('   ложные:', res['errors'][:8])
        for k in total:
            total[k] += t[k]
    print('ИТОГ', total)
    if a.save:
        Path(a.save).write_text(json.dumps(dict(total=total, runs=out), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
