import argparse
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_results(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    frames = data['frames'] if isinstance(data, dict) else data
    return {int(f['frame']): f for f in frames}


def overlap(bbox, gt, pad):
    lo, hi = np.asarray(bbox[0]), np.asarray(bbox[1])
    glo, ghi = np.asarray(gt['lo']) - pad, np.asarray(gt['hi']) + pad
    return bool(np.all(hi >= glo) and np.all(lo <= ghi))


def score(results, gt, pad=1.5, episode_gap=5, verbose=True):
    objects = {o['id']: o for o in gt['objects']}
    per_obj = {k: dict(first=None, alarm_frames=0, visible_frames=0, dist_err=[]) for k in objects}
    fp_frames = []
    n = max(results) + 1
    for i in range(n):
        r = results.get(i)
        if r is None:
            continue
        gts = gt['frames'].get(str(i), [])
        for g in gts:
            if g['n'] >= 3:
                per_obj[g['id']]['visible_frames'] += 1
        if r.get('status') != 'obstacle':
            continue
        confirmed = [c for c in r.get('candidates', []) if c.get('confirmed')]
        unmatched = False
        for c in confirmed:
            hit = [g for g in gts if overlap(c['bbox'], g, pad)]
            if not hit:
                unmatched = True
                continue
            for g in hit:
                o = per_obj[g['id']]
                o['alarm_frames'] += 1
                if o['first'] is None:
                    o['first'] = dict(frame=i, gt_x=g['xmin'], det=c['distance_m'], n=g['n'])
                o['dist_err'].append(c['distance_m'] - g['xmin'])
        if unmatched:
            fp_frames.append(i)
    # FP episodes
    episodes = []
    for f in fp_frames:
        if episodes and f - episodes[-1][-1] <= episode_gap:
            episodes[-1].append(f)
        else:
            episodes.append([f])
    summary = dict(objects=[], fp_frames=len(fp_frames), fp_episodes=len(episodes),
                   fp_episode_spans=[(e[0], e[-1]) for e in episodes])
    tp_ranges = []
    for k, o in per_obj.items():
        obj = objects[k]
        first = o['first']
        entry = dict(id=k, name=obj['name'], positive=obj['positive'],
                     first_frame=first['frame'] if first else None,
                     first_range_m=round(first['gt_x'], 1) if first else None,
                     alarm_frames=o['alarm_frames'], visible_frames=o['visible_frames'],
                     median_dist_err=round(float(np.median(o['dist_err'])), 2) if o['dist_err'] else None)
        summary['objects'].append(entry)
        if obj['positive']:
            tp_ranges.append(first['gt_x'] if first else 0.)
    pos = [e for e in summary['objects'] if e['positive']]
    neg = [e for e in summary['objects'] if not e['positive']]
    summary['detected'] = sum(e['first_frame'] is not None for e in pos)
    summary['positives'] = len(pos)
    summary['mean_first_range_m'] = round(float(np.mean(tp_ranges)), 1)
    summary['negative_alarm_frames'] = sum(e['alarm_frames'] for e in neg)
    if verbose:
        print(f"{'object':20s} {'pos':>4s} {'first_fr':>8s} {'range_m':>8s} {'alarms':>7s} {'visible':>7s} {'derr':>6s}")
        for e in summary['objects']:
            print(f"{e['name']:20s} {str(e['positive'])[0]:>4s} {str(e['first_frame']):>8s} {str(e['first_range_m']):>8s} {e['alarm_frames']:7d} {e['visible_frames']:7d} {str(e['median_dist_err']):>6s}")
        print(f"detected {summary['detected']}/{summary['positives']}  mean first range {summary['mean_first_range_m']} m  "
              f"alarms on negatives {summary['negative_alarm_frames']}  FP frames {summary['fp_frames']}  FP episodes {summary['fp_episodes']}")
        print('FP episodes:', summary['fp_episode_spans'])
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score detections against a user-provided truth JSON.')
    parser.add_argument('results', type=Path)
    parser.add_argument('--truth', type=Path, required=True)
    args = parser.parse_args()
    gt = json.loads(args.truth.read_text(encoding='utf-8'))
    score(load_results(args.results), gt)
