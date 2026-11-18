"""The approach warning (metro/approach.py) on all recordings: one replay of the core per recording, the band clusters
computed once per frame, several variants of the thresholds on the same frames.

Per variant: frames with a warning on the recordings without obstacles (the 20-minute ride in windows), the first
warning range of every marked target of the synthetic recordings (a warned box overlapping the target, pad 0.6), and
the markers score if the warnings were alarms.

    .venv/bin/python eval/approach_eval.py --workers 12
"""
import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]

VARIANTS = {
    'default': {},
    'loose': dict(min_slope=.004, min_t=4., min_span=15.),
    'strict': dict(min_slope=.008, min_t=6., min_span=25., frames=5),
    'absolute': dict(min_slope=.006, min_t=5., ref_min_age=10 ** 9),  # no static reference: the trend of the gap itself
}


def replay(run):
    from metro.approach import ApproachGate, band_clusters
    from metro.native import NativeTunnelDetector
    from frame_source import frames
    cfg, source = frames(run)
    det = NativeTunnelDetector(cfg, {})
    gates = {k: ApproachGate(**v) for k, v in VARIANTS.items()}
    if 'absolute' in gates:
        gates['absolute'].predict = absolute_predict.__get__(gates['absolute'])
    out = []
    for i, pts, t in source:
        r = det.process(pts, None, t)
        ego = (r.get('path_info') or {}).get('ego_s')
        clusters = band_clusters(pts, r.get('path') or [], 0.35, 110.) if ego is not None else []
        row = dict(frame=i, status=r['status'], candidates=[dict(bbox=c['bbox'], confirmed=True, distance_m=c['distance_m'])
                                                          for c in r.get('candidates', []) if c.get('confirmed') and c.get('bbox')])
        for k, g in gates.items():
            res = g.step(dict(status=r['status'], candidates=[]), ego, copy.deepcopy(clusters))
            row[k] = res.get('approaching', [])
        out.append(row)
    return run, out


def absolute_predict(self, t):
    import numpy as np
    obs = t.obs
    if len(obs) < self.min_obs:
        return None
    d, g = (np.array(v, float) for v in zip(*[(o[0], o[1]) for o in obs]))
    if d.max() - d.min() < self.min_span:
        return None
    A = np.column_stack([np.ones_like(d), d])
    coef, *_ = np.linalg.lstsq(A, g, rcond=None)
    resid = g - A @ coef
    s = np.sqrt(resid @ resid / max(len(d) - 2, 1))
    se = s / np.sqrt(((d - d.mean()) ** 2).sum())
    if coef[1] < self.min_slope or s > self.max_resid or coef[1] < self.min_t * se:
        return None
    return g[-1] - coef[1] * d[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean-runs', nargs='+', required=True)
    ap.add_argument('--labelled-runs', nargs='+', required=True)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--save', default=str(ROOT / 'output' / 'approach_eval.json'))
    a = ap.parse_args()
    from score_markers import overlap, score_run
    runs = a.labelled_runs + a.clean_runs
    with ProcessPoolExecutor(a.workers) as pool:
        results = dict(pool.map(replay, runs))
    report = {}
    for k in VARIANTS:
        clean = {run: sum(bool(f[k]) for f in results[run]) for run in a.clean_runs}
        targets, tp6, fp6 = {}, 0, 0
        for run in a.labelled_runs:
            truth = json.loads((ROOT / 'viewer' / 'data' / run / 'ground-truth.json').read_text(encoding='utf-8'))
            gt = {f['frame']: f['objects'] for f in truth['frames']}
            frames_alarm = []
            for f in results[run]:
                for w in f[k]:
                    for o in gt.get(f['frame'], []):
                        if o['label'] == 'positive' and any(overlap(w['bbox'], part, .6) for part in o['parts']):
                            key = f"{run}:{o['id']}"
                            targets.setdefault(key, round(w['distance_m'], 1))
                frames_alarm.append(dict(frame=f['frame'], candidates=f['candidates'] + [dict(bbox=w['bbox'], confirmed=True, distance_m=w['distance_m']) for w in f[k]]))
            sc = score_run(frames_alarm, truth, .6)
            tp6 += sc['totals']['tp']; fp6 += sc['totals']['fp']
        report[k] = dict(clean_warned_frames=clean, clean_total=sum(clean.values()), first_warning=targets,
                         as_alarm=dict(tp6=tp6, fp6=fp6))
        print(k, json.dumps(report[k], ensure_ascii=False), flush=True)
    base_tp = base_fp = 0
    for run in a.labelled_runs:
        truth = json.loads((ROOT / 'viewer' / 'data' / run / 'ground-truth.json').read_text(encoding='utf-8'))
        sc = score_run(results[run], truth, .6)
        base_tp += sc['totals']['tp']; base_fp += sc['totals']['fp']
    print('core alone', dict(tp6=base_tp, fp6=base_fp))
    Path(a.save).write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
