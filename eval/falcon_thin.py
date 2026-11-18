"""Thin returns inside detected objects and compare detections against an unthinned replay.

python eval/falcon_thin.py --dll core/build/libmetro_core.so --run sample --keep 1,.75,.5

This is an artificial density experiment, not a measured detection range.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAD = .4


def replay(dll, config, first, last, keep, boxes, seed=1, run=None):
    os.environ['METRO_CORE_LIB'] = str(ROOT / 'core/build' / dll)
    sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
    import numpy as np
    from metro.native import NativeTunnelDetector
    from metro.tunnel import TunnelConfig
    from source_clouds import SourceClouds
    cfg = TunnelConfig().as_dict()
    cfg.update(config)

    class Settings:
        def as_dict(self):
            return cfg

    manifest = json.loads((ROOT / 'viewer/data' / run / 'manifest.json').read_text(encoding='utf-8'))
    clouds, detector, rng = SourceClouds(), NativeTunnelDetector(Settings()), np.random.default_rng(seed)
    out, ms = {}, []
    for i, record in enumerate(manifest['frames']):
        if not first <= i < last:
            continue
        points = np.fromfile(ROOT / 'viewer' / record['full_file'], '<f4').reshape(-1, 3) if record.get('full_file') else np.frombuffer(clouds.read(manifest['summary'], i), '<f4').reshape(-1, 3)
        points = points[np.einsum('ij,ij->i', points, points) > 1.]
        frame = record['frame']
        if keep < 1. and frame in boxes:
            for lo, hi in boxes[frame]:
                inside = np.all((points >= np.array(lo) - PAD) & (points <= np.array(hi) + PAD), axis=1)
                points = points[~inside | (rng.random(len(points)) < keep)]
        began = time.perf_counter()
        result = detector.process(points, None, record['time_s'])
        ms.append((time.perf_counter() - began) * 1000)
        out[frame] = dict(alarm=bool(result['obstacle']), boxes=[c['bbox'] for c in result.get('candidates', []) if c.get('confirmed')],
                          far=max([c['distance_m'] for c in result.get('candidates', []) if c.get('confirmed')], default=None))
    return out, ms


def sweep(dll, config, keeps, first, last, seed=1, run=None):
    """one dict per keep: alarm frames, first alarm, how many of the unthinned alarm frames survive, stray alarms, farthest confirmed box, p95 time."""
    base, _ = replay(dll, config, first, last, 1., {}, run=run)
    boxes = {f: v['boxes'] for f, v in base.items() if v['boxes']}
    person_frames = [f for f, v in base.items() if v['alarm']]
    rows = []
    for k in keeps:
        observed, ms = replay(dll, config, first, last, k, boxes, seed, run=run)
        alarms = [f for f, v in observed.items() if v['alarm']]
        stray = sum(1 for f in alarms if f not in set(person_frames))
        far = max((v['far'] for v in observed.values() if v['far']), default=None)
        rows.append(dict(keep=k, alarm_frames=len(alarms), first_alarm=alarms[0] if alarms else None, unthinned_alarm_frames=len(person_frames),
                         kept_of_unthinned=sum(1 for f in person_frames if observed[f]['alarm']), stray=stray, farthest_confirmed_m=far,
                         p95_ms=round(sorted(ms)[int(.95 * len(ms))], 1)))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dll', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--keep', default='1,.75,.5')
    ap.add_argument('--config', default='{}')
    ap.add_argument('--frames', default='0:500')
    ap.add_argument('--seeds', type=int, default=1, help='thin with this many different random draws and print the mean and the range (one draw is chaotic)')
    ap.add_argument('--save', help='also write {"before": rows with far_confirm_hits 9, "after": rows with --config} as JSON')
    args = ap.parse_args()
    a, b = (int(v) for v in args.frames.split(':'))
    keeps = [float(v) for v in args.keep.split(',')]
    after = sweep(args.dll, json.loads(args.config), keeps, a, b, run=args.run)
    for row in after:
        print(json.dumps(row), flush=True)
    if args.seeds > 1:
        draws = [after] + [sweep(args.dll, json.loads(args.config), keeps, a, b, s, run=args.run) for s in range(2, args.seeds + 1)]
        for i, k in enumerate(keeps):
            rows = [d[i] for d in draws]
            summary = {key: (round(sum(r[key] for r in rows) / len(rows), 1), min(r[key] for r in rows), max(r[key] for r in rows))
                       for key in ('alarm_frames', 'kept_of_unthinned', 'stray')}
            print(json.dumps(dict(keep=k, seeds=len(rows), unthinned=rows[0]['unthinned_alarm_frames'], **{key: dict(mean=v[0], min=v[1], max=v[2]) for key, v in summary.items()})), flush=True)
    if args.save:
        before = sweep(args.dll, dict(json.loads(args.config), far_confirm_hits=9), keeps, a, b, run=args.run)
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(dict(before=before, after=after, frames=args.frames)), encoding='utf-8')
