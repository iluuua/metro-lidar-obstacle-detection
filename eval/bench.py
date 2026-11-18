"""Benchmark a core build and configuration against the organizers' criteria, on the raw core output.

What the ROS node publishes is the C++ core's answer, so this replays the saved full clouds through the core
only (no Python post-processing) and measures, per recording:

  - false positives (alarm frames on assumed-clean recordings, unmatched confirmed boxes on labelled ones);
  - false negatives, detection range (first confirmed distance of every target), coverage after the first detection;
  - envelope length and its stability (flicker, collapses), axis jumps and kinks, curvature reversal;
  - the person on the new lidar (Falcon): farthest confirmed detection and rate by range band;
  - frame time (the node must keep 10 Hz).

    METRO_CORE_LIB is set from --dll.   python eval/bench.py --dll core/build/libmetro_core.so --tag base
    python eval/bench.py --dll ... --tag x --config '{"soft_threshold": 0.2}' --set full

Results go to output/bench/<tag>/ (one compact JSON per recording plus scores.json).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEP = ('frame', 'time_s', 'status', 'obstacle', 'distance_m', 'analysis_range_m', 'envelope_range_m', 'latency_ms')
CAND = ('bbox', 'confirmed', 'attention', 'boundary', 'far_attention', 'confident_points', 'appeared_frac', 'held',
        'held_frames', 'point_count', 'distance_m', 'track_id', 'wall_like', 'unverified', 'edge_confirmed',
        'soft_confirmed', 'unique_point_count', 'unique_confident_points', 'decision_rule', 'margin_m', 'sigma_u',
        'u_range', 'h_range', 's_max', 'evidence', 'evidence_need', 'path_lateral_m')


def replay(job):
    sequence, dll, overrides, out, window = job[:5]
    stride = job[5] if len(job) > 5 else 1  # a busy computer gives the detector every stride-th frame
    name = sequence
    if '@' in sequence:  # recording@first:last = that part of the recording, cold start
        sequence, span = sequence.split('@')
        window = tuple(int(v) for v in span.split(':'))
    os.environ['METRO_CORE_LIB'] = str(dll)
    sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
    import numpy as np
    from metro.native import NativeTunnelDetector
    from metro.tunnel import TunnelConfig
    from source_clouds import SourceClouds
    folder = ROOT / 'viewer/data' / sequence
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    cfg = TunnelConfig().as_dict()
    cfg.update(overrides)

    class Settings:
        def as_dict(self):
            return cfg

    detector = NativeTunnelDetector(Settings())
    clouds = SourceClouds()
    rows = []
    started = time.perf_counter()
    for i, record in enumerate(manifest['frames']):
        if window and not window[0] <= i < window[1]:
            continue
        if i % stride:
            continue
        if record.get('full_file'):
            points = np.fromfile(ROOT / 'viewer' / record['full_file'], '<f4').reshape(-1, 3)
        else:
            points = np.frombuffer(clouds.read(manifest['summary'], i), '<f4').reshape(-1, 3)
        points = points[np.einsum('ij,ij->i', points, points) > 1.]
        began = time.perf_counter()
        result = detector.process(points, None, record['time_s'])
        wall_ms = (time.perf_counter() - began) * 1000
        row = {k: result[k] for k in KEEP if k in result}
        row['frame'], row['time_s'], row['wall_ms'] = record['frame'], record['time_s'], round(wall_ms, 2)
        info = result.get('path_info') or {}
        row['support_m'], row['ds'], row['ext'] = info.get('support_m'), info.get('odometry_ds'), bool(info.get('extension_ok'))
        row['candidates'] = [{k: c[k] for k in CAND if k in c} for c in result.get('candidates', [])]
        if result.get('path'):
            row['py'] = [round(p[1], 3) for p in result['path']]
        rows.append(row)
    (out / (name.replace(':', '-') + '.json')).write_text(json.dumps(dict(sequence=name, seconds=time.perf_counter() - started, frames=rows)), encoding='utf-8')
    return name, len(rows), time.perf_counter() - started


def reversal(x, y, floor=1 / 3000.):
    import numpy as np
    if len(x) < 4:
        return 0.0
    dx, dy = np.diff(x), np.diff(y)
    psi = np.arctan2(dy, dx)
    ds = np.hypot(dx, dy)
    mid = 0.5 * (ds[1:] + ds[:-1])
    k = np.diff(psi) / np.maximum(mid, 1e-6)
    turn = np.where(np.abs(k) >= floor, k * mid, 0.0)
    return float(2 * min(turn[turn > 0].sum(), -turn[turn < 0].sum()))


def axis_metrics(frames):
    """Envelope length, its stability, axis jumps and kinks from the compact frame records."""
    import numpy as np
    x = np.arange(0, 332, 2, dtype=float)
    env, sup, steps, kinks, rev, flick, collapse, osc, prev = [], [], {50: [], 100: [], 150: []}, [], 0, 0, 0, 0, None
    rigid, ends, travelled = {100: [], 150: []}, [], 0.
    for f in frames:
        e, py = f.get('envelope_range_m'), f.get('py')
        if py is None or f.get('status') == 'unknown' or e is None:
            prev = None
            continue
        y = np.asarray(py[:len(x)], float)
        xs = x[:len(y)]
        env.append(e)
        sup.append(f.get('support_m') or 0.)
        lim = max(10., min(e, f.get('support_m') or e))
        m = xs <= lim
        if m.sum() > 4:
            head = np.arctan2(np.diff(y[m]), np.diff(xs[m]))
            kinks.append(float(np.degrees(np.abs(np.diff(head)).max())))
            if reversal(xs[m], y[m]) > math.radians(5):
                rev += 1
        ds = f.get('ds')
        travelled += ds if ds is not None and 0 <= ds < 5 else 0.
        ends.append(travelled + e)  # where the drawn envelope ends along the path
        sup_now = f.get('support_m') or 0.
        if prev is not None and ds is not None and 0 <= ds < 5 and prev['frame'] == f['frame'] - 1:
            change = e - (prev['env'] - ds)
            if abs(change) > 20:
                flick += 1
            if abs(change) > 10 and prev.get('change') is not None and prev['change'] * change < 0 and abs(prev['change']) > 10:
                osc += 1
            if e < 0.6 * (prev['env'] - ds) and prev['env'] > 60:
                collapse += 1
            g = np.gradient(prev['y'], prev['x'])
            for d in steps:
                if min(sup_now, prev['sup'] - ds) >= d + 10 and d + ds < prev['x'][-1]:  # only where the axis is measured
                    s = lambda q: np.interp(q, prev['x'], prev['y'])
                    t = s(d + ds) - (s(ds) - prev['y'][0]) - (np.interp(ds, prev['x'], g) - np.interp(0., prev['x'], g)) * d
                    steps[d].append(abs(np.interp(d, xs, y) - t) * 100)
                    if d in rigid:  # the same step once a rigid offset and yaw change of the train are removed: the shape of the axis alone
                        q = np.arange(10., 61., 2.)
                        tq = np.array([s(v + ds) - (s(ds) - prev['y'][0]) - (np.interp(ds, prev['x'], g) - np.interp(0., prev['x'], g)) * v for v in q])
                        coef = np.linalg.lstsq(np.vstack([np.ones_like(q), q]).T, np.interp(q, xs, y) - tq, rcond=None)[0]
                        rigid[d].append(abs((np.interp(d, xs, y) - t) - (coef[0] + coef[1] * d)) * 100)
        prev = dict(x=xs, y=y, env=e, sup=sup_now, frame=f['frame'], change=(e - (prev['env'] - ds)) if prev is not None and ds is not None and 0 <= ds < 5 else None)
    pct = lambda v, q: float(np.percentile(v, q)) if len(v) else None
    out = dict(frames=len(env), env_median=pct(env, 50), env_p10=pct(env, 10), env_lt100=100 * float(np.mean(np.array(env) < 100)) if env else None,
               env_flicker=flick, env_oscillation=osc, env_collapse=collapse, reversal_frames=rev, kink_p99_deg=pct(kinks, 99), kink_max_deg=max(kinks) if kinks else None)
    # a dip: the drawn end falls well behind the best of the last 4 s and returns within 4 s
    dip, ends = 0, np.asarray(ends)
    for k in range(1, len(ends)):
        best = ends[max(0, k - 40):k].max()
        if best - ends[k] > max(15., .2 * env[k]) and ends[k + 1:k + 41].size and ends[k + 1:k + 41].max() >= best - 10.:
            dip += 1
    out.update(env_dips=dip, env_end_variation_m=float(np.abs(np.diff(ends)).mean()) if len(ends) > 1 else None)
    for d, v in rigid.items():
        out[f'rigid{d}_p99_cm'] = pct(v, 99)
    for d, v in steps.items():
        out[f'step{d}_p95_cm'], out[f'step{d}_p99_cm'], out[f'step{d}_max_cm'] = pct(v, 95), pct(v, 99), (max(v) if v else None)
    lat = [f['wall_ms'] for f in frames if 'wall_ms' in f]
    out.update(wall_p50_ms=pct(lat, 50), wall_p95_ms=pct(lat, 95), wall_max_ms=max(lat) if lat else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dll', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--config', default='{}')
    ap.add_argument('--sequences', required=True, help='comma-separated user-provided viewer runs')
    ap.add_argument('--controls', default='', help='comma-separated runs known to have no obstacles')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--window', help='first:last frame index (cold start there), for one long recording')
    ap.add_argument('--stride', type=int, default=1, help='feed every N-th frame only (a busy computer)')
    ap.add_argument('--rescore', action='store_true', help='score existing replays without running the core')
    args = ap.parse_args()
    dll = Path(args.dll)
    dll = dll if dll.is_absolute() or dll.exists() else ROOT / 'core/build' / args.dll
    out = ROOT / 'output/bench' / args.tag
    out.mkdir(parents=True, exist_ok=True)
    sequences = args.sequences.split(',')
    overrides = json.loads(args.config)
    (out / 'run.json').write_text(json.dumps(dict(dll=str(dll), config=overrides, sequences=sequences)), encoding='utf-8')
    if not args.rescore:
        window = tuple(int(v) for v in args.window.split(':')) if args.window else None
        jobs = [(s, str(dll), overrides, out, window, args.stride) for s in sequences]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for sequence, n, seconds in pool.map(replay, jobs):
                print(f'replayed {sequence}: {n} frames in {seconds:.0f} s', flush=True)
    scores = {}
    node = subprocess.run(['node', str(ROOT / 'eval/bench_score.cjs'), str(out), ','.join(sequences), args.controls], capture_output=True, text=True, encoding='utf-8')
    if node.returncode:
        sys.exit(node.stderr)
    scores = json.loads(node.stdout)
    for sequence in sequences:
        frames = json.loads((out / (sequence.replace(':', '-') + '.json')).read_text(encoding='utf-8'))['frames']
        scores[sequence]['axis'] = axis_metrics(frames)
    (out / 'scores.json').write_text(json.dumps(scores, ensure_ascii=False, indent=1), encoding='utf-8')
    print(summary(scores))


def summary(scores):
    lines = []
    ctrl_alarm = sum(s.get('alarm_frames', 0) for k, s in scores.items() if s.get('kind') == 'control')
    lab = [s for s in scores.values() if s.get('kind') == 'labelled']
    fp = sum(s['counts']['fp'] for s in lab)
    fn = sum(s['counts']['fn'] for s in lab)
    tp = sum(s['counts']['tp'] for s in lab)
    first = [o['first_distance'] for s in lab for o in s['objects'] if o['first_distance'] is not None]
    cover = [o['coverage_after_first'] for s in lab for o in s['objects'] if o['coverage_after_first'] is not None]
    miss = sum(1 for s in lab for o in s['objects'] if o['first_distance'] is None)
    lines.append(f"CONTROLS alarm frames {ctrl_alarm} | LABELLED TP {tp} FP {fp} FN {fn} | targets missed {miss} | first-detect distance mean {sum(first)/len(first):.0f} m min {min(first):.0f}" if first else 'no labelled objects')
    if cover:
        lines.append(f"coverage after first detection mean {100*sum(cover)/len(cover):.0f}%")
    for k, s in scores.items():
        a = s.get('axis', {})
        extra = ''
        if s.get('kind') == 'labelled':
            c = s['counts']
            extra = f" TP{c['tp']} FP{c['fp']}(stale{c.get('fpStale', 0)}) FN{c['fn']} dup{c.get('dup', 0)} shift{c.get('shifted', 0)} found {s['found']}"
        else:
            extra = f" alarm frames {s.get('alarm_frames')}"
        f = lambda v, n=0: '—' if v is None else f'{v:.{n}f}'
        lines.append(f"{k:36s}{extra} | env med {f(a.get('env_median'))} p10 {f(a.get('env_p10'))} flick {a.get('env_flicker')} coll {a.get('env_collapse')} | step100 p99 {f(a.get('step100_p99_cm'),1)} max {f(a.get('step100_max_cm'),0)} cm | kink p99 {f(a.get('kink_p99_deg'),1)}° | rev {a.get('reversal_frames')} | wall p50 {f(a.get('wall_p50_ms'))} p95 {f(a.get('wall_p95_ms'))} max {f(a.get('wall_max_ms'))} ms")
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
