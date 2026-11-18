"""Frame-to-frame jitter of the drawn axis at several ranges, per stage of the core (raw / fused / spatial / shown): the change of the axis
at distance d between two frames once the travel and the turn of the train are taken out, as in eval/bench.py (step100). Needs the core
library (METRO_CORE_LIB; for a Falcon recording also METRO_CORE_LIB_LONG).

    python eval/axis_jitter.py sample 0 100 ['{"display_far_smoothing": false}'] [stride]
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]
from metro.native import NativeTunnelDetector
from stability import frames_of, run_config
run, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
over = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
stride = int(sys.argv[5]) if len(sys.argv) > 5 else 1
over = dict(over, explain=True)
det = NativeTunnelDetector(run_config(run), over)
names = ['raw', 'fused', 'spatial', 'shown']
ranges = [100, 150, 200, 225, 250]
steps = {n: {d: [] for d in ranges} for n in names}
prev = None
n = 0
worst = []
for i, pts, t in frames_of(run):
    if i < first: continue
    if i > last: break
    if (i - first) % stride: continue
    r = det.process(pts, None, t)
    info = r.get('path_info') or {}; ex = info.get('explain')
    if ex is None or r['status'] == 'unknown': prev = None; continue
    st = np.asarray(ex['stages'], float)      # g, raw, fused, spatial, shown
    ds = info.get('odometry_ds'); sup = info.get('support_m') or 0.
    n += 1
    cur = dict(x=st[:, 0], y={nm: st[:, 1 + k] for k, nm in enumerate(names)}, sup=sup, frame=i)
    if prev is not None and ds is not None and 0 <= ds < 12 and cur['frame'] - prev['frame'] == stride:
        for nm in names:
            py, cy = prev['y'][nm], cur['y'][nm]
            g = np.gradient(py, prev['x'])
            s = lambda q: np.interp(q, prev['x'], py)
            for d in ranges:
                if min(sup, prev['sup'] - ds) >= d + 10 and d + ds < prev['x'][-1]:
                    tpred = s(d + ds) - (s(ds) - py[0]) - (np.interp(ds, prev['x'], g) - np.interp(0., prev['x'], g)) * d
                    steps[nm][d].append(abs(np.interp(d, cur['x'], cy) - tpred) * 100)
                    if nm == 'shown' and d == 100: worst.append((abs(np.interp(d, cur['x'], cy) - tpred) * 100, i, ds, sup))
    prev = cur
print(run, 'frames', n, 'stride', stride, over if len(over) > 1 else '')
print('range m | ' + ' | '.join(f'{nm:>16s}' for nm in names) + '    (median / p95 / p99 / max cm, n)')
for d in ranges:
    row = []
    for nm in names:
        v = steps[nm][d]
        row.append(f'{np.median(v):5.1f} /{np.percentile(v, 95):6.1f} {np.percentile(v, 99):6.1f} {np.max(v):6.1f} {len(v):5d}' if len(v) > 20 else '          -     ')
    print(f'{d:7d} | ' + ' | '.join(row))

worst.sort(reverse=True)
print('worst shown steps at 100 m (cm, frame, ds, support):', [(round(a), b, round(c,1), round(d)) for a, b, c, d in worst[:8]])
