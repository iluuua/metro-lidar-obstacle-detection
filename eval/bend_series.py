"""Per frame: how far the axis bend at b (from the tangent at a) was from what the train measured later, near (a bench run: eval/bench.py).

    python eval/bend_series.py experiment sample 10 50
"""
import json, sys
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
from axis_prediction import bend, slope_at
tag, seq = sys.argv[1], sys.argv[2]
lo, hi = int(sys.argv[3]), int(sys.argv[4])
a, near, bs = 40., 20., (80., 100., 120.)
frames = json.load(open(ROOT / 'output' / 'bench' / tag / f'{seq}.json'))['frames']
x = np.arange(0, 332, 2, dtype=float)
pose, trav = [], 0.
for f in frames:
    ds = f.get('ds')
    if f.get('py') is not None and ds is not None and 0 <= ds < 5: trav += ds
    pose.append(trav)
pose = np.asarray(pose); D = a - near
print('frame  t   sup  ver | error of bend at b=80/100/120 (m; +: axis bends more to + than later measured)')
for k, f in enumerate(frames):
    if not lo <= f['frame'] <= hi or f.get('py') is None: continue
    m = int(np.searchsorted(pose, pose[k] + D))
    if m >= len(frames) or frames[m].get('py') is None: continue
    yk, ym = np.asarray(f['py'], float), np.asarray(frames[m]['py'], float)
    xk, xm = x[:len(yk)], x[:len(ym)]
    row = []
    for b in bs:
        if (frames[m].get('support_m') or 0) < b - D + 5 or b > xk[-1]: row.append('   -  '); continue
        row.append(f'{bend(yk, xk, a, b) - bend(ym, xm, near, b - D):+6.2f}')
    if f['frame'] % 2 == 0:
        print(f"{f['frame']:5d} {f['time_s']:5.1f} {f.get('support_m') or 0:4.0f} {f.get('analysis_range_m') or 0:4.0f} | " + ' '.join(row))
