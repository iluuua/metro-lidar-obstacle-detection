"""How well does the far axis of a frame predict what the train measures once it has come closer?

The bench keeps, per frame, the lateral axis every 2 m (`py`), the travel since the last frame (`ds`) and the supported
range. A frame k predicts the bend of the track between a (a stretch ahead, default 60 m) and b (100..160 m):

    E_k(a, b) = y(b) - y(a) - (b - a) * slope(a)          # deviation of the axis at b from the tangent at a

E does not change when the sensor is rotated, so frames can be compared without knowing the pose. The frame m that has
travelled D = a - near metres farther sees the same stretch between near and b - D, close and well supported; its
E_m(near, b - D) is the reference. The difference E_k - E_m is the prediction error of the shape of the track.
It is split by where b lay in frame k: inside its supported range, or beyond (a pure continuation).

    python eval/axis_prediction.py baseline experiment --sequences sample
    python eval/axis_prediction.py experiment --sequences sample_a,sample_b --a 60 --b 100,130,160
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def slope_at(y, x, at, half=6.):
    lo, hi = at - half, at + half
    return (np.interp(hi, x, y) - np.interp(lo, x, y)) / (hi - lo)


def bend(y, x, a, b):
    return float(np.interp(b, x, y) - np.interp(a, x, y) - (b - a) * slope_at(y, x, a))


def errors(frames, a, bs, near, margin=5.):
    x = np.arange(0, 332, 2, dtype=float)
    pose, travelled, ok = [], 0., []
    for f in frames:
        ds = f.get('ds')
        if f.get('py') is None:
            ok.append(False)
            pose.append(travelled)
            continue
        if ds is not None and 0 <= ds < 5:
            travelled += ds
        pose.append(travelled)
        ok.append(True)
    pose = np.asarray(pose)
    rows = []
    D = a - near
    for k, f in enumerate(frames):
        if not ok[k]:
            continue
        m = int(np.searchsorted(pose, pose[k] + D))
        if m >= len(frames) or not ok[m] or abs(frames[m]['frame'] - f['frame']) > 1000:
            continue
        yk = np.asarray(f['py'], float)
        ym = np.asarray(frames[m]['py'], float)
        xk, xm = x[:len(yk)], x[:len(ym)]
        sup_k, sup_m = f.get('support_m') or 0., frames[m].get('support_m') or 0.
        for b in bs:
            if b > xk[-1] or b - D > xm[-1]:
                continue
            if sup_m < b - D + margin:  # the reference must itself be measured, not continued
                continue
            inside = b <= sup_k - margin
            outside = b >= sup_k + margin
            if not (inside or outside):
                continue
            err = bend(yk, xk, a, b) - bend(ym, xm, near, b - D)
            rows.append((b, 'in' if inside else 'ext', err))
    return rows


def summarize(tag, sequences, a, bs, near):
    out = {}
    for seq in sequences:
        path = ROOT / 'output' / 'bench' / tag / f'{seq}.json'
        if not path.exists():
            continue
        frames = json.loads(path.read_text(encoding='utf-8'))['frames']
        for b, kind, err in errors(frames, a, bs, near):
            out.setdefault((b, kind), []).append(err)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--sequences', required=True)
    ap.add_argument('--a', type=float, default=60.)
    ap.add_argument('--near', type=float, default=10.)
    ap.add_argument('--b', default='100,130,160')
    args = ap.parse_args()
    bs = [float(v) for v in args.b.split(',')]
    seqs = args.sequences.split(',')
    print(f'a={args.a:.0f} m, reference {args.near:.0f} m, error in cm of the bend E_k(a,b) - E_ref; n / median |e| / p90 |e| / mean signed')
    for tag in args.tags:
        res = summarize(tag, seqs, args.a, bs, args.near)
        print(f'\n{tag}')
        for b in bs:
            cells = []
            for kind in ('in', 'ext'):
                e = np.asarray(res.get((b, kind), []), float) * 100
                cells.append('   -' if not len(e) else f'{kind:>3} n{len(e):5d} med {np.median(np.abs(e)):6.1f} p90 {np.percentile(np.abs(e), 90):6.1f} bias {e.mean():+6.1f}')
            print(f'  b={b:4.0f}  ' + '  |  '.join(cells))


if __name__ == '__main__':
    main()
