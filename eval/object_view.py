"""Pictures of one labelled object (datasets/labels/<run>.json) for a review by eye: top and side views of the
cloud around its box in its farthest, middle and closest frames, with the drawn axis and the envelope edges
of that frame (viewer export, the core's own answer) and the organizers' markers (dashed: cyan target, grey
dust or exterior, magenta disputed).

    .venv/bin/python eval/object_view.py labelled_sample 268 /tmp/obj268.png
    .venv/bin/python eval/object_view.py labelled_sample 268 /tmp/obj268.png --frames 600,640,660
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HALF_WIDTH, HEIGHT, MIN_HEIGHT = 1.05, 3.0, .07


def load_frame(run, frame):
    root = ROOT / 'viewer' / 'data' / run
    full = root / f'{frame:05d}.full.bin'
    return np.fromfile(full, '<f4').reshape(-1, 3) if full.is_file() else np.zeros((0, 3), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('object', type=int)
    ap.add_argument('out')
    ap.add_argument('--frames', help='comma-separated frames (default: farthest, middle, closest of the object)')
    ap.add_argument('--pad', type=float, default=4., help='m around the box in the views')
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = json.loads((ROOT / 'datasets' / 'labels' / f'{a.run}.json').read_text())
    rows = sorted([r for r in labels['rows'] if r.get('object') == a.object], key=lambda r: r['frame'])
    if not rows:
        raise SystemExit(f'object {a.object} not found in {a.run}')
    obj = next(o for o in labels['tracks'] if o['object'] == a.object)
    by_frame = {}
    for r in rows:
        by_frame.setdefault(r['frame'], r)
    if a.frames:
        wanted = [int(v) for v in a.frames.split(',')]
    else:
        fs = sorted(by_frame)
        wanted = sorted({fs[0], fs[len(fs) // 2], min(fs, key=lambda f: by_frame[f]['distance_m'])})
    model = {f['frame']: f for f in json.loads((ROOT / 'viewer' / 'data' / a.run / 'model.tunnel.json').read_text())['frames']}
    gt_path = ROOT / 'viewer' / 'data' / a.run / 'ground-truth.json'
    gt = {f['frame']: f['objects'] for f in json.loads(gt_path.read_text())['frames']} if gt_path.is_file() else {}
    colours = dict(positive='cyan', negative='grey', ambiguous='magenta')
    fig, axes = plt.subplots(len(wanted), 2, figsize=(16, 4.2 * len(wanted)), squeeze=False)
    for row_i, f in enumerate(wanted):
        r = by_frame.get(f)
        pts = load_frame(a.run, f)
        res = model.get(f, {})
        path = np.asarray(res.get('path') or [[0, 0, 0, 0]], float)
        if r is None:
            # a frame of the range without this object: centre on the nearest labelled frame
            r = by_frame[min(by_frame, key=lambda k: abs(k - f))]
        (lo, hi) = r['bbox'] if r.get('bbox') else ([r['center'][0] - .5] * 3, [r['center'][0] + .5] * 3)
        x0, x1 = lo[0] - a.pad, hi[0] + a.pad
        m = (pts[:, 0] > x0) & (pts[:, 0] < x1)
        q = pts[m]
        xs = path[:, 0]
        sel = (xs >= x0 - 2) & (xs <= x1 + 2)
        lat, vert = path[sel, 1], path[sel, 2]
        ax = axes[row_i][0]
        near_axis = np.interp(q[:, 0], xs, path[:, 1]) if len(q) else np.zeros(0)
        keep = np.abs(q[:, 1] - near_axis) < 4 if len(q) else np.zeros(0, bool)
        ax.scatter(q[keep, 0], q[keep, 1], s=2, c=q[keep, 2], cmap='viridis')
        ax.plot(xs[sel], lat, 'w-', lw=1)
        ax.plot(xs[sel], lat + HALF_WIDTH, 'r-', lw=1)
        ax.plot(xs[sel], lat - HALF_WIDTH, 'r-', lw=1)
        ax.add_patch(plt.Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], fill=False, ec='orange', lw=1.5))
        for o in gt.get(f, []):
            for (glo, ghi) in o['parts']:
                if ghi[0] > x0 and glo[0] < x1:
                    ax.add_patch(plt.Rectangle((glo[0], glo[1]), ghi[0] - glo[0], ghi[1] - glo[1], fill=False, ec=colours.get(o['label'], 'white'), lw=1, ls='--'))
        mid = float(np.interp((x0 + x1) / 2, xs, path[:, 1]))
        ax.set_ylim(mid - 3.2, mid + 3.2)
        ax.set_xlim(x0, x1)
        ax.set_facecolor('#111418')
        ax.set_title(f"кадр {f}, сверху: {r['distance_m']} м, глубина {r['depth_m']} м, σ {r['sigma_u']}, {r['state']}"
                     + (f", маркер {r['markers'][0]['id']}" if r['markers'] else ''), fontsize=9)
        ax.set_xlabel('x вперёд, м'); ax.set_ylabel('y влево, м')
        ax = axes[row_i][1]
        sidek = keep & (np.abs(q[:, 1] - near_axis) < HALF_WIDTH + .6) if len(q) else keep
        ax.scatter(q[sidek, 0], q[sidek, 2], s=2, c=q[sidek, 1], cmap='coolwarm')
        ax.plot(xs[sel], vert, 'w-', lw=1)
        ax.plot(xs[sel], vert + MIN_HEIGHT, 'r--', lw=1)
        ax.plot(xs[sel], vert + HEIGHT, 'r-', lw=1)
        ax.add_patch(plt.Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2], fill=False, ec='orange', lw=1.5))
        for o in gt.get(f, []):
            for (glo, ghi) in o['parts']:
                if ghi[0] > x0 and glo[0] < x1:
                    ax.add_patch(plt.Rectangle((glo[0], glo[2]), ghi[0] - glo[0], ghi[2] - glo[2], fill=False, ec=colours.get(o['label'], 'white'), lw=1, ls='--'))
        ax.set_xlim(x0, x1)
        ax.set_facecolor('#111418')
        ax.set_title(f"сбоку (точки ±{HALF_WIDTH + .6:.2f} м от оси, цвет — бок): u {r['u_range']}, h {r['h_range']}", fontsize=9)
        ax.set_xlabel('x вперёд, м'); ax.set_ylabel('z вверх, м')
    fig.suptitle(f"{a.run}, объект {a.object}: кадры {obj['frames'][0]}–{obj['frames'][1]}, {obj['distance_m'][0]} → {obj['distance_m'][1]} м, "
                 f"вердикт кода {obj['verdict']}, тревог {obj['alarm_frames']}, ВНИМАНИЙ {obj['attention_frames']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=90)
    print(a.out)


if __name__ == '__main__':
    main()
