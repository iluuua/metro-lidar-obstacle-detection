"""Top and side views of chosen frames with the C++ core's envelope and candidates, for disputed places.

The detector runs from --start up to the last chosen frame (its history matters). Left: top view (x forward,
y left) with the envelope edges u = +-1.05 m, solid up to the verified range and dotted to the drawn length.
Right: side view of the points within 1.6 m of the axis, with the rail-head line and the 3 m top. Boxes:
red confirmed, orange attention (confident points or boundary), grey other candidates.

    .venv/bin/python eval/frame_views.py sample 10,20 out.png --start 0 --xmax 300
    .venv/bin/python eval/frame_views.py fake 560,576 out.png --xmin 0 --xmax 80 --config '{"margin_k": 1.0}'
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]
from metro.tunnel import TunnelConfig  # noqa: E402
from metro.native import NativeTunnelDetector  # noqa: E402
from stability import frames_of, run_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('frames', help='comma-separated frame numbers')
    ap.add_argument('out')
    ap.add_argument('--start', type=int, default=0, help='first frame the detector sees')
    ap.add_argument('--xmin', type=float, default=0.)
    ap.add_argument('--xmax', type=float, default=150.)
    ap.add_argument('--config', default='{}')
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    wanted = sorted(int(v) for v in a.frames.split(','))
    det = NativeTunnelDetector(run_config(a.run), json.loads(a.config))
    keep = {}
    for i, pts, t in frames_of(a.run):
        if i < a.start:
            continue
        if i > wanted[-1]:
            break
        r = det.process(pts, None, t)
        if i in wanted:
            keep[i] = (pts, r)
    fig, axes = plt.subplots(len(keep), 2, figsize=(22, 4.4 * len(keep)), squeeze=False)
    for row, k in enumerate(sorted(keep)):
        p, f = keep[k]
        q = p[(p[:, 0] > a.xmin) & (p[:, 0] < a.xmax) & (np.abs(p[:, 1]) < 12)]
        path = np.asarray(f.get('path') or [[0, 0, 0, 0]], float)
        verified = f.get('analysis_range_m') or 0.
        drawn = max(f.get('envelope_range_m') or 0., verified)
        ax = axes[row, 0]
        ax.scatter(q[:, 0], q[:, 1], s=.3, c=q[:, 2], cmap='viridis', vmin=-3, vmax=3)
        for limit, style in ((verified, '-'), (drawn, ':')):
            m = path[:, 3] <= limit
            for side in (1.05, -1.05):
                ax.plot(path[m, 0], path[m, 1] + side, 'm' + style, lw=.9)
        for c in f.get('candidates', []):
            lo, hi = c['bbox']
            col = 'r' if c.get('confirmed') else ('orange' if c.get('confident_points', 0) > 0 or c.get('boundary') or c.get('far_attention') else '0.5')
            ax.add_patch(plt.Rectangle((lo[0], lo[1]), hi[0] - lo[0] + .2, hi[1] - lo[1] + .2, fill=False, ec=col, lw=1.4))
            flags = ''.join(k[0] for k in ('boundary', 'unverified', 'wall_like', 'held') if c.get(k))
            ax.text(hi[0] + 1, hi[1], f"{c['distance_m']:.1f} n{c['point_count']}/{c.get('confident_points', 0)} "
                    f"m{c.get('margin_m', 0):.2f} {flags}", color=col, fontsize=7)
        info = f.get('path_info', {})
        ax.set_xlim(a.xmin, a.xmax), ax.set_ylim(-8, 8), ax.grid(alpha=.3)
        ax.set_title(f"{a.run} #{k} {f['status']} d={f.get('distance_m')} verified={verified:.0f} drawn={drawn:.0f} "
                     f"support={info.get('support_m', 0):.0f} rails={info.get('rail_support_m', 0):.0f} "
                     f"ext={info.get('extension_ok')}", fontsize=9)
        ax = axes[row, 1]
        near = np.abs(q[:, 1] - np.interp(q[:, 0], path[:, 0], path[:, 1])) < 1.6
        ax.scatter(q[near, 0], q[near, 2], s=.5, c='k')
        for c in f.get('candidates', []):
            lo, hi = c['bbox']
            col = 'r' if c.get('confirmed') else ('orange' if c.get('confident_points', 0) > 0 or c.get('boundary') or c.get('far_attention') else '0.5')
            ax.add_patch(plt.Rectangle((lo[0], lo[2]), hi[0] - lo[0] + .2, hi[2] - lo[2] + .2, fill=False, ec=col, lw=1.4))
        ax.plot(path[:, 0], path[:, 2], 'm-', lw=.8)
        ax.plot(path[:, 0], path[:, 2] + 3., 'm--', lw=.6)
        zc = np.interp((a.xmin + a.xmax) / 2, path[:, 0], path[:, 2])
        ax.set_xlim(a.xmin, a.xmax), ax.set_ylim(zc - 4, zc + 5), ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(a.out, dpi=70)
    print('saved', a.out)


if __name__ == '__main__':
    main()
