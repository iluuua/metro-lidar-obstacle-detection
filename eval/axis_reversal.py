"""Does the exported lateral axis bend back? Regression metric for the no-reversal rule.

Lidar looks along the track, so nothing measured can show a bend that reverses beyond the measured
support. For a frame's axis y(x) the heading is psi = atan(dy/dx), the curvature kappa = dpsi/ds, and

    reversal = 2 * min(integral of kappa+, integral of kappa-)           [rad]

is zero for any path that bends one way (or is straight) and equals the heading change that is
later undone for an S-bend or a hook. Curvature below 1/3000 1/m counts as straight, like the core's
own `extension_max_curvature`.

    python eval/axis_reversal.py viewer/data/labelled_sample/runs/<id>.json ...
"""
import json
import math
import sys

import numpy as np

STRAIGHT_KAPPA = 1 / 3000.0


def reversal(x, y, floor=STRAIGHT_KAPPA):
    """Heading change (rad) that a path undoes by bending the other way; 0 when it never does."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4:
        return 0.0
    dx, dy = np.diff(x), np.diff(y)
    psi = np.arctan2(dy, dx)
    ds = np.hypot(dx, dy)
    mid = 0.5 * (ds[1:] + ds[:-1])
    kappa = np.diff(psi) / np.maximum(mid, 1e-6)
    turn = np.where(np.abs(kappa) >= floor, kappa * mid, 0.0)
    return float(2 * min(turn[turn > 0].sum(), -turn[turn < 0].sum()))


def frame_reversal(frame):
    """(inside the measured support, beyond it, whole axis) in radians, or None without an axis."""
    path = frame.get('path')
    if not path or len(path) < 6 or frame.get('status') == 'unknown':
        return None
    a = np.asarray(path, float)
    x, y = a[:, 0], a[:, 1]
    support = max(float((frame.get('path_info') or {}).get('support_m') or 0.0), 30.0)
    beyond = x >= support
    return (reversal(x[x <= support], y[x <= support]),
            reversal(x[beyond], y[beyond]) if beyond.sum() >= 4 else 0.0,
            reversal(x, y))


def summarize(frames, threshold_deg=5.0):
    rows = [r for r in map(frame_reversal, frames) if r]
    if not rows:
        return None
    arr = np.degrees(np.array(rows))
    return dict(frames=len(rows),
                supported_over=int((arr[:, 0] > threshold_deg).sum()),
                beyond_support_over=int((arr[:, 1] > threshold_deg).sum()),
                axis_over=int((arr[:, 2] > threshold_deg).sum()),
                max_deg=float(arr[:, 2].max()))


def main(paths):
    for path in paths:
        data = json.load(open(path, encoding='utf-8'))
        report = summarize(data['frames'] if isinstance(data, dict) else data)
        print(path, json.dumps(report))


if __name__ == '__main__':
    main(sys.argv[1:])
