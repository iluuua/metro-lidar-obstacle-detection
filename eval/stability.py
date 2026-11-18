"""Envelope stability (organizers' criterion 3) and alarms on a viewer run.

For every frame the axis ahead at fixed distances is compared with the previous frame's axis
moved by the measured travel: in a static world they are the same line, so the difference is
the flicker of the envelope. Also the frame-to-frame change of the envelope length.

    .venv/bin/python eval/stability.py falcon --frames 2000:2600 [--no-fuse] [--python] [--config '{"fuse": false}']
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'eval'))
from dataclasses import replace  # noqa: E402
from metro.io import FALCON  # noqa: E402
from metro.tunnel import TunnelConfig, TunnelDetector  # noqa: E402

AHEAD = (20., 30., 40., 50., 100., 150., 200.)


def run_config(run, config=None):
    """Settings the ROS node would use for the run's lidar (metro/pipeline.py): a Falcon recording
    gets the long reach unless the range was set."""
    cfg = config or TunnelConfig()
    try:
        lidar = json.loads((ROOT / 'viewer' / 'data' / run / 'manifest.json').read_text())['summary'].get('lidar')
    except FileNotFoundError:
        lidar = None
    if lidar == FALCON and cfg.max_range == TunnelConfig.max_range:
        cfg = replace(cfg, max_range=cfg.long_range)
    return cfg


def frames_of(run):
    if (ROOT / 'output' / 'raw_cache' / run).is_dir() or (ROOT / 'output' / 'decoded_cache' / run).is_dir():
        # whole clouds cached by eval/cache_bag.py (or decoded by the old pipeline)
        from run_cached import iter_frames
        yield from iter_frames(run)
        return
    root = ROOT / 'viewer' / 'data' / run
    man = json.loads((root / 'manifest.json').read_text())['frames']
    for rec in man:
        yield rec['frame'], np.fromfile(root / f"{rec['frame']:05d}.full.bin", '<f4').reshape(-1, 3), rec['time_s']


def axis_at(res, x):
    path = np.asarray(res['path'])
    return float(np.interp(x, path[:, 0], path[:, 1])), float(np.interp(x, path[:, 0], path[:, 2]))


def shown(res):
    """Length of the envelope as drawn (the C++ core keeps it steady over short dropouts)."""
    return max(res.get('envelope_range_m') or 0., res.get('analysis_range_m') or 0.)


def moved_axis_at(res, ds, x):
    """The previous frame's axis seen after moving ds along it (as PathModel.shifted): the sensor
    keeps its offset and heading relative to the track, so in a curve the axis is not simply
    the old one read ds farther."""
    path = np.asarray(res['path'])
    g, out = path[:, 0], []
    for col in (1, 2):
        f = path[:, col]
        d = np.gradient(f, g)
        out.append(float(np.interp(x + ds, g, f) - (np.interp(ds, g, f) - f[0]) - (np.interp(ds, g, d) - d[0]) * x))
    return tuple(out)


def evaluate(run, start=0, stop=None, fuse=True, python=False, config=None, overrides=None):
    cfg = run_config(run, config)
    if python:
        det = TunnelDetector(cfg)
    else:
        from metro.native import NativeTunnelDetector
        over = dict(overrides or {})
        if not fuse:
            over['fuse'] = False
        det = NativeTunnelDetector(cfg, over)
    prev = None
    jit = {x: [] for x in AHEAD}
    jit_v = {x: [] for x in AHEAD}
    drange, lat, results = [], [], []
    end_jit, ext_flips, ranges = [], 0, []
    for i, pts, t in frames_of(run):
        if i < start or (stop is not None and i >= stop):
            continue
        res = det.process(pts, None, t)
        lat.append(res.get('latency_ms', 0.))
        if res.get('path'):
            ranges.append((shown(res), res.get('analysis_range_m') or 0.))
        results.append((i, res))
        if prev is not None and res.get('path') and prev.get('path'):
            ds = (res.get('path_info') or {}).get('odometry_ds') or 0.
            for x in AHEAD:
                if x + ds < min(shown(prev), shown(res)):
                    a, b = axis_at(res, x), moved_axis_at(prev, ds, x)
                    jit[x].append(abs(a[0] - b[0]))
                    jit_v[x].append(abs(a[1] - b[1]))
            drange.append(shown(res) - (shown(prev) - ds))
            # the far end of the envelope, where the expert saw it kick
            x = min(shown(prev) - ds, shown(res)) - 5.
            if x > 30:
                a, b = axis_at(res, x), moved_axis_at(prev, ds, x)
                end_jit.append(max(abs(a[0] - b[0]), abs(a[1] - b[1])))
            ext_flips += bool(prev['path_info'].get('extension_ok')) != bool(res['path_info'].get('extension_ok'))
        prev = res if res.get('path') else None
    out = dict(frames=len(results), latency_p50_ms=float(np.median(lat)), latency_p95_ms=float(np.percentile(lat, 95)))
    for x in AHEAD:
        if jit[x]:
            out[f'lat_jitter_{int(x)}m_p50'] = float(np.median(jit[x]))
            out[f'lat_jitter_{int(x)}m_p95'] = float(np.percentile(jit[x], 95))
            out[f'vert_jitter_{int(x)}m_p95'] = float(np.percentile(jit_v[x], 95))
    if drange:
        d = np.abs(drange)
        out.update(range_change_p50=float(np.median(d)), range_change_p95=float(np.percentile(d, 95)),
                   range_drops_over_20m=int(np.sum(np.asarray(drange) < -20)), extension_flips=ext_flips)
    if ranges:
        r = np.asarray(ranges)
        out.update(envelope_median_m=float(np.median(r[:, 0])), verified_median_m=float(np.median(r[:, 1])))
    if end_jit:
        out.update(end_jitter_p50=float(np.median(end_jit)), end_jitter_p95=float(np.percentile(end_jit, 95)))
    st = {}
    for _, r in results:
        st[r['status']] = st.get(r['status'], 0) + 1
    out['status'] = st
    return out, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--frames', default='0:')
    ap.add_argument('--no-fuse', action='store_true')
    ap.add_argument('--python', action='store_true')
    ap.add_argument('--config', default='{}', help='JSON overrides of the C++ detector config')
    a = ap.parse_args()
    lo, hi = a.frames.split(':')
    t0 = time.time()
    out, _ = evaluate(a.run, int(lo or 0), int(hi) if hi else None, fuse=not a.no_fuse, python=a.python,
                      overrides=json.loads(a.config))
    out['wall_s'] = round(time.time() - t0, 1)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
