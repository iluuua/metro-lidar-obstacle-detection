"""Run a detector over cached raw frames (output/raw_cache/<id>/NNNNN.npz) and score.

Development harness only: the ROS node and replay_bag.py read PointCloud2
directly. Cache files keep raw sensor XYZ; conversion to the analysis frame is
the same as metro.io.decode (forward=-y, lateral=x).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metro.tunnel import TunnelDetector, TunnelConfig  # noqa: E402


def load_frame(path):
    z = np.load(path)
    r = z['xyz']
    ok = np.isfinite(r).all(1) & ((r * r).sum(1) > 1.)
    r = r[ok]
    pts = np.column_stack([-r[:, 1], r[:, 0], r[:, 2]]).astype(np.float32)
    return pts, float(z['stamp'])


def iter_frames(source, start=0, stop=None):
    """Yield (index, points, stamp) from output/raw_cache/<id> or output/decoded_cache/<bag>."""
    raw = ROOT / 'output' / 'raw_cache' / source
    if raw.is_dir():
        files = [f for f in sorted(raw.glob('[0-9]*.npz'))
                 if int(f.stem) >= start and (stop is None or int(f.stem) < stop)]
        for f in files:
            pts, stamp = load_frame(f)
            yield int(f.stem), pts, stamp
        return
    dec = ROOT / 'output' / 'decoded_cache' / source
    sub = next(d for d in dec.iterdir() if (d / 'index.json').exists())
    index = json.loads((sub / 'index.json').read_text())['frames'][start:stop]
    for rec in index:
        pts = np.load(sub / rec['points']['file']).astype(np.float32)
        yield rec['index'], pts, rec['time_s']


def compact(res):
    keep = {k: res[k] for k in ('status', 'obstacle', 'distance_m', 'analysis_range_m', 'latency_ms') if k in res}
    keep['candidates'] = [{k: c[k] for k in ('distance_m', 'bbox', 'confirmed', 'hits', 'point_count',
                                             'confident_points', 'margin_m', 'path_lateral_m', 'h_range', 'u_range', 'track_id',
                                             'wall_like', 'held', 's_max', 'boundary')
                           if k in c} for c in res.get('candidates', [])]
    info = res.get('path_info', {})
    keep['support_m'] = info.get('support_m')
    keep['rail_support_m'] = info.get('rail_support_m')
    keep['ds'] = info.get('odometry_ds')
    keep['timing'] = res.get('timing')
    return keep


def run(cache_id, start=0, stop=None, config=None, verbose=True, save=None, stride=1, native=False):
    if native:
        from metro.native import NativeTunnelDetector
        det = NativeTunnelDetector(config or TunnelConfig())
    else:
        det = TunnelDetector(config or TunnelConfig())
    out = []
    lat = []
    t0 = time.perf_counter()
    for n, (i, pts, stamp) in enumerate(iter_frames(cache_id, start, stop)):
        if n % stride:
            continue    # a slow computer takes only every stride-th frame
        res = det.process(pts, None, stamp)
        lat.append(res['latency_ms'])
        c = compact(res)
        c['frame'] = i
        out.append(c)
        if verbose and i % 100 == 0:
            print(i, c['status'], c['distance_m'], 'support', c['support_m'], 'rail', c['rail_support_m'],
                  'ds', c['ds'], '%.0fms' % res['latency_ms'], flush=True)
    wall = time.perf_counter() - t0
    summary = dict(frames=len(out), latency_p50_ms=float(np.median(lat)), latency_p95_ms=float(np.percentile(lat, 95)),
                   wall_s=wall, obstacle_frames=sum(o['obstacle'] for o in out))
    if save:
        Path(save).write_text(json.dumps(dict(summary=summary, frames=out)))
    return summary, out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cache_id')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--stop', type=int)
    ap.add_argument('--save')
    ap.add_argument('--truth', type=Path, help='optional user-provided truth JSON for scoring')
    ap.add_argument('--stride', type=int, default=1, help='process every n-th frame, as a slow computer would')
    ap.add_argument('--native', action='store_true', help='C++ core instead of the Python reference')
    a = ap.parse_args()
    summary, frames = run(a.cache_id, a.start, a.stop, save=a.save, stride=a.stride, native=a.native)
    print(json.dumps(summary))
    if a.truth:
        from score_synthetic import score
        gt = json.loads(a.truth.read_text(encoding='utf-8'))
        score({f['frame']: f for f in frames}, gt)
