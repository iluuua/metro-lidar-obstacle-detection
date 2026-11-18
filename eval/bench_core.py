"""Time per frame of the C++ core and the Python reference on the same frames, one process at a time.

    .venv/bin/python eval/bench_core.py fake obstacle_sample falcon --frames 300
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]
from metro.tunnel import TunnelConfig, TunnelDetector  # noqa: E402
from metro.native import NativeTunnelDetector  # noqa: E402
from stability import frames_of, run_config  # noqa: E402


def bench(frames, det):
    wall, points = [], []
    for _, pts, t in frames:
        tic = time.perf_counter()
        det.process(pts, None, t)
        wall.append((time.perf_counter() - tic) * 1000)
        points.append(len(pts))
    if not wall:
        raise ValueError('no frames available for benchmark')
    # Full benchmarks omit template warmup; short smoke checks still have data.
    w = np.asarray(wall[5:] if len(wall) > 5 else wall)
    return dict(p50_ms=round(float(np.median(w)), 1), p95_ms=round(float(np.percentile(w, 95)), 1),
                points_k=round(float(np.median(points)) / 1000, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--native-only', action='store_true', help='measure only the C++ core, without the Python reference')
    ap.add_argument('--save', help='save the per-recording result dictionaries as JSON')
    a = ap.parse_args()
    if a.frames < 1:
        ap.error('--frames must be positive')
    out = {}
    for run in a.runs:
        frames = []
        for i, pts, t in frames_of(run):
            if i < a.start:
                continue
            if len(frames) >= a.frames:
                break
            frames.append((i, pts, t))
        cpp = bench(frames, NativeTunnelDetector(run_config(run)))
        out[run] = dict(cpp=cpp)
        if not a.native_only:
            py = bench(frames, TunnelDetector(run_config(run)))
            out[run].update(python=py, speedup=round(py['p50_ms'] / cpp['p50_ms'], 1))
        print(run, json.dumps(out[run]), flush=True)
    if a.save:
        path = Path(a.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
