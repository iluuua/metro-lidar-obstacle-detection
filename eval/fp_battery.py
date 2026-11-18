"""Count false alarms on recordings the caller identifies as obstacle-free."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]
from metro.tunnel import TunnelConfig  # noqa: E402
from metro.native import NativeTunnelDetector  # noqa: E402
from stability import frames_of  # noqa: E402
from run_cached import iter_frames  # noqa: E402

def make(overrides):
    return NativeTunnelDetector(TunnelConfig(), overrides)


def run_frames(frames, overrides, progress=0, label='PROGRESS'):
    det = make(overrides)
    alarms, cand, unknown, n, episodes, last = 0, 0, 0, 0, [], -99
    for i, pts, t in frames:
        r = det.process(pts, None, t)
        n += 1
        if r['status'] == 'obstacle':
            alarms += 1
            if i - last > 5:
                episodes.append((i, round(r['distance_m'], 1)))
            last = i
        cand += r['status'] == 'candidate'
        unknown += r['status'] == 'unknown'
        if progress and n % progress == 0:
            print(label, dict(processed_frames=n, frame=i, alarm_frames=alarms,
                              candidate_frames=cand, unknown_frames=unknown), flush=True)
    return dict(frames=n, alarm_frames=alarms, candidate_frames=cand, unknown_frames=unknown, episodes=episodes[:8])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-fuse', action='store_true')
    ap.add_argument('--config', default='{}')
    ap.add_argument('--runs', nargs='+', required=True, help='obstacle-free cache/viewer run IDs')
    ap.add_argument('--save', help='save all recording results and TOTAL as JSON')
    ap.add_argument('--progress', type=int, default=0, help='print progress every N processed frames; 0 disables it')
    a = ap.parse_args()
    if a.progress < 0:
        ap.error('--progress must be nonnegative')
    overrides = json.loads(a.config)
    if a.no_fuse:
        overrides['fuse'] = False
    total = dict(frames=0, alarm_frames=0, candidate_frames=0)
    results = {}
    for run in a.runs:
        r = run_frames(frames_of(run), overrides, a.progress, run)
        results[run] = r
        print(run, r, flush=True)
        for k in total:
            total[k] += r[k]
    print('TOTAL', total)
    if a.save:
        path = Path(a.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(runs=results, total=total, config=overrides),
                                   ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
