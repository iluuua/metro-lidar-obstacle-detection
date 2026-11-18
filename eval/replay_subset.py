"""Replays only some frames of a recording through the core: what the node saw when it dropped frames.

    python eval/replay_subset.py --dll core/build/libmetro_core.so --sequence sample --first 0 --last 100 --keep-from-ros output/ros_path/sample.log

--keep-from-ros takes the answer times of a ROS run (eval/ros_path_check.py log) as the frames the node processed; without it every
`--stride`-th frame is kept. Prints the alarm frames of the replay.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINE = re.compile(r'(?P<t>[\d.]+)с\s+(?P<state>ПРЕПЯТСТВИЕ|ВНИМАНИЕ(?: УСИЛЕННОЕ)?|СВОБОДНО)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dll', required=True)
    ap.add_argument('--sequence', required=True)
    ap.add_argument('--first', type=int, default=0)
    ap.add_argument('--last', type=int, default=10 ** 9)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--keep-from-ros', type=Path)
    ap.add_argument('--config', default='{}')
    args = ap.parse_args()
    os.environ['METRO_CORE_LIB'] = str(ROOT / 'core/build' / args.dll)
    sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
    import numpy as np
    from metro.native import NativeTunnelDetector
    from metro.tunnel import TunnelConfig
    cfg = TunnelConfig().as_dict()
    cfg.update(json.loads(args.config))

    class Settings:
        def as_dict(self):
            return cfg

    manifest = json.loads((ROOT / 'viewer/data' / args.sequence / 'manifest.json').read_text(encoding='utf-8'))
    frames = manifest['frames']
    keep = None
    if args.keep_from_ros:
        times = []
        for line in args.keep_from_ros.read_text(encoding='utf-8', errors='replace').splitlines():
            m = LINE.search(line)
            if m:
                times.append(float(m.group('t')))
        # the status echo prints the stamp relative to the first answer: frame number = round(10 * (t - t0))
        keep = {int(round(10 * t)) for t in times}
    from source_clouds import SourceClouds
    clouds = SourceClouds()
    detector = NativeTunnelDetector(Settings())
    alarms, fed = [], 0
    for i, record in enumerate(frames):
        if not args.first <= i < args.last:
            continue
        if keep is not None:
            if i not in keep:
                continue
        elif i % args.stride:
            continue
        if record.get('full_file'):
            pts = np.fromfile(ROOT / 'viewer' / record['full_file'], '<f4').reshape(-1, 3)
        else:
            pts = np.frombuffer(clouds.read(manifest['summary'], i), '<f4').reshape(-1, 3)
        pts = pts[np.einsum('ij,ij->i', pts, pts) > 1.]
        r = detector.process(pts, None, record['time_s'])
        fed += 1
        if r['obstacle']:
            alarms.append((i, round(r['distance_m'], 1)))
    print(f'fed {fed} frames, alarm frames {len(alarms)}: {alarms[:40]}')


if __name__ == '__main__':
    main()
