"""Publish the C++ core's own answer, with no Python post-processing, as a saved run for the viewer.

The ROS node publishes exactly this, so it is what the viewer should show and what FP/FN should judge.

    METRO_CORE_LIB=core/build/<lib> python eval/publish_core.py --label "Ядро по умолчанию" --sequences labelled_sample,falcon
    python eval/publish_core.py --dll core/build/libmetro_core.so --label "Ядро по умолчанию" --sequences sample
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def publish(job):
    sequence, dll, overrides, label = job
    os.environ['METRO_CORE_LIB'] = str(dll)
    sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
    import numpy as np
    from metro.native import NativeTunnelDetector
    from metro.tunnel import TunnelConfig
    from source_clouds import SourceClouds
    folder = ROOT / 'viewer/data' / sequence
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    summary = manifest['summary']
    cfg = TunnelConfig().as_dict()
    cfg.update(overrides)

    class Settings:
        def as_dict(self):
            return cfg

    detector = NativeTunnelDetector(Settings())
    clouds = SourceClouds()
    frames, latency = [], []
    for i, record in enumerate(manifest['frames']):
        if record.get('full_file'):
            points = np.fromfile(ROOT / 'viewer' / record['full_file'], '<f4').reshape(-1, 3)
        else:
            points = np.frombuffer(clouds.read(summary, i), '<f4').reshape(-1, 3)
        points = points[np.einsum('ij,ij->i', points, points) > 1.]
        result = detector.process(points, None, record['time_s'])
        result.update(frame=record['frame'], time_s=record['time_s'])
        frames.append(result)
        latency.append(result['latency_ms'])
    result_summary = dict(summary, config={k: v for k, v in cfg.items() if isinstance(v, (int, float, bool))},
                          detector_version='core', obstacle_frames=sum(f['obstacle'] for f in frames),
                          candidate_frames=sum(f['status'] == 'candidate' for f in frames),
                          unknown_frames=sum(f['status'] == 'unknown' for f in frames),
                          latency_p95_ms=float(np.percentile(latency, 95)))
    library = hashlib.sha256(Path(dll).read_bytes()).hexdigest()
    signature = dict(version='core', config=cfg, library_sha256=library, label=label)
    run_id = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    dest = folder / 'runs'
    dest.mkdir(exist_ok=True)
    temporary = dest / f'{run_id}.tmp'
    temporary.write_text(json.dumps(dict(summary=result_summary, frames=frames), ensure_ascii=False), encoding='utf-8')
    temporary.replace(dest / f'{run_id}.json')
    meta = dict(id=run_id, sequence=sequence, version='core', label=label, parameters={}, created_at=time.time(),
                file=f'data/{sequence}/runs/{run_id}.json', signature=signature)
    (dest / f'{run_id}.meta.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
    return sequence, run_id, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dll', default=os.environ.get('METRO_CORE_LIB'))
    ap.add_argument('--label', required=True)
    ap.add_argument('--config', default='{}')
    ap.add_argument('--sequences', required=True, help='comma-separated user-provided viewer runs')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    if not args.dll:
        sys.exit('Give --dll or set METRO_CORE_LIB')
    dll = Path(args.dll)
    dll = dll if dll.is_absolute() or dll.exists() else ROOT / 'core/build' / args.dll
    sequences = args.sequences.split(',')
    jobs = [(s, str(dll), json.loads(args.config), args.label) for s in sequences]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for sequence, run_id, n in pool.map(publish, jobs):
            print('published', sequence, run_id, n, flush=True)


if __name__ == '__main__':
    main()
