"""Generate a synthetic tunnel for the viewer without external recordings.

    python scripts/make_demo.py
    python scripts/serve.py
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from metro.tunnel import TunnelConfig, TunnelDetector


def cloud_frame(index, count=60000):
    """Generate a rail bed, rail heads, round walls and a box."""
    rng = np.random.default_rng(1000 + index)
    position = index * .25

    def along(size):
        return 3 + 117 * rng.random(size) ** 2

    distance = along(count // 3)
    lateral = rng.uniform(-1.45, 1.45, len(distance))
    height = np.where(np.abs(lateral) < .2, -.5, -.25)
    height += rng.normal(0, .005, len(distance))
    parts = [(distance, lateral, height)]
    for side in (-1, 1):
        distance = along(count // 8)
        lateral = side * .795 + rng.uniform(-.035, .035, len(distance))
        parts.append((distance, lateral, rng.normal(0, .004, len(distance))))
    distance = along(count // 2)
    angle = rng.uniform(-1.9, 1.9, len(distance))
    radius = 2.6 - .12 * (((distance + position) % 1.7) < .15)
    parts.append((distance, radius * np.sin(angle), 1.6 + radius * np.cos(angle)))
    parts.append((45 - position + rng.uniform(0, .15, 1800),
                  rng.uniform(-.25, .25, 1800), rng.uniform(.4, .9, 1800)))
    forward = np.concatenate([part[0] for part in parts])
    lateral = np.concatenate([part[1] for part in parts])
    height = np.concatenate([part[2] for part in parts]) - 1.12
    return np.column_stack((forward, lateral, height)).astype('<f4')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', default='synthetic_demo')
    parser.add_argument('--frames', type=int, default=35)
    args = parser.parse_args()
    if not args.id.replace('_', '').replace('-', '').isalnum() or not 1 <= args.frames <= 300:
        parser.error('use an alphanumeric id and between 1 and 300 frames')
    folder = ROOT / 'viewer' / 'data' / args.id
    folder.mkdir(parents=True, exist_ok=True)
    detector = TunnelDetector(TunnelConfig())
    model, frames = [], []
    for index in range(args.frames):
        points = cloud_frame(index)
        time_s = index / 10
        result = detector.process(points, None, time_s)
        result.update(frame=index, time_s=time_s)
        stem = f'{index:05d}'
        points.tofile(folder / f'{stem}.bin')
        model.append(result)
        frames.append(dict(
            frame=index, time_s=time_s, file=f'data/{args.id}/{stem}.bin',
            full_file=f'data/{args.id}/{stem}.bin', raw_points=len(points),
            display_points=len(points), full_points=len(points),
            latency_ms=result['latency_ms']))
    summary = dict(id=args.id, title='Синтетический тоннель с препятствием',
                   hint='Лидар движется по синтетическому тоннелю с ящиком на пути.',
                   kind='synthetic_demo', source_bag='', frames=len(frames),
                   duration_s=frames[-1]['time_s'], full_cloud_exported=True,
                   native_hz=10, backend='python', color_range_m=120)
    manifest = dict(summary=summary, frames=frames,
                    models=[dict(id='tunnel', name='Профиль тоннеля',
                                 file=f'data/{args.id}/model.tunnel.json')])
    (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
    (folder / 'model.tunnel.json').write_text(
        json.dumps(dict(summary=summary, frames=model), ensure_ascii=False), encoding='utf-8')
    index = [json.loads(path.read_text(encoding='utf-8'))['summary']
             for path in sorted(folder.parent.glob('*/manifest.json'))]
    (folder.parent / 'index.json').write_text(json.dumps(index, ensure_ascii=False), encoding='utf-8')
    print(f'Generated {len(frames)} synthetic frames. Run python scripts/serve.py')


if __name__ == '__main__':
    main()
