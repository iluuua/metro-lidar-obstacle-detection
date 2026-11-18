"""Run the detector over a ROS 2 bag and export the frames for the viewer.

    python scripts/replay_bag.py /path/to/your/recording --id my_run
    python scripts/replay_bag.py <bag> --id my_run --config '{"half_width": 1.1, "height": 3.2}'

Writes viewer/data/<id>/: a thinned preview and the full cloud of every frame,
the detector answer per frame and a manifest; viewer/data/index.json lists all runs.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from rosbags.highlevel import AnyReader  # noqa: E402
from rosbags.typesys import Stores, get_typestore  # noqa: E402

from metro.io import AXES, FALCON, decode, forward_axis, lidar_of  # noqa: E402
from metro.tunnel import TunnelConfig, TunnelDetector  # noqa: E402
from metro import native  # noqa: E402

# Parameters the viewer may change before a recompute.
EDITABLE = ('half_width', 'height', 'min_height', 'margin_k', 'min_range', 'max_range')


def preview(points, frame, limit=24000, reach=180.):
    """Keep the forward corridor before sampling surrounding tunnel context."""
    visible = points[(points[:, 0] > 0) & (points[:, 0] < reach)
                     & (abs(points[:, 1]) < 15) & (abs(points[:, 2]) < 15)]
    inside = visible[abs(visible[:, 1]) < 2]
    outside = visible[abs(visible[:, 1]) >= 2]
    rng = np.random.default_rng(frame)
    inside_limit = min(len(inside), limit // 2)
    if len(inside) > inside_limit:
        inside = inside[rng.choice(len(inside), inside_limit, replace=False)]
    outside_limit = max(0, limit - len(inside))
    if len(outside) > outside_limit:
        outside = outside[rng.choice(len(outside), outside_limit, replace=False)]
    return np.concatenate((inside, outside)).astype('<f4', copy=False)


def jsonable(v):
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def make_config(overrides=None):
    """TunnelConfig with validated overrides of the editable parameters."""
    cfg = TunnelConfig()
    for key, value in (overrides or {}).items():
        if key not in EDITABLE:
            raise ValueError(f'Unknown or read-only parameter: {key}')
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be a non-negative number')
        setattr(cfg, key, value)
    if not (.5 <= cfg.half_width <= 2.5 and 1. <= cfg.height <= 5. and cfg.min_range < cfg.max_range <= 400.):
        raise ValueError('half_width 0.5-2.5 m, height 1-5 m, min_range < max_range <= 400 m')
    return cfg


def sensor_of(msg):
    """Which lidar wrote the cloud: viewer colours and texts depend on its reach."""
    lidar = lidar_of(msg)
    return dict(lidar=lidar, color_range_m=300. if lidar == FALCON else 150.)


def replay(bag, sequence_id, *, max_frames=None, start_s=0., duration_s=None, config=None, title=None, every=1,
           backend='auto', preview_only=False):
    bag = Path(bag).resolve()
    if not (bag / 'metadata.yaml').is_file():
        raise FileNotFoundError(f'Expected an extracted ROS 2 bag directory: {bag}')
    if not sequence_id.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Sequence id must contain only letters, digits, _ or -')
    cfg = make_config(config)
    dest = ROOT / 'viewer' / 'data' / sequence_id
    dest.mkdir(parents=True, exist_ok=True)
    # a recompute keeps the human title of the run
    try:
        previous = json.loads((dest / 'manifest.json').read_text(encoding='utf-8')).get('summary', {})
    except (FileNotFoundError, ValueError):
        previous = {}
    rows, frames, elapsed = [], [], []
    began = time.perf_counter()
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        connections = [c for c in reader.connections if c.msgtype == 'sensor_msgs/msg/PointCloud2']
        if len(connections) != 1:
            raise ValueError('Expected exactly one PointCloud2 topic')
        if start_s < 0 or (duration_s is not None and duration_s <= 0):
            raise ValueError('start_s must be non-negative and duration_s positive')
        _, _, first_raw = next(reader.messages(connections=connections))
        first_msg = reader.deserialize(first_raw, connections[0].msgtype)
        # the forward axis of the sensor frame, found from the first cloud (z is up in all recordings)
        axis, _ = forward_axis(decode(first_msg, raw=True)[0])
        axis = axis or '-y'
        transform = None if axis == '-y' else np.asarray(AXES[axis]).reshape(4, 4)
        sensor = sensor_of(first_msg)
        # as metro/pipeline.py: the Falcon envelope reaches 400 m unless the range was given
        if sensor['lidar'] == FALCON and 'max_range' not in (config or {}):
            cfg.max_range = cfg.long_range
        if backend == 'cpp' or (backend == 'auto' and native.available()):
            det, backend = native.NativeTunnelDetector(cfg), 'cpp'
        else:
            det, backend = TunnelDetector(cfg), 'python'
        first_stamp = first_msg.header.stamp.sec * 1_000_000_000 + first_msg.header.stamp.nanosec
        start_ns = reader.start_time + round(start_s * 1e9)
        stop_ns = None if duration_s is None else start_ns + round(duration_s * 1e9)
        messages = reader.messages(connections=connections, start=start_ns, stop=stop_ns)
        for k, (connection, bag_ns, raw) in enumerate(messages):
            if k % every:
                continue
            i = k // every
            if max_frames is not None and i >= max_frames:
                break
            msg = reader.deserialize(raw, connection.msgtype)
            points, rings = decode(msg, source_to_analysis=transform)
            header_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            t = (header_ns - first_stamp) / 1e9
            result = det.process(points, rings, t)
            result.update(frame=i, time_s=t, header_stamp_ns=header_ns, bag_stamp_ns=bag_ns,
                          point_step=msg.point_step, has_ring=rings is not None)
            rows.append(jsonable(result))
            elapsed.append(result['latency_ms'])
            stem = f'{i:05d}'
            view = preview(points, i, reach=max(180., 1.4 * sensor['color_range_m']))
            view.tofile(dest / f'{stem}.bin')
            full = None
            if not preview_only:
                full, _ = decode(msg, min_distance=0., source_to_analysis=transform)
                full.astype('<f4', copy=False).tofile(dest / f'{stem}.full.bin')
            frames.append(dict(frame=i, time_s=t, status=result['status'], obstacle=result['obstacle'],
                               distance_m=result['distance_m'], candidates=result['candidates'], points=len(points),
                               latency_ms=result['latency_ms'], raw_points=msg.width * msg.height,
                               file=f'data/{sequence_id}/{stem}.bin', display_points=len(view)))
            if full is not None:
                frames[-1].update(full_file=f'data/{sequence_id}/{stem}.full.bin', full_points=len(full),
                                  full_bounds=[full.min(0).tolist(), full.max(0).tolist()] if len(full) else None)
            if i % 50 == 0:
                print(f'{sequence_id}: {i} {result["status"]} {result["distance_m"]}', flush=True)
                print(json.dumps({'event': 'progress', 'done': i + 1, 'total': 0}), flush=True)
    if not rows:
        raise ValueError('No PointCloud2 frames in bag')
    source = str(bag.relative_to(ROOT)) if bag.is_relative_to(ROOT) else str(bag)
    summary = dict(id=sequence_id, title=title or previous.get('title') or sequence_id, source_bag=source,
                   source_start_s=start_s, source_duration_s=duration_s, source_max_frames=max_frames,
                   source_topic=connections[0].topic, frames=len(rows), duration_s=rows[-1]['time_s'],
                   source_every=every, full_cloud_exported=not preview_only, native_hz=10. / every, sensor_forward_axis=axis, backend=backend, **sensor,
                   obstacle_frames=sum(r['obstacle'] for r in rows),
                   candidate_frames=sum(r['status'] == 'candidate' for r in rows),
                   unknown_frames=sum(r['status'] == 'unknown' for r in rows),
                   latency_p50_ms=float(np.median(elapsed)), latency_p95_ms=float(np.percentile(elapsed, 95)),
                   wall_s_including_export=time.perf_counter() - began,
                   config={k: getattr(cfg, k) for k in EDITABLE})
    (dest / 'model.tunnel.json').write_text(json.dumps(dict(summary=summary, frames=rows),
                                                       ensure_ascii=False, allow_nan=False), encoding='utf-8')
    models = [dict(id='tunnel', name='Профиль тоннеля', file=f'data/{sequence_id}/model.tunnel.json')]
    (dest / 'manifest.json').write_text(json.dumps(dict(summary=summary, frames=frames, models=models),
                                                   ensure_ascii=False, allow_nan=False), encoding='utf-8')
    data_root = ROOT / 'viewer' / 'data'
    index = [json.loads(p.read_text(encoding='utf-8'))['summary'] for p in sorted(data_root.glob('*/manifest.json'))]
    (data_root / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'event': 'complete', 'summary': summary}, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', type=Path, help='directory with metadata.yaml and .db3')
    parser.add_argument('--id', required=True, help='name of the run in the viewer')
    parser.add_argument('--title', help='human-readable title in the viewer')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--preview-only', action='store_true', help='export light clouds; detector still receives all points in every frame')
    parser.add_argument('--start-s', type=float, default=0.)
    parser.add_argument('--duration-s', type=float)
    parser.add_argument('--every', type=int, default=1, help='keep every n-th cloud (long recordings)')
    parser.add_argument('--config', help='JSON with detector parameters: ' + ', '.join(EDITABLE))
    parser.add_argument('--backend', choices=('auto', 'cpp', 'python'), default='auto',
                        help='detector implementation: C++ core (when built) or the Python reference')
    args = parser.parse_args()
    replay(args.bag, args.id, max_frames=args.max_frames, start_s=args.start_s, duration_s=args.duration_s,
           config=json.loads(args.config) if args.config else None, title=args.title, every=max(1, args.every),
           backend=args.backend, preview_only=args.preview_only)


if __name__ == '__main__':
    main()
