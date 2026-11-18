"""Cache PointCloud2 frames of a ROS2 bag as raw per-frame .npz for fast experiments.

Raw sensor XYZ (+intensity, ring, per-point time when present) are stored in
the original point order; conversion to the analysis frame happens in
eval/run_cached.load_frame exactly as metro.io.decode does.

    .venv/bin/python eval/cache_bag.py datasets/sample fake
    .venv/bin/python eval/cache_bag.py datasets/additional/sample sample --start-s 600 --duration-s 60
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metro.io import TYPES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('bag', type=Path)
    ap.add_argument('cache_id')
    ap.add_argument('--start-s', type=float, default=0.)
    ap.add_argument('--duration-s', type=float)
    a = ap.parse_args()
    out = ROOT / 'output' / 'raw_cache' / a.cache_id
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with AnyReader([a.bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        con = [c for c in reader.connections if c.msgtype == 'sensor_msgs/msg/PointCloud2']
        start = reader.start_time + int(a.start_s * 1e9)
        stop = None if a.duration_s is None else start + int(a.duration_s * 1e9)
        for k, (c, t, raw) in enumerate(reader.messages(connections=con, start=start, stop=stop)):
            msg = reader.deserialize(raw, c.msgtype)
            names = {f.name: f for f in msg.fields}
            dt = np.dtype({'names': list(names), 'formats': ['<' + TYPES[f.datatype] for f in names.values()],
                           'offsets': [f.offset for f in names.values()], 'itemsize': msg.point_step})
            arr = np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)
            data = dict(xyz=np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float32))
            if 'intensity' in names:
                data['intensity'] = arr['intensity'].astype(np.float32)
            if 'ring' in names:
                data['ring'] = arr['ring'].astype(np.uint16)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            index = int(round(a.start_s * 10)) + k
            np.savez(out / f'{index:05d}.npz', stamp=stamp, bag_ns=t, **data)
    print('cached', a.cache_id, 'in %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
