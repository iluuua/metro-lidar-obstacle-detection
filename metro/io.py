"""PointCloud2 decoding that honours field offsets, endianness and row padding."""
import os

import numpy as np

# PointField datatype -> numpy type
TYPES = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
FIELDS = ('x', 'y', 'z', 'intensity', 'ring', 'timestamp')

# Source frame -> analysis frame (x forward, y left, z up) for a sensor whose z axis is up,
# by the horizontal axis that looks along the track. The axis can be supplied explicitly or inferred from the cloud.
AXES = {
    '+x': [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
    '-x': [-1., 0., 0., 0., 0., -1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
    '+y': [0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
    '-y': [0., -1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
}


PANDAR, FALCON = 'Hesai Pandar128', 'Seyond Falcon Prime'


def lidar_of(msg):
    """Which of the task's lidars wrote the cloud: the Seyond driver adds scan fields and names its frame."""
    names = {f.name for f in msg.fields}
    frame = msg.header.frame_id.lower()
    return FALCON if 'scan_id' in names or 'seyond' in frame or 'innovusion' in frame else PANDAR


def forward_axis(xyz, min_range=40., half_angle_deg=20., min_points=200):
    """The horizontal sensor axis that looks along the tunnel, or None if the cloud cannot tell.

    Along the tunnel the view is open for a hundred metres and more; sideways it ends at
    the wall and backwards at the train. So the axis with the most far returns inside a
    narrow horizontal cone is the forward one.
    """
    p = np.asarray(xyz, np.float64)
    p = p[np.isfinite(p).all(1)]
    far = p[np.hypot(p[:, 0], p[:, 1]) > min_range]
    tan = np.tan(np.radians(half_angle_deg))
    counts = {}
    for name, (i, sign) in {'+x': (0, 1.), '-x': (0, -1.), '+y': (1, 1.), '-y': (1, -1.)}.items():
        a, b = sign * far[:, i], far[:, 1 - i]
        counts[name] = int(np.count_nonzero((a > 0) & (np.abs(b) < a * tan)))
    best = max(counts, key=counts.get)
    ranked = sorted(counts.values(), reverse=True)
    if ranked[0] < min_points or ranked[0] < 3 * max(ranked[1], 1):
        return None, counts
    return best, counts


def decode(msg, min_distance=1., source_to_analysis=None, raw=False):
    """Points of a PointCloud2 message in the analysis frame (x forward, y left, z up).

    Returns (xyz float32 array, ring array or None). Points that are not finite or
    closer than min_distance to the sensor (self-returns, zero points) are dropped.
    raw=True keeps the sensor frame (used to find the forward axis).
    """
    if (msg.width < 0 or msg.height < 0 or msg.point_step <= 0
            or msg.row_step < msg.width * msg.point_step
            or memoryview(msg.data).nbytes < msg.height * msg.row_step):
        raise ValueError('Truncated or invalid PointCloud2 layout')
    endian = '>' if msg.is_bigendian else '<'
    fields = [f for f in msg.fields if f.name in FIELDS]
    if not {'x', 'y', 'z'} <= {f.name for f in fields}:
        raise ValueError('PointCloud2 requires XYZ fields')
    for f in fields:
        if (f.datatype not in TYPES or getattr(f, 'count', 1) != 1 or f.offset < 0
                or f.offset + np.dtype(TYPES[f.datatype]).itemsize > msg.point_step):
            raise ValueError('Invalid scalar PointCloud2 field: ' + f.name)
    if not raw and os.environ.get('METRO_NATIVE_DECODE', '1') != '0':
        # the C++ parser gives the same points in the same order, several times faster
        from . import native
        by = {f.name: f for f in fields}
        ring = by.get('ring')
        out = native.decode_cloud(
            msg, [by['x'].offset, by['y'].offset, by['z'].offset, ring.offset if ring else -1],
            [by['x'].datatype, by['y'].datatype, by['z'].datatype, ring.datatype if ring else 0],
            min_distance, source_to_analysis, ring is not None)
        if out is not None:
            return out
    dtype = np.dtype({'names': [f.name for f in fields],
                      'formats': [endian + TYPES[f.datatype] for f in fields],
                      'offsets': [f.offset for f in fields],
                      'itemsize': msg.point_step})
    cloud = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                       strides=(msg.row_step, msg.point_step)).reshape(-1)
    p = np.column_stack([cloud[n] for n in ('x', 'y', 'z')])
    keep = np.isfinite(p).all(1) & (np.einsum('ij,ij->i', p, p) > min_distance ** 2)
    if raw:
        xyz = p[keep].astype(np.float32, copy=False)
    elif source_to_analysis is None:
        # default sensor frame: forward along -y, left along +x
        xyz = p[keep][:, [1, 0, 2]].copy()
        xyz[:, 0] *= -1
    else:
        xyz = p[keep] @ source_to_analysis[:3, :3].T + source_to_analysis[:3, 3]
    rings = cloud['ring'][keep] if 'ring' in cloud.dtype.names else None
    return xyz, rings
