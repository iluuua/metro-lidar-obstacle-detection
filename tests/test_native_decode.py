"""The C++ PointCloud2 parser (core/src/cloud.cpp) gives exactly the points of the numpy decoder and
refuses malformed messages; random payloads never crash it."""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from metro import native
from metro.io import decode, AXES


def field(name, offset, datatype):
    return SimpleNamespace(name=name, offset=offset, datatype=datatype, count=1)


def cloud(points, ring=None, big=False, f64=False, pad=0, height=1):
    """A PointCloud2-like message: x, y, z (float32 or float64), optional uint16 ring, row padding."""
    fmt = ('>' if big else '<') + ('f8' if f64 else 'f4')
    size = 8 if f64 else 4
    names = ['x', 'y', 'z'] + (['ring'] if ring is not None else [])
    formats = [fmt] * 3 + ([('>' if big else '<') + 'u2'] if ring is not None else [])
    offsets = [0, size, 2 * size] + ([3 * size] if ring is not None else [])
    step = 3 * size + (2 if ring is not None else 0) + 3  # odd padding inside a point
    dtype = np.dtype({'names': names, 'formats': formats, 'offsets': offsets, 'itemsize': step})
    n = len(points)
    width = n // height
    arr = np.zeros(n, dtype)
    for k, nm in enumerate('xyz'):
        arr[nm] = points[:, k]
    if ring is not None:
        arr['ring'] = ring
    rows = arr.reshape(height, width).view(np.uint8).reshape(height, width * step)
    data = np.concatenate([rows, np.full((height, pad), 0xAB, np.uint8)], 1).reshape(-1).tobytes()
    types = {'f4': 7, 'f8': 8}[fmt[1:]]
    fields = [field(nm, o, types) for nm, o in zip('xyz', offsets)] + \
             ([field('ring', offsets[3], 4)] if ring is not None else [])
    return SimpleNamespace(width=width, height=height, point_step=step, row_step=width * step + pad,
                           is_bigendian=big, fields=fields, data=data)


def run_both(msg, transform=None):
    os.environ['METRO_NATIVE_DECODE'] = '0'
    a, ra = decode(msg, source_to_analysis=transform)
    os.environ['METRO_NATIVE_DECODE'] = '1'
    b, rb = decode(msg, source_to_analysis=transform)
    return np.asarray(a, np.float32), ra, b, rb


@unittest.skipUnless(native.available() and hasattr(native.library(), 'metro_decode'), 'libmetro_core is not built')
class NativeDecodeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.p = rng.normal(0, 30, (600, 3)).astype(np.float32)
        self.p[::17] = np.nan            # invalid returns
        self.p[5::31] = [.1, .2, 0.]     # inside the minimum distance
        self.p[7::41, 1] = np.inf
        self.ring = rng.integers(0, 128, 600).astype(np.uint16)

    def tearDown(self):
        os.environ.pop('METRO_NATIVE_DECODE', None)

    def check(self, msg, transform=None):
        a, ra, b, rb = run_both(msg, transform)
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_array_equal(a, b)
        if ra is None:
            self.assertIsNone(rb)
        else:
            np.testing.assert_array_equal(ra, rb)

    def test_layouts_match_numpy(self):
        for big in (False, True):
            for f64 in (False, True):
                for ring in (None, self.ring):
                    with self.subTest(big=big, f64=f64, ring=ring is not None):
                        self.check(cloud(self.p, ring, big=big, f64=f64, pad=5, height=3))

    def test_transform_matches_numpy(self):
        for axis in ('+x', '-x', '+y'):
            self.check(cloud(self.p, self.ring), np.asarray(AXES[axis]).reshape(4, 4))

    def test_truncated_message_is_refused(self):
        msg = cloud(self.p, self.ring)
        msg.data = msg.data[:-10]
        with self.assertRaises(ValueError):
            decode(msg)

    def test_layout_rechecked_in_cpp(self):
        msg = cloud(self.p)
        bad = [msg.point_step, 4, 8, -1]  # x past the end of the point
        with self.assertRaises(ValueError):
            native.decode_cloud(msg, bad, [7, 7, 7, 0], 1., None, False)

    def test_random_payloads_do_not_crash(self):
        rng = np.random.default_rng(11)
        for k in range(300):
            step = int(rng.integers(12, 64))
            width = int(rng.integers(0, 400))
            data = rng.integers(0, 256, width * step, dtype=np.uint8).tobytes()
            msg = SimpleNamespace(width=width, height=1, point_step=step, row_step=width * step,
                                  is_bigendian=bool(rng.integers(0, 2)), data=data,
                                  fields=[field('x', 0, 7), field('y', 4, 7), field('z', 8, 7)])
            xyz, _ = decode(msg)
            self.assertTrue(np.isfinite(xyz).all())


if __name__ == '__main__':
    unittest.main()
