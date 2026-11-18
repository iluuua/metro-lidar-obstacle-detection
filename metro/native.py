"""ctypes binding of the C++ core (core/, libmetro_core): the same detector as metro.tunnel, faster.

The library is looked up in $METRO_CORE_LIB, core/build/, the lib/ folders of the ROS install
spaces (colcon builds the core with metro_perception) and /opt/metro/lib. libmetro_core_long, the
same core with the axis grid to 430 m, serves a reach beyond SHORT_REACH_M (the Falcon's 400 m);
$METRO_CORE_LIB_LONG overrides its path.
"""
import ctypes
import json
import os
from pathlib import Path

import numpy as np

from .tunnel import TunnelConfig

_libs = {}
SHORT_REACH_M = 325.  # the default grid ends at 330 m; the region of interest reaches max_range + 25


def _candidates(long=False):
    env = os.environ.get('METRO_CORE_LIB_LONG' if long else 'METRO_CORE_LIB')
    if env:
        yield Path(env)
    root = Path(__file__).resolve().parents[1]
    prefixes = [Path(p) / 'lib' for p in os.environ.get('AMENT_PREFIX_PATH', '').split(os.pathsep) if p]
    stem = 'libmetro_core_long' if long else 'libmetro_core'
    for folder in (root / 'core' / 'build', *prefixes, Path('/opt/metro/lib')):
        for ext in ('.so', '.dylib'):
            yield folder / (stem + ext)


def library(long=False):
    """The loaded core library (long: with the 430 m grid), or None when it is not built."""
    if long not in _libs:
        path = next((p for p in _candidates(long) if p.is_file()), None)
        if path is None:
            return None
        lib = ctypes.CDLL(str(path))
        lib.metro_create.argtypes = [ctypes.c_char_p]
        lib.metro_create.restype = ctypes.c_void_p
        lib.metro_create_error.restype = ctypes.c_char_p
        lib.metro_destroy.argtypes = [ctypes.c_void_p]
        lib.metro_reset.argtypes = [ctypes.c_void_p]
        lib.metro_process.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                                      ctypes.c_double, ctypes.c_int]
        lib.metro_process.restype = ctypes.c_char_p
        lib.metro_error.argtypes = [ctypes.c_void_p]
        lib.metro_error.restype = ctypes.c_char_p
        if hasattr(lib, 'metro_decode'):
            lib.metro_decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32,
                                         ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int,
                                         ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_double,
                                         ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_float),
                                         ctypes.POINTER(ctypes.c_uint16)]
            lib.metro_decode.restype = ctypes.c_long
        _libs[long] = lib
    return _libs[long]


def library_for(max_range):
    """The library whose grid covers this reach; without the long build the reach is cut to it."""
    if max_range > SHORT_REACH_M:
        lib = library(True)
        if lib is not None:
            return lib, max_range
    return library(), min(max_range, SHORT_REACH_M)


def available():
    return library() is not None


def decode_cloud(msg, offsets, types, min_distance, transform, want_ring):
    """XYZ (and ring) of a PointCloud2 payload parsed in C++; None when the library has no decoder.

    offsets, types: x, y, z, ring (offset -1 when absent). transform: 4x4 source -> analysis or None
    to use the default sensor convention. The layout was validated by the caller and is checked again.
    """
    lib = library()
    if lib is None or not hasattr(lib, 'metro_decode'):
        return None
    data = msg.data
    buf = data if isinstance(data, (bytes, bytearray)) else memoryview(data).cast('B')
    raw = np.frombuffer(buf, dtype=np.uint8)
    total = msg.width * msg.height
    xyz = np.empty((total, 3), np.float32)
    ring = np.empty(total, np.uint16) if want_ring else None
    off = (ctypes.c_int * 4)(*offsets)
    typ = (ctypes.c_int * 4)(*types)
    tr = None if transform is None else np.ascontiguousarray(transform, dtype=np.float64).reshape(16)
    n = lib.metro_decode(raw.ctypes.data_as(ctypes.c_void_p), raw.nbytes, msg.height, msg.width, msg.point_step,
                         msg.row_step, int(bool(msg.is_bigendian)), off, typ, float(min_distance),
                         None if tr is None else tr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                         xyz.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                         None if ring is None else ring.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
    if n < 0:
        raise ValueError('Truncated or invalid PointCloud2 layout')
    return xyz[:n], (ring[:n] if ring is not None else None)


class NativeTunnelDetector:
    """Drop-in replacement of metro.tunnel.TunnelDetector backed by the C++ core."""
    _gate = _approach = None  # optional rules after the core (enter_model, approach overrides)

    def __init__(self, config=None, overrides=None):
        """overrides: core keys beyond TunnelConfig (spatial_path, explain, soft_*...) or changed values."""
        self.cfg = config or TunnelConfig()
        d = self.cfg.as_dict()
        d.update(overrides or {})
        enter_model = d.pop('enter_model', None)  # path of an early-alarm model (metro/enter.py), applied after the core
        self._gate = None
        if enter_model:
            from .enter import EnterGate
            self._gate = EnterGate(enter_model, d.pop('enter_min_distance_m', 0.))
        d.pop('enter_min_distance_m', None)
        approach = d.pop('approach', None)  # True or the keyword arguments of metro.approach.ApproachGate
        self._approach = None
        if approach:
            from .approach import ApproachGate
            self._approach = ApproachGate(**(approach if isinstance(approach, dict) else {}))
        lib, d['max_range'] = library_for(d['max_range'])
        if lib is None:
            raise RuntimeError('libmetro_core is not built: cmake -S core -B core/build && cmake --build core/build')
        self._lib = lib
        self._h = lib.metro_create(json.dumps(d).encode())
        if not self._h:
            raise ValueError(lib.metro_create_error().decode())

    def reset(self):
        self._lib.metro_reset(self._h)
        if self._gate:
            self._gate.reset()
        if self._approach:
            self._approach.reset()

    def process(self, points, rings=None, timestamp=None, return_debug=False):
        p = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
        out = self._lib.metro_process(self._h, p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(p),
                                      float(timestamp) if timestamp is not None else 0., timestamp is not None)
        if not out:
            raise RuntimeError(self._lib.metro_error(self._h).decode())
        result = json.loads(out)
        if self._gate:
            result = self._gate.apply(result)
        if self._approach:
            result = self._approach.apply(result, p)
        return (result, None) if return_debug else result

    def __del__(self):
        h, self._h = getattr(self, '_h', None), None
        if h:
            self._lib.metro_destroy(h)
