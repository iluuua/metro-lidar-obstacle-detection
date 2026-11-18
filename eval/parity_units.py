"""Numerical helpers of the C++ core against numpy/scipy."""
import ctypes
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metro.native import library  # noqa: E402
from metro.tunnel import _penalized_fit  # noqa: E402

lib = library()
D = ctypes.POINTER(ctypes.c_double)
lib.metro_test_fit.argtypes = [ctypes.c_int, D, D, D, ctypes.c_size_t, ctypes.c_double, D]
lib.metro_test_uniform.argtypes = [D, ctypes.c_int, ctypes.c_int, D]
rng = np.random.default_rng(1)
# distance transform
occ = (rng.random((400, 160)) > .985)
out = np.zeros((400, 160), np.float32)
lib.metro_test_edt(occ.astype(np.uint8).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), 400, 160,
                   out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
ref = ndimage.distance_transform_edt(~occ)
print('edt max abs diff', float(np.abs(out - ref).max()))
# penalized fit
g = np.arange(0., 331.)
x = np.r_[np.arange(3., 40.), rng.uniform(20, 300, 60), g]
y = np.sin(x / 40.) + rng.normal(0, .05, len(x))
w = np.r_[np.full(37, 400.), rng.uniform(0, 60, 60), np.clip(1 - (g - 120) / 20, 0, 1)]
for lam in (4e4, 4e5, 4e6):
    f = np.zeros(331)
    lib.metro_test_fit(331, x.ctypes.data_as(D), y.ctypes.data_as(D), w.ctypes.data_as(D), len(x), lam, f.ctypes.data_as(D))
    print('fit lam', lam, 'max abs diff', float(np.abs(f - _penalized_fit(g, x, y, w, lam)).max()))
# uniform filter
v = rng.normal(0, 1, 461)
o = np.zeros(461)
lib.metro_test_uniform(v.ctypes.data_as(D), 461, 20, o.ctypes.data_as(D))
print('uniform max abs diff', float(np.abs(o - ndimage.uniform_filter1d(v, 20)).max()))
