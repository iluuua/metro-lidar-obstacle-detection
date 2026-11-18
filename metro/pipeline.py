"""PointCloud2 -> tunnel detector: the entry point shared by the ROS worker and tests."""
from dataclasses import replace
import time
import numpy as np
from .tunnel import TunnelConfig, TunnelDetector
from .io import AXES, FALCON, decode, forward_axis, lidar_of
from . import native

BACKENDS = ('auto', 'cpp', 'python')

# Analysis frame: x forward, y left, z up. The default sensor frame uses -y forward and +x left.
DEFAULT_TRANSFORM = AXES['-y']


def rigid(values):
    matrix = np.asarray(values, dtype=float).reshape(4, 4)
    rotation = matrix[:3, :3]
    if (not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)):
        raise ValueError('source_to_analysis must be a finite rigid 4x4 transform')
    return matrix


class PointCloudPipeline:
    """Decodes a PointCloud2 message and runs the detector on it.

    The same object serves the ROS worker process and the tests, so both see
    exactly the same decoding and the same detector state.

    source_to_analysis is a 4x4 matrix, a forward-axis name from io.AXES, or 'auto':
    then the forward axis is found from the first clouds (z is taken as up).

    backend: 'cpp' is the C++ core (libmetro_core), 'python' the reference implementation,
    'auto' the core when it is built.
    """

    def __init__(self, config=None, source_to_analysis=None, analysis_frame='metro_lidar_analysis', backend='auto'):
        self.config = config or TunnelConfig()
        if backend not in BACKENDS:
            raise ValueError('backend must be one of ' + ', '.join(BACKENDS))
        if backend == 'cpp' and not native.available():
            raise ValueError('backend cpp: libmetro_core is not built')
        self.backend = 'cpp' if backend == 'cpp' or (backend == 'auto' and native.available()) else 'python'
        self.axis = None
        self.auto_axes = isinstance(source_to_analysis, str) and source_to_analysis == 'auto'
        if self.auto_axes:
            self.transform = None
        elif isinstance(source_to_analysis, str):
            if source_to_analysis not in AXES:
                raise ValueError('sensor axes must be auto or one of ' + ', '.join(AXES))
            self.axis = source_to_analysis
            self.transform = rigid(AXES[source_to_analysis])
        else:
            self.transform = rigid(DEFAULT_TRANSFORM if source_to_analysis is None else source_to_analysis)
        if not analysis_frame:
            raise ValueError('analysis_frame must be non-empty')
        self.analysis_frame = analysis_frame
        self.reset()

    def reset(self):
        if self.auto_axes:  # a reset may mean another sensor: find its axes again
            self.transform, self.axis = None, None
        self.lidar = None
        self._build(self.config)

    def _build(self, config):
        if self.backend == 'cpp':
            self.detector = native.NativeTunnelDetector(replace(config))
        else:
            self.detector = TunnelDetector(replace(config))

    def _identify(self, msg):
        """Select the longer default range for a Falcon cloud and preserve explicit range overrides."""
        self.lidar = lidar_of(msg)
        if self.lidar == FALCON and self.config.max_range == TunnelConfig.max_range:
            self._build(replace(self.config, max_range=self.config.long_range))

    def _resolve_axes(self, msg):
        """Infer the forward axis from the cloud, falling back to DEFAULT_TRANSFORM."""
        raw, _ = decode(msg, raw=True)
        axis, _ = forward_axis(raw)
        if axis is None:
            return rigid(DEFAULT_TRANSFORM)
        self.axis = axis
        self.transform = rigid(AXES[axis])
        return self.transform

    def process(self, msg):
        tic = time.perf_counter()
        try:
            if self.lidar is None:
                self._identify(msg)
            transform = self.transform if self.transform is not None else self._resolve_axes(msg)
            points, rings = decode(msg, source_to_analysis=None if np.array_equal(transform.ravel(), DEFAULT_TRANSFORM)
                                   else transform)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            result = self.detector.process(points, rings, stamp)
        except Exception:
            # a broken cloud must not leave half-updated history behind
            self.reset()
            raise
        result.update(frame_id=self.analysis_frame, source_frame_id=msg.header.frame_id,
                      source_to_analysis=transform.ravel().tolist(), sensor_forward_axis=self.axis, lidar=self.lidar,
                      backend=self.backend,
                      stamp_s=stamp, stamp_sec=int(msg.header.stamp.sec),
                      stamp_nanosec=int(msg.header.stamp.nanosec),
                      valid=result['status'] != 'unknown',
                      decode_and_detect_ms=(time.perf_counter() - tic) * 1000)
        return result
