"""Tunnel-profile detector.

Idea: a metro tunnel is an extruded cross-section around the track.  Near the
train the rail pair fixes the track frame (centre, heading, rail-head plane
and cant).  The near cross-section becomes a template of the *normal* tunnel.
Far ahead, where rails are invisible, every slab of points is registered to
that template; a smooth centre line through the registered slabs gives the
clearance envelope measured from the rail heads.
Points inside the envelope that the normal tunnel cannot explain are obstacle
evidence.  Evidence is accumulated in a world coordinate along the track, so
a static object is confirmed by repeated observations at the same place while
sensor noise is not.  Path uncertainty grows with distance and is subtracted
from the penetration depth, so objects near the boundary are only reported
once the geometry is accurate enough to decide.

Only the current and previous frames are used; no labels, no bag names.
"""
from dataclasses import dataclass, field, asdict, replace
import time
import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.linalg import solveh_banded


@dataclass
class TunnelConfig:
    # Envelope relative to the rail heads, normal to the centre line.
    half_width: float = 1.05
    height: float = 3.0
    min_range: float = 1.0          # the lidar sees a person from ~1 m; closer is its blind zone
    # Default range limits; the pipeline uses long_range for Falcon clouds.
    max_range: float = 300.0
    long_range: float = 400.0
    # Initial track and sensor geometry before a cloud is available.
    gauge: float = 1.59            # rail-head centre distance for 1520 mm track
    rail_z_prior: float = -1.12
    rail_over_bed: float = .25
    # Near rail search.
    rail_max_x: float = 45.0
    # Far registration.
    template_decay: float = .75
    template_near: tuple = (4.0, 32.0)
    slab_edges: tuple = tuple(float(v) for v in np.r_[np.arange(20., 100., 5.), np.arange(100., 310., 10.)])
    slab_points: int = 260
    chunk_edges: tuple = (0., 50., 80., 120., 180., 310.)
    min_slab_inliers: float = 6.
    max_ambiguity: float = .98
    min_explained: float = .7
    break_explained: float = .6
    # Uncertainty (1 sigma) used for margin-aware decisions.
    sigma_lat0: float = .02
    sigma_lat_per_m: float = .003
    sigma_lat_per_m_far: float = .0015
    sigma_vert0: float = .03
    sigma_vert_per_m: float = .002
    sigma_extrap_per_m: float = .012
    margin_k: float = 1.25
    max_sigma_support: float = .35
    support_buffer: float = 5.
    # Straight extension beyond the supported geometry into empty space.
    extension: bool = True
    extension_min_support: float = 80.
    extension_max_curvature: float = 1 / 3000.
    extension_sigma_per_m: float = .004
    extension_corridor: float = 2.0
    min_height: float = .07        # above rail heads; rails themselves are not obstacles
    rail_exclusion: float = .07
    # Clusters and confirmation.
    voxel_s: float = .5
    voxel_uh: float = .25
    strong_points: int = 6
    strong_hits: int = 2
    weak_hits: int = 4
    window: int = 8
    hold_frames: int = 3            # a confirmed obstacle survives short observation gaps
    max_hold_s: float = 15.         # ... and is kept while its place is beyond the verified range
    hold_min_range: float = 10.     # closer objects leave the lidar field of view: no hold
    hold_min_travel: float = 10.    # long hold only after confirmations from positions this far apart
    # Objects that appear while the train stands still (change detection in the sensor frame).
    still_ds: float = .01           # m per frame: the train is standing
    change_voxel: float = .2
    change_ref_s: float = 2.        # a voxel empty during the first 2 s of the stop may count as new
    change_min_age_s: float = .4    # ... once it has been occupied this long (5 frames at 10 Hz)
    change_occupancy: float = .6    # ... in at least this share of frames since it appeared
    change_persist: int = 8         # the cluster is seen in >= 8 of the last change_window frames
    change_window: int = 12
    change_new_frac: float = .5
    # Moving objects (a person walking on the track, a train ahead): confirmed when the
    # track moves along the path at a steady speed and gives strong evidence.
    move_min_obs: int = 4           # seen in at least 4 of the last `window` frames
    move_strong: int = 3            # ... with >= strong_points confident points in 3 of them
    move_evidence_k: float = 2.     # ... and twice the evidence a static object needs
    move_resid: float = .3          # m; S(t) follows a straight line within this (+ .004 m per m of range)
    move_min_speed: float = .5      # m/s along the path, in the world
    move_max_speed: float = 30.
    move_rel_min: float = 1.        # m/s; slower relative to the train is how clutter moving with the view looks
    move_hold_s: float = 2.         # a moving obstacle is carried on by its speed this long when it is not seen
    gate_s: float = 1.6
    static_tol: float = 2.0         # large objects give fragments with a wide scatter of S; calibrated on the long ride
    wall_extent_s: float = 8.
    wall_extent_uh: float = 1.4
    wall_clear_center: float = .4
    wall_end_margin: float = 10.
    gate_h: float = .6
    evidence_near: float = 8.
    evidence_per_m: float = .16
    frame_period_s: float = .1
    max_coast_frames: int = 5
    max_gap_s: float = 3.           # longer pauses reset history; shorter ones come from frames skipped on a busy CPU
    coast_sigma_h: float = .06
    coast_sigma_u: float = .03
    temporal_weight: float = 1.

    def as_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = [float(x) for x in v]
        return d


# --------------------------------------------------------------------------
# small helpers
def _shift(a, k, fill):
    out = np.full_like(a, fill)
    if k > 0:
        out[:, k:] = a[:, :-k]
    elif k < 0:
        out[:, :k] = a[:, -k:]
    else:
        out[:] = a
    return out


def _robust_lstsq(A, b, w=None, floor=.02, iters=3):
    w = np.ones(len(b)) if w is None else np.asarray(w, float)
    coef = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)[0]
    for _ in range(iters):
        r = b - A @ coef
        s = max(floor, 1.4826 * np.median(np.abs(r)))
        ww = w * np.minimum(1., 2. * s / np.maximum(np.abs(r), 1e-9))
        coef = np.linalg.lstsq(A * ww[:, None], b * ww, rcond=None)[0]
    return coef


class PathModel:
    """Centre line y=c(x), rail-head height z=r(x)+phi*u, sampled on a grid."""
    GRID = np.arange(0., 331., 1.)

    def __init__(self, lat, vert, phi, support=0., sigma_u=None, sigma_h=None, info=None):
        self.lat = np.asarray(lat, float)        # c on GRID
        self.vert = np.asarray(vert, float)      # r on GRID
        self.phi = float(phi)
        self.support = float(support)
        g = self.GRID
        self.dlat = np.gradient(self.lat, g)
        self.arc = np.r_[0., np.cumsum(np.sqrt(1 + ((self.dlat[1:] + self.dlat[:-1]) / 2) ** 2))]
        self.sigma_u = sigma_u if sigma_u is not None else np.full(len(g), .05)
        self.sigma_h = sigma_h if sigma_h is not None else np.full(len(g), .05)
        self.info = info or {}

    def _idx(self, x):
        g = self.GRID
        xf = np.clip(np.asarray(x, dtype=np.float64), g[0], g[-1] - 1e-9)
        i = xf.astype(np.int64)          # GRID starts at 0 with 1 m step
        return i, xf - i

    def _at(self, arr, i, f):
        return arr[i] * (1 - f) + arr[i + 1] * f

    def c(self, x):
        i, f = self._idx(x)
        return self._at(self.lat, i, f)

    def dc(self, x):
        i, f = self._idx(x)
        return self._at(self.dlat, i, f)

    def r(self, x):
        i, f = self._idx(x)
        return self._at(self.vert, i, f)

    def to_track(self, p):
        i, f = self._idx(p[:, 0])
        c = self._at(self.lat, i, f)
        dc = self._at(self.dlat, i, f)
        cos = 1. / np.sqrt(1. + dc * dc)
        dy = p[:, 1] - c
        u = dy * cos
        h = p[:, 2] - self._at(self.vert, i, f) - self.phi * u
        s = self._at(self.arc, i, f) + dy * dc * cos
        return s, u, h

    def shifted(self, ds):
        """Same world track seen after moving ds forward along it.

        The sensor keeps its mounting offset and yaw/pitch relative to the
        track, so offset and heading at x=0 are preserved and only the shape
        (curvature) ahead is transported.
        """
        g = self.GRID
        if not ds:
            return PathModel(self.lat, self.vert, self.phi, self.support, self.sigma_u, self.sigma_h, dict(self.info))
        x_old = np.minimum(g + ds, g[-1])
        lat = np.interp(x_old, g, self.lat)
        vert = np.interp(x_old, g, self.vert)
        dvert = np.gradient(self.vert, g)
        lat = lat - (np.interp(ds, g, self.lat) - self.lat[0]) - (np.interp(ds, g, self.dlat) - self.dlat[0]) * g
        vert = vert - (np.interp(ds, g, self.vert) - self.vert[0]) - (np.interp(ds, g, dvert) - dvert[0]) * g
        return PathModel(lat, vert, self.phi, max(0., self.support - ds), self.sigma_u, self.sigma_h,
                         dict(self.info))

    def samples(self, step=1.):
        g = self.GRID
        return np.column_stack([g, self.lat, self.vert, self.arc])

    def section_frames(self, max_s):
        """[s, centre xyz, lateral xyz, up xyz] every 2 m for the viewer."""
        k = np.arange(0, len(self.GRID), 2)
        k = k[self.arc[k] <= max_s + 2]
        t = np.column_stack([np.ones(len(k)), self.dlat[k], np.zeros(len(k))])
        t /= np.linalg.norm(t, axis=1)[:, None]
        lat = np.column_stack([-t[:, 1], t[:, 0], np.full(len(k), self.phi)])
        lat /= np.linalg.norm(lat, axis=1)[:, None]
        up = np.cross(t, lat)
        centre = np.column_stack([self.GRID[k], self.lat[k], self.vert[k]])
        return np.column_stack([self.arc[k], centre, lat, up]).round(4).tolist()


# --------------------------------------------------------------------------
class NearTrack:
    """Rail-pair ridges in slabs, de-skewed by the previous centre line."""
    DY = .02
    Y0 = -1.9
    NY = 190
    EDGES = np.r_[np.arange(3., 20., .75), np.arange(20., 34., 1.5), np.arange(34., 46.1, 2.5)]

    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, p, prior):
        cfg = self.cfg
        E = self.EDGES
        x = p[:, 0]
        sel = (x >= E[0]) & (x < E[-1]) & (np.abs(p[:, 1]) < 4.)
        q = _rows(p, sel)
        x = q[:, 0]
        yc = prior.c(x)
        zr = prior.r(x)
        y = q[:, 1] - yc
        z = q[:, 2] - zr
        m = (np.abs(y) < 1.85) & (z > -.45) & (z < .35)
        x, y, z = x[m], y[m], z[m]
        ns = len(E) - 1
        ix = np.searchsorted(E, x, side='right') - 1
        iy = ((y - self.Y0) / self.DY).astype(np.int32)
        ok = (ix >= 0) & (ix < ns) & (iy >= 0) & (iy < self.NY)
        M = _group_max(ix[ok] * self.NY + iy[ok], z[ok], ns * self.NY).reshape(ns, self.NY)
        with np.errstate(invalid='ignore'):
            nb = np.maximum(np.maximum.reduce([_shift(M, k, -np.inf) for k in range(5, 11)]),
                            np.maximum.reduce([_shift(M, -k, -np.inf) for k in range(5, 11)]))
            R = M - nb
        R = np.where(np.isfinite(R), R, 0.)
        R = np.maximum.reduce([R, _shift(R, 1, 0.), _shift(R, -1, 0.), _shift(R, 2, 0.), _shift(R, -2, 0.)])
        half = int(round(cfg.gauge / 2 / self.DY))
        L = _shift(R, half, 0.)
        Rr = _shift(R, -half, 0.)
        Hl = _shift(M, half, -np.inf)
        Hr = _shift(M, -half, -np.inf)
        with np.errstate(invalid='ignore'):
            dz = np.abs(Hl - Hr)
        P = np.minimum(L, Rr)
        P = np.where((dz < .2) & (P > .02), np.minimum(P, .25), 0.)
        # Viterbi over slabs; deviation from the prior is nearly constant.
        centres = self.Y0 + self.DY * (np.arange(self.NY) + .5)
        score = np.log1p(40 * P)
        D = score[0] - .5 * (centres / .35) ** 2
        back = np.zeros((ns, self.NY), np.int32)
        idx = np.arange(self.NY)
        for j in range(1, ns):
            best = D.copy()
            arg = idx.copy()
            for d in (-2, -1, 1, 2):
                cand = np.full(self.NY, -1e9)
                if d > 0:
                    cand[d:] = D[:-d] - .15 * d * d
                else:
                    cand[:d] = D[-d:] - .15 * d * d
                better = cand > best
                best = np.where(better, cand, best)
                arg = np.where(better, idx - d, arg)
            D = best + score[j] - .03
            back[j] = arg
        path = np.zeros(ns, np.int32)
        path[-1] = int(np.argmax(D))
        for j in range(ns - 1, 0, -1):
            path[j - 1] = back[j, path[j]]
        rows = np.arange(ns)
        sup = P[rows, path]
        xs = (E[:-1] + E[1:]) / 2
        good = sup > .03
        if good.sum() < 5 or xs[good].min() > 15:
            return None
        dev = centres[path]
        A = np.column_stack([np.ones(good.sum()), xs[good], xs[good] ** 2])
        coef = _robust_lstsq(A, dev[good], sup[good])
        coef[2] = np.clip(coef[2], -1 / 500., 1 / 500.)
        resid = dev[good] - A @ coef
        # rail-head plane (relative to prior r): z = a + b x + phi u
        hl = Hl[rows, path][good]
        hr = Hr[rows, path][good]
        xg = xs[good]
        zz = np.r_[hl, hr]
        uu = np.r_[np.full(len(xg), -cfg.gauge / 2), np.full(len(xg), cfg.gauge / 2)]
        xx = np.r_[xg, xg]
        fin = np.isfinite(zz)
        B = np.column_stack([np.ones(fin.sum()), xx[fin], uu[fin]])
        rc = _robust_lstsq(B, zz[fin], floor=.008)
        return dict(dev=coef, rail=rc, support=float(xs[good].max()), n=int(good.sum()),
                    resid=float(np.median(np.abs(resid))) if len(resid) else 0., xs=xs[good], c=dev[good])


# --------------------------------------------------------------------------
class Template:
    """Occupancy of the normal cross-section in (u,h), exponentially updated."""
    U0, U1, H0, H1, D = -10., 10., -1.5, 6.5, .05

    def __init__(self, cfg):
        self.cfg = cfg
        self.nu = int(round((self.U1 - self.U0) / self.D))
        self.nh = int(round((self.H1 - self.H0) / self.D))
        self.count = np.zeros((self.nu, self.nh), np.float32)
        self.dist = None

    def reset(self):
        self.count[:] = 0
        self.dist = None

    def update(self, u, h, envelope_clear=True):
        cfg = self.cfg
        if envelope_clear:
            inside = (np.abs(u) < cfg.half_width + .05) & (h > .04) & (h < cfg.height + .1)
            u, h = u[~inside], h[~inside]
        iu = ((u - self.U0) / self.D).astype(np.int32)
        ih = ((h - self.H0) / self.D).astype(np.int32)
        ok = (iu >= 0) & (iu < self.nu) & (ih >= 0) & (ih < self.nh)
        now = np.zeros_like(self.count)
        now[iu[ok], ih[ok]] = 1.
        self.count *= self.cfg.template_decay
        self.count += now
        occ = self.count > .5
        if occ.sum() < 50:
            self.dist = None
            return
        d = ndimage.distance_transform_edt(~occ)
        self.dist = (d * self.D).astype(np.float32)

    def lookup(self, u, h):
        iu = np.clip(((u - self.U0) / self.D).astype(np.int32), 0, self.nu - 1)
        ih = np.clip(((h - self.H0) / self.D).astype(np.int32), 0, self.nh - 1)
        return self.dist[iu, ih]


# --------------------------------------------------------------------------
class WallOdometry:
    """Forward displacement from wall depth profiles of two consecutive frames.

    Both frames are expressed with the same (latest) centre line, so the
    comparison is free of path-estimate changes.  Cells are tall (0.5 m) and
    short along the track (0.1 m); a 2 m high-pass removes the smooth lining,
    leaving brackets, joints and niches that move with the world.
    """
    X0, X1, DX = 6., 52., .1
    H0, H1, DH = -.3, 3.9, .5
    MAX_DT = 3.       # s; a busy computer skips frames (the detector resets after longer pauses)
    MAX_SPEED = 25.   # m/s; bounds the first search, before the speed is known

    def __init__(self):
        self.prev = None
        self.v = None

    def _select(self, p, path):
        x = p[:, 0]
        m = (x > self.X0) & (x < self.X1)
        q = _rows(p, m)
        u = q[:, 1] - path.c(q[:, 0])
        h = q[:, 2] - path.r(q[:, 0])
        k = (np.abs(u) > 1.3) & (np.abs(u) < 4.5) & (h > self.H0) & (h < self.H1)
        return _rows(q, k)

    def _image(self, q, path):
        s = q[:, 0]
        u = q[:, 1] - path.c(s)
        h = q[:, 2] - path.r(s)
        nx = int(np.ceil((self.X1 - self.X0) / self.DX)) + 1
        nh = int(np.ceil((self.H1 - self.H0) / self.DH)) + 1
        side = (u > 0).astype(np.int64)
        ix = ((s - self.X0) / self.DX).astype(np.int64)
        ih = ((h - self.H0) / self.DH).astype(np.int64)
        ok = (ix >= 0) & (ix < nx) & (ih >= 0) & (ih < nh)
        key = ((side * nx + ix) * nh + ih)[ok]
        size = 2 * nx * nh
        tot = np.bincount(key, np.abs(u[ok]), size)
        cnt = np.bincount(key, None, size)
        img = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan).reshape(2, nx, nh)
        w = int(2. / self.DX)
        v = np.nan_to_num(img)
        n = np.isfinite(img).astype(float)
        ms = ndimage.uniform_filter1d(v, w, axis=1)
        mn = ndimage.uniform_filter1d(n, w, axis=1)
        return np.where(np.isfinite(img), img - ms / np.maximum(mn, 1e-6), np.nan)

    def prime(self, p, path):
        """Wall image of a frame processed without odometry (the first after a reset),
        so that the next frame already gets a displacement."""
        q = self._select(p, path)
        self.prev = self._image(q, path) if len(q) >= 200 else None

    def update(self, p, path, dt):
        q = self._select(p, path)
        b = self._image(q, path) if len(q) >= 200 else None
        a, self.prev = self.prev, b
        if a is None or b is None or dt is None or dt <= 0 or dt > self.MAX_DT:
            return None, 0.
        nx = b.shape[1]
        if self.v is None:
            cands = np.arange(0., max(3.2, min(20., self.MAX_SPEED * dt)) + .01, .05)
        else:
            pred = self.v * dt
            cands = np.arange(max(0., pred - .6), pred + .61, .05)
        costs = np.full(len(cands), np.inf)
        for i, ds in enumerate(cands):
            k = ds / self.DX
            k0 = int(np.floor(k))
            f = k - k0
            if k0 + 1 >= nx:
                continue
            sh = (1 - f) * a[:, k0:nx - 1, :] + f * a[:, k0 + 1:nx, :]
            c = b[:, :nx - 1 - k0, :]
            ok = np.isfinite(sh) & np.isfinite(c)
            if ok.sum() < 60:
                continue
            costs[i] = float(np.mean(np.minimum(np.abs(sh[ok] - c[ok]), .1)))
        if not np.isfinite(costs).any():
            return None, 0.
        i = int(np.argmin(costs))
        ds = float(cands[i])
        if 0 < i < len(costs) - 1 and np.isfinite(costs[i - 1]) and np.isfinite(costs[i + 1]):
            den = costs[i - 1] - 2 * costs[i] + costs[i + 1]
            if den > 1e-9:
                ds += float(np.clip(.5 * (costs[i - 1] - costs[i + 1]) / den, -.5, .5)) * .05
        finite = costs[np.isfinite(costs)]
        quality = float((np.median(finite) - costs[i]) / max(np.median(finite), 1e-6))
        v = max(0., ds) / dt
        self.v = v if self.v is None else .5 * self.v + .5 * v
        return max(0., ds), quality


# --------------------------------------------------------------------------
class Track:
    __slots__ = ('S', 'u', 'h', 'first', 'last', 'hits', 'strong', 'frames', 'best', 'id', 'Ss', 'conf',
                 'confirmed_at', 'confirmed_t', 'offset', 'last_c', 'S_ego_c', 'obs_ego', 'appeared', 'ts', 'moving',
                 'speed', 'n_moving')

    def __init__(self, tid, S, u, h, frame, strong, obs):
        self.id = tid
        self.S, self.u, self.h = S, u, h
        self.first = self.last = frame
        self.hits = 1
        self.strong = int(strong)
        self.frames = [frame]
        self.best = obs
        self.Ss = [S]
        self.conf = [(frame, obs.get('evidence_obs', obs['confident_points']))]
        self.confirmed_at = None      # last frame the track was confirmed
        self.confirmed_t = None       # ... and its time, s
        self.offset = 0.              # nearest point minus median along the track
        self.last_c = None
        self.S_ego_c = 0.
        self.obs_ego = []             # train positions of observations with confident points
        self.appeared = False         # confirmed as an object that appeared while the train stood
        self.ts = []                  # times of the observations in Ss
        self.moving = False           # confirmed as a moving object
        self.speed = 0.               # its speed along the path, m/s
        self.n_moving = 0             # frames confirmed as moving


class TunnelDetector:
    def __init__(self, config=None):
        self.cfg = config or TunnelConfig()
        # the reference grid ends at 330 m; the C++ core reaches 430 m
        if self.cfg.max_range > PathModel.GRID[-1]:
            self.cfg = replace(self.cfg, max_range=float(PathModel.GRID[-1]))
        self.near = NearTrack(self.cfg)
        self.template = Template(self.cfg)
        self.odometry = WallOdometry()
        self.reset()

    def reset(self):
        self.surface_ref = None
        self.miss = 0
        self.path = None
        self.template.reset()
        self.odometry.prev = None
        self.odometry.v = None
        self.tracks = []
        self.next_id = 0
        self.frame = -1
        self.now = 0.
        self.S_ego = 0.
        self.last_time = None
        self._reset_change()

    def _reset_change(self):
        self.still_f0 = None
        self.still_t0 = None
        self.vox_keys = np.zeros(0, np.int64)
        self.vox_first = np.zeros(0, np.int32)
        self.vox_first_t = np.zeros(0)
        self.vox_count = np.zeros(0, np.int32)

    # ------------------------------------------------------------------
    def _prior(self, ds, p=None):
        cfg = self.cfg
        if self.path is None:
            g = PathModel.GRID
            z0 = self._rail_height_guess(p) if p is not None else cfg.rail_z_prior
            return PathModel(np.zeros(len(g)), np.full(len(g), z0), 0., 0.)
        return self.path.shifted(ds or 0.)

    def _rail_height_guess(self, p):
        """Rail-head height from the dominant track-bed level near the train.

        Mounting height differs between vehicles/recordings; the bed is the
        largest horizontal surface 4-15 m ahead within the track width.
        """
        cfg = self.cfg
        m = (p[:, 0] > 4) & (p[:, 0] < 15) & (np.abs(p[:, 1]) < 1.5) & (p[:, 2] < -.3) & (p[:, 2] > -4.)
        z = p[m, 2]
        if len(z) < 200:
            return cfg.rail_z_prior
        hist, edges = np.histogram(z, bins=np.arange(-4., -.29, .05))
        k = int(np.argmax(hist))
        bed = (edges[k] + edges[k + 1]) / 2
        return float(bed + cfg.rail_over_bed)

    def _slab_points(self, p):
        """Deterministic per-slab subsample of far points, computed once per frame."""
        cfg = self.cfg
        E = np.asarray(cfg.slab_edges)
        x = p[:, 0]
        idx = np.flatnonzero((x >= E[0]) & (x < E[-1]))
        slab = np.searchsorted(E, x[idx], side='right') - 1
        counts = np.bincount(slab, minlength=len(E) - 1)
        keep_frac = np.minimum(1., 2. * cfg.slab_points / np.maximum(counts, 1))
        h = (idx.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(1000)
        sel = h.astype(np.float64) < keep_frac[slab] * 1000.
        return np.take(p, idx[sel], axis=0), slab[sel]

    def _register_slabs(self, p, base, width_scale=1., xlo=0., xhi=1e9, pre=None):
        """Per-slab lateral template registration and floor/ceiling vertical estimate."""
        cfg = self.cfg
        E = np.asarray(cfg.slab_edges)
        if self.template.dist is None:
            return None
        q, slab = pre if pre is not None else self._slab_points(p)
        xc_all = (E[:-1] + E[1:]) / 2
        in_chunk = (xc_all[slab] >= xlo) & (xc_all[slab] < xhi)
        q, slab = q[in_chunk], slab[in_chunk]
        if len(q) == 0:
            return None
        s, u, h = base.to_track(q)
        slab_all, u_all, h_all = slab, u, h
        # ignore points in the predicted envelope for the lateral fit: potential obstacles
        keep = ~((np.abs(u) < cfg.half_width + .35) & (h > .05) & (h < cfg.height + .3))
        keep &= (np.abs(u) < 9.8) & (h > -1.45) & (h < 6.45)
        slab, u, h = slab[keep], u[keep], h[keep]
        if len(slab) == 0:
            return None
        nsl = len(E) - 1
        xc = (E[:-1] + E[1:]) / 2
        W = np.clip((.5 + .012 * xc) * width_scale, .4, 3.5)
        sig = .09

        def best_shift(du_grid, u_, h_, slab_):
            # du_grid: (K,) shared offsets scaled per slab by W
            K = len(du_grid)
            du = du_grid[None, :] * W[slab_][:, None]
            d = self.template.lookup((u_[:, None] - du).ravel(), np.repeat(h_, K))
            sc = np.exp(-.5 * (d / sig) ** 2)
            key = (slab_[:, None] * K + np.arange(K)[None, :]).ravel()
            return np.bincount(key, sc, nsl * K).reshape(nsl, K)

        grid = np.linspace(-1, 1, 41)
        tot = best_shift(grid, u, h, slab)
        k = np.argmax(tot, axis=1)
        du = grid[k] * W
        peak = tot[np.arange(nsl), k]
        # vertical search with lateral applied
        uu = u - du[slab]
        Wh = np.clip(.15 + .004 * xc, .15, 1.2)
        hgrid = np.linspace(-1, 1, 21)
        K = len(hgrid)
        dh_all = hgrid[None, :] * Wh[slab][:, None]
        d = self.template.lookup(np.repeat(uu, K), (h[:, None] - dh_all).ravel())
        sc = np.exp(-.5 * (d / sig) ** 2)
        toth = np.bincount((slab[:, None] * K + np.arange(K)[None, :]).ravel(), sc, nsl * K).reshape(nsl, K)
        kh = np.argmax(toth, axis=1)
        dh = hgrid[kh] * Wh
        # refine lateral with fine grid around du
        hh = h - dh[slab]
        fine = np.linspace(-.12, .12, 13)
        K2 = len(fine)
        dfine = du[slab][:, None] + fine[None, :]
        d = self.template.lookup((u[:, None] - dfine).ravel(), np.repeat(hh, K2))
        sc = np.exp(-.5 * (d / sig) ** 2)
        tf = np.bincount((slab[:, None] * K2 + np.arange(K2)[None, :]).ravel(), sc, nsl * K2).reshape(nsl, K2)
        kf = np.argmax(tf, axis=1)
        du = du + fine[kf]
        peak = tf[np.arange(nsl), kf]
        # ambiguity: best score outside +-0.4 m of the peak (lateral)
        alt = np.where(np.abs(grid[None, :] * W[:, None] - du[:, None]) > .25, tot, 0.).max(axis=1)
        n = np.bincount(slab, minlength=nsl).astype(float)
        # vertical: floor (lower band) and ceiling (crown) against near-field references
        dhv, wv = self._vertical_from_surfaces(slab_all, u_all - du[slab_all], h_all, nsl)
        use_t = wv <= 0
        dh = np.where(use_t, dh, dhv)
        wv = np.where(use_t, .3, wv)
        cos = 1. / np.sqrt(1. + base.dc(xc) ** 2)
        sel_slabs = (xc >= xlo) & (xc < xhi)
        return dict(x=xc[sel_slabs], y=(base.c(xc) + du / cos)[sel_slabs], z=(base.r(xc) + dh)[sel_slabs],
                    du=du[sel_slabs], dh=dh[sel_slabs], inl=peak[sel_slabs], alt=alt[sel_slabs], n=n[sel_slabs],
                    wv=wv[sel_slabs])

    def _vertical_from_surfaces(self, slab, u, h, nsl):
        """Per-slab vertical offset from the track-bed band and the crown.

        A real grade change moves both surfaces; a low object only the floor
        statistic, a hanging one only the crown. Disagreement lowers weight.
        """
        ref = self.surface_ref
        dh = np.zeros(nsl)
        w = np.zeros(nsl)
        if ref is None:
            return dh, w
        fl = (np.abs(u) < 1.3) & (h > -1.3) & (h < 1.6)
        if ref['ceil'] is None:     # no crown in view (open section, high hall)
            ce = np.zeros(len(h), bool)
        else:
            ce = (np.abs(u) < 1.0) & (h > ref['ceil'] - 1.4) & (h < ref['ceil'] + 1.6)
        for j in np.unique(slab):
            a = h[fl & (slab == j)]
            b = h[ce & (slab == j)]
            ests = []
            if len(a) >= 6:
                ests.append((np.quantile(a, .3) - ref['floor'], min(len(a), 40) / 40.))
            if len(b) >= 4 and ref['ceil'] is not None:
                ests.append((np.quantile(b, .9) - ref['ceil'], min(len(b), 30) / 30.))
            if not ests:
                continue
            if len(ests) == 2:
                d0, d1 = ests[0][0], ests[1][0]
                if abs(d0 - d1) < .3:
                    dh[j] = (d0 * ests[0][1] + d1 * ests[1][1]) / (ests[0][1] + ests[1][1])
                    w[j] = 1.
                else:
                    dh[j] = d0 if abs(d0) < abs(d1) else d1
                    w[j] = .2
            else:
                dh[j] = ests[0][0]
                w[j] = .5 * ests[0][1]
        return dh, w

    def _smooth_path(self, prior, near, slabs, first, quick=False):
        """Smooth centre line/rail height through rails, slabs and the moved previous path."""
        cfg = self.cfg
        g = PathModel.GRID
        phi = prior.phi
        ox, oy, ow = [], [], []
        vx, vy, vw = [], [], []
        support = 0.
        rail_support = 0.
        if near is not None:
            xs = np.arange(3., min(near['support'], 40.) + .1, 1.)
            d0, d1, d2 = near['dev']
            ox.append(xs); oy.append(prior.c(xs) + d0 + d1 * xs + d2 * xs ** 2); ow.append(np.full(len(xs), 400.))
            a, b, ph = near['rail']
            vx.append(xs); vy.append(prior.r(xs) + a + b * xs); vw.append(np.full(len(xs), 400.))
            phi = ph
            support = rail_support = near['support']
        good = np.zeros(0, bool)
        if slabs is not None:
            q = slabs['inl']
            ratio = slabs['alt'] / np.maximum(q, 1e-6)
            # a slab counts only if the normal-tunnel template explains most of
            # its points; a changed cross-section (station, widening, portal)
            # is not trusted to position the track
            frac = q / np.maximum(slabs['n'], 1.)
            good = (q >= cfg.min_slab_inliers) & (ratio < cfg.max_ambiguity) & (frac >= cfg.min_explained)
            w = np.where(good, np.minimum(q, 60.) * np.maximum(1 - ratio, .1) * frac, 0.)
            # a slab the template clearly cannot explain marks a changed tunnel:
            # nothing beyond it supports the track position
            broken = (slabs['n'] >= 20) & (frac < cfg.break_explained)
            if broken.any():
                cut = float(slabs['x'][broken].min())
                good &= slabs['x'] < cut
                w = np.where(slabs['x'] < cut, w, 0.)
            ox.append(slabs['x']); oy.append(slabs['y']); ow.append(w)
            vx.append(slabs['x']); vy.append(slabs['z']); vw.append(np.where(good | (slabs['wv'] > .4), 20. * slabs['wv'], 0.))
            last = support
            for xv in slabs['x'][good]:
                if xv - last > 25.:
                    break
                last = xv
            support = max(support, last)
        cat = lambda a: np.concatenate(a) if a else np.zeros(0)
        ox, oy, ow = cat(ox), cat(oy), cat(ow)
        vx, vy, vw = cat(vx), cat(vy), cat(vw)
        # robust weights from a data-only fit: a wrong previous path must not
        # turn consistent new observations into "outliers"
        if not quick:
            ow = _robust_weights(g, ox, oy, ow, 4e4, .25)
            vw = _robust_weights(g, vx, vy, vw, 4e6, .2)
        # temporal prior: the previous path moved by odometry, trusted only inside
        # the range it was itself supported by data; beyond that it fades out so
        # an unsupported bend cannot propagate towards the train.
        if first:
            tw = np.zeros(len(g))
        else:
            tw = cfg.temporal_weight * np.clip(1. - (g - prior.support) / 20., 0., 1.)
        ox = np.r_[ox, g]; oy = np.r_[oy, prior.lat]; ow = np.r_[ow, tw]
        vx = np.r_[vx, g]; vy = np.r_[vy, prior.vert]; vw = np.r_[vw, tw]
        lat = _penalized_fit(g, ox, oy, ow, 4e4)
        curv = np.gradient(np.gradient(lat, g), g)
        if np.max(np.abs(curv[:int(support) + 1])) > 1 / 250.:
            lat = _penalized_fit(g, ox, oy, ow, 4e5)
        vert = _penalized_fit(g, vx, vy, vw, 4e6)
        su = cfg.sigma_lat0 + cfg.sigma_lat_per_m * np.minimum(g, 100.) + cfg.sigma_lat_per_m_far * np.maximum(0., g - 100.)
        sh = cfg.sigma_vert0 + cfg.sigma_vert_per_m * g
        if slabs is not None and good.any():
            # empirical consistency of the registered slabs around the smooth path
            xs = slabs['x'][good]
            ru = slabs['y'][good] - np.interp(xs, g, lat)
            rz = slabs['z'][good] - np.interp(xs, g, vert)
            wq = np.minimum(slabs['inl'][good], 60.)
            eu = _local_rms(g, xs, ru, wq, 12.)
            ez = _local_rms(g, xs, rz, wq, 12.)
            su = np.sqrt(su ** 2 + eu ** 2)
            sh = np.sqrt(sh ** 2 + ez ** 2)
            # support ends where the lateral geometry becomes inconsistent
            bad = (g > rail_support) & (eu > cfg.max_sigma_support)
            if bad.any():
                support = min(support, float(g[np.argmax(bad)]))
        beyond = np.maximum(0., g - support)
        # Straight continuation is allowed only when the last 60 m of supported
        # path are straight; otherwise a curve may start right at the support end.
        k0 = int(max(0., support - 60.))
        k1 = int(min(len(g) - 1, support))
        straight = False
        if cfg.extension and support >= cfg.extension_min_support and k1 - k0 >= 30:
            curv = np.gradient(np.gradient(lat[k0:k1 + 1]))
            straight = bool(np.max(np.abs(curv)) < cfg.extension_max_curvature)
        if straight:
            su = su + cfg.extension_sigma_per_m * beyond
            sh = sh + cfg.extension_sigma_per_m * beyond
        else:
            su = su + cfg.sigma_extrap_per_m * beyond + beyond ** 2 / (2 * 800.)
            sh = sh + .5 * cfg.sigma_extrap_per_m * beyond
        info = dict(support_m=float(support), rail_support_m=float(rail_support), extension_ok=straight,
                    good_slabs=int(good.sum()), phi=float(phi))
        return PathModel(lat, vert, phi, support, su, sh, info)

    def _update_surface_ref(self, s, u, h):
        fl = (s > 10) & (s < 32) & (np.abs(u) < 1.3) & (h > -1.3) & (h < 1.6)
        ce = (s > 12) & (s < 40) & (np.abs(u) < 1.0) & (h > 2.3) & (h < 7.)
        if fl.sum() < 50:
            return
        floor = float(np.quantile(h[fl], .3))
        ceil = float(np.quantile(h[ce], .9)) if ce.sum() >= 20 else None
        r = self.surface_ref
        if r is None:
            self.surface_ref = dict(floor=floor, ceil=ceil)
        else:
            r['floor'] = .8 * r['floor'] + .2 * floor
            if ceil is not None:
                r['ceil'] = ceil if r['ceil'] is None else .8 * r['ceil'] + .2 * ceil

    # ------------------------------------------------------------------
    def process(self, points, rings=None, timestamp=None, return_debug=False):
        tic = time.perf_counter()
        cfg = self.cfg
        self.frame += 1
        dt = None
        if timestamp is not None and self.last_time is not None:
            dt = timestamp - self.last_time
            if dt <= 0 or dt > cfg.max_gap_s:
                self.reset()
                self.frame = 0
                dt = None
        self.last_time = timestamp
        # clock for time windows; frames may be skipped when the computer is busy
        self.now = float(timestamp) if timestamp is not None else self.frame * cfg.frame_period_s
        p = np.asarray(points, np.float32)
        X, Y, Z = p[:, 0], p[:, 1], p[:, 2]
        m = (X > .5) & (X < cfg.max_range + 25) & (np.abs(Y) < 45) & (Z > -6) & (Z < 10)
        if not m.all():
            p = _rows(p, m)
        timing = {}
        # 1) odometry on previous path frame (wall depth images)
        ds = None
        oq = 0.
        had_path = self.path is not None
        if had_path:
            ds, oq = self.odometry.update(p, self.path, dt)
            if self.miss:
                oq = 0.
            if ds is None and self.odometry.v is not None and dt:
                ds = self.odometry.v * dt
        timing['odometry_ms'] = (time.perf_counter() - tic) * 1000
        prior = self._prior(ds, p)
        # 2) rails: with the moved previous path, then with a neutral prior
        t1 = time.perf_counter()
        near = self.near.fit(p, prior)
        reacquired = False
        if near is None and self.path is not None:
            neutral = PathModel(np.zeros(len(PathModel.GRID)),
                                np.full(len(PathModel.GRID), self._rail_height_guess(p)), 0., 0.)
            near = self.near.fit(p, neutral)
            if near is not None:
                prior = neutral
                reacquired = True
                self.template.reset()
        if near is None:
            self.miss += 1
            if self.path is None or self.miss > cfg.max_coast_frames:
                self.reset()
                result = self._empty('Не найдены головки рельсов вблизи поезда', tic)
                return (result, None) if return_debug else result
        else:
            self.miss = 0
        timing['rails_ms'] = (time.perf_counter() - t1) * 1000
        # provisional near path for template construction (only from fresh rails)
        t2 = time.perf_counter()
        first = self.path is None or reacquired
        prov = self._smooth_path(prior, near, None, first)
        if near is not None:
            near_idx = np.flatnonzero((p[:, 0] > cfg.template_near[0]) & (p[:, 0] < cfg.template_near[1]))
            sn, un, hn = prov.to_track(p[near_idx[::3]])
            self.template.update(un, hn)
            self._update_surface_ref(sn, un, hn)
        timing['template_ms'] = (time.perf_counter() - t2) * 1000
        # 3) far registration, two passes
        t3 = time.perf_counter()
        # progressive extension: each chunk is searched around the path fitted
        # from all nearer chunks, so strong curves are followed step by step
        path = prov
        slabs = None
        pre = self._slab_points(p)
        edges = list(cfg.chunk_edges)
        for a, b in zip(edges[:-1], edges[1:]):
            sl = self._register_slabs(p, path, 1.4 if first else 1., a, b, pre)
            if sl is None or len(sl['x']) == 0:
                continue
            slabs = sl if slabs is None else {k: np.r_[slabs[k], sl[k]] for k in sl}
            path = self._smooth_path(prior, near, slabs, first, quick=True)
        if slabs is not None:
            path = self._smooth_path(prior, near, slabs, first)
        timing['register_ms'] = (time.perf_counter() - t3) * 1000
        if ds is not None:
            self.S_ego += ds
        self._cur_ds = ds
        if near is None:
            # Coasting on odometry: the rail datum is not re-observed, so the
            # near-field geometry is less certain with every such frame.
            path.info['coasting_frames'] = self.miss
            path.support = min(path.support, 40.)
            path.sigma_h = path.sigma_h + cfg.coast_sigma_h * self.miss
            path.sigma_u = path.sigma_u + cfg.coast_sigma_u * self.miss
        self.path = path
        if not had_path:
            self.odometry.prime(p, path)
        # 4) envelope membership
        t4 = time.perf_counter()
        corridor = self._corridor(p, path)
        cand = self._clusters(corridor, path)
        timing['clusters_ms'] = (time.perf_counter() - t4) * 1000
        # 5) tracking and decision
        analysis = float(path.info.get('analysis_limit_m', min(cfg.max_range, np.interp(max(0., path.support - cfg.support_buffer), path.GRID, path.arc))))
        confirmed = self._track(cand, analysis)
        result = self._result(path, cand, confirmed, near, slabs, analysis, ds, oq, timing, tic, len(points), len(p))
        if return_debug:
            return result, dict(path=path, near=near, slabs=slabs, corridor=corridor)
        return result

    # ------------------------------------------------------------------
    def _corridor(self, p, path):
        cfg = self.cfg
        x = p[:, 0]
        xi = np.clip(x.astype(np.int64), 0, len(path.GRID) - 1)
        rough = (np.abs(p[:, 1] - path.lat[xi]) < cfg.half_width + 1.2 + np.abs(path.dlat[xi])) & (x > cfg.min_range - .5)
        q = _rows(p, rough)
        # standstill occupancy history over the whole corridor, independent of the verified range
        self._update_change(q, getattr(self, '_cur_ds', None))
        s, u, h = path.to_track(q)
        limit = min(cfg.max_range, float(np.interp(max(0., path.support - cfg.support_buffer), path.GRID, path.arc)))
        ext = self._extension_limit(p, path) if path.info.get('extension_ok') else None
        if ext is not None:
            limit = max(limit, ext)
        path.info['analysis_limit_m'] = limit
        keep = (np.abs(u) < cfg.half_width + .4) & (h > -.05) & (h < cfg.height + .4) & (s > cfg.min_range) & (s < limit)
        return dict(p=q[keep], s=s[keep], u=u[keep], h=h[keep])

    def _extension_limit(self, p, path):
        """How far the straight continuation is observably free of tunnel structure.

        Returns in the corridor |u| < extension_corridor beyond the supported
        geometry are grouped; the first wall-like group (long along the track
        or wide across it) ends the extension - e.g. the outer wall of a curve
        that starts beyond the support. Compact groups are left to the
        detector as candidates.
        """
        cfg = self.cfg
        s0 = float(path.support)
        x = p[:, 0]
        sel = x > s0 - 2.
        if not sel.any():
            return float(cfg.max_range)
        q = _rows(p, sel)
        s, u, h = path.to_track(q)
        m = (np.abs(u) < cfg.extension_corridor) & (h > -.3) & (h < cfg.height + .8) & (s > s0)
        if not m.any():
            return float(cfg.max_range)
        S, U = s[m], u[m]
        o = np.argsort(S)
        S, U = S[o], U[o]
        cuts = np.flatnonzero(np.diff(S) > 3.) + 1
        for grp in np.split(np.arange(len(S)), cuts):
            ext_s = S[grp[-1]] - S[grp[0]]
            ext_u = U[grp].max() - U[grp].min()
            if ext_s > 6. or ext_u > 2.6:
                return float(max(s0, S[grp[0]] - 2.))
        return float(cfg.max_range)

    def _clusters(self, cor, path):
        cfg = self.cfg
        s, u, h, q = cor['s'], cor['u'], cor['h'], cor['p']
        if len(s) == 0:
            return []
        gi = np.clip(np.searchsorted(path.arc, s), 0, len(path.arc) - 1)
        su = path.sigma_u[gi]
        sh = path.sigma_h[gi]
        k = cfg.margin_k
        d_lat = cfg.half_width - np.abs(u)
        d_top = cfg.height - h
        d_bot = h
        rail = (np.abs(np.abs(u) - cfg.gauge / 2) < cfg.rail_exclusion + np.minimum(.03, su)) & (h < .12 + sh)
        inside = (d_lat > 0) & (d_top > 0) & (d_bot > cfg.min_height) & ~rail
        conf = (d_lat > k * su) & (d_top > k * sh) & (d_bot > np.maximum(cfg.min_height, k * sh)) & ~rail
        if not inside.any():
            return []
        S, U, H = s[inside], u[inside], h[inside]
        C = conf[inside]
        Q = q[inside]
        order = np.argsort(S, kind='stable')
        S, U, H, C, Q = S[order], U[order], H[order], C[order], Q[order]
        labels = _cluster_labels(S, U, H)
        support_arc = float(np.interp(path.support, path.GRID, path.arc))
        out = []
        for lab in np.unique(labels):
            idx = np.flatnonzero(labels == lab)
            n = len(idx)
            nc = int(C[idx].sum())
            a = Q[idx]
            # distance to the nearest point that is confidently inside; low,
            # uncertain points only contribute to the extent
            conf_idx = idx[C[idx]]
            smin = float(S[conf_idx].min()) if len(conf_idx) else float(S[idx].min())
            cand = dict(distance_m=smin, s=smin, s_max=float(S[idx].max()), s_med=float(np.median(S[idx])),
                        path_lateral_m=float(np.median(U[idx])), u_range=[float(U[idx].min()), float(U[idx].max())],
                        h_range=[float(H[idx].min()), float(H[idx].max())],
                        center=np.median(a, axis=0).tolist(), bbox=[a.min(axis=0).tolist(), a.max(axis=0).tolist()],
                        point_count=n, confident_points=nc, appeared_frac=self._appeared_frac(Q[idx]),
                        margin_m=float(np.max(np.minimum(np.minimum(cfg.half_width - np.abs(U[idx]), cfg.height - H[idx]), H[idx]))),
                        sigma_u=float(path.sigma_u[min(len(path.arc) - 1, np.searchsorted(path.arc, smin))]))
            # A wall crossing the envelope (curve end, mis-estimated far axis) is
            # seen obliquely: long along the track AND wide AND tall. It either
            # enters from the side or stands where the supported axis ends (the
            # tunnel turns away there). A real obstacle is seen by its facing
            # side: short along the track. A long object that blocks the track
            # centre well inside the supported range is never taken for a wall.
            ext_s = float(S[idx].max() - S[idx].min())
            ext_u = float(U[idx].max() - U[idx].min())
            ext_h = float(H[idx].max() - H[idx].min())
            off_centre = float(np.abs(U[idx]).min()) > cfg.wall_clear_center
            at_axis_end = float(S[idx].max()) > support_arc - cfg.wall_end_margin
            if ext_s > cfg.wall_extent_s and ext_u > cfg.wall_extent_uh and ext_h > cfg.wall_extent_uh \
                    and (off_centre or at_axis_end):
                cand.update(wall_like=True, confident_points=0)
            out.append(cand)
        out.sort(key=lambda c: c['distance_m'])
        return out

    def _voxel_keys(self, q):
        v = self.cfg.change_voxel
        ix = np.floor(q[:, 0] / v).astype(np.int64)
        iy = np.floor(q[:, 1] / v).astype(np.int64) + 512
        iz = np.floor(q[:, 2] / v).astype(np.int64) + 512
        return (ix << 22) | (iy << 11) | iz

    def _update_change(self, p, ds):
        """Occupancy history of the scene in sensor voxels while the train stands."""
        cfg = self.cfg
        if ds is None or abs(ds) >= cfg.still_ds:
            if self.still_f0 is not None:
                self._reset_change()
            return
        if self.still_f0 is None:
            self.still_f0, self.still_t0 = self.frame, self.now
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        m = (x < 150.) & (np.abs(y) < 12.) & (z > -8.) & (z < 8.)
        keys = np.unique(self._voxel_keys(_rows(p, m)))
        n = len(self.vox_keys)
        idx = np.searchsorted(self.vox_keys, keys)
        found = idx < n
        found[found] = self.vox_keys[idx[found]] == keys[found]
        self.vox_count[idx[found]] += 1
        new = keys[~found]
        if len(new):
            k = np.concatenate([self.vox_keys, new])
            o = np.argsort(k, kind='stable')
            self.vox_keys = k[o]
            self.vox_first = np.concatenate([self.vox_first, np.full(len(new), self.frame, np.int32)])[o]
            self.vox_first_t = np.concatenate([self.vox_first_t, np.full(len(new), self.now)])[o]
            self.vox_count = np.concatenate([self.vox_count, np.ones(len(new), np.int32)])[o]

    def _appeared_frac(self, q):
        """Share of points in voxels that were empty early in the stop and are
        now steadily occupied: something appeared while the train stood still."""
        cfg = self.cfg
        eps = .5 * cfg.frame_period_s     # timestamp jitter
        if self.still_f0 is None or len(self.vox_keys) == 0 or \
                self.now - self.still_t0 < cfg.change_ref_s + cfg.change_min_age_s - eps:
            return 0.
        keys = self._voxel_keys(q)
        idx = np.clip(np.searchsorted(self.vox_keys, keys), 0, len(self.vox_keys) - 1)
        found = self.vox_keys[idx] == keys
        first_t = self.vox_first_t[idx]
        age = self.frame - self.vox_first[idx] + 1
        new = found & (first_t >= self.still_t0 + cfg.change_ref_s - eps) & \
            (self.now - first_t >= cfg.change_min_age_s - eps) & (age >= 2) & \
            (self.vox_count[idx] >= cfg.change_occupancy * age)
        return float(new.mean())

    def _holdable(self, t):
        limit = self.cfg.move_hold_s if t.moving else self.cfg.max_hold_s
        return t.confirmed_at is not None and self.now - t.confirmed_t <= limit

    def _expected_S(self, t):
        """Where the track's object is now: a moving one has travelled on at its speed."""
        if t.moving and t.ts:
            return t.Ss[-1] + t.speed * (self.now - t.ts[-1])
        return t.S

    def _track(self, cands, analysis=None):
        cfg = self.cfg
        f = self.frame
        for c in cands:
            c['S'] = c['s_med'] + self.S_ego
            best = None
            for t in self.tracks:
                if f - t.last > cfg.window and not self._holdable(t):
                    continue
                gate = cfg.gate_s + .01 * c['s']
                dS = abs(c['S'] - self._expected_S(t))
                if dS < gate and abs(c['path_lateral_m'] - t.u) < .9 and abs(np.mean(c['h_range']) - t.h) < cfg.gate_h:
                    # an already confirmed track wins ties, so one object keeps one track
                    score = dS + (0. if t.confirmed_at is not None else .5)
                    if best is None or score < best[0]:
                        best = (score, t)
            strong = c['confident_points'] >= cfg.strong_points
            c['evidence_obs'] = self._evidence(c)
            if best is None:
                t = Track(self.next_id, c['S'], c['path_lateral_m'], float(np.mean(c['h_range'])), f, strong, c)
                t.ts = [self.now]
                self.next_id += 1
                self.tracks.append(t)
            else:
                t = best[1]
                if t.last != f:
                    t.hits += 1
                    t.frames.append(f)
                t.last = f
                a = 1. if t.moving else .5          # a moving track follows its object without lag
                t.S = (1 - a) * t.S + a * c['S']
                t.u = (1 - a) * t.u + a * c['path_lateral_m']
                t.h = (1 - a) * t.h + a * float(np.mean(c['h_range']))
                t.strong += int(strong)
                t.best = c
                t.Ss.append(c['S'])
                del t.Ss[:-cfg.window]
                t.ts.append(self.now)
                del t.ts[:-cfg.window]
                t.conf.append((f, self._evidence(c)))
                del t.conf[:-cfg.window]
            c['track_id'] = t.id
            if c['confident_points'] > 0 and (not t.obs_ego or t.obs_ego[-1] != self.S_ego):
                t.obs_ego.append(self.S_ego)
                del t.obs_ego[:-100]
        # prune
        self.tracks = [t for t in self.tracks if f - t.last <= max(cfg.window, cfg.hold_frames) + 2 or self._holdable(t)]
        confirmed = []
        for c in cands:
            t = next(t for t in self.tracks if t.id == c['track_id'])
            recent = [x for x in t.frames if f - x < cfg.window]
            conf_pts = c['confident_points'] > 0
            drift = (max(t.Ss) - min(t.Ss)) if len(t.Ss) > 1 else 0.
            static = drift < cfg.static_tol + .006 * c['s']
            evidence = sum(min(n, 12) for fr_, n in t.conf if f - fr_ < cfg.window)
            need = cfg.evidence_near + cfg.evidence_per_m * max(0., c['s'] - 40.)
            ok = conf_pts and static and evidence >= need and \
                ((t.strong >= cfg.strong_hits and len(recent) >= cfg.strong_hits) or len(recent) >= cfg.weak_hits)
            if not ok and static and self.still_f0 is not None and \
                    (t.appeared or c.get('appeared_frac', 0.) >= cfg.change_new_frac):
                # the train stands still and the object appeared where the sensor
                # saw nothing before: path uncertainty does not matter here
                seen = sum(1 for x in t.frames if f - x < cfg.change_window)
                gi = min(len(self.path.arc) - 1, int(np.searchsorted(self.path.arc, c['s'])))
                lateral_ok = abs(c['path_lateral_m']) < cfg.half_width - float(self.path.sigma_u[gi])
                if lateral_ok and (seen >= cfg.change_persist or (t.appeared and seen >= 3)):
                    ok = True
                    t.appeared = True
                    c['appeared'] = True
            if not ok and not static and conf_pts:
                speed = self._moving(t, c, evidence, need, recent)
                if speed is not None:
                    ok = True
                    if t.confirmed_at != f:
                        t.n_moving += 1
                    t.moving, t.speed = True, speed
                    c.update(moving=True, speed_mps=speed)
            c['evidence'] = float(evidence)
            c['S_drift'] = float(drift)
            c['hits'] = len(recent)
            c['confirmed'] = bool(ok)
            if ok:
                confirmed.append(c)
                t.confirmed_at, t.confirmed_t = f, self.now
                t.offset, t.last_c, t.S_ego_c = c['distance_m'] - c['s_med'], c, self.S_ego
        confirmed += self._held(cands, analysis)
        confirmed.sort(key=lambda c: c['distance_m'])
        return confirmed

    def _held(self, cands, analysis):
        """Keep a confirmed static obstacle that is not confirmed in this frame.

        Short gaps (<= hold_frames) are bridged anywhere. Longer ones only while
        the obstacle's place lies beyond the verified range: the train cannot see
        that it has gone, so it must not report the track as clear. The distance
        follows the train by odometry. Nothing is held closer than hold_min_range,
        where an object may simply leave the lidar's vertical field of view.
        The long hold needs observations (with confident points) from train
        positions at least hold_min_travel apart: when the train stands still,
        "static in the world" proves nothing, and a single spurious
        confirmation must not become a long alarm.
        """
        cfg = self.cfg
        f = self.frame
        out = []
        by_track = {c['track_id']: c for c in cands if 'track_id' in c}
        for t in self.tracks:
            if t.confirmed_at is None or t.confirmed_at == f:
                continue
            age = f - t.confirmed_at
            if t.moving:
                c = self._held_moving(t, age, analysis, by_track.get(t.id))
                if c is not None:
                    if c is not by_track.get(t.id):
                        cands.append(c)
                    out.append(c)
                continue
            short = age <= cfg.hold_frames
            d = t.S - self.S_ego + t.offset
            if d < max(cfg.min_range, cfg.hold_min_range):
                continue
            beyond = analysis is not None and d > analysis - 2.
            seen = by_track.get(t.id)
            if seen is not None and seen['confident_points'] > 0:
                # observed with confident points, just not confirmed this frame
                if short:
                    seen.update(confirmed=True, held=True)
                    out.append(seen)
                continue
            # not observed, or only as unconfident points: keep the last confirmed state
            parallax = (max(t.obs_ego) - min(t.obs_ego)) if t.obs_ego else 0.
            long_hold = beyond and parallax >= cfg.hold_min_travel and self.now - t.confirmed_t <= cfg.max_hold_s
            if short or long_hold:
                dx = self.S_ego - t.S_ego_c
                c = dict(t.last_c)
                lo, hi = list(c['bbox'][0]), list(c['bbox'][1])
                lo[0] -= dx
                hi[0] -= dx
                center = list(c['center'])
                center[0] -= dx
                c.update(distance_m=float(d), s=float(d), s_med=float(t.S - self.S_ego), s_max=float(c['s_max'] - dx),
                         S=float(t.S), bbox=[lo, hi], center=center, point_count=0, confident_points=0, hits=0,
                         confirmed=True, held=True, held_frames=int(age), track_id=t.id)
                cands.append(c)
                out.append(c)
        cands.sort(key=lambda c: c['distance_m'])
        return out

    def _held_moving(self, t, age, analysis, seen):
        """A confirmed moving obstacle that is not confirmed in this frame, carried on by its speed.

        Short gaps are bridged anywhere. Longer ones, up to move_hold_s, only where
        the predicted place lies beyond the verified range: a large body ahead
        hides the tunnel behind it, so the verified range ends in front of it.
        """
        cfg = self.cfg
        if t.n_moving < 2 or not t.ts:
            return None
        S_pred = self._expected_S(t)
        d = S_pred - self.S_ego + t.offset
        if d < max(cfg.min_range, cfg.hold_min_range):
            return None
        if seen is not None and seen['confident_points'] > 0:
            if age > cfg.hold_frames:
                return None
            seen.update(confirmed=True, held=True, moving=True, speed_mps=t.speed)
            return seen
        beyond = analysis is not None and d > analysis - 2.
        if not (age <= cfg.hold_frames or (beyond and self.now - t.confirmed_t <= cfg.move_hold_s)):
            return None
        c = dict(t.last_c)
        shift = d - c['distance_m']
        lo, hi = list(c['bbox'][0]), list(c['bbox'][1])
        lo[0] += shift
        hi[0] += shift
        center = list(c['center'])
        center[0] += shift
        c.update(distance_m=float(d), s=float(d), s_med=float(S_pred - self.S_ego), s_max=float(c['s_max'] + shift),
                 S=float(S_pred), bbox=[lo, hi], center=center, point_count=0, confident_points=0, hits=0,
                 confirmed=True, held=True, held_frames=int(age), track_id=t.id, moving=True, speed_mps=t.speed)
        return c

    def _moving(self, t, c, evidence, need, recent):
        """World speed of a track that moves steadily along the path with strong evidence, else None.

        Clutter that moves together with the view (the far end of a curve) has the
        train's own speed; only a large body across the track centre (a train ahead)
        is accepted at that relative speed.
        """
        cfg = self.cfg
        f = self.frame
        if len(t.Ss) < cfg.move_min_obs or len(recent) < cfg.move_min_obs or len(t.ts) != len(t.Ss):
            return None
        ts, Ss = np.asarray(t.ts), np.asarray(t.Ss)
        if np.ptp(ts) < .2:
            return None
        A = np.column_stack([ts - ts.mean(), np.ones(len(ts))])
        coef = np.linalg.lstsq(A, Ss, rcond=None)[0]
        speed = float(coef[0])
        if np.sqrt(np.mean((Ss - A @ coef) ** 2)) > cfg.move_resid + .004 * c['s']:
            return None
        if not cfg.move_min_speed <= abs(speed) <= cfg.move_max_speed:
            return None
        if abs(speed - (self.odometry.v or 0.)) < cfg.move_rel_min:
            wide = c['u_range'][1] - c['u_range'][0] > 1.2 and c['h_range'][1] - c['h_range'][0] > 1.2
            if not (wide and abs(c['path_lateral_m']) < .6):
                return None
        if evidence < cfg.move_evidence_k * need:
            return None
        if sum(1 for fr_, n in t.conf if f - fr_ < cfg.window and n >= cfg.strong_points) < cfg.move_strong:
            return None
        gi = min(len(self.path.arc) - 1, int(np.searchsorted(self.path.arc, c['s'])))
        if abs(c['path_lateral_m']) >= cfg.half_width - float(self.path.sigma_u[gi]):
            return None
        return speed

    def _evidence(self, c):
        """Evidence of one observation: confident points, discounted when a close
        cluster has fewer returns than any real object of >=10 cm would give."""
        n = c['confident_points']
        expected = min(8., max(1., 40. / max(c['s'], 1.)))
        return float(min(n, 12) * (1. if n >= expected else .3))

    def _empty(self, reason, tic):
        return dict(model_id='tunnel', model_revision='tunnel-profile-v1', obstacle=False, distance_m=None,
                    status='unknown', candidates=[], geometry_reason=reason,
                    latency_ms=(time.perf_counter() - tic) * 1000)

    def _result(self, path, cands, confirmed, near, slabs, analysis, ds, oq, timing, tic, npoints, nroi):
        cfg = self.cfg
        # a cluster without confident (or newly appeared) points is uncertain
        # geometry, e.g. distant rail points in a curve: it is not a candidate
        significant = any(c['confident_points'] > 0 or c.get('appeared_frac', 0.) > 0 for c in cands)
        status = 'obstacle' if confirmed else ('candidate' if significant else 'clear')
        res = dict(model_id='tunnel', model_revision='tunnel-profile-v1', status=status, obstacle=bool(confirmed),
                   distance_m=float(confirmed[0]['distance_m']) if confirmed else None,
                   candidates=cands, points=int(npoints), roi_points=int(nroi),
                   analysis_range_m=analysis,
                   path=path.samples()[::2].tolist(), path_info=dict(path.info, odometry_ds=ds, odometry_quality=oq,
                                                                     ego_s=self.S_ego, method='rails + tunnel profile'),
                   floor=[0., 0., float(path.vert[0])],
                   section_frames=path.section_frames(analysis),
                   envelope=[[float(a), -cfg.half_width, cfg.half_width, -cfg.half_width, cfg.half_width]
                             for a in np.arange(0, analysis + 1, 2.)],
                   vehicle=dict(envelope_mode='section', car_width=2 * cfg.half_width, max_height=cfg.height,
                                body_bottom=0., safety_margin=0., projection_mode='rail_section'),
                   half_width_m=cfg.half_width, min_height_m=cfg.min_height,
                   geometry_contract=dict(height_datum='rail_heads', section_width_m=2 * cfg.half_width,
                                          section_height_m=cfg.height, curve_margin_m=0.,
                                          lower_boundary_tolerance_m=cfg.min_height,
                                          datum_observed=near is not None),
                   timing=timing)
        res['latency_ms'] = (time.perf_counter() - tic) * 1000
        return res


def _rows(a, mask):
    """Row selection; np.take on indices is ~4x faster than boolean indexing of (N,3)."""
    return np.take(a, np.flatnonzero(mask), axis=0)


def _local_rms(g, x, r, w, half):
    """Weighted RMS of residuals r(x) around each grid node within +-half metres."""
    out = np.zeros(len(g))
    if len(x) == 0:
        return out
    d = np.abs(g[:, None] - x[None, :])
    k = np.where(d <= half, w[None, :], 0.)
    num = (k * r[None, :] ** 2).sum(1)
    den = k.sum(1)
    near = den > 0
    out[near] = np.sqrt(num[near] / den[near])
    # beyond the last observation keep the last value
    if (~near).any() and near.any():
        idx = np.flatnonzero(near)
        out[~near] = out[idx[np.clip(np.searchsorted(idx, np.flatnonzero(~near)) - 1, 0, len(idx) - 1)]]
    return out


def _group_max(key, val, size):
    """Per-key maximum (-inf where empty) via one sort of packed (key, value) integers."""
    out = np.full(size, -np.inf, np.float32)
    if len(key) == 0:
        return out
    lo = float(val.min())
    q = np.clip(((val - lo) / .001).astype(np.int64), 0, (1 << 20) - 1)
    packed = np.sort(key.astype(np.int64) * (1 << 20) + q)
    k = packed >> 20
    last = np.r_[np.flatnonzero(np.diff(k)), len(k) - 1]
    out[k[last]] = (packed[last] & ((1 << 20) - 1)).astype(np.float32) * .001 + lo
    return out


def _robust_weights(g, x, y, w, lam, tau, iters=1):
    """Huber weights of observations against a data-only smooth fit."""
    if len(x) == 0 or not np.any(w > 0):
        return w
    f = _penalized_fit(g, x, y, w, lam)
    ww = w
    for _ in range(iters):
        r = np.abs(y - np.interp(x, g, f))
        ww = w * np.minimum(1., tau / np.maximum(r, 1e-6))
        f = _penalized_fit(g, x, y, ww, lam)
    return ww


def _irls_fit(g, x, y, w, lam, tau, iters=2):
    """Penalized fit with Huber reweighting of observations (outlier slabs lose weight)."""
    f = _penalized_fit(g, x, y, w, lam)
    for _ in range(iters):
        r = np.abs(y - np.interp(x, g, f))
        ww = w * np.minimum(1., tau / np.maximum(r, 1e-6))
        f = _penalized_fit(g, x, y, ww, lam)
    return f


def _penalized_fit(g, x, y, w, lam):
    """Smooth f on uniform grid g: min sum w (f(x)-y)^2 + lam * sum (second difference)^2."""
    n = len(g)
    idx = np.clip(np.searchsorted(g, x) - 1, 0, n - 2)
    t = (x - g[idx]) / (g[idx + 1] - g[idx])
    diag = np.zeros(n)
    off1 = np.zeros(n - 1)
    b = np.zeros(n)
    diag += np.bincount(idx, w * (1 - t) ** 2, n)
    diag += np.bincount(idx + 1, w * t * t, n)
    off1 += np.bincount(idx, w * t * (1 - t), n - 1)
    b += np.bincount(idx, w * (1 - t) * y, n)
    b += np.bincount(idx + 1, w * t * y, n)
    # lam * D^T D for second differences: bands [1,-4,6,-4,1] with edge corrections
    d0 = np.full(n, 6.); d0[[0, -1]] = 1.; d0[[1, -2]] = 5.
    d1 = np.full(n - 1, -4.); d1[[0, -1]] = -2.
    d2 = np.ones(n - 2)
    ab = np.zeros((3, n))
    ab[2] = diag + lam * d0 + 1e-6
    ab[1, 1:] = off1 + lam * d1
    ab[0, 2:] = lam * d2
    return solveh_banded(ab, b, check_finite=False)


def _cluster_labels(S, U, H):
    """Gap-based grouping: along-track gaps first, then lateral and vertical gaps.

    Point spacing grows with range (0.15 deg azimuth, ~0.125 deg elevation), so
    gaps are scaled with distance. Returns labels for points sorted by S.
    """
    n = len(S)
    labels = np.zeros(n, np.int64)
    if n == 0:
        return labels
    gap_s = np.maximum(1.0, .012 * S)
    cut = np.flatnonzero(np.diff(S) > gap_s[1:]) + 1
    next_label = 0
    for grp in np.split(np.arange(n), cut):
        gu = np.maximum(.45, .006 * S[grp].mean())
        o = grp[np.argsort(U[grp], kind='stable')]
        for sub in np.split(o, np.flatnonzero(np.diff(U[o]) > gu) + 1):
            o2 = sub[np.argsort(H[sub], kind='stable')]
            for sub2 in np.split(o2, np.flatnonzero(np.diff(H[o2]) > gu) + 1):
                labels[sub2] = next_label
                next_label += 1
    return labels
