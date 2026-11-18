"""Warning about an object that is moving toward the envelope and will be inside it when the train reaches it.

The core looks for obstacles inside the current envelope. This optional module also tracks objects approaching it.

This rule looks at the returns in a band just outside the envelope (beside it and above it), using the axis the core
returned for the frame, so the core and its decisions are not touched. The band returns are grouped into clusters,
followed along the track (world position S, side, height) and, within the range where the axis is measured well enough,
their gap to the envelope is fitted against the distance. The trend is taken relative to the static things beside the
object (walls, cable trays: the same side, about the same range, the same frame): an error of the far axis shifts both
and cancels, a real motion of the object stays. When the gap shrinks steadily and the trend puts the object inside by
the arrival, the rule warns: the frame gets the attention status (mode attention, the default) or the object becomes a
confirmed obstacle (mode alarm). The core's own decisions are never removed.
"""
import numpy as np

HALF, TOP, FLOOR = 1.05, 3., .07


def gap(u_lo, u_hi, h_lo, h_hi):
    """Distance beyond the envelope on the limiting axis (negative: inside)."""
    lateral = -HALF if u_lo <= 0. <= u_hi else min(abs(u_lo), abs(u_hi)) - HALF
    return max(lateral, h_lo - TOP, FLOOR - h_hi)


def band_clusters(points, path, band=.35, max_range=110., cell=(.5, .1, .1), min_points=2):
    """Clusters of returns in the band outside the envelope, in track coordinates of this frame's axis."""
    from scipy import ndimage
    path = np.asarray(path, float)
    if len(path) < 3 or not len(points):
        return []
    p = points[(points[:, 0] > 3.) & (points[:, 0] < min(max_range, path[-1, 0]))]
    if not len(p):
        return []
    u = p[:, 1] - np.interp(p[:, 0], path[:, 0], path[:, 1])
    h = p[:, 2] - np.interp(p[:, 0], path[:, 0], path[:, 2])
    s = np.interp(p[:, 0], path[:, 0], path[:, 3])
    au = np.abs(u)
    side = (au > HALF) & (au <= HALF + band) & (h > FLOOR) & (h < TOP)
    top = (au <= HALF) & (h >= TOP) & (h <= TOP + band)
    m = side | top
    if m.sum() < min_points:
        return []
    p, s, u, h = p[m], s[m], u[m], h[m]
    i = np.floor(s / cell[0]).astype(int)
    j = np.floor((u + HALF + band) / cell[1]).astype(int)
    k = np.floor((h - FLOOR) / cell[2]).astype(int)
    i0 = i.min()
    grid = np.zeros((i.max() - i0 + 1, j.max() + 1, k.max() + 1), bool)
    grid[i - i0, j, k] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3), bool))
    lab = labels[i - i0, j, k]
    out = []
    for c in range(1, n + 1):
        sel = lab == c
        if sel.sum() < min_points:
            continue
        cs, cu, ch, cp = s[sel], u[sel], h[sel], p[sel]
        if (cu.min() < 0) != (cu.max() < 0) and ch.max() < TOP:  # beside the envelope a cluster keeps to one side
            continue
        out.append(dict(distance_m=float(cs.min()), s_mid=float(np.median(cs)), u_range=[float(cu.min()), float(cu.max())],
                        h_range=[float(ch.min()), float(ch.max())], point_count=int(sel.sum()), length=float(cs.max() - cs.min()),
                        side=float(np.sign(np.median(cu))) if ch.max() < TOP else 0.,
                        bbox=[cp.min(axis=0).tolist(), cp.max(axis=0).tolist()]))
    return out


class Track:
    def __init__(self, S, side, u, h, frame):
        self.S, self.side, self.u, self.h, self.last = S, side, u, h, frame
        self.obs = []  # (distance, gap, gap of the static things beside it)
        self.above = 0
        self.age = 0


class ApproachGate:
    def __init__(self, mode='attention', band=.35, max_range=110., max_length=2.5, sigma0=.02, sigma_per_m=.003, max_sigma=.3,
                 min_points=2, min_span=20., min_obs=8, min_slope=.006, max_resid=.06, min_t=5., max_gap_now=.4,
                 inside_by=.05, frames=3, forget=10, ref_window=8., ref_min_age=10):
        self.mode, self.band, self.max_range, self.max_length = mode, band, max_range, max_length
        self.sigma0, self.sigma_per_m, self.max_sigma, self.min_points = sigma0, sigma_per_m, max_sigma, min_points
        self.min_span, self.min_obs, self.min_slope, self.max_resid, self.min_t = min_span, min_obs, min_slope, max_resid, min_t
        self.max_gap_now, self.inside_by, self.frames, self.forget = max_gap_now, inside_by, frames, forget
        self.ref_window, self.ref_min_age = ref_window, ref_min_age
        self.reset()

    def reset(self):
        self.tracks = []
        self.frame = 0

    def _track(self, S, c):
        u, h = sum(c['u_range']) / 2, sum(c['h_range']) / 2
        best = None
        for t in self.tracks:
            if (t.last < self.frame and t.side == c['side'] and abs(t.S - S) < 1. and abs(t.u - u) < .25 and abs(t.h - h) < .5
                    and (best is None or abs(t.S - S) < abs(best.S - S))):
                best = t
        if best is None:
            best = Track(S, c['side'], u, h, self.frame)
            self.tracks.append(best)
        best.S, best.u, best.h, best.last = S, u, h, self.frame
        best.age += 1
        return best

    def predict(self, t):
        """The present gap carried by the trend relative to the static things to the arrival (distance 0), or None."""
        obs = [o for o in t.obs if o[2] is not None]
        if len(obs) < self.min_obs:
            return None
        d, g, ref = (np.array(v, float) for v in zip(*obs))
        if d.max() - d.min() < self.min_span:
            return None
        y = g - ref
        A = np.column_stack([np.ones_like(d), d])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        s = np.sqrt(resid @ resid / max(len(d) - 2, 1))
        se = s / np.sqrt(((d - d.mean()) ** 2).sum())
        slope = coef[1]  # > 0: the gap shrinks as the train comes closer
        if slope < self.min_slope or s > self.max_resid or slope < self.min_t * se:
            return None
        return g[-1] - slope * d[-1]

    def apply(self, result, points):
        ego = (result.get('path_info') or {}).get('ego_s')
        clusters = band_clusters(points, result.get('path') or [], self.band, self.max_range) if ego is not None else []
        return self.step(result, ego, clusters)

    def step(self, result, ego, clusters):
        self.frame += 1
        rows, statics = [], []
        for c in clusters:
            g = gap(*c['u_range'], *c['h_range'])
            if c['length'] > self.max_length:  # a wall or a cable tray along the track: only the static reference
                statics.append((c['side'], c['distance_m'], g, None))
                continue
            t = self._track(float(ego) + c['s_mid'], c)
            rows.append((c, t, g))
            if t.age >= self.ref_min_age:
                statics.append((c['side'], c['distance_m'], g, t))
        warned = []
        for c, t, g in rows:
            d = c['distance_m']
            if self.sigma0 + self.sigma_per_m * d <= self.max_sigma and c['point_count'] >= self.min_points:
                near = [gs for sd, dd, gs, ts in statics if sd == c['side'] and ts is not t and abs(dd - d) <= self.ref_window]
                t.obs = (t.obs + [(d, g, float(np.median(near)) if near else None)])[-40:]
            g0 = self.predict(t) if 0. < g <= self.max_gap_now else None
            t.above = t.above + 1 if g0 is not None and g0 <= -self.inside_by else 0
            if t.above >= self.frames:
                warned.append(dict(distance_m=round(d, 2), u_range=c['u_range'], h_range=c['h_range'], bbox=c['bbox'],
                                   gap_m=round(g, 3), gap_at_arrival_m=round(float(g0), 3)))
        self.tracks = [t for t in self.tracks if self.frame - t.last <= self.forget]
        if warned:
            result['approaching'] = warned
            if self.mode == 'alarm':
                result.setdefault('candidates', []).extend(dict(w, confirmed=True, decision_rule='approach') for w in warned)
                if result.get('status') != 'obstacle':
                    result['status'], result['obstacle'] = 'obstacle', True
                    result['distance_m'] = min(w['distance_m'] for w in warned)
            elif result.get('status') == 'clear':
                result['status'] = 'candidate'  # attention: an object is coming toward the envelope
        return result
