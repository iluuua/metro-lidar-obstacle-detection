"""Physically consistent synthetic obstacles injected into real, clean recordings.

Mimics the organizer generator (ray grid, occluded returns removed) but only
removes returns *behind* the object, so objects are never seen through walls.
Objects are static in the world: their along-track position is S_obj, their
position in each frame is S_obj - S_ego(t) on the track estimated from the
near rails (straight windows only), plus lateral/vertical offsets.

Pass 1 (clean run) provides S_ego(t) and the near track line per frame.
Pass 2 runs a fresh detector on the modified clouds and scores each object.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]
from metro.tunnel import TunnelDetector, TunnelConfig  # noqa: E402
from run_cached import iter_frames  # noqa: E402

# Analytic test grid, independent of any recording or measured sensor calibration.
ELEV = np.linspace(-25., 15., 256)


def set_elevations(path):
    """Use a calibration provided by the caller for this process."""
    global ELEV
    values = np.asarray(np.load(path), dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('elevations must be a nonempty finite 1D array in degrees')
    ELEV = np.sort(values)

AZ_STEP = .15


def track_pass(source, start, stop, observed_only=False):
    """Clean run -> per-frame S_ego, near offset/heading, and world curvature samples.

    Only the near, rail-supported part of each frame's path (5-30 m) is used,
    so the reconstructed track does not depend on far-range estimation.
    """
    det = TunnelDetector(TunnelConfig())
    rows = {}
    kappa, kappa_v = {}, {}
    for i, pts, stamp in iter_frames(source, start, stop):
        res, dbg = det.process(pts, None, stamp, return_debug=True)
        if dbg is None or dbg.get('near') is None:
            continue
        path = dbg['path']
        rail_support = float(dbg['near']['support'])
        x = np.arange(5., min(30., rail_support) + .01, 1.) if observed_only else np.arange(5., 31., 1.)
        c = np.polyfit(x, path.c(x), 2)
        r = np.polyfit(x, path.r(x), 2)
        rows[i] = dict(S=det.S_ego, c0=float(c[2]), c1=float(c[1]), r0=float(r[2]), r1=float(r[1]),
                       phi=path.phi, stamp=stamp, obstacle=res['obstacle'], analysis=res['analysis_range_m'],
                       rail_support_m=rail_support, reference_x=x.tolist(), reference_y=path.c(x).tolist(),
                       reference_z=path.r(x).tolist())
        for xs in (10., 20., 30.):
            if observed_only and xs > rail_support:
                continue
            key = int(round(det.S_ego + xs))
            kappa.setdefault(key, []).append(2 * c[0])
            kappa_v.setdefault(key, []).append(2 * r[0])
    return rows, kappa, kappa_v


class WorldTrack:
    """Curvature profile along the track built from near observations."""
    def __init__(self, kappa, kappa_v):
        keys = np.array(sorted(kappa))
        self.S = keys.astype(float)
        self.k = np.array([np.median(kappa[k]) for k in keys])
        self.kv = np.array([np.median(kappa_v[k]) for k in keys])
        # light smoothing (5 m)
        w = np.ones(5) / 5
        self.k = np.convolve(np.pad(self.k, 2, mode='edge'), w, 'valid')
        self.kv = np.convolve(np.pad(self.kv, 2, mode='edge'), w, 'valid')

    def offsets(self, S0, s):
        """Lateral/vertical deviation from the tangent line at S0, at distance s ahead."""
        g = np.arange(0., max(s) + 1.1, 1.)
        k = np.interp(S0 + g, self.S, self.k)
        kv = np.interp(S0 + g, self.S, self.kv)
        slope = np.r_[0., np.cumsum((k[1:] + k[:-1]) / 2)]
        lat = np.r_[0., np.cumsum((slope[1:] + slope[:-1]) / 2)]
        slope_v = np.r_[0., np.cumsum((kv[1:] + kv[:-1]) / 2)]
        vert = np.r_[0., np.cumsum((slope_v[1:] + slope_v[:-1]) / 2)]
        return np.interp(s, g, lat), np.interp(s, g, vert), np.interp(s, g, slope)

    def anchored_frame(self, frame):
        """Keep the measured near track when replacing frame curvature by world curvature.

        A quadratic's intercept/slope cannot be reused with a different curvature:
        that moves the whole observed near track. Fit those two pose terms again,
        against clean near-rail observations only, before placing any object.
        """
        x = np.asarray(frame['reference_x'])
        y, z = np.asarray(frame['reference_y']), np.asarray(frame['reference_z'])
        dl, dv, _ = self.offsets(frame['S'], x)
        c1, c0 = np.polyfit(x, y - dl, 1)
        r1, r0 = np.polyfit(x, z - dv, 1)
        residual_u, residual_h = c0 + c1 * x + dl - y, r0 + r1 * x + dv - z
        return dict(frame, c0=float(c0), c1=float(c1), r0=float(r0), r1=float(r1),
                    reference_near_rms_u_m=float(np.sqrt(np.mean(residual_u ** 2))),
                    reference_near_max_u_m=float(np.max(np.abs(residual_u))),
                    reference_near_rms_h_m=float(np.sqrt(np.mean(residual_h ** 2))),
                    reference_near_max_h_m=float(np.max(np.abs(residual_h))))


def box_hits(center, size, max_range=250., axes=None):
    """Ray-cast a box on the configured test grid; columns of axes are its local unit vectors."""
    c = np.asarray(center, float)
    half = np.asarray(size, float) / 2
    axes = np.eye(3) if axes is None else np.asarray(axes, float)
    corners = c + np.array([[x, y, z] for x in (-half[0], half[0]) for y in (-half[1], half[1])
                            for z in (-half[2], half[2])]) @ axes.T
    az = np.degrees(np.arctan2(corners[:, 1], corners[:, 0]))
    el = np.degrees(np.arctan2(corners[:, 2], np.hypot(corners[:, 0], corners[:, 1])))
    az_grid = np.arange(np.floor(az.min() / AZ_STEP), np.ceil(az.max() / AZ_STEP) + 1) * AZ_STEP
    el_grid = ELEV[(ELEV >= el.min() - .01) & (ELEV <= el.max() + .01)]
    if len(az_grid) == 0 or len(el_grid) == 0:
        return np.zeros((0, 3)), None
    A, E = np.meshgrid(np.radians(az_grid), np.radians(el_grid))
    d = np.column_stack([(np.cos(E) * np.cos(A)).ravel(), (np.cos(E) * np.sin(A)).ravel(), np.sin(E).ravel()])
    local_origin, local_ray = -c @ axes, d @ axes
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (-half[None, :] - local_origin) / local_ray
        t2 = (half[None, :] - local_origin) / local_ray
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (tmax >= tmin) & (tmin > 0) & (tmin < max_range)
    pts = d[hit] * tmin[hit][:, None]
    footprint = (az.min(), az.max(), el.min(), el.max(), float(np.linalg.norm(c) - np.linalg.norm(half)))
    return pts, footprint


def inject(pts, objects, return_added=False):
    """Add object returns hidden neither by nearer real returns nor by each other;
    remove real returns behind each object."""
    if not objects:
        return (pts, []) if return_added else pts
    rng = np.linalg.norm(pts, axis=1)
    az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    el = np.degrees(np.arcsin(pts[:, 2] / np.maximum(rng, 1e-6)))
    # nearest real return per (azimuth, elevation) cell -> occlusion of object hits
    ia = np.floor(az / AZ_STEP).astype(np.int64)
    ie = np.searchsorted(ELEV, el)
    keep = np.ones(len(pts), bool)
    add = []
    added = []
    for obj in objects:
        center, size = obj[:2]
        axes = obj[2] if len(obj) > 2 else None
        hits, fp = box_hits(center, size, axes=axes)
        if fp is None or len(hits) == 0:
            added.append(np.empty((0, 3), np.float32))
            continue
        a0, a1, e0, e1, rmin = fp
        hr = np.linalg.norm(hits, axis=1)
        haz = np.degrees(np.arctan2(hits[:, 1], hits[:, 0]))
        hel = np.degrees(np.arcsin(hits[:, 2] / hr))
        hia = np.floor(haz / AZ_STEP).astype(np.int64)
        hie = np.searchsorted(ELEV, hel)
        near = (az >= a0 - .2) & (az <= a1 + .2) & (el >= e0 - .3) & (el <= e1 + .3) & (rng < hr.max())
        blocked = np.zeros(len(hits), bool)
        if near.any():
            cells = set(zip(ia[near].tolist(), ie[near].tolist()))
            rmin_cell = {}
            for (a_, e_), r_ in zip(zip(ia[near].tolist(), ie[near].tolist()), rng[near].tolist()):
                rmin_cell[(a_, e_)] = min(r_, rmin_cell.get((a_, e_), 1e9))
            blocked = np.array([rmin_cell.get((a_, e_), 1e9) < r_ - .3 for a_, e_, r_ in zip(hia.tolist(), hie.tolist(), hr.tolist())])
        hits = hits[~blocked]
        added.append(hits.astype(np.float32))
        if len(hits) == 0:
            continue
        keep &= ~((az >= a0) & (az <= a1) & (el >= e0) & (el <= e1) & (rng > rmin))
        add.append(hits.astype(np.float32))
    cloud = np.concatenate([pts[keep]] + add) if add else pts[keep]
    return (cloud, added) if return_added else cloud


SCENARIOS = [  # name, lateral u, bottom height over rail heads, size (along, across, up), positive
    ('box2_center', 0., .0, (1., 2., 2.), True),
    ('box05_center', 0., .3, (.5, .5, .5), True),
    ('box03_center', 0., 1., (.3, .3, .3), True),
    ('box03_on_rail', .8, .0, (.3, .3, .3), True),
    ('box03_edge_in', .95, 1., (.3, .3, .3), True),
    ('box03_out_15cm', -1.35, 1., (.3, .3, .3), False),
    ('box2_out_30cm', -2.35, .0, (1., 2., 2.), False),
    ('low_across', 0., .0, (.3, 2., .15), True),
]


def run(source, start, stop, spacing=130., first=250., offset=0, trace=None, native=False):
    track, kappa, kappa_v = track_pass(source, start, stop)
    frames = sorted(track)
    if not frames:
        raise SystemExit('no frames with rails')
    world = WorldTrack(kappa, kappa_v)
    S0 = track[frames[0]]['S']
    S_end = track[frames[-1]]['S']
    objects = []
    for k, sc in enumerate((SCENARIOS[offset:] + SCENARIOS[:offset]) * 8):
        S_obj = S0 + first + k * spacing
        if S_obj > S_end + 5:
            break
        objects.append((S_obj,) + sc)
    if native:  # the C++ core; the track itself is still reconstructed by the reference above
        from metro.native import NativeTunnelDetector
        det = NativeTunnelDetector(TunnelConfig())
    else:
        det = TunnelDetector(TunnelConfig())
    per = {k: dict(name=o[1], positive=o[5], first_range=None, alarms=0, visible=0, S=o[0])
           for k, o in enumerate(objects)}
    fp_frames = 0
    for i, pts, stamp in iter_frames(source, start, stop):
        if i not in track:
            continue
        t = track[i]
        placed, meta = [], []
        live = [(k, o) for k, o in enumerate(objects) if 2. < o[0] - t['S'] < 250.]
        if live:
            ss = np.array([o[0] - t['S'] for _, o in live])
            dl, dv, slope = world.offsets(t['S'], ss)
            for (k, (S_obj, name, u, h0, size, positive)), s, a, b, sl in zip(live, ss, dl, dv, slope):
                y = t['c0'] + t['c1'] * s + a + u
                zr = t['r0'] + t['r1'] * s + b + t['phi'] * u
                center = (s + size[0] / 2, y + (t['c1'] + sl) * size[0] / 2, zr + h0 + size[2] / 2)
                placed.append((center, size))
                meta.append((k, s, y, zr + h0, size))
        cloud = inject(pts, placed)
        res = det.process(cloud, None, stamp)
        if trace is not None:
            for k, s, y, zb, size in meta:
                if objects[k][1] == trace:
                    cs = [c for c in res.get('candidates', []) if abs(c['distance_m'] - s) < 3.]
                    print(f"f{i} s={s:6.1f} analysis={res.get('analysis_range_m', 0):6.1f} support={res.get('path_info', {}).get('support_m')} "
                          + ' '.join(f"[n{c['point_count']} conf{c['confident_points']} ev{c.get('evidence', 0):.0f} hits{c['hits']} {'C' if c['confirmed'] else ''} u{c['u_range'][0]:.2f}..{c['u_range'][1]:.2f}]" for c in cs), flush=True)
        matched_fp = False
        for c in res.get('candidates', []):
            if not c.get('confirmed'):
                continue
            hit = None
            for k, s, y, zb, size in meta:
                if abs(c['distance_m'] - s) < 3. and abs(c['center'][1] - y) < max(size[1], 1.) + .5:
                    hit = k
            if hit is None:
                matched_fp = True
                print(f"FP f{i} d={c['distance_m']:.1f} u={c['u_range'][0]:.2f}..{c['u_range'][1]:.2f} "
                      f"h={c['h_range'][0]:.2f}..{c['h_range'][1]:.2f} n={c['point_count']} conf={c['confident_points']} "
                      f"hits={c['hits']} analysis={res.get('analysis_range_m', 0):.0f} "
                      f"support={res.get('path_info', {}).get('support_m')} ext={res.get('path_info', {}).get('extension_ok')}", flush=True)
                continue
            o = per[hit]
            o['alarms'] += 1
            if o['first_range'] is None:
                o['first_range'] = round(float(c['distance_m']), 1)
        for k, s, *_ in meta:
            per[k]['visible'] += 1
        fp_frames += int(matched_fp)
    return dict(source=source, start=start, stop=stop, objects=list(per.values()), fp_frames=fp_frames)


def boundary_cases(sizes=(.3, .5), ranges=(30., 60., 90., 120., 150.),
                   offsets=(.95, 1., 1.05, 1.1, 1.15), top_offsets=(2.9, 2.95, 3., 3.05, 3.1),
                   sides=(-1, 1), axes=('side', 'top'), lateral_reference='nearest-face',
                   top_reference='bottom-face'):
    """Independent boxes. Range is the front face; offsets have an explicit datum.

    nearest-face: unsigned distance from the axis to the inward lateral face.
    center: unsigned distance from the axis to the centre of the cube.
    bottom-face: height of its bottom face above the rail heads (the top suite).
    Merely touching an envelope boundary has no positive-volume intersection.
    """
    out = []
    if any(not np.isfinite(v) or v < 0 for v in (*offsets, *top_offsets)):
        raise ValueError('boundary offsets must be finite and nonnegative')
    for size in sizes:
        if not np.isfinite(size) or size <= 0:
            raise ValueError('cube sizes must be finite and positive')
        for distance in ranges:
            if not 1. < distance <= 150.:
                raise ValueError('boundary-suite front-face ranges must be in (1, 150] m')
            for axis in axes:
                values = offsets if axis == 'side' else top_offsets
                for offset in values:
                    for side in (sides if axis == 'side' else (0,)):
                        u = side * (offset + size / 2 if lateral_reference == 'nearest-face' else offset)
                        bottom = 1. if axis == 'side' else (offset if top_reference == 'bottom-face' else offset - size / 2)
                        u_lo, u_hi = u - size / 2, u + size / 2
                        h_lo, h_hi = bottom, bottom + size
                        overlap_u = max(0., min(u_hi, 1.05) - max(u_lo, -1.05))
                        overlap_h = max(0., min(h_hi, 3.) - max(h_lo, .07))
                        positive = overlap_u > 1e-9 and overlap_h > 1e-9
                        touching = (abs(abs(u) - size / 2 - 1.05) < 1e-9 or abs(h_lo - 3.) < 1e-9)
                        out.append(dict(id=f'{axis}_size{size:g}_front{distance:g}_offset{offset:g}_side{side:+d}',
                                        axis=axis, size_m=size, initial_front_distance_m=distance,
                                        initial_center_distance_m=distance + size / 2,
                                        offset_m=offset, offset_reference=lateral_reference if axis == 'side' else top_reference,
                                        side=side, center_u_m=u, center_h_m=bottom + size / 2,
                                        u_bounds_m=[u_lo, u_hi], h_bounds_m=[h_lo, h_hi],
                                        inward_face_abs_u_m=max(0., abs(u) - size / 2),
                                        envelope_overlap_u_m=overlap_u, envelope_overlap_h_m=overlap_h,
                                        positive=positive, geometry='inside' if positive else ('touching' if touching else 'outside')))
    return out


def boundary_placement(case, frame, world, world_center):
    """Place the cube normal to the reconstructed track, including grade and roll.

    The reconstruction is an evaluation reference from the near rails, not GT.
    Unlike the legacy injection mode, the cube is not aligned with sensor XYZ.
    """
    s = world_center - frame['S']
    # The requested distance is arc length, not sensor x, including on grades.
    x_grid = np.arange(0., s + 1., .5)
    lateral_grid, vertical_grid, _ = world.offsets(frame['S'], x_grid)
    line = np.column_stack([x_grid, frame['c0'] + frame['c1'] * x_grid + lateral_grid,
                            frame['r0'] + frame['r1'] * x_grid + vertical_grid])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    x = float(np.interp(s, arc, x_grid))
    at = np.array([max(0., x - .25), x, x + .25])
    dl, dv, slopes = world.offsets(frame['S'], at)
    grade = frame['r1'] + (dv[2] - dv[0]) / (at[2] - at[0])
    heading = frame['c1'] + slopes[1]
    forward = np.array([1., heading, grade]); forward /= np.linalg.norm(forward)
    lateral = np.array([-heading, 1., 0.]); lateral /= np.linalg.norm(lateral)
    up = np.cross(forward, lateral)
    roll = np.arctan(frame['phi'])
    lateral, up = lateral * np.cos(roll) + up * np.sin(roll), up * np.cos(roll) - lateral * np.sin(roll)
    basis = np.column_stack([forward, lateral, up])
    rail_center = np.array([x, frame['c0'] + frame['c1'] * x + dl[1],
                            frame['r0'] + frame['r1'] * x + dv[1]])
    center = rail_center + case['center_u_m'] * lateral + case['center_h_m'] * up
    return center, (case['size_m'],) * 3, basis


def _matches_injected(candidate, added, tolerance=.12):
    """A candidate must contain an actual injected return, not just share its range."""
    if not len(added) or 'bbox' not in candidate:
        return False
    lo, hi = np.asarray(candidate['bbox'], float)
    return bool(np.any(np.all((added >= lo - tolerance) & (added <= hi + tolerance), axis=1)))


def _boundary_reference(source, start, stop, cache=None):
    """An optional frozen clean pass shared by baseline/final detector runs."""
    identity = dict(version=1, source=source, start=start, stop=stop)
    path = Path(cache) if cache else None
    if path is not None and path.exists():
        data = json.loads(path.read_text())
        if data.get('identity') != identity:
            raise ValueError('reference-cache source/window/version differs; choose a different cache path')
        return ({int(k): v for k, v in data['track'].items()},
                {int(k): v for k, v in data['kappa'].items()},
                {int(k): v for k, v in data['kappa_v'].items()}, True)
    track, kappa, kappa_v = track_pass(source, start, stop, observed_only=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(identity=identity, track=track, kappa=kappa, kappa_v=kappa_v),
                                   ensure_ascii=False, allow_nan=False))
    return track, kappa, kappa_v, False


def geometry_breakdown(cases):
    """Keep tangency visible instead of mixing it with truly exterior objects."""
    out = {}
    for geometry in ('inside', 'touching', 'outside'):
        rows = [c for c in cases if c['geometry'] == geometry]
        out[geometry] = dict(cases=len(rows), alarm_cases=sum(c['alarm_frames'] > 0 for c in rows),
                             alarm_frames=sum(c['alarm_frames'] for c in rows),
                             attention_frames=sum(c['attention_frames'] for c in rows),
                             unmatched_alarm_frames=sum(c['unmatched_alarm_frames'] for c in rows),
                             reference_extrapolated_cases=sum(c['reference_extrapolated'] for c in rows))
    return out


def boundary_suite(source, start, stop, cases, native=False, reference_stop=None, reference_cache=None):
    """Each box/range gets a fresh detector over the same clean input window.

    Report visible-return frames separately: a cube with no intersecting beam
    cannot be a detector miss. The legacy run() visible counter means placement.
    """
    reference_stop = stop if reference_stop is None else reference_stop
    if reference_stop < stop:
        raise ValueError('reference-stop must cover the full detector window')
    track, kappa, kappa_v, cache_reused = _boundary_reference(source, start, reference_stop, reference_cache)
    frames = sorted(i for i in track if start <= i < stop)
    if not frames:
        raise ValueError('no frames with rails in the requested window')
    world = WorldTrack(kappa, kappa_v)
    track = {i: world.anchored_frame(t) for i, t in track.items() if i < stop}
    print('REFERENCE', dict(start=start, stop=reference_stop, observed_world_s_min_m=float(world.S[0]),
                            observed_world_s_max_m=float(world.S[-1]), cache_reused=cache_reused,
                            near_fit_max_u_m=max(track[i]['reference_near_max_u_m'] for i in frames)), flush=True)
    if native:
        from metro.native import NativeTunnelDetector
        detector_type = NativeTunnelDetector
    else:
        detector_type = TunnelDetector
    # Short evaluation windows fit in memory. Reuse immutable clean clouds;
    # every case still gets its own detector and newly injected output array.
    cached_inputs = list(iter_frames(source, start, stop)) if stop - start <= 100 else None
    if cached_inputs is not None:
        for _, points, _ in cached_inputs:
            points.flags.writeable = False
    results = []
    for case in cases:
        det = detector_type(TunnelConfig())
        world_center = track[frames[0]]['S'] + case['initial_center_distance_m']
        world_front, world_back = world_center - case['size_m'] / 2, world_center + case['size_m'] / 2
        extrapolated = world_front < world.S[0] or world_back > world.S[-1]
        records, alarms, attention, fp = [], [], [], []
        for i, pts, stamp in cached_inputs if cached_inputs is not None else iter_frames(source, start, stop):
            if i not in track:
                continue
            front_distance = world_center - track[i]['S'] - case['size_m'] / 2
            placed = [boundary_placement(case, track[i], world, world_center)] if front_distance > 1. else []
            cloud, additions = inject(pts, placed, return_added=True)
            added = additions[0] if additions else np.empty((0, 3), np.float32)
            res = det.process(cloud, None, stamp)
            candidates = res.get('candidates', [])
            matched = [c for c in candidates if _matches_injected(c, added)]
            confirmed = [c for c in matched if c.get('confirmed')]
            attended = [c for c in matched if c.get('confirmed') or c.get('boundary') or c.get('far_attention')
                        or c.get('confident_points', 0) > 0 or c.get('appeared_frac', 0) > 0]
            unmatched = [c for c in candidates if c.get('confirmed') and not _matches_injected(c, added)]
            record = dict(frame=i, front_distance_m=float(front_distance), inserted_returns=len(added),
                          analysis_range_m=res.get('analysis_range_m'), alarm=bool(confirmed),
                          attention=bool(attended), unmatched_alarms=len(unmatched),
                          reference_near_max_u_m=track[i]['reference_near_max_u_m'],
                          reference_near_max_h_m=track[i]['reference_near_max_h_m'])
            if confirmed:
                nearest = min(c['distance_m'] for c in confirmed)
                record.update(distance_m=nearest, distance_error_m=nearest - front_distance)
                alarms.append(record)
            if attended:
                attention.append(record)
            if unmatched:
                fp.append(i)
            fields = ('distance_m', 'point_count', 'confident_points', 'u_range', 'h_range', 'margin_m',
                      'sigma_u', 'sigma_h', 'evidence', 'confirmed', 'boundary', 'unverified', 'wall_like',
                      'track_id', 'held', 'bbox', 'far_attention', 'appeared_frac')
            if matched:
                record['candidates'] = [{k: c[k] for k in fields if k in c} for c in matched]
            if unmatched:
                record['unmatched_candidates'] = [{k: c[k] for k in fields if k in c} for c in unmatched]
            records.append(record)
        errors = [r['distance_error_m'] for r in alarms]
        row = dict(case, reference_extrapolated=bool(extrapolated), object_world_s_bounds_m=[world_front, world_back],
                   reference_observed_world_s_min_m=float(world.S[0]), reference_observed_world_s_max_m=float(world.S[-1]),
                   processed_frames=len(records), visible_frames=sum(r['inserted_returns'] > 0 for r in records),
                   injected_returns=sum(r['inserted_returns'] for r in records), alarm_frames=len(alarms),
                   attention_frames=len(attention), unmatched_alarm_frames=len(fp),
                   first_alarm=alarms[0] if alarms else None, first_attention=attention[0] if attention else None,
                   max_alarm_front_distance_m=max((r['front_distance_m'] for r in alarms), default=None),
                   median_distance_error_m=float(np.median(errors)) if errors else None, frames=records)
        results.append(row)
        print(f"{case['id']} truth={case['geometry']} visible={row['visible_frames']} "
              f"alarms={row['alarm_frames']} first={row['first_alarm']['front_distance_m'] if alarms else None} "
              f"unmatched={len(fp)} reference={'extrapolated' if extrapolated else 'observed'}", flush=True)
    inside = [r for r in results if r['positive']]
    outside = [r for r in results if not r['positive']]
    summary = dict(cases=len(results), inside_cases=len(inside), outside_or_touching_cases=len(outside),
                   inside_cases_with_alarm=sum(r['alarm_frames'] > 0 for r in inside),
                   outside_alarm_frames=sum(r['alarm_frames'] for r in outside),
                   unmatched_alarm_frames=sum(r['unmatched_alarm_frames'] for r in results),
                   cases_without_returns=sum(r['visible_frames'] == 0 for r in results),
                   reference_extrapolated_cases=sum(r['reference_extrapolated'] for r in results),
                   outside_confirmed_reference_alarm_frames=sum(r['alarm_frames'] for r in outside if not r['reference_extrapolated']),
                   outside_extrapolated_reference_alarm_frames=sum(r['alarm_frames'] for r in outside if r['reference_extrapolated']))
    return dict(mode='boundary-suite', source=source, start=start, stop=stop,
                truth_reference='near-rail reconstruction by Python reference; not manually labelled ground truth',
                reference=dict(start=start, stop=reference_stop, cache=str(reference_cache) if reference_cache else None,
                               cache_reused=cache_reused, observed_world_s_min_m=float(world.S[0]),
                               observed_world_s_max_m=float(world.S[-1]),
                               near_fit_max_u_m=max(track[i]['reference_near_max_u_m'] for i in frames),
                               near_fit_max_h_m=max(track[i]['reference_near_max_h_m'] for i in frames),
                               coverage_note='Observed interval is curvature sampled from near rails, not survey ground truth; outside it curvature is extrapolated.'),
                sensor_model=dict(name='Pandar128 channel grid', azimuth_step_deg=AZ_STEP),
                range_reference='nearest front face along the reconstructed track',
                envelope=dict(half_width_m=1.05, bottom_m=.07, top_m=3.), summary=summary,
                summary_by_geometry=geometry_breakdown(results), cases=results)


def _numbers(value):
    return [float(v) for v in value.split(',')]


def boundary_main(argv):
    ap = argparse.ArgumentParser(description='Isolated boxes at the fixed 2.1 x 3.0 m envelope boundaries.')
    ap.add_argument('source')
    ap.add_argument('start', type=int)
    ap.add_argument('stop', type=int)
    ap.add_argument('--native', action='store_true')
    ap.add_argument('--save', required=True, help='JSON report, normally under output/')
    ap.add_argument('--sizes', type=_numbers, default=[.3, .5])
    ap.add_argument('--ranges', type=_numbers, default=[30., 60., 90., 120., 150.], help='initial nearest front-face distances in m')
    ap.add_argument('--offsets', type=_numbers, default=[.95, 1., 1.05, 1.1, 1.15], help='unsigned side offsets in m')
    ap.add_argument('--top-offsets', type=_numbers, default=[2.9, 2.95, 3., 3.05, 3.1], help='top-suite heights above the rail heads')
    ap.add_argument('--lateral-reference', choices=['nearest-face', 'center'], default='nearest-face')
    ap.add_argument('--top-reference', choices=['bottom-face', 'center'], default='bottom-face')
    ap.add_argument('--sides', type=_numbers, default=[-1, 1], help='-1,1 by default; use --sides=-1 for the right side')
    ap.add_argument('--axes', default='side,top', help='side,top or one of them')
    ap.add_argument('--case-limit', type=int, help='run only the first cases for a smoke check')
    ap.add_argument('--reference-stop', type=int, help='end of the separate clean near-rail pass; may extend beyond detector stop')
    ap.add_argument('--reference-cache', help='reuse/save a frozen clean reference JSON, normally under output/')
    ap.add_argument('--elevations', type=Path, help='optional user-provided .npy beam elevations in degrees')
    a = ap.parse_args(argv)
    if a.elevations:
        set_elevations(a.elevations)
    if a.stop <= a.start:
        ap.error('stop must be after start')
    axes = a.axes.split(',')
    if not axes or any(v not in ('side', 'top') for v in axes):
        ap.error('--axes must contain side and/or top')
    if any(v not in (-1, 1) for v in a.sides):
        ap.error('--sides must contain only -1 and/or 1')
    if a.case_limit is not None and a.case_limit < 1:
        ap.error('--case-limit must be positive')
    try:
        cases = boundary_cases(a.sizes, a.ranges, a.offsets, a.top_offsets, [int(v) for v in a.sides],
                               axes, a.lateral_reference, a.top_reference)
        out = boundary_suite(a.source, a.start, a.stop, cases[:a.case_limit], a.native,
                             a.reference_stop, a.reference_cache)
    except ValueError as exc:
        ap.error(str(exc))
    path = Path(a.save)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False))
    print(json.dumps(out['summary'], ensure_ascii=False))


if __name__ == '__main__':
    if sys.argv[1:2] == ['boundary-suite']:
        boundary_main(sys.argv[2:])
        raise SystemExit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    ap.add_argument('start', type=int)
    ap.add_argument('stop', type=int)
    ap.add_argument('--save')
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--first', type=float, default=250.)
    ap.add_argument('--spacing', type=float, default=130.)
    ap.add_argument('--trace')
    ap.add_argument('--native', action='store_true', help='run the C++ core instead of the Python reference')
    ap.add_argument('--elevations', type=Path, help='optional user-provided .npy beam elevations in degrees')
    a = ap.parse_args()
    if a.elevations:
        set_elevations(a.elevations)
    out = run(a.source, a.start, a.stop, a.spacing, a.first, a.offset, a.trace, a.native)
    for o in out['objects']:
        print(f"{o['name']:16s} pos={str(o['positive'])[0]}  first={o['first_range']}  alarms={o['alarms']}  visible_frames={o['visible']}")
    print('unmatched alarm frames', out['fp_frames'])
    if a.save:
        Path(a.save).write_text(json.dumps(out, ensure_ascii=False, indent=1))
