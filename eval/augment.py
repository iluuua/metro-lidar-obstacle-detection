"""Synthetic obstacles ray-cast into recordings (eval/inject.py): a test of the core on objects whose truth is known.

Pass 1 replays the recording through the core and keeps, per frame, the travelled distance (ego_s, kept continuous
over its restarts), the near rail-supported axis (5-30 m) and the cant; the world track (curvature along the line) is
built from those near observations, so an object far ahead is placed by the curve that the train later sees from close.
Pass 2 places static boxes along the line (from a 0.3 m cube to a standing person), ray-casts them on the Pandar128 grid
with occlusion by the tunnel and by each other (the tunnel itself is not changed), replays the core on the modified
clouds and writes every candidate with the truth of the box it holds and, per frame, the status of the core.

What the truth means. `depth_m` is how far the box is inside the 2.1 x 3.0 m envelope on the limiting axis (negative:
outside). The inside / outside verdict has a dead band of 5 cm, and the gap to the envelope is drawn from bins that
cover everything from 0.8 m inside to 1 m outside (it is the result by depth bin that matters: an object 5 cm inside at
80 m cannot be told from one 5 cm outside, an object 40 cm inside can). On a recording without obstacles every other
candidate is track equipment and counts as a false alarm if the core confirms it.

Returns are ideal rays by default (`--degrade 0`); `--degrade 1` and `2` drop a share of them (growing with range) and
add range noise, so the result can be bracketed: the real count of returns on an object is not known.

Perturbations that a single replay hides: `--offset N` starts the core at frame N (cold start), `--stride K` feeds every
K-th frame, `--dll` runs another core on the same clouds (`--ref-dll` fixes the core of pass 1, so two cores see exactly
the same objects).

    python eval/augment.py clean_sample --seed 1
    python eval/augment.py sample@1000:3000 --seed 2 --dll core/build/libmetro_core.so

Output: one JSON line per object (kind=object), per candidate (kind=row), per frame (kind=frame) and per frame in which
a live object has no candidate (kind=miss). Summary: eval/augment_eval.py.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]

HALF, TOP, FLOOR = 1.05, 3., .07
DEAD_BAND = .05
FIELDS = ('distance_m', 'point_count', 'unique_point_count', 'confident_points', 'unique_confident_points', 'margin_m',
          'sigma_u', 'sigma_h', 'evidence', 'evidence_need', 'appeared_frac', 's_max', 'path_lateral_m', 'hits',
          'held_frames', 'confirmed', 'attention', 'boundary', 'far_attention', 'held', 'wall_like', 'unverified',
          'edge_confirmed', 'soft_confirmed', 'decision_rule', 'track_id', 'u_range', 'h_range', 'bbox')
# inner face of the box beyond the side of the envelope, m (negative: inside), with the probability of each bin
GAP_BINS = [((-.8, -.4), .20), ((-.4, -.2), .20), ((-.2, -.1), .12), ((-.1, -.05), .04), ((.05, .1), .04),
            ((.1, .2), .10), ((.2, .5), .15), ((.5, 1.), .15)]
DEGRADE = {0: (0., 0., 0.), 1: (.10, .20, .02), 2: (.30, .30, .03)}  # dropout at 0 m, added by 150 m, range noise (m)


def frames(run):
    """Settings and a source of frames that can be iterated again (eval/frame_source.frames)."""
    from frame_source import frames as source
    return source(run)[0], lambda: source(run)[1]


def reference(run, overrides):
    """Per frame: ego_s, near axis (x, y, z at 5-30 m within the rail support), cant; curvature samples."""
    from metro.native import NativeTunnelDetector
    cfg, source = frames(run)
    det = NativeTunnelDetector(cfg, overrides)
    track, kappa, kappa_v = {}, {}, {}
    S, last_ego = 0., None  # the core restarts ego_s after a loss of the track: the distance is kept continuous
    for i, pts, t in source():
        r = det.process(pts, None, t)
        info = r.get('path_info') or {}
        ego = info.get('ego_s')
        if ego is not None and last_ego is not None and ego >= last_ego - .01:
            S += ego - last_ego
        else:
            S += float(info.get('odometry_ds') or 0.)
        last_ego = ego
        path = np.asarray(r.get('path') or [], float)
        rail = float(info.get('rail_support_m') or 0.)
        if len(path) < 5 or rail < 12. or ego is None:
            continue
        x = np.arange(5., min(30., rail) + .01, 1.)
        y, z = np.interp(x, path[:, 0], path[:, 1]), np.interp(x, path[:, 0], path[:, 2])
        c, q = np.polyfit(x, y, 2), np.polyfit(x, z, 2)
        track[i] = dict(S=S, c0=float(c[2]), c1=float(c[1]), r0=float(q[2]), r1=float(q[1]), phi=float(info.get('phi') or 0.),
                        reference_x=x.tolist(), reference_y=y.tolist(), reference_z=z.tolist())
        for xs in (10., 20., 30.):
            if xs <= rail:
                key = int(round(S + xs))
                kappa.setdefault(key, []).append(2 * c[0])
                kappa_v.setdefault(key, []).append(2 * q[0])
    return track, kappa, kappa_v


def reference_cached(run, ref_dll):
    """Pass 1 of the core in use, kept in output/augment/ref/. With --ref-dll the pass 1 of that core is read (it must have
    been run before: a process loads one core library), so two cores are tested on exactly the same objects."""
    stem = Path(ref_dll).stem if ref_dll else Path(os.environ.get('METRO_CORE_LIB', 'default')).stem
    path = ROOT / 'output' / 'augment' / 'ref' / f"{run.replace('@', '_').replace(':', '-')}__{stem}.pkl"
    if path.is_file():
        with path.open('rb') as f:
            return pickle.load(f)
    if ref_dll:
        raise SystemExit(f'no pass 1 of {ref_dll} for {run} yet: run eval/augment.py with --dll {ref_dll} first')
    got = reference(run, {})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(got, f)
    return got


def placement(obj, frame, world):
    """Box centre, size and axes, normal to the reconstructed track with grade and cant (inject.boundary_placement)."""
    s = obj['S'] - frame['S']
    x_grid = np.arange(0., s + 1., .5)
    lat, vert, _ = world.offsets(frame['S'], x_grid)
    line = np.column_stack([x_grid, frame['c0'] + frame['c1'] * x_grid + lat, frame['r0'] + frame['r1'] * x_grid + vert])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    x = float(np.interp(s, arc, x_grid))
    at = np.array([max(0., x - .25), x, x + .25])
    dl, dv, slopes = world.offsets(frame['S'], at)
    grade = frame['r1'] + (dv[2] - dv[0]) / max(at[2] - at[0], 1e-6)
    heading = frame['c1'] + slopes[1]
    forward = np.array([1., heading, grade]); forward /= np.linalg.norm(forward)
    lateral = np.array([-heading, 1., 0.]); lateral /= np.linalg.norm(lateral)
    up = np.cross(forward, lateral)
    roll = np.arctan(frame['phi'])
    lateral, up = lateral * np.cos(roll) + up * np.sin(roll), up * np.cos(roll) - lateral * np.sin(roll)
    rail_center = np.array([x, frame['c0'] + frame['c1'] * x + dl[1], frame['r0'] + frame['r1'] * x + dv[1]])
    center = rail_center + obj['u'] * lateral + obj['h'] * up
    return center, tuple(obj['size']), np.column_stack([forward, lateral, up])


def scenario(rng, k, S):
    kind = rng.choice(['cube', 'person', 'low'], p=[.55, .30, .15])
    if kind == 'cube':
        size = (float(rng.choice([.3, .5, .8])),) * 3
    elif kind == 'person':
        size = (.4, .5, 1.7)
    else:
        size = (.3, 1., .2)
    bins, p = zip(*GAP_BINS)
    lo, hi = bins[int(rng.choice(len(bins), p=np.array(p) / sum(p)))]
    gap = float(rng.uniform(lo, hi))  # inner face beyond the side
    side = float(rng.choice([-1., 1.]))
    floor = rng.random() < .7 or kind == 'person'
    bottom = float(rng.uniform(-.2, .1) if floor else rng.uniform(.3, 1.5))
    u = side * (HALF + gap + size[1] / 2)
    u_lo, u_hi = u - size[1] / 2, u + size[1] / 2
    h_lo, h_hi = bottom, bottom + size[2]
    over_u = min(u_hi, HALF) - max(u_lo, -HALF)
    over_h = min(h_hi, TOP) - max(h_lo, FLOOR)
    depth = min(over_u, over_h)  # > 0: inside by this much; < 0: outside by this much (on the limiting axis)
    truth = 'inside' if depth > DEAD_BAND else ('outside' if depth < -DEAD_BAND else 'boundary')
    return dict(id=k, S=S, shape=kind, size=size, gap_m=gap, side=side, u=u, h=bottom + size[2] / 2,
                u_range=[u_lo, u_hi], h_range=[h_lo, h_hi], depth_m=depth, truth=truth)


def degrade(hits, rng, level):
    """Fewer returns with range and a little range noise (the share of returns on a real object is not known)."""
    p0, p1, sigma = DEGRADE[level]
    if level == 0 or not len(hits):
        return hits
    r = np.linalg.norm(hits, axis=1)
    keep = rng.random(len(hits)) >= p0 + p1 * np.minimum(r / 150., 1.)
    hits, r = hits[keep], r[keep]
    return hits * (1. + rng.normal(0., sigma, len(hits)) / np.maximum(r, 1.))[:, None]


def inject(pts, objects, rng=None, level=0):
    """inject.inject with the occlusion test vectorised: object returns hidden by a nearer real return, or by a nearer
    object, in the same beam cell are dropped; real returns behind an object are removed."""
    from inject import AZ_STEP, ELEV, box_hits
    rng = rng or np.random.default_rng(0)
    dist = np.linalg.norm(pts, axis=1)
    az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    el = np.degrees(np.arcsin(pts[:, 2] / np.maximum(dist, 1e-6)))
    cell = np.floor(az / AZ_STEP).astype(np.int64) * 1000 + np.searchsorted(ELEV, el)
    keep = np.ones(len(pts), bool)
    add, added = [], [np.empty((0, 3), np.float32)] * len(objects)
    nearest_object = {}  # beam cell -> range of the nearest object return so far
    order = sorted(range(len(objects)), key=lambda k: np.linalg.norm(objects[k][0]))
    for k in order:
        center, size, axes = objects[k]
        hits, fp = box_hits(center, size, axes=axes)
        if fp is None or len(hits) == 0:
            continue
        a0, a1, e0, e1, rmin = fp
        hr = np.linalg.norm(hits, axis=1)
        hcell = (np.floor(np.degrees(np.arctan2(hits[:, 1], hits[:, 0])) / AZ_STEP).astype(np.int64) * 1000
                 + np.searchsorted(ELEV, np.degrees(np.arcsin(hits[:, 2] / hr))))
        near = (az >= a0 - .2) & (az <= a1 + .2) & (el >= e0 - .3) & (el <= e1 + .3) & (dist < hr.max())
        blocked = np.zeros(len(hits), bool)
        if near.any():
            keys, r = cell[near], dist[near]
            srt = np.lexsort((r, keys))
            uniq, first = np.unique(keys[srt], return_index=True)
            nearest = r[srt][first]
            pos = np.clip(np.searchsorted(uniq, hcell), 0, len(uniq) - 1)
            blocked = (uniq[pos] == hcell) & (nearest[pos] < hr - .3)
        blocked |= np.array([nearest_object.get(int(c), 1e9) < r - .3 for c, r in zip(hcell, hr)], bool)
        hits, hr, hcell = hits[~blocked], hr[~blocked], hcell[~blocked]
        for c, r in zip(hcell, hr):
            nearest_object[int(c)] = min(nearest_object.get(int(c), 1e9), float(r))
        hits = degrade(hits, rng, level).astype(np.float32)
        added[k] = hits
        if len(hits):
            keep &= ~((az >= a0) & (az <= a1) & (el >= e0) & (el <= e1) & (dist > rmin))
            add.append(hits)
    return (np.concatenate([pts[keep]] + add) if add else pts[keep]), added


def contains(c, added, tol=.12):
    if not len(added) or not c.get('bbox'):
        return False
    lo, hi = np.asarray(c['bbox'], float)
    return bool(np.any(np.all((added >= lo - tol) & (added <= hi + tol), axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--clean', action='store_true', help='source is known to contain no obstacles')
    ap.add_argument('--elevations', type=Path, help='optional user-provided .npy beam elevations in degrees')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--spacing', type=float, default=70.)
    ap.add_argument('--first', type=float, default=200.)
    ap.add_argument('--out')
    ap.add_argument('--dll', help='core of pass 2 (default: $METRO_CORE_LIB)')
    ap.add_argument('--ref-dll', help='core of pass 1, which fixes where the objects stand (default: --dll)')
    ap.add_argument('--config', default='{}', help='core settings of pass 2 (JSON)')
    ap.add_argument('--drop', default='', help='settings the core of pass 2 does not know (an older library), comma-separated')
    ap.add_argument('--degrade', type=int, default=0, choices=sorted(DEGRADE))
    ap.add_argument('--offset', type=int, default=0, help='the core of pass 2 starts at this frame (cold start)')
    ap.add_argument('--stride', type=int, default=1, help='the core of pass 2 gets every K-th frame')
    a = ap.parse_args()
    if a.elevations:
        from inject import set_elevations
        set_elevations(a.elevations)
    overrides = json.loads(a.config)

    def use(dll):
        if dll:
            path = Path(dll)
            os.environ['METRO_CORE_LIB'] = str(path if path.is_absolute() or path.exists() else ROOT / 'core' / 'build' / dll)
    use(a.dll)
    from metro.native import NativeTunnelDetector
    started = time.time()
    rng = np.random.default_rng(a.seed)
    track, kappa, kappa_v = reference_cached(a.run, a.ref_dll)
    if not track:
        raise SystemExit('no frames with rails')
    from inject import WorldTrack
    world = WorldTrack(kappa, kappa_v)
    track = {i: world.anchored_frame(t) for i, t in track.items()}
    ids = sorted(track)
    S0, S1 = track[ids[0]]['S'], track[ids[-1]]['S']
    first = a.first + rng.uniform(0., a.spacing)
    objects = [scenario(rng, k, S0 + first + k * a.spacing) for k in range(int((S1 - 5. - S0 - first) / a.spacing) + 1)]
    objects = [o for o in objects if o['S'] < S1 - 5.]
    out = Path(a.out or ROOT / 'datasets' / 'augment' / f"{a.run.replace('@', '_').replace(':', '-')}_s{a.seed}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg, source = frames(a.run)
    if a.drop:
        settings = {k: v for k, v in cfg.as_dict().items() if k not in a.drop.split(',')}

        class Settings:
            def as_dict(self):
                return dict(settings)
        cfg = Settings()
    det = NativeTunnelDetector(cfg, overrides)
    clean = a.clean
    by_id, track_owner = {o['id']: o for o in objects}, {}
    n_rows = 0
    with out.open('w', encoding='utf-8') as f:
        for o in objects:
            f.write(json.dumps(dict(kind='object', run=a.run, seed=a.seed, degrade=a.degrade, **o)) + '\n')
        for i, pts, t in source():
            if i < a.offset or (i - a.offset) % a.stride:
                continue
            fr = track.get(i)
            live = []
            if fr is not None:
                for o in objects:
                    front = o['S'] - fr['S'] - o['size'][0] / 2
                    if 1. < front < 300.:
                        live.append((o, front))
            placed = [placement(o, fr, world) for o, _ in live]
            cloud, added = inject(pts, placed, np.random.default_rng([a.seed, i]), a.degrade) if placed else (pts, [])
            r = det.process(cloud, None, t)
            seen, owners, foreign = set(), set(), 0
            for c in r.get('candidates', []):
                owner = [k for k, (o, _) in enumerate(live) if contains(c, added[k])]
                tid = c.get('track_id')
                if owner and tid is not None:
                    track_owner[tid] = live[owner[0]][0]['id']
                if owner:
                    o, front = live[owner[0]]
                    seen.add(o['id'])
                    label = o['truth']
                    if c.get('confirmed'):
                        owners.add(o['id'])
                elif tid in track_owner:
                    # a held box of a track that held the object's returns: it is the object even when its returns are hidden now
                    o = by_id[track_owner[tid]]
                    front = next((fr_m for oo, fr_m in live if oo['id'] == o['id']), None)
                    label = o['truth']
                    if c.get('confirmed'):
                        owners.add(o['id'])
                elif clean:
                    o, front, label = None, None, 'structure'
                    foreign += bool(c.get('confirmed'))
                else:
                    continue
                row = dict(kind='row', run=a.run, seed=a.seed, frame=i, time_s=t, object=o['id'] if o else None,
                           front_m=front, label=label, status=r['status'],
                           **{k: c.get(k) for k in FIELDS if k in c})
                f.write(json.dumps(row) + '\n')
                n_rows += 1
            f.write(json.dumps(dict(kind='frame', run=a.run, seed=a.seed, frame=i, status=r['status'], clean=clean,
                                    distance_m=r.get('distance_m'), analysis_m=r.get('analysis_range_m'),
                                    support_m=(r.get('path_info') or {}).get('support_m'), break_m=(r.get('path_info') or {}).get('break_m'),
                                    extension=(r.get('path_info') or {}).get('extension_ok'),
                                    live=[o['id'] for o, _ in live], owners=sorted(owners), foreign=foreign)) + '\n')
            for k, (o, front) in enumerate(live):
                if o['id'] not in seen:
                    f.write(json.dumps(dict(kind='miss', run=a.run, seed=a.seed, frame=i, object=o['id'], front_m=front,
                                            returns=int(len(added[k])) if k < len(added) else 0,
                                            analysis_m=r.get('analysis_range_m'))) + '\n')
    print(f"{a.run} seed {a.seed}: {len(objects)} objects ({sum(o['truth'] == 'inside' for o in objects)} inside), "
          f"{n_rows} rows, {time.time() - started:.0f}s -> {out}", flush=True)


if __name__ == '__main__':
    main()
