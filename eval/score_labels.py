"""Score detections against user-supplied approximate labels.

Provide --labels JSON with a sequences object. Each sequence lists objects with
visible_interval_s, side, label and optional approx_distance_start_end_m.
Approximate matches should be checked visually.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]

SIDE_Y = .3      # m: a box centred more than this to the other side of the lidar is not on the labelled side
CEILING_H = 1.8  # m above the rail heads: a box reaching higher is at the ceiling side
TOL_S = 1.       # s: tolerance of the hand-made interval ends


def side_ok(side, c):
    y = c['center'][1]  # lidar frame, + left
    if side == 'right':
        return y < SIDE_Y
    if side == 'left':
        return y > -SIDE_Y
    if side == 'ceiling':
        return c['h_range'][1] > CEILING_H
    return True  # on_rails, across_most_of_track, centre


def distance_error(o, t, d):
    """Normalised disagreement with the rough distances; None when they rule the box out."""
    d0, d1 = o.get('approx_distance_start_end_m') or (None, None)
    t0, t1 = o['visible_interval_s']
    if d0 is None:
        return .5
    if d1 is None:  # the train may stop: only an upper bound
        return .5 if d <= 1.25 * d0 + 10 else None
    expected = d0 + (d1 - d0) * min(max((t - t0) / max(t1 - t0, 1e-6), 0.), 1.)
    err = abs(d - expected) / max(20., .3 * expected)
    return err if err <= 1. else None


def frames_live(run, overrides):
    from metro.tunnel import TunnelConfig
    from metro.native import NativeTunnelDetector
    from stability import frames_of
    det = NativeTunnelDetector(TunnelConfig(), overrides)
    for i, pts, t in frames_of(run):
        r = det.process(pts, None, t)
        yield i, t, r


def frames_saved(run):
    root = ROOT / 'viewer' / 'data' / run
    times = {f['frame']: f['time_s'] for f in json.loads((root / 'manifest.json').read_text())['frames']}
    for r in json.loads((root / 'model.tunnel.json').read_text())['frames']:
        yield r['frame'], times.get(r['frame'], r.get('time_s', 0.)), r


def score_run(run, spec, frames):
    objs = spec['objects']
    stats = {o['id']: dict(label=o['label'], side=o['side'], interval=o['visible_interval_s'], frames=0,
                           alarm_frames=0, attention_frames=0, first_alarm=None) for o in objs}
    unmatched, n = [], 0
    for i, t, r in frames:
        n += 1
        live = [o for o in objs if o['visible_interval_s'][0] - TOL_S <= t <= o['visible_interval_s'][1] + TOL_S]
        inner = [o for o in objs if o['visible_interval_s'][0] <= t <= o['visible_interval_s'][1]]
        for o in inner:
            stats[o['id']]['frames'] += 1
        alarm_now, attention_now = set(), set()
        for c in r.get('candidates', []):
            scored = [(distance_error(o, t, c['distance_m']), o) for o in live if side_ok(o['side'], c)]
            scored = [(e, o) for e, o in scored if e is not None]
            best = min(scored, key=lambda x: x[0])[1] if scored else None
            if c.get('confirmed'):
                if best is None:
                    unmatched.append((i, round(t, 1), round(c['distance_m'], 1), [round(v, 2) for v in c['center']],
                                      [round(v, 2) for v in c['h_range']]))
                    continue
                s = stats[best['id']]
                if best in inner:
                    alarm_now.add(best['id'])
                if s['first_alarm'] is None:
                    s['first_alarm'] = (i, round(t, 1), round(c['distance_m'], 1))
            elif best is not None and best in inner and (c.get('confident_points', 0) > 0 or c.get('boundary')
                                                         or c.get('far_attention')):
                attention_now.add(best['id'])
        for oid in alarm_now:
            stats[oid]['alarm_frames'] += 1
        for oid in attention_now - alarm_now:
            stats[oid]['attention_frames'] += 1
    # an unmatched box may repeat for several frames: count episodes as well
    episodes, last = 0, -99
    for f in sorted({u[0] for u in unmatched}):
        episodes += f - last > 5
        last = f
    return dict(frames=n, objects=stats, unmatched_alarm_frames=len({u[0] for u in unmatched}),
                unmatched_episodes=episodes, unmatched=unmatched[:30])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', type=Path, required=True)
    ap.add_argument('--live', action='store_true', help='run the C++ core instead of reading viewer results')
    ap.add_argument('--config', default='{}')
    ap.add_argument('--runs', nargs='*')
    ap.add_argument('--save')
    a = ap.parse_args()
    labels = json.loads(a.labels.read_text(encoding='utf-8'))['sequences']
    overrides = json.loads(a.config)
    out, total = {}, dict(targets=0, detected=0, unmatched_alarm_frames=0, unmatched_episodes=0)
    for run in a.runs or list(labels):
        frames = frames_live(run, overrides) if a.live else frames_saved(run)
        res = score_run(run, labels[run], frames)
        out[run] = res
        print(f"== {run}: {res['frames']} frames")
        for oid, s in res['objects'].items():
            share = s['alarm_frames'] / s['frames'] if s['frames'] else 0.
            att = s['attention_frames'] / s['frames'] if s['frames'] else 0.
            mark = '' if s['label'] == 'target_by_user' else '  (спорный, вне итога)'
            print(f"  {oid:11s} {s['side']:21s} {s['interval']}  тревога {share:5.0%} кадров, ВНИМАНИЕ {att:5.0%}, "
                  f"первая тревога {s['first_alarm']}{mark}")
            if s['label'] == 'target_by_user':
                total['targets'] += 1
                total['detected'] += s['first_alarm'] is not None
        print(f"  тревог без размеченного предмета: {res['unmatched_alarm_frames']} кадров, "
              f"{res['unmatched_episodes']} эпизодов {res['unmatched'][:6]}")
        total['unmatched_alarm_frames'] += res['unmatched_alarm_frames']
        total['unmatched_episodes'] += res['unmatched_episodes']
    print('ИТОГ', total)
    if a.save:
        Path(a.save).write_text(json.dumps(dict(total=total, runs=out), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
