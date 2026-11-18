"""Compare detections with user-supplied automatic reference samples.

Exact samples and optional interpolation are reported separately. The reference
JSON contains rows of [frame_index, distance_m]. An automatic reference remains
a proxy and cannot establish correctness outside its coverage.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]
from metro.tunnel import TunnelConfig  # noqa: E402
from metro.native import NativeTunnelDetector  # noqa: E402
from stability import frames_of, run_config  # noqa: E402

BANDS = ((0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 320))
MATCH_M = 8.  # a candidate within this distance of the expected person position belongs to the person


def reference(path):
    rows = np.asarray(json.loads(Path(path).read_text(encoding='utf-8'))['rows'], float)
    return rows[:, 0], rows[:, 1]


def reference_distance(frame, frames, distances, interpolate=False):
    """Expected range only where this reference mode provides a position."""
    if not len(frames) or frame < frames[0] or frame > frames[-1]:
        return None
    if interpolate:
        return float(np.interp(frame, frames, distances))
    pos = int(np.searchsorted(frames, frame))
    return float(distances[pos]) if pos < len(frames) and frames[pos] == frame else None


def classify_reference(result, expected):
    """Pure matching helper; 'foreign' means unmatched by the proxy, not a proven FP."""
    has_alarm = result.get('status') == 'obstacle'
    if expected is None:
        return dict(near=[], attention=False, alarm=False, foreign=False,
                    unscored=has_alarm)
    candidates = result.get('candidates', [])
    near = [c for c in candidates if abs(c['distance_m'] - expected) < MATCH_M]
    attention = any(c.get('confident_points', 0) > 0 or c.get('boundary') or
                    c.get('far_attention') or c.get('confirmed') for c in near)
    alarm = has_alarm and any(c.get('confirmed') for c in near)
    unmatched = any(c.get('confirmed') and abs(c['distance_m'] - expected) >= MATCH_M
                    for c in candidates)
    return dict(near=near, attention=attention, alarm=alarm,
                foreign=has_alarm and (unmatched or not alarm), unscored=False)


def compact_candidate(c):
    row = dict(d=round(c['distance_m'], 1), n=c.get('point_count', 0),
               conf=c.get('confident_points', 0),
               u=[round(v, 2) for v in c.get('u_range', [])],
               h=[round(v, 2) for v in c.get('h_range', [])],
               margin=round(c.get('margin_m', 0.), 2), sigma_u=round(c.get('sigma_u', 0.), 2),
               ev=round(c.get('evidence', 0.), 1), hits=c.get('hits', 0),
               explain=c.get('explain'),
               flags=[k for k in ('confirmed', 'boundary', 'far_attention', 'unverified', 'wall_like', 'held')
                      if c.get(k)])
    if 'bbox' in c:
        row['bbox'] = [[round(v, 3) for v in corner] for corner in c['bbox']]
    return row


def update_bands(stats, expected, match):
    if expected is None:
        return
    for band in BANDS:
        if band[0] <= expected < band[1]:
            s = stats[band]
            s['frames'] += 1
            s['attention'] += match['attention']
            s['alarm'] += match['alarm']


def print_bands(stats):
    for lo, hi in BANDS:
        s = stats[(lo, hi)]
        if s['frames']:
            print(f"{lo:3d}-{hi:3d} m: frames {s['frames']:4d}  attention {s['attention'] / s['frames']:6.1%}  "
                  f"alarm {s['alarm'] / s['frames']:6.1%}")


def make(overrides, run):
    return NativeTunnelDetector(run_config(run), overrides)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--reference', type=Path, required=True, help='JSON with rows of [frame, distance_m]')
    ap.add_argument('--config', default='{}')
    ap.add_argument('--save')
    ap.add_argument('--stop', type=int, help='exclusive frame limit for a diagnostic run; default is the complete recording')
    a = ap.parse_args()
    rf, rx = reference(a.reference)
    det = make(json.loads(a.config), a.run)
    stats = {b: dict(frames=0, attention=0, alarm=0) for b in BANDS}
    exact_stats = {b: dict(frames=0, attention=0, alarm=0) for b in BANDS}
    first_attention = first_alarm = None
    exact_first_attention = exact_first_alarm = None
    foreign, unscored, exact_foreign, exact_unscored, rows = [], [], [], [], []
    exact_frames = 0
    for i, pts, t in frames_of(a.run):
        if a.stop is not None and i >= a.stop:
            break
        r = det.process(pts, None, t)
        x = reference_distance(i, rf, rx, interpolate=True)
        exact_x = reference_distance(i, rf, rx)
        match = classify_reference(r, x)
        exact = classify_reference(r, exact_x)
        # Preserve the historical interpolated foreign rule for comparisons:
        # a frame with any matched alarm is not foreign, even if another alarm
        # is unmatched. The exact view additionally counts such mixed frames.
        if x is not None and r['status'] == 'obstacle' and not match['alarm']:
            foreign.append((i, round(r['distance_m'], 1), round(x, 1)))
        if match['unscored']:
            unscored.append((i, round(r['distance_m'], 1)))
        if exact['foreign']:
            exact_foreign.append((i, round(r['distance_m'], 1), round(exact_x, 1)))
        if exact['unscored']:
            exact_unscored.append((i, round(r['distance_m'], 1)))
        exact_frames += exact_x is not None
        update_bands(stats, x, match)
        update_bands(exact_stats, exact_x, exact)
        if match['attention'] and first_attention is None:
            first_attention = (i, round(x, 1))
        if match['alarm'] and first_alarm is None:
            first_alarm = (i, round(x, 1))
        if exact['attention'] and exact_first_attention is None:
            exact_first_attention = (i, round(exact_x, 1))
        if exact['alarm'] and exact_first_alarm is None:
            exact_first_alarm = (i, round(exact_x, 1))
        rows.append(dict(frame=i, expected_m=round(x, 2) if x is not None else None,
                         exact_expected_m=round(exact_x, 2) if exact_x is not None else None,
                         status=r['status'],
                         support_m=r.get('path_info', {}).get('support_m'),
                         geometry=r.get('path_info', {}).get('explain'),
                         envelope_m=round(r.get('envelope_range_m') or 0., 1),
                         verified_m=round(r.get('analysis_range_m') or 0., 1),
                         person=[compact_candidate(c) for c in match['near']],
                         exact_person=[compact_candidate(c) for c in exact['near']],
                         confirmed=[compact_candidate(c) for c in r.get('candidates', []) if c.get('confirmed')]))
    print('interpolated automatic proxy, historical comparison only:')
    print('first attention on the person (frame, expected m):', first_attention)
    print('first alarm on the person (frame, expected m):', first_alarm)
    print_bands(stats)
    print('alarm frames unmatched by interpolated proxy:', len(foreign), foreign[:10])
    print('unscored alarms outside interpolation coverage:', len(unscored), unscored[:10])
    print(f'exact automatic proxy samples, not manual ground truth: {exact_frames}/{len(rf)} frames')
    print('first attention on an exact sample (frame, expected m):', exact_first_attention)
    print('first alarm on an exact sample (frame, expected m):', exact_first_alarm)
    print_bands(exact_stats)
    print('alarm frames with a proxy-unmatched confirmed candidate:', len(exact_foreign), exact_foreign[:10])
    print('unscored alarms without an exact reference sample:', len(exact_unscored), exact_unscored[:10])
    if a.save:
        Path(a.save).write_text(json.dumps(dict(first_attention=first_attention, first_alarm=first_alarm,
                                                bands={f'{lo}-{hi}': v for (lo, hi), v in stats.items()},
                                                foreign=foreign, unscored_alarm_frames=unscored,
                                                reference=dict(kind='automatic_proxy_not_ground_truth',
                                                               samples=len(rf),
                                                               max_gap_frames=int(max(np.diff(rf), default=0)),
                                                               legacy_interpolates_missing_frames=True),
                                                exact_reference=dict(frames=exact_frames,
                                                    first_attention=exact_first_attention, first_alarm=exact_first_alarm,
                                                    bands={f'{lo}-{hi}': v for (lo, hi), v in exact_stats.items()},
                                                    foreign=exact_foreign, unscored_alarm_frames=exact_unscored,
                                                    foreign_rule='any_unmatched_confirmed_candidate_on_exact_sample'),
                                                frames=rows), ensure_ascii=False))


if __name__ == '__main__':
    main()
