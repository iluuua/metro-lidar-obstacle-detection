"""Track labels for the recordings: every cloud the core looked at, followed from far to near.

The core's tracker already groups the returns of one object over frames (track_id). For every candidate
of every frame the core gives the box, the range along the track, the position against the axis and the
axis uncertainty. Far away the axis is uncertain and the envelope floats, so a track is judged where it is
close (the rails are seen, the axis error is a few cm): its points enter the envelope deeper than the error
-> "enters", stay outside -> "outside", in between -> "boundary". The verdict is carried to every frame of
the track. Tracks that never come close are "unresolved". On the synthetic recordings the track is also
matched with the organizers' object markers (target, dust, exterior cube, disputed cube).

    .venv/bin/python eval/label_tracks.py                    # all recordings with a full viewer export
    .venv/bin/python eval/label_tracks.py --runs labelled_sample falcon

Writes datasets/labels/<run>.json (per-frame track rows and per-track verdicts) and
datasets/labels/review.md (the tracks a person or an agent should look at, with viewer links).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]

NEAR_M = 50.        # the verdict is taken from frames closer than this
MIN_NEAR_FRAMES = 3
HALF_WIDTH = 1.05   # m, the organizers' envelope 2.1 x 3.0 m
HEIGHT = 3.0
MIN_HEIGHT = .07
MARKER_PAD = .6


def depth(c):
    """How deep the cluster enters the nominal envelope, m (negative: outside), from its box against the axis."""
    u_lo, u_hi = c['u_range']
    h_lo, h_hi = c['h_range']
    lateral = HALF_WIDTH - min(abs(u_lo), abs(u_hi)) if u_lo * u_hi > 0 else HALF_WIDTH  # straddles the axis
    vertical = min(HEIGHT - h_lo, h_hi - MIN_HEIGHT)
    return min(lateral, vertical)


def overlap(box, part, pad=MARKER_PAD):
    (lo, hi), (glo, ghi) = box, part
    return all(hi[k] >= glo[k] - pad and lo[k] <= ghi[k] + pad for k in range(3))


def run_core(run):
    from metro.native import NativeTunnelDetector
    from stability import frames_of, run_config
    det = NativeTunnelDetector(run_config(run), {'explain': True})
    for i, pts, t in frames_of(run):
        yield i, t, det.process(pts, None, t)


def collect(run):
    gt_path = ROOT / 'viewer' / 'data' / run / 'ground-truth.json'
    gt = {f['frame']: f['objects'] for f in json.loads(gt_path.read_text())['frames']} if gt_path.is_file() else {}
    rows, frames = [], 0
    for i, t, r in run_core(run):
        frames += 1
        info = r.get('path_info') or {}
        for c in r.get('candidates', []):
            ex = c.get('explain') or {}
            markers = [o for o in gt.get(i, []) if c.get('bbox') and any(overlap(c['bbox'], p) for p in o['parts'])]
            rows.append(dict(
                frame=i, time_s=round(t, 3), track=c.get('track_id', -1), S=round(c.get('S', 0.), 2), lateral_m=round(c.get('path_lateral_m', 0.), 3),
                distance_m=round(c['distance_m'], 2), center=[round(v, 3) for v in c['center']],
                bbox=[[round(v, 3) for v in corner] for corner in c['bbox']] if c.get('bbox') else None,
                u_range=[round(v, 3) for v in c['u_range']], h_range=[round(v, 3) for v in c['h_range']],
                depth_m=round(depth(c), 3), margin_m=round(c.get('margin_m', 0.), 3),
                sigma_u=round(c.get('sigma_u', 0.), 3), sigma_h=round(ex.get('sigma_h', 0.), 3),
                points=c.get('point_count', 0), unique_points=c.get('unique_point_count', 0),
                confident_points=c.get('confident_points', 0),
                state='obstacle' if c.get('confirmed') else 'attention' if (c.get('boundary') or c.get('far_attention') or c.get('confident_points', 0) > 0) else 'candidate',
                rule=ex.get('rule'), held=bool(c.get('held')),
                verified_m=round(r.get('analysis_range_m') or 0., 1), support_m=round(info.get('support_m') or 0., 1),
                markers=[dict(id=o['id'], label=o['label']) for o in markers]))
    return rows, frames


CLOSE_M = 30.       # closest frames: the rails carry the axis, its error is a few cm
SIGMA_CLOSE = (.03, .08)


def merge_tracks(rows):
    """The core's tracks of one object (it breaks them when the object is lost for a while) joined by their
    place in the world: arc position S and lateral offset, close in time."""
    tracks = defaultdict(list)
    for row in rows:
        tracks[row['track']].append(row)
    info = []
    for tid, rs in tracks.items():
        rs.sort(key=lambda r: r['frame'])
        info.append(dict(track=tid, f0=rs[0]['frame'], f1=rs[-1]['frame'],
                         S=float(np.median([r['S'] for r in rs])), u=float(np.median([r['lateral_m'] for r in rs]))))
    info.sort(key=lambda d: d['f0'])
    parent = {d['track']: d['track'] for d in info}

    def root(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t
    for a in range(len(info)):
        for b in range(a + 1, len(info)):
            da, db = info[a], info[b]
            if db['f0'] > da['f1'] + 30:
                continue
            if abs(da['S'] - db['S']) < 3. and abs(da['u'] - db['u']) < .6:
                parent[root(db['track'])] = root(da['track'])
    objects = defaultdict(list)
    for tid, rs in tracks.items():
        objects[root(tid)].extend(rs)
    return objects


def verdicts(rows):
    objects = merge_tracks(rows)
    out = {}
    for oid, rs in objects.items():
        rs.sort(key=lambda r: (r['frame'], r['distance_m']))
        live = [r for r in rs if not r['held']]
        close = [r for r in live if r['distance_m'] <= CLOSE_M]
        near = close if len(close) >= 2 else [r for r in live if r['distance_m'] <= NEAR_M]
        marker_ids = defaultdict(int)
        for r in rs:
            for m in r['markers']:
                marker_ids[(m['id'], m['label'])] += 1
        marker = max(marker_ids, key=marker_ids.get) if marker_ids else None
        if len(near) >= 2:
            d = np.array([r['depth_m'] for r in near])
            sig = np.array([np.clip(r['sigma_u'], *SIGMA_CLOSE) if r['distance_m'] <= CLOSE_M else max(r['sigma_u'], .03) for r in near])
            if np.median(d - sig) > 0:
                verdict = 'enters'
            elif np.median(d + sig) < 0:
                verdict = 'outside'
            else:
                verdict = 'boundary'
        else:
            verdict = 'unresolved'
        states = [r['state'] for r in rs]
        first_alarm = next((r for r in rs if r['state'] == 'obstacle'), None)
        out[oid] = dict(
            object=oid, tracks=sorted({r['track'] for r in rs}), frames=[rs[0]['frame'], rs[-1]['frame']], n=len({r['frame'] for r in rs}),
            distance_m=[rs[0]['distance_m'], rs[-1]['distance_m']], min_distance_m=min(r['distance_m'] for r in rs),
            S=round(float(np.median([r['S'] for r in rs])), 1), lateral_m=round(float(np.median([r['lateral_m'] for r in rs])), 2),
            judged_at_m=round(float(np.median([r['distance_m'] for r in near])), 1) if len(near) >= 2 else None,
            near_depth_m=round(float(np.median([r['depth_m'] for r in near])), 3) if len(near) >= 2 else None,
            near_sigma_m=round(float(np.median([r['sigma_u'] for r in near])), 3) if len(near) >= 2 else None,
            verdict=verdict, marker=dict(id=marker[0], label=marker[1]) if marker else None,
            alarm_frames=states.count('obstacle'), attention_frames=states.count('attention'),
            first_alarm=dict(frame=first_alarm['frame'], distance_m=first_alarm['distance_m']) if first_alarm else None,
            max_points=max(r['points'] for r in rs))
    # every row carries its object
    for oid, rs in objects.items():
        for r in rs:
            r['object'] = oid
    return out


def needs_review(v):
    """An object to look at: it matters (an alarm, an organizers' marker, a long attention) and the verdict is
    undecided or disagrees with the markers or with the core."""
    matters = v['alarm_frames'] > 0 or v['marker'] is not None or v['attention_frames'] >= 10
    if not matters:
        return None
    if v['verdict'] in ('boundary', 'unresolved'):
        return 'verdict ' + v['verdict']
    if v['marker'] and v['marker']['label'] == 'positive' and v['verdict'] == 'outside':
        return 'marker target, judged outside'
    if v['marker'] and v['marker']['label'] == 'negative' and v['verdict'] == 'enters':
        return 'marker exterior/dust, judged inside'
    if v['alarm_frames'] and v['verdict'] == 'outside':
        return 'alarm on an object judged outside'
    if v['verdict'] == 'enters' and not v['alarm_frames']:
        return 'judged inside, never an alarm'
    return None


def build_review(dest):
    review = ['# Треки на проверку', '',
              'Вердикт по треку принят вблизи (до 50 м) и перенесён на все его кадры. Здесь треки, где он не принят '
              '(не подошёл близко или на границе), спорит с маркерами организаторов или с тревогой ядра. '
              'Ссылки открывают визуализатор на кадре середины трека.', '']
    for path in sorted(dest.glob('*.json')):
        if path.name.endswith('.review.json'):
            continue
        payload = json.loads(path.read_text())
        todo = [v for v in payload['tracks'] if v.get('review')]
        if not todo:
            continue
        run = payload['run']
        review += [f'## {run}', '']
        for v in todo:
            mid = (v['frames'][0] + v['frames'][1]) // 2
            m = f", маркер {v['marker']['id']} ({v['marker']['label']})" if v['marker'] else ''
            review.append(f"- объект {v['object']} (треки {v['tracks'][:6]}): кадры {v['frames'][0]}–{v['frames'][1]}, {v['distance_m'][0]} → {v['distance_m'][1]} м, "
                          f"вердикт {v['verdict']} (на {v['judged_at_m']} м глубина {v['near_depth_m']} м, σ {v['near_sigma_m']}), "
                          f"тревог {v['alarm_frames']}, ВНИМАНИЙ {v['attention_frames']}{m}; причина: {v['review']}; "
                          f"http://localhost:8766/viewer/?seq={run}&frame={mid}&camera=cab")
        review.append('')
    (dest / 'review.md').write_text('\n'.join(review), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--review-only', action='store_true', help='only rebuild review.md from the saved labels')
    a = ap.parse_args()
    dest = ROOT / 'datasets' / 'labels'
    dest.mkdir(parents=True, exist_ok=True)
    for run in ([] if a.review_only else a.runs):
        if not (ROOT / 'viewer' / 'data' / run / 'manifest.json').is_file():
            print(run, 'no viewer export, skipped')
            continue
        rows, frames = collect(run)
        tv = verdicts(rows)
        for v in tv.values():
            v['review'] = needs_review(v)
        payload = dict(run=run, frames=frames, near_m=NEAR_M, envelope=dict(half_width=HALF_WIDTH, height=HEIGHT, min_height=MIN_HEIGHT),
                       note='Rows: every candidate of every frame with its core track id. Verdicts per track: enters / outside / '
                            'boundary (decided at <= near_m) or unresolved; carried to all frames of the track.',
                       tracks=sorted(tv.values(), key=lambda v: v['frames'][0]), rows=rows)
        (dest / f'{run}.json').write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
        counts = defaultdict(int)
        for v in tv.values():
            counts[v['verdict']] += 1
        print(f"{run}: {frames} frames, {len(tv)} tracks {dict(counts)}, to review {sum(1 for v in tv.values() if v['review'])}", flush=True)
    build_review(dest)


if __name__ == '__main__':
    main()
