"""Early alarm of a candidate that will enter the envelope (eval/enter_model.py), on top of the core's answer.

The model is a small set of boosted trees saved as plain JSON and evaluated with numpy. Per track of the core it
keeps causal aggregates of the track's frames; a candidate whose probability stays above the threshold for K frames
in a row is confirmed, and the frame's status becomes an obstacle. The core's own decisions are never removed.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HALF, TOP, FLOOR = 1.05, 3., .07


def frame_features(c):
    u_lo, u_hi = c['u_range']
    h_lo, h_hi = c['h_range']
    inner = 0. if u_lo <= 0. <= u_hi else min(abs(u_lo), abs(u_hi))
    edge = HALF - inner
    v = min(TOP - h_lo, h_hi - FLOOR)
    depth = min(edge, v)
    sigma = float(c.get('sigma_u') or .05)
    return dict(distance=float(c['distance_m']), points=float(c.get('point_count') or 0),
                unique=float(c.get('unique_point_count') or 0), confident=float(c.get('confident_points') or 0),
                unique_confident=float(c.get('unique_confident_points') or 0), edge_depth=edge, v_depth=v, depth=depth,
                width=u_hi - u_lo, h_lo=h_lo, h_hi=h_hi, sigma=sigma, ratio=depth / max(sigma, .02),
                margin=float(c.get('margin_m') or 0.), evidence=float(c.get('evidence') or 0.),
                hits=float(c.get('hits') or 0), appeared=float(c.get('appeared_frac') or 0.), u_mid=(u_lo + u_hi) / 2)


class History:
    """Causal aggregates of one track (the same in training, eval/enter_model.py)."""
    def __init__(self):
        self.k, self.max_points, self.sum_depth, self.min_depth, self.sum_ratio = 0, 0., 0., 9., 0.
        self.u, self.first, self.confident_frames = [], None, 0
        self.above, self.last_frame = 0, -1

    def add(self, f):
        self.k += 1
        self.max_points = max(self.max_points, f['points'])
        self.sum_depth += f['depth']
        self.min_depth = min(self.min_depth, f['depth'])
        self.sum_ratio += f['ratio']
        self.u = (self.u + [f['u_mid']])[-10:]
        self.first = f['distance'] if self.first is None else self.first
        self.confident_frames += f['confident'] > 0
        return dict(f, k=self.k, max_points=self.max_points, mean_depth=self.sum_depth / self.k, min_depth=self.min_depth,
                    mean_ratio=self.sum_ratio / self.k, u_std=float(np.std(self.u)) if len(self.u) > 1 else 0.,
                    approach=self.first - f['distance'], confident_frames=self.confident_frames)


class Trees:
    def __init__(self, model):
        self.baseline = model['baseline']
        self.trees = [{k: np.asarray(v) for k, v in t.items()} for t in model['trees']]

    def proba(self, x):
        raw = self.baseline
        for t in self.trees:
            n = 0
            while not t['leaf'][n]:
                n = t['left'][n] if x[t['feature'][n]] <= t['threshold'][n] else t['right'][n]
            raw += t['value'][n]
        return 1. / (1. + np.exp(-raw))


class EnterGate:
    def __init__(self, path, min_distance=0., forget_frames=30):
        spec = json.loads(Path(path).read_text(encoding='utf-8'))
        self.features, self.theta, self.k = spec['features'], spec['theta'], spec['k']
        self.trees = Trees(spec['model'])
        self.min_distance, self.forget = min_distance, forget_frames
        self.tracks = defaultdict(History)
        self.frame = 0

    def reset(self):
        self.tracks.clear()
        self.frame = 0

    def apply(self, result):
        self.frame += 1
        fired = []
        for c in result.get('candidates', []):
            if not c.get('u_range') or not c.get('h_range') or c.get('track_id') is None:
                continue
            h = self.tracks[c['track_id']]
            if h.last_frame != self.frame - 1:
                h.above = 0
            h.last_frame = self.frame
            f = h.add(frame_features(c))
            p = self.trees.proba([f[k] for k in self.features])
            h.above = h.above + 1 if p > self.theta else 0
            c['enter_p'] = round(float(p), 3)
            if not c.get('confirmed') and h.above >= self.k and f['distance'] >= self.min_distance:
                c['confirmed'] = True
                c['decision_rule'] = 'enter_model'
                fired.append(c)
        if fired and result.get('status') != 'obstacle':
            result['status'] = 'obstacle'
            result['obstacle'] = True
            result['distance_m'] = min(c['distance_m'] for c in result['candidates'] if c.get('confirmed'))
        for t in [t for t, h in self.tracks.items() if self.frame - h.last_frame > self.forget]:
            del self.tracks[t]
        return result
