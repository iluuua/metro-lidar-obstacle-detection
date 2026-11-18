"""Will the object enter the envelope? A classifier of candidate tracks from their first, far frames.

Data: datasets/augment/*.jsonl (eval/augment.py): candidates holding the returns of ray-cast boxes with the truth of
the box (inside / outside / boundary by more than 5 cm), and on the recordings without obstacles every other
candidate (track equipment, a hard negative). A row's features use only its track's frames up to that frame, so the
model sees what the detector would see then. Boundary boxes are left out of training.

The test is by recording groups (a model never sees the tunnel it is tested on): per object, the first frame of a
model alarm (probability above the threshold in K frames of the track in a row) against the core's first confirmed
frame, and every track of an outside box or of equipment where the model would alarm. The threshold is the lowest
with no such track in the training folds.

    .venv/bin/python eval/enter_model.py
    .venv/bin/python eval/enter_model.py --save output/enter_model.json     # trained on everything, for the core wrapper
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
from metro.enter import History, frame_features  # noqa: E402
FEATURES = ['distance', 'points', 'unique', 'confident', 'unique_confident', 'edge_depth', 'v_depth', 'depth', 'width',
            'h_lo', 'h_hi', 'sigma', 'ratio', 'margin', 'evidence', 'hits', 'appeared',
            'k', 'max_points', 'mean_depth', 'min_depth', 'mean_ratio', 'u_std', 'approach', 'confident_frames']


def group_of(run):
    return run  # the windows of the 20-minute ride are separate stretches of line


def load(pattern):
    objects, rows = {}, []
    for path in sorted(glob.glob(pattern)):
        hist = defaultdict(History)
        for line in open(path, encoding='utf-8'):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # the last line of a file still being written
                continue
            if r['kind'] == 'object':
                objects[(r['run'], r['seed'], r['id'])] = r
            elif r['kind'] == 'row' and r.get('u_range') and r.get('h_range'):
                key = (r['run'], r['seed'], r.get('track_id'))
                f = hist[key].add(frame_features(r))
                rows.append(dict(f, run=r['run'], seed=r['seed'], frame=r['frame'], track=key, object=r.get('object'),
                                 label=r['label'], front=r.get('front_m'), confirmed=bool(r.get('confirmed'))))
    return objects, rows


def matrix(rows):
    return np.array([[r[k] for k in FEATURES] for r in rows], float)


def first_alarm(rows, p, theta, k_in_row):
    """Per track: distance of the first frame with K frames in a row above theta (None if never)."""
    out, run_len = {}, defaultdict(int)
    for r, q in zip(rows, p):
        t = r['track']
        run_len[t] = run_len[t] + 1 if q > theta else 0
        if run_len[t] >= k_in_row and t not in out:
            out[t] = r['distance']
    return out


def evaluate(train_rows, test_rows, k_in_row, model_kw):
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr = [r for r in train_rows if r['label'] != 'boundary']
    y = np.array([r['label'] == 'inside' for r in tr])
    model = HistGradientBoostingClassifier(**model_kw).fit(matrix(tr), y)
    p_tr = model.predict_proba(matrix(tr))[:, 1]
    neg_tracks = {r['track'] for r in tr if r['label'] != 'inside'}
    theta = .5
    for theta in np.arange(.5, 1., .01):
        fired = first_alarm(tr, p_tr, theta, k_in_row)
        if not any(t in neg_tracks for t in fired):
            break
    p = model.predict_proba(matrix(test_rows))[:, 1]
    fired = first_alarm(test_rows, p, theta, k_in_row)
    res = dict(theta=round(float(theta), 2), fp_tracks=[], gains=[], objects=0, found_core=0, found_model=0, found_any=0)
    by_obj = defaultdict(list)
    for r in test_rows:
        if r['label'] in ('outside', 'structure') and r['track'] in fired:
            res['fp_tracks'].append((r['run'], r['seed'], r['label'], round(fired[r['track']], 1)))
            fired.pop(r['track'])
        if r['object'] is not None:
            by_obj[(r['run'], r['seed'], r['object'])].append(r)
    for key, rs in by_obj.items():
        if rs[0]['label'] != 'inside':
            continue
        res['objects'] += 1
        core = max([r['distance'] for r in rs if r['confirmed']], default=None)
        model_first = max([fired[r['track']] for r in rs if r['track'] in fired], default=None)
        both = max([v for v in (core, model_first) if v is not None], default=None)
        res['found_core'] += core is not None
        res['found_model'] += model_first is not None
        res['found_any'] += both is not None
        res['gains'].append((core or 0., both or 0.))
    return res, model, theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'datasets' / 'augment' / '*.jsonl'))
    ap.add_argument('--k', type=int, default=3, help='frames in a row above the threshold')
    ap.add_argument('--save')
    a = ap.parse_args()
    objects, rows = load(a.data)
    labels = defaultdict(int)
    for r in rows:
        labels[r['label']] += 1
    print(f'{len(objects)} boxes, {len(rows)} candidate rows: {dict(labels)}')
    kw = dict(max_depth=4, max_iter=200, learning_rate=.05, l2_regularization=1., class_weight='balanced', random_state=0)
    groups = sorted({group_of(r['run']) for r in rows})
    tot = defaultdict(float)
    fps = []
    for g in groups:
        test = [r for r in rows if group_of(r['run']) == g]
        train = [r for r in rows if group_of(r['run']) != g]
        if not test or not train:
            continue
        res, _, theta = evaluate(train, test, a.k, kw)
        core = np.array([c for c, _ in res['gains']]) if res['gains'] else np.zeros(0)
        both = np.array([b for _, b in res['gains']]) if res['gains'] else np.zeros(0)
        print(f"{g:40s} theta {res['theta']:.2f}  inside boxes {res['objects']:3d}: core finds {res['found_core']}, "
              f"with model {res['found_any']}; first alarm mean {core.mean() if len(core) else 0:.1f} -> {both.mean() if len(both) else 0:.1f} m; "
              f"false tracks {len(res['fp_tracks'])} {res['fp_tracks'][:4]}")
        tot['objects'] += res['objects']
        tot['core'] += core.sum()
        tot['both'] += both.sum()
        fps += res['fp_tracks']
    n = max(tot['objects'], 1)
    print(f"ALL: {int(tot['objects'])} inside boxes, first alarm mean {tot['core'] / n:.1f} -> {tot['both'] / n:.1f} m, false tracks {len(fps)}")
    if a.save:
        _, model, theta = evaluate(rows, rows[:1], a.k, kw)
        Path(a.save).write_text(json.dumps(dict(features=FEATURES, theta=theta, k=a.k, model=export(model)), ensure_ascii=False))
        print('saved', a.save)


def export(model):
    """The trees of a HistGradientBoostingClassifier as plain lists (evaluated without scikit-learn)."""
    trees = []
    for (pred,) in model._predictors:
        n = pred.nodes
        trees.append(dict(value=n['value'].tolist(), feature=n['feature_idx'].tolist(), threshold=n['num_threshold'].tolist(),
                          left=n['left'].tolist(), right=n['right'].tolist(), leaf=n['is_leaf'].astype(int).tolist(),
                          missing_left=n['missing_go_to_left'].astype(int).tolist()))
    return dict(baseline=float(model._baseline_prediction.ravel()[0]), trees=trees)


if __name__ == '__main__':
    main()
