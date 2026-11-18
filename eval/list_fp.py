from pathlib import Path
import argparse
import json
import sys
import numpy as np
sys.path.insert(0, 'eval')
from score_synthetic import overlap
parser = argparse.ArgumentParser()
parser.add_argument('results', type=Path)
parser.add_argument('--truth', type=Path, required=True)
args = parser.parse_args()
d = json.loads(args.results.read_text(encoding='utf-8'))
fr = {f['frame']: f for f in d['frames']}
gt = json.loads(args.truth.read_text(encoding='utf-8'))
names = {o['id']: o['name'] for o in gt['objects']}
pos = {o['id']: o['positive'] for o in gt['objects']}
for i, f in sorted(fr.items()):
    if f['status'] != 'obstacle': continue
    gts = gt['frames'].get(str(i), [])
    for c in f['candidates']:
        if not c['confirmed']: continue
        hit = [g for g in gts if overlap(c['bbox'], g, 1.5)]
        if hit and all(pos[g['id']] for g in hit): continue
        gtpos = [(names[g['id']], round(g['xmin'], 1), np.round(g['lo'][1:], 2).tolist(), np.round(g['hi'][1:], 2).tolist()) for g in hit]
        print(i, 'd %.1f' % c['distance_m'], 'u', np.round(c['u_range'], 2), 'h', np.round(c['h_range'], 2), 'n', c['point_count'], 'conf', c['confident_points'], 'hits', c['hits'], 'sup', f['support_m'], 'match', gtpos)
