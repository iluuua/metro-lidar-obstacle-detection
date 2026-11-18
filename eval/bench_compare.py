"""One line per bench run: the criteria that matter, summed over the recordings of the run.

    python eval/bench_compare.py baseline experiment
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]


def line(tag):
    scores = json.loads((ROOT / 'output/bench' / tag / 'scores.json').read_text(encoding='utf-8'))
    lab = [s for s in scores.values() if s.get('kind') == 'labelled']
    ctrl = sum(s['alarm_frames'] for s in scores.values() if s.get('kind') == 'control')
    fp = sum(s['counts']['fp'] for s in lab)
    tp = sum(s['counts']['tp'] + s['counts'].get('shifted', 0) for s in lab)
    fn = sum(s['counts']['fn'] for s in lab)
    first = [o['first_distance'] for s in lab for o in s['objects'] if o['first_distance'] is not None]
    cover = [o['coverage_after_first'] for s in lab for o in s['objects'] if o['coverage_after_first'] is not None]
    axes = []
    for name, s in scores.items():
        if 'axis' not in s:
            continue
        if 'env_dips' not in s['axis']:  # a run made before these metrics: take them from the saved frames
            import bench
            frames = json.loads((ROOT / 'output/bench' / tag / (name.replace(':', '-') + '.json')).read_text(encoding='utf-8'))['frames']
            s['axis'] = bench.axis_metrics(frames)
        axes.append(s['axis'])
    flick = sum(a['env_flicker'] or 0 for a in axes)
    coll = sum(a['env_collapse'] or 0 for a in axes)
    osc = sum(a.get('env_oscillation') or 0 for a in axes)
    step = max((a.get('step100_p99_cm') or 0) for a in axes if a)
    stepmax = max((a.get('step100_max_cm') or 0) for a in axes if a)
    kink = max((a.get('kink_p99_deg') or 0) for a in axes)
    rev = sum(a['reversal_frames'] or 0 for a in axes)
    dips = sum(a.get('env_dips') or 0 for a in axes)
    rigid = max((a.get('rigid100_p99_cm') or 0) for a in axes)
    p95 = max((a.get('wall_p95_ms') or 0) for a in axes)
    return (f"{tag:22s} ctrlAlarm {ctrl:3d} | FP {fp:3d} TP {tp:5d} FN {fn:5d} | first {sum(first)/max(len(first),1):5.1f} m cov {100*sum(cover)/max(len(cover),1):4.1f}% "
            f"| env flick {flick:4d} osc {osc:4d} coll {coll:3d} dips {dips:4d} | step100 p99 max {step:5.1f} (rigid {rigid:4.1f}) max {stepmax:5.0f} cm | kink {kink:5.1f}° rev {rev:4d} | wall p95 max {p95:4.0f} ms "
            )


if __name__ == '__main__':
    for tag in sys.argv[1:]:
        print(line(tag))
