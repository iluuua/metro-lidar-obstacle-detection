"""Coordinate search using user-selected training, control and held-out runs.

python eval/bench_search.py --dll core/build/libmetro_core.so --space eval/search_space.json \
    --train labelled_a --controls clean_a --held-out labelled_b

Labelled runs require user-supplied ground-truth.json alongside their viewer export.
The held-out recordings do not take part in parameter selection.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def evaluate(dll, config, sequences, tag, controls=()):
    started = time.time()
    done = subprocess.run([sys.executable, str(ROOT / 'eval/bench.py'), '--dll', dll, '--tag', tag, '--config', json.dumps(config),
                           '--sequences', ','.join(sequences), '--controls', ','.join(controls), '--workers', '8'], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if done.returncode:
        raise RuntimeError(done.stderr[-2000:])
    scores = json.loads((ROOT / 'output/bench' / tag / 'scores.json').read_text(encoding='utf-8'))
    return scores, time.time() - started


def digest(scores, sequences):
    lab = [scores[s] for s in sequences if scores[s].get('kind') == 'labelled']
    ctrl = sum(scores[s]['alarm_frames'] for s in sequences if scores[s].get('kind') == 'control')
    fp = sum(s['counts']['fp'] for s in lab)
    tp = sum(s['counts']['tp'] + s['counts'].get('shifted', 0) for s in lab)
    objs = [o for s in lab for o in s['objects']]
    first = [o['first_distance'] or 0. for o in objs]
    cover = [o['coverage_after_first'] or 0. for o in objs]
    return dict(ctrl=ctrl, fp=fp, tp=tp, first=sum(first) / max(len(first), 1), cover=sum(cover) / max(len(cover), 1))


def better(a, b, fp_cap=None):
    """a beats b: no control alarm, then fewer false positives (they come first), then more detected frames, then a farther first detection."""
    if a['ctrl'] != 0:
        return False
    return (a['fp'], -a['tp'], -a['first']) < (b['fp'], -b['tp'], -b['first'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dll', required=True)
    ap.add_argument('--base', default='{}')
    ap.add_argument('--space', required=True, help='JSON {param: [values]}')
    ap.add_argument('--name', default='search')
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--train', required=True, help='comma-separated labelled runs for parameter selection')
    ap.add_argument('--controls', required=True, help='comma-separated runs known to be obstacle-free')
    ap.add_argument('--held-out', required=True, help='comma-separated independent labelled runs')
    args = ap.parse_args()
    space = json.loads(Path(args.space).read_text(encoding='utf-8'))
    log = ROOT / 'output/bench_search' / f'{args.name}.jsonl'
    log.parent.mkdir(parents=True, exist_ok=True)
    train = args.train.split(',')
    controls = args.controls.split(',')
    held_out = args.held_out.split(',')
    if set(train) & set(held_out) or set(controls) & (set(train) | set(held_out)):
        ap.error('train, controls and held-out runs must be disjoint')
    sequences = train + controls
    config = json.loads(args.base)
    scores, seconds = evaluate(args.dll, config, sequences, f'{args.name}_cur', controls)
    best = digest(scores, sequences)
    fp_cap = best['fp']
    print('start', json.dumps(best), f'{seconds:.0f}s', flush=True)
    with log.open('a', encoding='utf-8') as out:
        out.write(json.dumps(dict(config=config, train=best)) + '\n')
    for round_ in range(args.rounds):
        improved = False
        for key, values in space.items():
            for value in values:
                if config.get(key) == value:
                    continue
                trial = dict(config, **{key: value})
                scores, seconds = evaluate(args.dll, trial, sequences, f'{args.name}_try', controls)
                result = digest(scores, sequences)
                out_line = dict(config=trial, change={key: value}, train=result)
                accepted = better(result, best, fp_cap)
                out_line['accepted'] = accepted
                with log.open('a', encoding='utf-8') as out:
                    out.write(json.dumps(out_line) + '\n')
                print(f"round {round_} {key}={value}: ctrl {result['ctrl']} fp {result['fp']} tp {result['tp']} first {result['first']:.1f} cover {100*result['cover']:.1f}% "
                      f"{'ACCEPT' if accepted else ''} ({seconds:.0f}s)", flush=True)
                if accepted:
                    config, best, improved = trial, result, True
        if not improved:
            break
    print('best config', json.dumps(config))
    held, _ = evaluate(args.dll, config, held_out + controls, f'{args.name}_heldout', controls)
    print('held-out', json.dumps(digest(held, held_out + controls)))
    (ROOT / 'output/bench_search' / f'{args.name}_best.json').write_text(json.dumps(config, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
