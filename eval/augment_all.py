"""Run augmentation jobs on user-selected recordings with reproducible random seeds.

python eval/augment_all.py --tag experiment --runs sample_a sample_b --clean-runs sample_a
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def jobs(runs, seeds):
    return [(run, seed) for seed in range(1, seeds + 1) for run in runs]


def output(tag, run, seed):
    return ROOT / 'output' / 'augment' / tag / f"{run.replace('@', '_').replace(':', '-')}_s{seed}.jsonl"


def run_job(job):
    run, seed, a = job
    path = output(a.tag, run, seed)
    if path.is_file():
        return f'{run} seed {seed}: exists'
    began = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix('.part')
    cmd = [sys.executable, str(ROOT / 'eval' / 'augment.py'), run, '--seed', str(seed), '--out', str(part),
           '--degrade', str(a.degrade), '--offset', str(a.offset), '--stride', str(a.stride), '--config', a.config]
    if run.split('@')[0] in a.clean_runs:
        cmd += ['--clean']
    if a.elevations:
        cmd += ['--elevations', str(a.elevations)]
    if a.dll:
        cmd += ['--dll', a.dll]
    if a.ref_dll:
        cmd += ['--ref-dll', a.ref_dll]
    if a.drop:
        cmd += ['--drop', a.drop]
    if a.first is not None:
        cmd += ['--first', str(a.first)]
    if a.spacing is not None:
        cmd += ['--spacing', str(a.spacing)]
    done = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if done.returncode:
        return f'{run} seed {seed}: FAILED {done.stderr[-600:] or done.stdout[-300:]}'
    part.replace(path)
    return f'{done.stdout.strip()} ({time.time() - began:.0f}s)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--dll')
    ap.add_argument('--ref-dll')
    ap.add_argument('--config', default='{}')
    ap.add_argument('--drop', default='', help='settings omitted when testing an older core library')
    ap.add_argument('--first', type=float, help='distance of the first object from the start, m (default 200-270)')
    ap.add_argument('--spacing', type=float, help='distance between objects, m (default 70); the short recordings need 35 and --first 60')
    ap.add_argument('--degrade', type=int, default=0)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--runs', nargs='+', required=True, help='viewer/cache run IDs, optionally run@first:last')
    ap.add_argument('--clean-runs', nargs='*', default=[])
    ap.add_argument('--elevations', type=Path)
    a = ap.parse_args()
    todo = jobs(a.runs, a.seeds)
    print(f'{len(todo)} jobs on {a.workers} processes', flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        for line in pool.map(run_job, [(r, s, a) for r, s in todo]):
            print(time.strftime('%H:%M:%S'), line, flush=True)


if __name__ == '__main__':
    main()
