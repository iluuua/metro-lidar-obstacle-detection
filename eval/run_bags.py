"""Run the detector over user-provided cached recordings and print a summary."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
from run_cached import run  # noqa: E402

parser = argparse.ArgumentParser(description='Summarise user-provided cached recordings.')
parser.add_argument('runs', nargs='+')
parser.add_argument('--out', type=Path, default=ROOT / 'output' / 'tunnel_runs')
args = parser.parse_args()
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)

out = {}
for bag in args.runs:
    summary, frames = run(bag, verbose=False, save=OUT / f'{bag}.json')
    st = [f['status'] for f in frames]
    an = np.array([f.get('analysis_range_m') or 0 for f in frames])
    alarms = [(f['frame'], round(f['distance_m'], 1)) for f in frames if f['status'] == 'obstacle']
    out[bag] = dict(frames=len(frames), obstacle=st.count('obstacle'), unknown=st.count('unknown'),
                    candidate=st.count('candidate'), analysis_median=float(np.median(an)),
                    p50_ms=round(summary['latency_p50_ms']), p95_ms=round(summary['latency_p95_ms']),
                    alarm_frames=alarms[:6] + (['...'] if len(alarms) > 6 else []))
    print(bag, json.dumps(out[bag], ensure_ascii=False), flush=True)
(OUT / 'bags_summary.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
