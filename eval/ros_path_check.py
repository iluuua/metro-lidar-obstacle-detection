"""Compares what the ROS node answered while a bag was played in the Docker image with the benchmark of the core on the same recording.

The logs come from `ros2 launch metro_bringup demo.launch.py bag:=... rate:=1.0` (one status line per answer):

      1.6с  ПРЕПЯТСТВИЕ           95.9 м   габарит до  300 м   расчёт   18 мс

    python eval/ros_path_check.py output/ros_path --tag experiment

Writes output/ros_path/summary.json and prints a markdown table (answers vs. frames, alarm frames vs. the benchmark, frame time, node warnings).
"""
import argparse
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINE = re.compile(r'(?P<t>[\d.]+)с\s+(?P<state>ПРЕПЯТСТВИЕ|ВНИМАНИЕ(?: УСИЛЕННОЕ)?|СВОБОДНО|НЕДОСТАТОЧНО ДАННЫХ)\s+(?P<dist>—|[\d.]+ м)?\s*габарит до\s+(?P<env>\d+)\s*м\s+расчёт\s+(?P<ms>\d+)\s*мс')
def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q / 100 * len(values)))] if values else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', type=Path)
    ap.add_argument('--tag', default='experiment')
    args = ap.parse_args()
    bench = json.loads((ROOT / 'output/bench' / args.tag / 'scores.json').read_text(encoding='utf-8'))
    summary, rows = {}, []
    for seq in sorted(bench):
        path = args.logs / f'{seq}.log'
        if not path.exists():
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        answers = [m.groupdict() for m in (LINE.search(l) for l in text.splitlines()) if m]
        states = {}
        for a in answers:
            states[a['state']] = states.get(a['state'], 0) + 1
        ms = [int(a['ms']) for a in answers]
        first = next((float(a['dist'].split()[0]) for a in answers if a['state'] == 'ПРЕПЯТСТВИЕ' and a['dist'] and a['dist'] != '—'), None)
        warnings = [l for l in text.splitlines() if ('[WARN]' in l or '[ERROR]' in l) and 'rosbag2_player' not in l]
        frames = json.loads((ROOT / 'output/bench' / args.tag / f'{seq}.json').read_text(encoding='utf-8'))['frames']
        expected = bench[seq]['alarm_frames']
        entry = dict(frames=len(frames), answers=len(answers), alarm=states.get('ПРЕПЯТСТВИЕ', 0), attention=states.get('ВНИМАНИЕ', 0) + states.get('ВНИМАНИЕ УСИЛЕННОЕ', 0), clear=states.get('СВОБОДНО', 0),
                     unknown=states.get('НЕДОСТАТОЧНО ДАННЫХ', 0), bench_alarm=expected, first_alarm_m=first, p50_ms=statistics.median(ms) if ms else None,
                     p95_ms=percentile(ms, 95), max_ms=max(ms) if ms else None, node_warnings=len(warnings), player_starved=text.count('Message queue starved'))
        summary[seq] = entry
        rows.append(entry | dict(seq=seq))
    (args.logs / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print('| Запись | Кадров | Ответов ноды | Тревожных (ROS) | Тревожных (ядро) | Первая тревога, м | расчёт p50 / p95 / max, мс | Предупреждения ноды |')
    print('|---|---:|---:|---:|---:|---:|---|---:|')
    for r in rows:
        print(f"| {r['seq']} | {r['frames']} | {r['answers']} ({100 * r['answers'] / r['frames']:.0f} %) | {r['alarm']} | {r['bench_alarm']} | {r['first_alarm_m'] or '—'} | "
              f"{r['p50_ms']:.0f} / {r['p95_ms']} / {r['max_ms']} | {r['node_warnings']} |")


if __name__ == '__main__':
    main()
