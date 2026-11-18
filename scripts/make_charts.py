"""Plot per-object detection distances from a local results JSON.

Input format is a list of {"name": "object", "distance_m": number}.
    python scripts/make_charts.py /path/to/results.json --output output/ranges.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--title', default='Дальность обнаружения')
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding='utf-8'))
    if not rows or any(not np.isfinite(row['distance_m']) or row['distance_m'] < 0 for row in rows):
        parser.error('input must contain finite non-negative distances')
    labels = [str(row['name']) for row in rows]
    values = [float(row['distance_m']) for row in rows]
    figure, axis = plt.subplots(figsize=(10, max(3, len(rows) * .5)))
    axis.barh(labels, values, color='#505050')
    axis.invert_yaxis()
    axis.set_xlabel('Дальность первого обнаружения, м')
    axis.set_title(args.title)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)


if __name__ == '__main__':
    main()
