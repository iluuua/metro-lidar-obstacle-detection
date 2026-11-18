"""Read user-provided cached clouds or viewer exports, optionally run@first:last."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]

def frames(run):
    from metro.tunnel import TunnelConfig
    from run_cached import iter_frames
    from stability import frames_of, run_config
    if '@' in run:
        name, window = run.split('@', 1)
        first, last = (int(value) for value in window.split(':'))
        source = frames(name)[1]
        return run_config(name), ((i, points, stamp) for i, points, stamp in source if first <= i < last)
    if (ROOT / 'output' / 'raw_cache' / run).is_dir() or (ROOT / 'output' / 'decoded_cache' / run).is_dir():
        return run_config(run), frames_of(run)
    return run_config(run), viewer_frames(run)


def viewer_frames(sequence, first=0, last=None):
    """The recording from viewer/data/<sequence>: exported whole clouds, or the source bag as eval/bench.py reads it."""
    import numpy as np
    root = ROOT / 'viewer' / 'data' / sequence
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    clouds = None
    for i, record in enumerate(manifest['frames']):
        if i < first or (last is not None and i >= last):
            continue
        exported = root / f"{record.get('frame', i):05d}.full.bin"
        if record.get('full_file') and (ROOT / 'viewer' / record['full_file']).is_file():
            points = np.fromfile(ROOT / 'viewer' / record['full_file'], '<f4').reshape(-1, 3)
        elif exported.is_file():
            points = np.fromfile(exported, '<f4').reshape(-1, 3)
        else:
            if clouds is None:
                sys.path[:0] = [str(ROOT / 'scripts')]
                from source_clouds import SourceClouds
                clouds = SourceClouds()
            points = np.frombuffer(clouds.read(manifest['summary'], i), '<f4').reshape(-1, 3)
        yield record.get('frame', i), points, record['time_s']
