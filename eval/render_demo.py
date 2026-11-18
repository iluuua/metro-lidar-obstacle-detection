"""A demo video from a recording: the core runs over it, eval/render_still.py draws every chosen frame, ffmpeg joins them.

    python eval/render_demo.py labelled_sample 40 230 output/demo/station.mp4 --title "Станция, человек справа"

The envelope is drawn on the rail plane and the axis of the core itself (its answer, with the vertical profile), and the camera
follows the rail plane, so the picture is level with the track whatever the tilt of the sensor. The core is warmed up on the 30
frames before the first one. Needs METRO_CORE_LIB (and METRO_CORE_LIB_LONG for a Falcon recording) and ffmpeg. The video plays at
the speed of the recording (10 frames a second), `--step N` makes it N times faster.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]


def render(job):
    seq, frame, out, core, title, pitch, what = job
    cmd = [sys.executable, str(ROOT / 'eval' / 'render_still.py'), seq, str(frame), str(out), '--core-result', str(core),
           '--title', title, '--what', what]
    if pitch is not None:
        cmd += ['--pitch', str(pitch)]
    done = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return frame, done.returncode, done.stderr[-300:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sequence')
    ap.add_argument('first', type=int)
    ap.add_argument('last', type=int)
    ap.add_argument('output', type=Path)
    ap.add_argument('--title', default='')
    ap.add_argument('--pitch', type=float, default=None, help='camera pitch (default: along the rail plane of the core)')
    ap.add_argument('--what', default='предмет')
    ap.add_argument('--step', type=int, default=1, help='every N-th frame (the video then plays N times faster)')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--fps', type=float, default=10.)
    a = ap.parse_args()
    from metro.native import NativeTunnelDetector
    from frame_source import frames
    cfg, source = frames(a.sequence)
    det = NativeTunnelDetector(cfg, {})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.output.parent) as tmp:
        jobs = []
        for i, pts, t in source:
            if i < a.first - 30:
                continue
            if i > a.last:
                break
            r = det.process(pts, None, t)
            if i >= a.first and (i - a.first) % a.step == 0:
                core = Path(tmp) / f'core_{i}.json'
                core.write_text(json.dumps(r, default=float), encoding='utf-8')
                jobs.append((a.sequence, i, Path(tmp) / f'{len(jobs):05d}.png', core, a.title or a.sequence, a.pitch, a.what))
        with ThreadPoolExecutor(a.workers) as pool:
            for frame, code, err in pool.map(render, jobs):
                if code:
                    print('frame', frame, 'failed', err.encode('ascii', 'replace').decode(), flush=True)
        done = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(a.fps), '-i', str(Path(tmp) / '%05d.png'),
                               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', str(a.output)], capture_output=True, text=True)
        print(done.stderr[-400:] or f'{len(jobs)} frames -> {a.output}')


if __name__ == '__main__':
    main()
