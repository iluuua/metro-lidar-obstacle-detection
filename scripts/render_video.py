"""Render a demo video: cab view of the point cloud, clearance envelope,
detections and a HUD, plus a top-view inset. Frames come from cached decodes
(output/raw_cache/<id> or output/decoded_cache/<bag>) and the live detector.

Example:
    python scripts/render_video.py output/demo.mp4 --source /path/to/cache --stop 100
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]
from metro.tunnel import TunnelDetector, TunnelConfig  # noqa: E402
from run_cached import iter_frames  # noqa: E402

W, H = 1280, 720
FONT = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
if not Path(FONT).exists():
    FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


F_BIG, F_MED, F_SMALL, F_TINY = font(34), font(22), font(17), font(14)
BG = (17, 20, 24)                  # classic palette: graphite, greys, one red accent
PANEL, LINE, INK_HI, INK_LO = (27, 31, 36), (70, 78, 88), (236, 238, 241), (160, 168, 178)
RED, AMBER, GREEN, ENV = (214, 64, 69), (214, 160, 72), (116, 184, 138), (246, 247, 249)


def distance_ramp(x, z):
    """Point colour by distance: light cyan near the train, blue at 60 m, violet far away.
    Red and amber stay free for obstacles and candidates; height adds a little shading."""
    t = np.clip(x / 150., 0, 1)[:, None]
    stops = np.array([[150., 236., 255.], [58., 138., 255.], [128., 84., 226.]])
    lo = np.where(t < .4, stops[0], stops[1])
    hi = np.where(t < .4, stops[1], stops[2])
    k = np.where(t < .4, t / .4, (t - .4) / .6)
    rgb = lo + (hi - lo) * k
    shade = .72 + .28 * np.clip((z + 1.6) / 5.5, 0, 1)[:, None]
    return (rgb * shade).astype(np.uint8)




class Camera:
    def __init__(self, pos=(-2., 0., .45), pitch_deg=-1.5, fov_deg=55.):
        self.pos = np.asarray(pos, float)
        p = np.radians(pitch_deg)
        self.R = np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]])  # world->cam (x fwd)
        self.f = (W / 2) / np.tan(np.radians(fov_deg) / 2)

    def project(self, P):
        q = (np.asarray(P, float) - self.pos) @ self.R.T
        depth = q[:, 0]
        u = W / 2 - self.f * q[:, 1] / np.maximum(depth, 1e-3)
        v = H * .46 - self.f * q[:, 2] / np.maximum(depth, 1e-3)
        return u, v, depth


def draw_points(img, cam, pts, color):
    u, v, d = cam.project(pts)
    ok = (d > .5) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    u, v, d, c = u[ok].astype(int), v[ok].astype(int), d[ok], color[ok]
    order = np.argsort(-d)
    u, v, c = u[order], v[order], c[order]
    img[v, u] = c
    near = d[order] < 40
    img[v[near] + 1, u[near]] = c[near]
    img[v[near], u[near] + 1] = c[near]


def line3d(draw, cam, a, b, fill, width=2):
    P = np.array([a, b], float)
    u, v, d = cam.project(P)
    if (d < .6).any():
        return
    draw.line([(u[0], v[0]), (u[1], v[1])], fill=fill, width=width)


def box3d(draw, cam, lo, hi, fill, width=3):
    lo, hi = np.asarray(lo), np.asarray(hi)
    c = [np.array([hi[0] if i & 1 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 4 else lo[2]]) for i in range(8)]
    for i in range(8):
        for bit in (1, 2, 4):
            if not i & bit:
                line3d(draw, cam, c[i], c[i | bit], fill, width)


def envelope(draw, cam, res, color=ENV):
    rows = res.get('section_frames') or []
    lim = max(res.get('envelope_range_m') or 0, res.get('analysis_range_m') or 0)
    hw = res.get('half_width_m', 1.05)
    top = res.get('vehicle', {}).get('max_height', 3.)
    prev = None
    for r in rows:
        s = r[0]
        if s < 1 or s > lim:
            continue
        ctr, lat, up = np.array(r[1:4]), np.array(r[4:7]), np.array(r[7:10])
        corners = [ctr + lat * x + up * z for x, z in ((-hw, 0), (hw, 0), (hw, top), (-hw, top))]
        if prev is not None:
            for a, b in zip(prev, corners):
                line3d(draw, cam, a, b, color, 1)
        if int(round(s)) % 20 == 0 or prev is None:
            for k in range(4):
                line3d(draw, cam, corners[k], corners[(k + 1) % 4], color, 2)
        prev = corners


def inset(draw, pts, res, x0=W - 330, y0=70, w=310, h=210, xmax=200.):
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=PANEL, outline=LINE)
    sx = w / xmax
    sy = h / 40.

    def to(x, y):
        return x0 + x * sx, y0 + h / 2 - y * sy
    sel = (pts[:, 0] > 0) & (pts[:, 0] < xmax) & (np.abs(pts[:, 1]) < 20)
    q = pts[sel][::6]
    for x, y in q[:, :2]:
        px, py = to(x, y)
        draw.point((px, py), fill=(112, 120, 132))
    path = res.get('path')
    lim = max(res.get('envelope_range_m') or 0, res.get('analysis_range_m') or 0)
    if path:
        P = np.asarray(path)
        k = P[:, 3] <= lim
        pl = [to(x, y) for x, y in P[k][:, :2]]
        if len(pl) > 1:
            draw.line(pl, fill=INK_HI, width=2)
    for c in res.get('candidates', []):
        lo, hi = c['bbox']
        col = RED if c.get('confirmed') else AMBER
        a, b = to(lo[0], hi[1]), to(hi[0] + .5, lo[1])
        draw.rectangle([a[0] - 2, a[1] - 2, b[0] + 2, b[1] + 2], outline=col, width=2)
    draw.text((x0 + 8, y0 + 6), 'Карта пути на 200 м вперед', font=F_TINY, fill=INK_LO)


def hud(draw, res, title, subtitle, t_s, ms):
    draw.rectangle([0, 0, W, 54], fill=PANEL)
    draw.text((22, 12), title, font=F_MED, fill=INK_HI)
    draw.text((W - 22 - draw.textlength(subtitle, font=F_SMALL), 16), subtitle, font=F_SMALL, fill=INK_LO)
    st = res.get('status')
    if res.get('obstacle'):
        text, col = f"ПРЕПЯТСТВИЕ  {res['distance_m']:.1f} м".replace('.', ','), RED
    elif st == 'unknown':
        text, col = 'НЕДОСТАТОЧНО ДАННЫХ', INK_LO
    else:
        text, col = 'ПУТЬ СВОБОДЕН', GREEN
    draw.rectangle([18, H - 118, 560, H - 18], fill=PANEL, outline=col, width=2)
    draw.text((36, H - 108), text, font=F_BIG, fill=col)
    info = res.get('path_info', {})
    v = info.get('odometry_ds')
    lim = res.get('analysis_range_m') or 0
    line = f"путь проверен на {lim:.0f} м   ·   скорость {abs(v) * 10 * 3.6:.0f} км/ч   ·   расчёт {ms:.0f} мс" if v is not None else \
        f"путь проверен на {lim:.0f} м   ·   расчёт {ms:.0f} мс"
    draw.text((38, H - 56), line, font=F_SMALL, fill=INK_LO)
    draw.text((W - 170, H - 40), f'{t_s:5.1f} с'.replace('.', ','), font=F_SMALL, fill=INK_LO)


def card(text, sub, n):
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((80, H / 2 - 60), text, font=font(40), fill=INK_HI)
    y = H / 2
    for s in sub:
        d.text((80, y), s, font=F_MED, fill=INK_LO)
        y += 34
    return [np.asarray(img)] * n


def segment(source, start, stop, title, subtitle, step=1, warm=15, max_px=None, modify=None):
    det = TunnelDetector(TunnelConfig())
    cam = Camera()
    frames = []
    t0 = None
    for i, pts, stamp in iter_frames(source, max(0, start - warm), stop):
        if modify is not None:
            pts = modify(i, pts)
        res = det.process(pts, None, stamp)
        if i < start or (i - start) % step:
            continue
        t0 = stamp if t0 is None else t0
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = BG
        sel = (pts[:, 0] > 0) & (pts[:, 0] < 220) & (np.abs(pts[:, 1]) < 30)
        q = pts[sel]
        col = distance_ramp(q[:, 0], q[:, 2])
        # highlight confirmed obstacle points
        for c in res.get('candidates', []):
            if not c.get('confirmed'):
                continue
            lo, hi = np.asarray(c['bbox'][0]) - .05, np.asarray(c['bbox'][1]) + .05
            inb = np.all((q >= lo) & (q <= hi), axis=1)
            col[inb] = RED
        draw_points(img, cam, q, col)
        pil = Image.fromarray(img)
        dr = ImageDraw.Draw(pil)
        envelope(dr, cam, res)
        for c in res.get('candidates', []):
            box3d(dr, cam, c['bbox'][0], c['bbox'][1], RED if c.get('confirmed') else AMBER,
                  3 if c.get('confirmed') else 1)
            if c.get('confirmed'):
                lo, hi = np.asarray(c['bbox'][0]), np.asarray(c['bbox'][1])
                u, v, dpt = cam.project(np.array([[lo[0], (lo[1] + hi[1]) / 2, hi[2] + .3]]))
                if dpt[0] > 1 and 0 < u[0] < W - 120 and 60 < v[0] < H:
                    dr.text((u[0] + 8, v[0] - 26), f"{c['distance_m']:.1f} м".replace('.', ','), font=F_MED, fill=RED)
        inset(dr, pts, res)
        hud(dr, res, title, subtitle, stamp - t0, res.get('latency_ms', 0.))
        frames.append(np.asarray(pil))
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--source', required=True, help='local decoded cache directory')
    ap.add_argument('--start', type=int, default=0, help='first frame to render')
    ap.add_argument('--stop', type=int, required=True, help='exclusive last frame')
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--title', default='Обнаружение препятствий')
    ap.add_argument('--subtitle', default='Локальная запись')
    args = ap.parse_args()
    if args.start < 0 or args.stop <= args.start or args.step < 1 or args.fps < 1:
        ap.error('require 0 <= start < stop, step >= 1 and fps >= 1')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = segment(args.source, args.start, args.stop, args.title, args.subtitle, step=args.step)
    if not frames:
        ap.error('no frames in the requested interval')
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{W}x{H}', '-r', str(args.fps), '-i', '-', '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p', '-crf', '20', str(args.output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for image in frames:
            encoder.stdin.write(np.ascontiguousarray(image).tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError('ffmpeg failed')


if __name__ == '__main__':
    main()
