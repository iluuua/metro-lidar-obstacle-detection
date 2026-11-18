"""Render a cab-view still from a user-provided viewer export and benchmark result.

python eval/render_still.py sample 10 output/still.png --tag experiment
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
W, H = 1920, 1080
BG = (17, 20, 24)
PANEL, LINE, INK_HI, INK_LO = (27, 31, 36), (70, 78, 88), (236, 238, 241), (160, 168, 178)
RED, AMBER, GREEN, ENV = (232, 92, 82), (232, 168, 72), (116, 200, 150), (246, 247, 249)
FONTS = ['C:/Windows/Fonts/segoeui.ttf', 'C:/Windows/Fonts/arial.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf',
         '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']


def font(size, bold=False):
    names = (['C:/Windows/Fonts/segoeuib.ttf', 'C:/Windows/Fonts/arialbd.ttf', '/System/Library/Fonts/Supplemental/Arial Bold.ttf'] if bold else []) + FONTS
    for n in names:
        if Path(n).exists():
            return ImageFont.truetype(n, size)
    return ImageFont.load_default()


def ramp(x, z):
    """light cyan near, blue at 60 m, violet far; red and amber stay free for the answer"""
    t = np.clip(x / 150., 0, 1)[:, None]
    stops = np.array([[150., 236., 255.], [58., 138., 255.], [128., 84., 226.]])
    lo, hi = np.where(t < .4, stops[0], stops[1]), np.where(t < .4, stops[1], stops[2])
    k = np.where(t < .4, t / .4, (t - .4) / .6)
    shade = .72 + .28 * np.clip((z - z.min()) / max(np.ptp(z), 1.), 0, 1)[:, None]
    return ((lo + (hi - lo) * k) * shade).astype(np.uint8)


class Cam:
    def __init__(self, pos, pitch_deg, fov_deg=62.):
        self.pos = np.asarray(pos, float)
        p = np.radians(pitch_deg)
        self.R = np.array([[np.cos(p), 0, -np.sin(p)], [0, 1, 0], [np.sin(p), 0, np.cos(p)]])
        self.f = (W / 2) / np.tan(np.radians(fov_deg) / 2)

    def project(self, pts):
        c = (np.asarray(pts, float) - self.pos) @ self.R.T
        ok = c[:, 0] > 1.
        u = W / 2 - self.f * c[:, 1] / np.where(ok, c[:, 0], 1.)
        v = H * .46 - self.f * c[:, 2] / np.where(ok, c[:, 0], 1.)
        return u, v, c[:, 0], ok


def load_points(seq, frame):
    manifest = json.loads((ROOT / 'viewer/data' / seq / 'manifest.json').read_text(encoding='utf-8'))
    for i, rec in enumerate(manifest['frames']):
        if rec['frame'] == frame:
            if rec.get('full_file'):
                pts = np.fromfile(ROOT / 'viewer' / rec['full_file'], '<f4').reshape(-1, 3)
            else:
                from source_clouds import SourceClouds
                pts = np.frombuffer(SourceClouds().read(manifest['summary'], i), '<f4').reshape(-1, 3)
            return pts[np.einsum('ij,ij->i', pts, pts) > 1.]
    raise SystemExit(f'frame {frame} not in {seq}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sequence')
    ap.add_argument('frame', type=int)
    ap.add_argument('output', type=Path)
    ap.add_argument('--tag', default='experiment')
    ap.add_argument('--pitch', type=float, default=None, help='camera pitch, degrees (down is positive); default: along the rail plane of the core, or -1.5 without it')
    ap.add_argument('--title', default='')
    ap.add_argument('--cam', default='-3,0,0.4', help='camera x,y,z in the analysis frame')
    ap.add_argument('--max-range', type=float, default=300.)
    ap.add_argument('--what', default='предмет', help='what the inset shows, in the caption')
    ap.add_argument('--inset', action='store_true', help='magnified inset of the first confirmed box')
    ap.add_argument('--core-result', type=Path, help="the answer of the core for this frame (JSON): the envelope is then drawn on its own axis and rail plane")
    args = ap.parse_args()
    core = json.loads(args.core_result.read_text(encoding='utf-8')) if args.core_result else None
    if core:
        rec = dict(core)
        path_rows = np.asarray(core['path'], float)  # rows [x, lateral, rail height, arc], every 2 m
        rec['py'] = path_rows[:, 1].tolist()
    else:
        rec = next(f for f in json.loads((ROOT / 'output/bench' / args.tag / (args.sequence + '.json')).read_text(encoding='utf-8'))['frames'] if f['frame'] == args.frame)
    pts = load_points(args.sequence, args.frame)
    pitch = args.pitch
    if pitch is None:
        # the sensor is not level with the track (a Falcon looks 2-3 degrees off it): the view follows the rail plane of the core
        pitch = float(np.degrees(np.arctan2(-(np.interp(60., path_rows[:, 0], path_rows[:, 2]) - np.interp(10., path_rows[:, 0], path_rows[:, 2])), 50.))) if core else -1.5
    cam = Cam([float(v) for v in args.cam.split(',')], pitch)
    img = Image.new('RGB', (W, H), BG)
    px = np.asarray(img).copy()
    u, v, depth, ok = cam.project(pts)
    keep = ok & (depth < args.max_range) & (u > 0) & (u < W - 3) & (v > 0) & (v < H - 3)
    order = np.argsort(-depth[keep])
    uu, vv, dd = u[keep][order].astype(int), v[keep][order].astype(int), depth[keep][order]
    col = ramp(pts[keep][order][:, 0], pts[keep][order][:, 2])
    size = np.where(dd < 12, 4, np.where(dd < 45, 3, 2))
    for s in (4, 3, 2):
        m = size == s
        for dx in range(s):
            for dy in range(s):
                px[np.clip(vv[m] + dy, 0, H - 1), np.clip(uu[m] + dx, 0, W - 1)] = col[m]
    img = Image.fromarray(px)
    d = ImageDraw.Draw(img, 'RGBA')

    # floor height along the path, from the points near the axis
    py = rec.get('py') or []
    xs = np.arange(0, 2 * len(py), 2.)
    floor_x = np.arange(4., args.max_range, 6.)
    floor = []
    for x0 in floor_x:
        sel = (pts[:, 0] > x0 - 3) & (pts[:, 0] < x0 + 3) & (np.abs(pts[:, 1] - np.interp(x0, xs, py)) < 1.0) if py else None
        floor.append(np.percentile(pts[sel][:, 2], 8) if sel is not None and sel.sum() > 3 else np.nan)
    floor = np.array(floor)
    good = ~np.isnan(floor)
    floor = np.interp(floor_x, floor_x[good], floor[good]) if good.sum() > 1 else np.full_like(floor_x, -1.1)
    if core:
        # the rail plane of the core itself (its smooth vertical profile), not a floor guessed from the points of this frame:
        # the envelope stands 0.07 m above it and is 3 m high
        floor_x = path_rows[:, 0]
        floor = path_rows[:, 2] - .03
    reach = rec.get('envelope_range_m') or 100.
    for dist in [8, 14, 22, 34, 50, 70, 95, 125, 160, 200, 250, 295]:
        if dist > reach or not py:
            continue
        y0 = float(np.interp(dist, xs, py))
        z0 = float(np.interp(dist, floor_x, floor)) + .1
        corners = np.array([[dist, y0 + 1.05, z0], [dist, y0 - 1.05, z0], [dist, y0 - 1.05, z0 + 3.], [dist, y0 + 1.05, z0 + 3.]])
        cu, cv, cd, cok = cam.project(corners)
        if cok.all():
            a = int(np.clip(255 * (1.2 - dist / 320.), 70, 255))
            d.polygon([tuple(p) for p in zip(cu, cv)], outline=ENV + (a,), width=2 if dist < 120 else 1)
    for z in (0., 3.):  # the four edges of the envelope tube
        for sgn in (1, -1):
            seg = []
            for dist in np.arange(6., min(reach, args.max_range), 12.):
                y0 = float(np.interp(dist, xs, py)) if py else 0.
                z0 = float(np.interp(dist, floor_x, floor)) + .1 + z
                seg.append([dist, y0 + sgn * 1.05, z0])
            su, sv, _, sok = cam.project(np.array(seg))
            pts_line = [(a, b) for a, b, k in zip(su, sv, sok) if k]
            if len(pts_line) > 1:
                d.line(pts_line, fill=ENV + (110,), width=1)

    # confirmed boxes
    focus = None
    for c in rec.get('candidates', []):
        if not c.get('confirmed'):
            continue
        (x0, y0, z0), (x1, y1, z1) = c['bbox']
        pad = max(.25, .002 * x0)
        cs = np.array([[x, y, z] for x in (x0 - pad, x1 + pad) for y in (y0 - pad, y1 + pad) for z in (z0 - pad, z1 + pad)])
        cu, cv, cd, cok = cam.project(cs)
        if not cok.all():
            continue
        edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        for a, b in edges:
            d.line([(cu[a], cv[a]), (cu[b], cv[b])], fill=RED + (255,), width=3)
        focus = focus or (float(cu.mean()), float(cv.mean()), c['distance_m'])
        d.text((cu.max() + 14, cv.min() - 6), f"{c['distance_m']:.0f} м".replace('.', ','), font=font(34, True), fill=RED)

    if focus and args.inset:
        fx, fy, fd = focus
        half = 55
        shift = 24  # the distance label stands to the right of the box
        box = (int(fx - half * 1.33 + shift), int(fy - half), int(fx + half * 1.33 + shift), int(fy + half))
        crop = img.copy().crop(box).resize((560, 420), Image.NEAREST)  # the same picture, with its box, magnified
        ox, oy = W - 560 - 36, 96
        img.paste(crop, (ox, oy))
        d = ImageDraw.Draw(img, 'RGBA')
        d.rectangle([ox, oy, ox + 560, oy + 420], outline=RED + (255,), width=3)
        d.rectangle([ox, oy, ox + 560, oy + 46], fill=PANEL + (235,))
        d.text((ox + 14, oy + 6), f'увеличение ×4: {args.what} на {fd:.0f} м'.replace('.', ','), font=font(28), fill=INK_HI)
        d.rectangle([box[0], box[1], box[2], box[3]], outline=RED + (200,), width=1)

    # HUD
    d.rectangle([0, 0, W, 64], fill=PANEL + (255,))
    d.text((36, 14), args.title or args.sequence, font=font(34), fill=INK_HI)
    state = rec['status']
    color = RED if state == 'obstacle' else AMBER if state == 'candidate' else GREEN
    head = (f"ПРЕПЯТСТВИЕ  {rec['distance_m']:.1f} м".replace('.', ',') if state == 'obstacle' else
            {'candidate': 'ВНИМАНИЕ', 'clear': 'ПУТЬ СВОБОДЕН'}.get(state, 'НЕДОСТАТОЧНО ДАННЫХ'))
    d.rectangle([36, H - 190, 900, H - 40], fill=PANEL + (235,), outline=color + (255,), width=3)
    d.text((62, H - 176), head, font=font(58), fill=color)
    d.text((62, H - 100), f"путь проверен на {rec['analysis_range_m']:.0f} м   ·   расчёт {rec['latency_ms']:.0f} мс".replace('.', ','), font=font(30), fill=INK_LO)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.output)
    print('saved', args.output)


if __name__ == '__main__':
    main()
