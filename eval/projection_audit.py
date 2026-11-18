"""Compare projection methods on user-provided points and path arrays.

Input NPZ contains points (N, 3), path (M, >=3), and optional scalar phi.
"""
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def closest_coordinates(points, path, phi=0., spatial=False):
    """10 cm global seed followed by continuous Newton refinement.

    Horizontal mode isolates nearest-point search from changes to the height
    convention. Spatial mode also uses grade and an orthonormal banked section.
    This is an experimental reference, not a production projector.
    """
    x = path[:, 0]
    curves = [CubicHermiteSpline(x, path[:, k], np.gradient(path[:, k], x)) for k in (1, 2)]
    grid = np.linspace(x[0], x[-1], int(np.ceil((x[-1]-x[0])/.1))+1)
    centers = np.column_stack((grid, curves[0](grid), curves[1](grid)))
    dimensions = 3 if spatial else 2
    _, seed = cKDTree(centers[:, :dimensions]).query(points[:, :dimensions])
    t = grid[seed]
    for _ in range(8):
        numerator = t-points[:, 0]
        denominator = np.ones(len(t))
        for j in range(2 if spatial else 1):
            residual = curves[j](t)-points[:, j+1]
            derivative = curves[j](t, 1)
            numerator += residual*derivative
            denominator += derivative**2+residual*curves[j](t, 2)
        step = numerator/np.maximum(denominator, 1e-6)
        t = np.clip(t-np.clip(step, -.3, .3), x[0], x[-1])
    center = np.column_stack((t, curves[0](t), curves[1](t)))
    residual = points-center
    dy = curves[0](t, 1)
    lateral = np.column_stack((-dy, np.ones(len(t)), np.zeros(len(t))))
    lateral /= np.linalg.norm(lateral, axis=1)[:, None]
    if spatial:
        tangent = np.column_stack((np.ones(len(t)), dy, curves[1](t, 1)))
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        up = np.cross(tangent, lateral)
        lateral = (lateral+phi*up)/np.sqrt(1+phi**2)
        up = np.cross(tangent, lateral)
        u = np.sum(residual*lateral, axis=1)
        h = np.sum(residual*up, axis=1)
    else:
        u = np.sum(residual*lateral, axis=1)
        h = residual[:, 2]-phi*u
    return u, h, t


def native_coordinates(points, path, phi):
    x = points[:, 0]
    d = np.interp(x, path[:, 0], np.gradient(path[:, 1]))
    u = (points[:, 1]-np.interp(x, path[:, 0], path[:, 1]))/np.sqrt(1+d*d)
    h = points[:, 2]-np.interp(x, path[:, 0], path[:, 2])-phi*u
    return u, h


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with np.load(args.input) as data:
        points = np.asarray(data['points'], float)
        path = np.asarray(data['path'], float)
        phi = float(data['phi']) if 'phi' in data else 0.
    original_u, original_h = native_coordinates(points, path, phi)
    report = {}
    for name, spatial in [('nearest_xy', False), ('nearest_xyz', True)]:
        u, h, _ = closest_coordinates(points, path, phi, spatial=spatial)
        report[name] = dict(samples=len(points),
                            max_lateral_change_m=float(np.max(np.abs(u - original_u))),
                            max_vertical_change_m=float(np.max(np.abs(h - original_h))))
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
