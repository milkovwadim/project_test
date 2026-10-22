# -*- coding: utf-8 -*-
"""
Мышцы реальной голени -> пучки трубок.

Источник анатомии: BodyParts3D, (c) The Database Center for Life Science,
CC Attribution-Share Alike 2.1 Japan (https://github.com/Kevin-Mattheus-Moerman/BodyParts3D).

Каждая мышца режется горизонтальными сечениями. В каждом сечении N центров трубок
расставляются равномерно по площади мышцы (k-means по случайным точкам сечения),
стартуя с положения в предыдущем сечении, — так трубки непрерывно «текут» вдоль
мышцы, повторяя её реальную форму: расходятся в брюшке и сходятся в сухожилие.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Координаты атласа: мм, Z вверх от подошвы, -Y вперёд, правая нога при x < 0.
# Наша система: Z вверх, +Y вперёд, +X наружу (латерально) для правой ноги,
# начало — центр дистального конца большеберцовой кости, z = 0 на уровне низа узла щиколотки.
ATLAS_ORIGIN = np.array([-77.0, -63.0, 85.0])


def atlas_to_model(p):
    """Поворот на 180° вокруг Z (перед -> +Y) и перенос начала координат."""
    q = np.asarray(p, float) - ATLAS_ORIGIN
    return np.stack([-q[..., 0], -q[..., 1], q[..., 2]], axis=-1)


@dataclass
class MuscleSpec:
    key: str
    fma: int
    name: str
    tube_r: float        # радиус трубок, мм
    fill: float = 0.55   # доля площади сечения, занятая трубками
    twist: float = 0.0   # закрутка пучка по всей длине, градусы


MUSCLES = [
    MuscleSpec("gm", 45957, "икроножная, внутренняя головка", 2.6, 0.55, 14.0),
    MuscleSpec("gl", 45960, "икроножная, наружная головка", 2.6, 0.55, -14.0),
    MuscleSpec("sol", 22558, "камбаловидная", 2.4, 0.5, 8.0),
    MuscleSpec("ach", 258847, "ахиллово сухожилие", 1.7, 0.6, 25.0),
    MuscleSpec("ta", 22544, "передняя большеберцовая", 2.3, 0.55, -10.0),
    MuscleSpec("edl", 22548, "длинный разгибатель пальцев", 1.9, 0.55, 6.0),
    MuscleSpec("ehl", 22546, "длинный разгибатель большого пальца", 1.7, 0.55, 0.0),
    MuscleSpec("pl", 22552, "длинная малоберцовая", 2.0, 0.55, 12.0),
    MuscleSpec("pb", 22554, "короткая малоберцовая", 2.0, 0.55, -8.0),
]
BONES = {"tib": 24477, "fib": 24480}


def load_atlas_mesh(stl_dir, fma):
    import trimesh
    m = trimesh.load(f"{stl_dir}/FMA{fma}.stl")
    m.vertices = atlas_to_model(m.vertices)      # поворот на 180° — ориентация граней сохраняется
    if m.volume < 0:
        m.invert()
    return m


def slice_polygons(mesh, zs):
    """Сечения меша плоскостями z = const -> список shapely-геометрий (или None)."""
    import trimesh
    from shapely.ops import unary_union

    out = []
    for z in zs:
        lines = trimesh.intersections.mesh_plane(mesh, [0, 0, 1], [0, 0, z])
        if len(lines) < 3:
            out.append(None)
            continue
        try:
            polys = trimesh.load_path(lines[:, :, :2]).polygons_full
        except Exception:
            polys = []
        g = unary_union(polys) if len(polys) else None
        out.append(g if g is not None and not g.is_empty and g.area > 1.0 else None)
    return out


def _sample_inside(geom, n, rng):
    from shapely import contains_xy

    x0, y0, x1, y1 = geom.bounds
    pts = np.empty((0, 2))
    for _ in range(40):
        c = rng.uniform((x0, y0), (x1, y1), size=(max(n, 64) * 3, 2))
        pts = np.vstack([pts, c[contains_xy(geom, c[:, 0], c[:, 1])]])
        if len(pts) >= n:
            break
    return pts[:n]


def _lloyd(P, S, iters):
    """k-means: центры P подтягиваются к равномерному распределению точек S."""
    for _ in range(iters):
        d = ((S[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        cnt = np.bincount(lab, minlength=len(P)).astype(float)
        sx = np.bincount(lab, S[:, 0], minlength=len(P))
        sy = np.bincount(lab, S[:, 1], minlength=len(P))
        ok = cnt > 0
        P[ok, 0], P[ok, 1] = sx[ok] / cnt[ok], sy[ok] / cnt[ok]
        if (~ok).any():                                 # пустые центры — к ближайшей точке сечения
            P[~ok] = S[d[:, ~ok].argmin(0)]
    return P


def muscle_tubes(mesh, spec: MuscleSpec, z_lo, z_hi, dz=3.0, seed=0, smooth_sigma=3.0, n=None):
    """Пути трубок для одной мышцы: список массивов (k, 3) и радиусы трубок."""
    rng = np.random.default_rng(seed)
    zmin, zmax = mesh.bounds[0, 2] + 0.5, mesh.bounds[1, 2] - 0.5
    zs = np.arange(max(zmin, z_lo), min(zmax, z_hi), dz)
    if len(zs) < 3:
        return [], []
    secs = slice_polygons(mesh, zs)
    areas = np.array([g.area if g is not None else 0.0 for g in secs])
    if areas.max() <= 0:
        return [], []
    if n is None:
        n = max(3, int(round(areas.max() * spec.fill / (math.pi * spec.tube_r ** 2))))
    # идём от самого толстого сечения вверх и вниз, чтобы трубки «раскладывались» из брюшка
    k0 = int(areas.argmax())
    P = np.full((len(zs), n, 2), np.nan)
    S = _sample_inside(secs[k0], 3000, rng)
    init = S[rng.choice(len(S), n, replace=False)].copy()
    P[k0] = _lloyd(init, S, 25)
    for order in (range(k0 + 1, len(zs)), range(k0 - 1, -1, -1)):
        prev = P[k0].copy()
        for k in order:
            g = secs[k]
            if g is None:
                break
            S = _sample_inside(g, 1500, rng)
            if len(S) < 10:
                break
            prev = _lloyd(prev.copy(), S, 4)
            P[k] = prev
    valid = ~np.isnan(P[:, 0, 0])
    idx = np.nonzero(valid)[0]
    k_a, k_b = idx.min(), idx.max()
    P = P[k_a:k_b + 1]
    z = zs[k_a:k_b + 1]
    # сглаживание вдоль трубок + закрутка пучка
    if smooth_sigma > 0 and len(z) > 5:
        from scipy.ndimage import gaussian_filter1d
        P = gaussian_filter1d(P, smooth_sigma, axis=0, mode="nearest")
    c = P.mean(axis=1, keepdims=True)
    ang = np.radians(spec.twist) * (z - z.mean()) / max(z[-1] - z[0], 1.0)
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    d = P - c
    P = c + np.stack([d[..., 0] * ca - d[..., 1] * sa, d[..., 0] * sa + d[..., 1] * ca], axis=-1)
    paths = [np.column_stack([P[:, i, 0], P[:, i, 1], z]) for i in range(n)]
    radii = spec.tube_r * (1.0 + 0.15 * rng.uniform(-1, 1, n))
    return paths, radii


def tube_mesh(path, r, sides=10, cap=True):
    """Труба радиуса r вдоль ломаной path (k, 3): кольца с параллельным переносом рамки."""
    path = np.asarray(path, float)
    t = np.gradient(path, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None] + 1e-12
    a = np.array([1.0, 0.0, 0.0]) if abs(t[0, 0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    nrm = np.cross(t[0], a)
    nrm /= np.linalg.norm(nrm)
    N = [nrm]
    for i in range(1, len(path)):
        v = N[-1] - t[i] * (N[-1] @ t[i])
        N.append(v / (np.linalg.norm(v) + 1e-12))
    N = np.array(N)
    B = np.cross(t, N)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    rr = np.broadcast_to(np.asarray(r, float), (len(path),))
    ring = (path[:, None, :] + rr[:, None, None] * (np.cos(ang)[None, :, None] * N[:, None, :]
                                                     + np.sin(ang)[None, :, None] * B[:, None, :]))
    V = ring.reshape(-1, 3)
    k = len(path)
    F = []
    for i in range(k - 1):
        for j in range(sides):
            a0, a1 = i * sides + j, i * sides + (j + 1) % sides
            b0, b1 = a0 + sides, a1 + sides
            F += [(a0, b1, b0), (a0, a1, b1)]
    if cap:
        c0, c1 = len(V), len(V) + 1
        V = np.vstack([V, path[0], path[-1]])
        for j in range(sides):
            F.append((c0, (j + 1) % sides, j))
            F.append((c1, (k - 1) * sides + j, (k - 1) * sides + (j + 1) % sides))
    return V, np.array(F, dtype=np.int64)
