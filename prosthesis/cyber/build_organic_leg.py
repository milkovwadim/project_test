#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Протез правой голени: 60 % — стиль референса (толстые гладкие ветви, плавные слияния,
гладкая гильза, узкая горловина и цилиндр щиколотки), 40 % — реальная анатомия:
ветви идут по мышцам и костям настоящей правой голени (BodyParts3D).

Каждая мышца даёт 1–4 толстые ветви по ходу её волокон (пути — из anatomy_tubes),
большеберцовая и малоберцовая кости — стойки по их реальным осям. Выше низа гильзы
ветви обнимают её и мягкими «корнями» врастают в стенку; внизу сходятся в горловину
над щиколоткой. Всё сливается полем расстояний (smooth-min), как у литой скульптуры.

Запуск:
    python build_organic_leg.py --anatomy <папка со STL BodyParts3D> [--out stl]
Анатомия: BodyParts3D, (c) The Database Center for Life Science, CC BY-SA 2.1 Japan.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import generate_prosthesis as gp  # noqa: E402
from anatomy_tubes import BONES, MUSCLES, load_atlas_mesh, muscle_tubes, slice_polygons  # noqa: E402
from build_cyber_leg import SOCKET_AXIS, deform_table, socket_params, stylize_cables  # noqa: E402

# ветви: мышца -> (сколько ветвей, радиус, мм)
BRANCHES = {
    "gm": (2, 5.9),    # икроножная, внутренняя головка
    "gl": (2, 5.5),    # икроножная, наружная головка
    "sol": (2, 5.3),   # камбаловидная
    "ach": (2, 4.8),   # ахиллово сухожилие — свитая пара
    "ta": (2, 5.2),    # передняя большеберцовая
    "edl": (1, 4.3),   # длинный разгибатель пальцев
    "pl": (1, 4.5),    # длинная малоберцовая
}
BONE_R = {"tib": 6.6, "fib": 4.3}
ROOT_RANGE = (10.0, 34.0)   # «корни» поднимаются на столько мм выше нижней точки «яйца» гильзы
ROOT_LEN = 40.0       # длина участка, на котором ветвь вливается в гильзу
BRIDGE_LEVELS = (0.15, 0.35, 0.55, 0.75, 0.92)   # уровни перемычек между соседними ветвями
NECK_Z = (36.0, 88.0) # внизу ветви сходятся к оси щиколотки на этом участке
SLIM = 0.8            # голень стройнее анатомической (как на референсе)
MEANDER = 5.0         # амплитуда «извилин» ветвей, мм
K_BRANCH = 7.0        # радиус слияния ветвей между собой
K_BODY = 15.0         # радиус слияния с гильзой и щиколоткой
STEM = (30.0, 21.0, 92.0, 13.0)   # гладкий «стебель» над щиколоткой: (z низа, r низа, z верха, r верха)
# низ гильзы — овал («яйцо»), как на референсе
EGG_UP = 18.0         # овал переходит в конус гильзы (по касательной) на столько выше дна полости
EGG_C = 64.0          # вертикальная полуось овала
HOLE = (-45.0, 0.45, 4.5)   # маленькое круглое отверстие на овале, как на фото: угол, высота (доля), радиус


# ----------------------------------------------------------------------
def sd_round_cone_seg(px, py, pz, a, b, r1, r2):
    """Отрезок a-b с радиусами r1, r2 на концах (конус со сферическими торцами)."""
    ba = b - a
    l2 = float(ba @ ba)
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pax, pay, paz = px - a[0], py - a[1], pz - a[2]
    y = pax * ba[0] + pay * ba[1] + paz * ba[2]
    zz = y - l2
    cx, cy, cz = pax * l2 - ba[0] * y, pay * l2 - ba[1] * y, paz * l2 - ba[2] * y
    x2 = cx * cx + cy * cy + cz * cz
    y2, z2 = y * y * l2, zz * zz * l2
    k = np.sign(rr) * rr * rr * x2
    d_top = np.sqrt(x2 + z2) * il2 - r2
    d_bot = np.sqrt(x2 + y2) * il2 - r1
    d_mid = (np.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1
    return np.where(np.sign(zz) * a2 * z2 > k, d_top, np.where(np.sign(y) * a2 * y2 < k, d_bot, d_mid))


def resample(path, step=4.0):
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    t = np.linspace(0, d[-1], max(3, int(d[-1] / step) + 1))
    return np.column_stack([np.interp(t, d, path[:, i]) for i in range(3)])


def egg(D):
    """Овал низа гильзы, касающийся её конуса без излома.
    Возвращает (z центра овала, его радиус, z нижней точки, z касания с конусом)."""
    z_t = D.z_d + EGG_UP
    rho_t = float(D.rho_out(z_t))
    m = -D.cb / D.ca                                  # наклон конуса гильзы dR/dz
    q = m * EGG_C / rho_t
    sq = (-1.0 + math.sqrt(1.0 + 4.0 * q * q)) / (2.0 * q)
    r_e = rho_t / math.sqrt(1.0 - sq * sq)
    z_e = z_t + sq * EGG_C
    return z_e, r_e, z_e - EGG_C, z_t


def egg_radius(D, z):
    """Наружный радиус гильзы (в «эллиптических» координатах) на высоте z: овал внизу, конус выше."""
    z_e, r_e, _, z_t = egg(D)
    z = np.asarray(z, float)
    s = np.clip((z_e - z) / EGG_C, 0.0, 1.0)
    return np.where(z >= z_t, D.rho_out(z), r_e * np.sqrt(np.maximum(1.0 - s * s, 0.0)))


def egg_socket_sdf(P, D, rho, f_trim, n_trim, z, Xs, Ys):
    """Гильза: полость под культю, гладкий конус сверху и овальное «яйцо» снизу, как на референсе."""
    z_eq, r_eq, z_tip, z_t = egg(D)
    qx = np.maximum(rho - P.socket_flat_radius, 0.0)
    cav = gp.sd_round_cone_2d(qx, z - D.z_c1, D.r_b, D.r_top_rc, D.h_rc)
    d_os = D.ca * (rho - P.socket_flat_radius) + D.cb * (z - D.z_c1) - D.r_b - P.wall
    k0 = np.sqrt((rho / r_eq) ** 2 + ((z - z_eq) / EGG_C) ** 2)
    k1 = np.sqrt((rho / r_eq ** 2) ** 2 + ((z - z_eq) / EGG_C ** 2) ** 2) + 1e-9
    d_egg = k0 * (k0 - 1.0) / k1
    outer = np.minimum(np.maximum(d_os, z_t - z), d_egg)          # овал касается конуса — без шва
    hollow = gp.round_intersect(outer + P.skirt, (z - D.z_fu + rho) / gp.SQRT2, 6.0)
    shell = np.maximum(np.maximum(outer, -cav), -hollow)
    # отверстия: маленькое круглое спереди-изнутри (как на фото) и сток снизу — пустота не запечатана
    a = math.radians(HOLE[0])
    ux, uy = math.sin(a), math.cos(a)
    zh = z_tip + HOLE[1] * EGG_C
    vu = Xs * ux + Ys * uy
    hole = np.maximum(np.sqrt((Xs - vu * ux) ** 2 + (Ys - vu * uy) ** 2 + (z - zh) ** 2) - HOLE[2], -vu)
    drain = np.maximum(np.maximum(np.hypot(Xs, Ys) - 3.5, z - (z_tip + 30.0)), (z_tip - 8.0) - z)
    shell = np.maximum(shell, -np.minimum(hole, drain))
    trim = (z - D.z_rim - f_trim) * n_trim
    return gp.round_intersect(shell, trim, 0.45 * P.wall)


def ray_socket(P, D, c, u, z):
    """Расстояние от c вдоль u до наружной поверхности гильзы (с овалом снизу)."""
    r = np.maximum(egg_radius(D, z), 12.0)
    dx, dy = (c[..., 0] - SOCKET_AXIS[0]) / P.socket_ml_scale, (c[..., 1] - SOCKET_AXIS[1]) / P.socket_ap_scale
    ux, uy = u[..., 0] / P.socket_ml_scale, u[..., 1] / P.socket_ap_scale
    a = ux * ux + uy * uy
    b = dx * ux + dy * uy
    cc = dx * dx + dy * dy - r * r
    return (-b + np.sqrt(np.maximum(b * b - a * cc, 0.0))) / a


# ----------------------------------------------------------------------
def shape_branch(path, r0, P, D, zs, tab, z_root_end, rng=None):
    """Стройнее, «извилины», обжатие вокруг гильзы, «корни» в овал, схождение в горловину."""
    from scipy.ndimage import gaussian_filter1d, map_coordinates

    path = path[path[:, 2] <= z_root_end].copy()
    if len(path) < 4:
        return None, None
    z = path[:, 2]
    z_tip = egg(D)[2]
    # голень стройнее: к оси «щиколотка -> гильза»
    axis = np.outer(gp.smoothstep(40.0, z_tip, z), SOCKET_AXIS)
    path[:, :2] = axis + (path[:, :2] - axis) * SLIM
    # --- выше низа гильзы: слой поверх гильзы (как в кабельной версии), к концу — прижать к стенке
    phis, C, RI, RO = tab
    fi = np.clip(np.interp(z, zs, np.arange(len(zs))), 0, len(zs) - 1)
    c = np.column_stack([np.interp(fi, np.arange(len(zs)), C[:, 0]), np.interp(fi, np.arange(len(zs)), C[:, 1])])
    d = path[:, :2] - c
    t = np.hypot(d[:, 0], d[:, 1])
    u = d / np.maximum(t, 1e-6)[:, None]
    fj = (np.arctan2(u[:, 0], u[:, 1]) % (2 * np.pi)) / (2 * np.pi) * len(phis)
    RIw, ROw = np.concatenate([RI, RI[:, :1]], 1), np.concatenate([RO, RO[:, :1]], 1)
    ri = map_coordinates(RIw, [fi, fj], order=1, mode="nearest")
    ro = np.maximum(map_coordinates(ROw, [fi, fj], order=1, mode="nearest"), ri + 1.0)
    s_sock = ray_socket(P, D, c, u, z)
    thick = np.clip(0.25 * (ro - ri), 1.2 * r0, 14.0)
    root = gp.smoothstep(z_root_end - ROOT_LEN, z_root_end, z)       # 0 -> 1 к концу корня
    t_layer = s_sock + r0 * (0.6 - 1.5 * root) + np.clip((t - ri) / (ro - ri), 0.0, 1.0) * thick * (1.0 - root)
    w = gp.smoothstep(z_tip - 20.0, z_tip + 25.0, z)
    t_f = t + w * (t_layer - t)
    out = path.copy()
    out[:, :2] = c + u * t_f[:, None]
    # «извилины»: волна поперёк ветви, гаснет у горловины и у корня
    if rng is not None and MEANDER > 0:
        lam, ph0 = rng.uniform(70.0, 115.0), rng.uniform(0, 2 * np.pi)
        amp = MEANDER * gp.smoothstep(NECK_Z[0], NECK_Z[1], z) * (1.0 - gp.smoothstep(z_root_end - ROOT_LEN, z_root_end, z))
        off = amp * np.sin(2 * np.pi * z / lam + ph0)
        out[:, 0] += -u[:, 1] * off
        out[:, 1] += u[:, 0] * off
    # --- внизу: ветви сходятся к оси щиколотки (как узкая горловина на референсе)
    f = 0.42 + 0.58 * gp.smoothstep(NECK_Z[0], NECK_Z[1], z)
    out[:, :2] *= f[:, None]
    out[:, :2] = gaussian_filter1d(out[:, :2], 2.5, axis=0, mode="nearest")
    out = resample(out, 4.0)
    zz = out[:, 2]
    # радиус: корень сужается и «тает» в стенке гильзы; к щиколотке ветви чуть толще
    rad = r0 * (1.0 - 0.45 * gp.smoothstep(z_root_end - ROOT_LEN, z_root_end, zz))
    rad = rad * (1.0 + 0.3 * gp.smoothstep(NECK_Z[1], NECK_Z[0], zz))
    return out, rad


def tie_loose_ends(branches, z_tip):
    """Мышцы, которые кончаются посреди голени (низ икры, верх ахиллова сухожилия), не висят
    «веточками»: свободный конец плавно уходит к ближайшей точке соседней ветви."""
    out = []
    for i, (p, r, key) in enumerate(branches):
        others = [b for j, b in enumerate(branches) if j != i]
        pts = np.vstack([b[0] for b in others])
        rad = np.concatenate([b[1] for b in others])
        segs = [p]
        rads = [r]
        for end, sign in ((0, -1.0), (-1, 1.0)):
            e = p[end]
            anchored = e[2] < NECK_Z[1] if sign < 0 else e[2] > z_tip - 5.0
            if anchored:
                continue
            # ближайшая точка другой ветви «по ходу» (ниже — для нижнего конца, выше — для верхнего)
            ok = (pts[:, 2] - e[2]) * sign > 6.0
            if not ok.any():
                continue
            d = np.linalg.norm(pts - e, axis=1) + np.where(ok, 0.0, 1e9)
            k = int(np.argmin(d))
            tgt = pts[k]
            ctrl = e + (p[end] - p[end - 3 if end == -1 else 3]) * 0.8          # продолжение направления ветви
            tt = np.linspace(0, 1, 12)[1:, None]
            curve = (1 - tt) ** 2 * e + 2 * (1 - tt) * tt * ctrl + tt ** 2 * tgt
            rr = np.linspace(r[end], 0.8 * min(r[end], rad[k]), len(curve))
            if sign < 0:
                segs.insert(0, curve[::-1])
                rads.insert(0, rr[::-1])
            else:
                segs.append(curve)
                rads.append(rr)
        q = np.vstack(segs)
        order = np.argsort(q[:, 2], kind="stable") if False else np.arange(len(q))
        out.append((q[order], np.concatenate(rads)[order], key))
    return out


def make_bridges(branches, D, rng, max_gap=34.0):
    """Гладкие перемычки между соседними (по окружности) ветвями разных мышц на нескольких уровнях:
    анатомически — межмышечные перегородки, визуально — окна, как на референсе."""
    out = []
    z_lo, z_hi = D.z_n1 + 12.0, egg(D)[2] + 6.0
    for li, fr in enumerate(BRIDGE_LEVELS):
        z = z_lo + fr * (z_hi - z_lo) + rng.uniform(-8.0, 8.0)
        pts = []
        for pth, rad, key in branches:
            if pth[0, 2] <= z <= pth[-1, 2]:
                k = int(np.searchsorted(pth[:, 2], z))
                pts.append((pth[min(k, len(pth) - 1)], float(rad[min(k, len(rad) - 1)]), key))
        if len(pts) < 3:
            continue
        c = np.mean([p[0][:2] for p in pts], axis=0)
        order = sorted(range(len(pts)), key=lambda i: math.atan2(pts[i][0][0] - c[0], pts[i][0][1] - c[1]))
        for a_i, b_i in zip(order, order[1:] + order[:1]):
            (pa, ra, ka), (pb, rb, kb) = pts[a_i], pts[b_i]
            gap = np.linalg.norm(pa[:2] - pb[:2])
            if ka == kb or gap > max_gap or gap < ra + rb + 1.0:
                continue
            dz = (1.0 if (li + a_i) % 2 else -1.0) * rng.uniform(10.0, 18.0)  # диагональ: то вверх, то вниз
            q0, q1 = pa.copy(), pb.copy()
            q1[2] += dz
            mid = 0.5 * (q0 + q1)
            mid[:2] += (mid[:2] - c) / max(np.linalg.norm(mid[:2] - c), 1e-6) * 0.5   # по поверхности голени
            path = resample(np.vstack([q0, mid, q1]), 3.0)
            r = 0.75 * min(ra, rb)
            out.append((path, np.full(len(path), r), "bridge"))
    return out


def bone_path(mesh, z_lo, z_hi, step=8.0):
    from scipy.ndimage import gaussian_filter1d
    zs = np.arange(z_lo, z_hi, step)
    secs = slice_polygons(mesh, zs)
    pts = np.array([(g.centroid.x, g.centroid.y, z) for z, g in zip(zs, secs) if g is not None])
    pts[:, :2] = gaussian_filter1d(pts[:, :2], 1.0, axis=0, mode="nearest")
    return pts


# ----------------------------------------------------------------------
def build(anatomy_dir, out_dir, seed=7, voxel=0.6, max_faces=400_000, log=print):
    from scipy.spatial import cKDTree  # noqa: F401  (проверка зависимостей)

    t0 = time.time()
    rng = np.random.default_rng(seed)
    P = socket_params()
    D = gp.derive(P)
    meshes = {s.key: load_atlas_mesh(anatomy_dir, s.fma) for s in MUSCLES}
    z_eq, r_eq, z_tip, _ = egg(D)
    zs = np.arange(z_tip - 44.0, D.z_top + 5.0, 3.0)
    log(f"• гильза: овал R={r_eq:.0f} мм, низ овала z={z_tip:.0f}, дно полости z={D.z_d:.0f}, край z={D.z_rim:.0f}")
    tab = deform_table(P, D, list(meshes.values()), zs)

    branches = []                                  # (путь, радиусы, группа)
    for i, spec in enumerate(MUSCLES):
        if spec.key not in BRANCHES:
            continue
        n, r0 = BRANCHES[spec.key]
        sp = replace(spec, tube_r=r0, twist=spec.twist * 0.5)
        paths, radii = muscle_tubes(meshes[spec.key], sp, z_lo=20.0, z_hi=z_tip + ROOT_RANGE[1] + 30.0,
                                    seed=seed + i, n=n, smooth_sigma=5.0)
        if n >= 2:
            paths = stylize_cables(paths, rng, group=2, twist_range=(150.0, 300.0), shrink=1.0)
        for p, r in zip(paths, radii):
            z_end = z_tip + rng.uniform(*ROOT_RANGE)
            q, rad = shape_branch(p, float(r), P, D, zs, tab, z_end, rng)
            if q is not None:
                branches.append((q, rad, spec.key))
    branches = tie_loose_ends(branches, z_tip)
    branches += make_bridges(branches, D, rng)
    for k, f in BONES.items():
        # кости-стойки: по реальной оси кости, вверху так же вливаются в гильзу, как мышцы
        b = load_atlas_mesh(anatomy_dir, f)
        q = bone_path(b, 14.0, z_tip + 80.0)
        z_end = z_tip + (40.0 if k == "tib" else 26.0)
        q, rad = shape_branch(q, BONE_R[k], P, D, zs, tab, z_end, rng)
        if q is not None:
            branches.append((q, rad, k))
    log(f"• ветвей: {len(branches)} ({time.time() - t0:.0f} с)")

    # ---------------- поле расстояний ----------------
    allp = np.vstack([b[0] for b in branches])
    rmax = max(float(b[1].max()) for b in branches)
    zt = D.z_top + 2.0
    rs = float(D.rho_out(zt)) * max(P.socket_ml_scale, P.socket_ap_scale) + 4.0
    lo = np.minimum(allp.min(0) - rmax - 6.0, [SOCKET_AXIS[0] - rs, SOCKET_AXIS[1] - rs, -2.0])
    hi = np.maximum(allp.max(0) + rmax + 6.0, [SOCKET_AXIS[0] + rs, SOCKET_AXIS[1] + rs, zt + 2.0])
    lo[2], hi[2] = -2.0, zt + 2.0
    grid = gp.Grid(lo, hi, voxel)
    log(f"  сетка {grid.shape[0]}×{grid.shape[1]}×{grid.shape[2]} = {np.prod(grid.shape) / 1e6:.1f} млн вокселей")
    X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
    Xs, Ys = X - SOCKET_AXIS[0], Y - SOCKET_AXIS[1]
    Xs3, Ys3 = Xs[..., None], Ys[..., None]
    rho_c = np.hypot(X, Y)[..., None]
    rho_e = np.hypot(Xs / P.socket_ml_scale, Ys / P.socket_ap_scale)[..., None]
    phi = np.arctan2(Xs, Ys)
    f_trim = (D.c0 + D.c1 * np.cos(phi) + D.c2 * np.cos(2 * phi))[..., None]
    fp = -D.c1 * np.sin(phi) - 2.0 * D.c2 * np.sin(2 * phi)
    n_trim = (1.0 / np.sqrt(1.0 + (fp / np.maximum(np.hypot(Xs, Ys), 20.0)) ** 2))[..., None]
    base = np.full(grid.shape, gp.BIG, np.float32)
    for k0 in range(0, grid.shape[2], 24):
        z = grid.z[k0:k0 + 24][None, None, :]
        res = np.full((grid.shape[0], grid.shape[1], z.shape[2]), gp.BIG)
        if z.max() >= z_tip - 10.0:
            res = np.minimum(res, egg_socket_sdf(P, D, rho_e, f_trim, n_trim, z, Xs3, Ys3))
        if z.min() <= STEM[2] + 12.0:
            ank = gp.ankle_sdf(P, D, rho_c, z)
            stem = gp.sd_round_cone_2d(rho_c, z - STEM[0], STEM[1], STEM[3], STEM[2] - STEM[0])
            res = np.minimum(res, gp.smin(ank, stem, 8.0))
        base[:, :, k0:k0 + 24] = res

    lat = np.full(grid.shape, gp.BIG, np.float32)
    for pts, rad, _ in branches:
        pad = float(rad.max()) + K_BRANCH + 2 * voxel
        sl = []
        for ax, l, h in zip((grid.x, grid.y, grid.z), pts.min(0) - pad, pts.max(0) + pad):
            sl.append(slice(int(np.searchsorted(ax, l)), int(np.searchsorted(ax, h, side="right"))))
        sl = tuple(sl)
        xs, ys, zz = grid.x[sl[0]], grid.y[sl[1]], grid.z[sl[2]]
        local = np.full((len(xs), len(ys), len(zz)), gp.BIG)
        for j in range(len(pts) - 1):
            a, b = pts[j], pts[j + 1]
            if np.linalg.norm(b - a) < 1e-6:
                continue
            ps = max(rad[j], rad[j + 1]) + K_BRANCH + 2 * voxel
            l_, h_ = np.minimum(a, b) - ps, np.maximum(a, b) + ps
            i0, i1 = np.searchsorted(xs, l_[0]), np.searchsorted(xs, h_[0], side="right")
            j0, j1 = np.searchsorted(ys, l_[1]), np.searchsorted(ys, h_[1], side="right")
            q0, q1 = np.searchsorted(zz, l_[2]), np.searchsorted(zz, h_[2], side="right")
            if i1 <= i0 or j1 <= j0 or q1 <= q0:
                continue
            dd = sd_round_cone_seg(xs[i0:i1, None, None], ys[None, j0:j1, None], zz[None, None, q0:q1],
                                   a, b, float(rad[j]), float(rad[j + 1]))
            sub = local[i0:i1, j0:j1, q0:q1]
            np.minimum(sub, dd, out=sub)
        lat[sl] = gp.smin(lat[sl], local.astype(np.float32), np.float32(K_BRANCH))
    for k0 in range(0, grid.shape[2], 64):
        base[:, :, k0:k0 + 64] = gp.smin(lat[:, :, k0:k0 + 64], base[:, :, k0:k0 + 64], np.float32(K_BODY))
    del lat
    sections = gp.section_profile(grid, base, P.ankle_height, D.z_d, 2.0)
    weakest = min(sections, key=lambda r: r["W_min_mm3"])
    log(f"  поле готово ({time.time() - t0:.0f} с); самое слабое сечение: z={weakest['z']} мм, "
        f"W={weakest['W_min_mm3']} мм³; сетка…")

    body = gp.decimate(gp.field_to_mesh(grid, base), max_faces)
    del base
    man = gp.to_manifold(body)
    # запечатанные пустоты (компоненты с отрицательным объёмом) заполняем
    man = gp.m3d_union([c for c in man.decompose() if c.volume() > 0]) - gp.ankle_cut(P)
    for cpt in sorted(man.decompose(), key=lambda m: -m.volume())[1:]:
        bb = cpt.bounding_box()
        log(f"  ! отдельный кусок {cpt.volume() / 1000:.2f} см³: x {bb[0]:.0f}..{bb[3]:.0f} y {bb[1]:.0f}..{bb[4]:.0f} z {bb[2]:.0f}..{bb[5]:.0f}")
    full = gp.from_manifold(man)
    out_dir.mkdir(parents=True, exist_ok=True)
    full.export(out_dir / "organic_leg_full.stl")
    ext = full.bounds[1] - full.bounds[0]
    info = {"branches": len(branches), "size_mm": [round(float(e), 1) for e in ext],
            "volume_cm3": round(full.volume / 1000, 1), "triangles": int(len(full.faces)),
            "watertight": bool(full.is_watertight), "bodies": len(man.decompose()),
            "weakest_section": weakest, "socket_split_plane_z": round(D.z_split, 1)}
    json.dump(info, open(out_dir / "organic_info.json", "w"), ensure_ascii=False, indent=2)
    log(f"• готово за {time.time() - t0:.0f} с: {ext[0]:.0f}×{ext[1]:.0f}×{ext[2]:.0f} мм, {info['volume_cm3']} см³, "
        f"{info['triangles']:,} треуг., замкнута: {info['watertight']}, тел: {info['bodies']}")
    return info


def main():
    ap = argparse.ArgumentParser(description="Протез голени: ветви по реальной анатомии в стиле референса")
    ap.add_argument("--anatomy", required=True, help="папка со STL BodyParts3D (FMA*.stl)")
    ap.add_argument("--out", type=Path, default=HERE / "stl")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--voxel", type=float, default=0.6)
    a = ap.parse_args()
    build(a.anatomy, a.out, a.seed, a.voxel)


if __name__ == "__main__":
    main()
