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
from dataclasses import dataclass, field, replace
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
BRIDGE_LEVELS = (0.18, 0.5, 0.8)   # уровни плавных развилок между соседними ветвями
SCULPT = 1.7          # «скульптурная» доводка решётки: размытие поля расстояний, мм (щели заплывают,
                      # бугорки сглаживаются, в стыках — галтели); гильзу не трогает
SCULPT_GROW = 0.35    # компенсация: размытие чуть утончает ветви, возвращаем столько мм
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


# связки не обрываются у низа гильзы, а поднимаются по ней рельефом и тают в стенке —
# гильза и решётка читаются одной конструкцией. Доля высоты до края гильзы, на которую
# поднимается связка (по анатомии: икроножная — к колену сзади, передняя большеберцовая и
# гребень большеберцовой кости — спереди, малоберцовые — сбоку)
CLIMB = {"gm": 0.9, "gl": 0.85, "sol": 0.55, "ach": 0.35, "ta": 0.85, "edl": 0.6, "ehl": 0.5, "pl": 0.7,
         "pb": 0.45, "tib": 0.95, "fib": 0.6}
RIM_MARGIN = 22.0     # рельеф кончается не ближе этого к краю гильзы, мм
K_RELIEF = 4.5        # радиус слияния рельефа со стенкой гильзы (ниже, у «яйца», — K_BODY)
# низ гильзы («яйцо») не глухой: окна решётки заходят в него и закругляются, перемычки между
# окнами переходят в ветви — гильза сама расходится на ветви, как на референсе
CARVE_W = 10.0        # «ширина» оставляемой стенки вокруг ветвей у дна яйца, мм (растёт кверху)
CARVE_TOP = 6.0       # окна кончаются не ближе этого ниже дна полости под культю, мм
CARVE_ROUND = 4.0     # скругление краёв окон, мм
# направление «встречной» спирали: икроножная группа в одну сторону, передняя и наружная — в другую
SPIN = {"gm": 1, "gl": 1, "sol": 1, "ach": 1, "ta": -1, "edl": -1, "ehl": -1, "pl": -1, "pb": -1, "tib": 0, "fib": 0}


@dataclass(frozen=True)
class Layout:
    """Раскладка связок: сколько их, какой толщины и как они идут."""
    title: str
    branches: dict = field(default_factory=lambda: dict(BRANCHES))   # мышца -> (сколько связок, радиус, мм)
    thick: float = 1.0              # общий множитель толщины
    thick_by: dict = field(default_factory=dict)   # множители толщины по мышцам и костям
    bones: bool = True              # кости-стойки (большеберцовая и малоберцовая)
    twist: float = None             # закрутка связок одной мышцы вокруг её оси, градусы (None — анатомическая)
    cable: tuple = ()               # (радиус пучка, мм; закрутка, градусы): связки мышцы собраны в свитый «кабель»
    spiral: float = 0.0             # общий винт всех связок вокруг оси голени (от гильзы к щиколотке), градусы
    counter: bool = False           # винт встречный: задняя группа мышц в одну сторону, передняя — в другую
    fan: float = 0.0                # 0..1: насколько развести связки по кругу равномерно (1 — ровный шаг)
    meander: float = MEANDER        # амплитуда «извилин», мм
    wave: tuple = (70.0, 115.0)     # длина волны «извилин», мм
    taper: float = 0.0              # >0: внизу толще, вверху тоньше (ствол -> ветки), <0 — наоборот
    bridges: tuple = BRIDGE_LEVELS  # уровни Y-развилок между соседними связками
    climb: float = 1.0              # насколько высоко связки поднимаются рельефом по гильзе (0 — обрываются у низа)
    carve: float = 1.0              # окна решётки заходят в низ гильзы (0 — «яйцо» глухое)
    sculpt: float = SCULPT          # мягкость «литья»: размытие поля расстояний, мм
    k_branch: float = K_BRANCH      # радиус слияния связок между собой


LAYOUTS = {
    "anatomy": Layout("анатомия: связки по ходу мышц"),
    "spiral": Layout("спираль: весь пучок винтом вокруг голени", spiral=110.0),
    "fan": Layout("веер: связки разведены по кругу ровнее, окна между ними чище", fan=0.75),
    "delta": Layout("дельта: извилистее и больше развилок", meander=6.5, bridges=(0.1, 0.28, 0.46, 0.64, 0.82)),
    "web": Layout("паутина: вдвое больше связок, вдвое тоньше",
                  branches={"gm": (4, 3.6), "gl": (3, 3.4), "sol": (4, 3.4), "ach": (3, 3.0), "ta": (3, 3.4),
                            "edl": (2, 3.0), "ehl": (1, 2.8), "pl": (2, 3.0), "pb": (1, 3.0)},
                  thick_by={"tib": 0.6, "fib": 0.75}, meander=4.5, bridges=(0.1, 0.28, 0.46, 0.64, 0.82),
                  sculpt=1.3, k_branch=5.0),
    "ropes": Layout("канаты: по одной толстой связке на мышцу, большие окна",
                    branches={"gm": (1, 8.0), "gl": (1, 7.0), "sol": (1, 7.5), "ach": (1, 6.5), "ta": (1, 7.5),
                              "pl": (1, 6.0)},
                    meander=3.0, bridges=(0.3, 0.7), sculpt=2.0, k_branch=9.0),
    "tornado": Layout("торнадо: сильный винт, пучок закручен на 220°", spiral=220.0, meander=3.0, thick=0.9),
    "cables": Layout("кабели: связки каждой мышцы свиты в тугой трёхжильный кабель",
                     branches={"gm": (3, 4.4), "gl": (3, 4.2), "sol": (3, 4.2), "ach": (2, 4.0), "ta": (3, 4.0),
                               "edl": (1, 4.0), "pl": (2, 3.8)},
                     cable=(3.6, 1300.0), meander=2.5, bridges=(0.3, 0.7), sculpt=1.0, k_branch=3.0),
    "weave": Layout("плетёнка: задние и передние мышцы закручены навстречу — связки перекрещиваются",
                    spiral=140.0, counter=True, meander=2.0, bridges=(0.5,), thick=0.9),
    "strings": Layout("струны: прямые тонкие связки ровным шагом, без извилин и развилок",
                      branches={"gm": (3, 3.6), "gl": (2, 3.6), "sol": (3, 3.4), "ach": (2, 3.2), "ta": (2, 3.6),
                                "edl": (1, 3.2), "pl": (2, 3.2), "pb": (1, 3.0)},
                      thick_by={"tib": 0.6, "fib": 0.75}, twist=0.0, fan=0.9, meander=0.0, bridges=(),
                      sculpt=1.2, k_branch=4.0),
    "tree": Layout("дерево: толстый ствол у щиколотки ветвится кверху в тонкие веточки",
                   taper=0.45, bridges=(0.08, 0.22, 0.36, 0.5, 0.64, 0.78, 0.9)),
    "snakes": Layout("змеи: сильные частые извилины", meander=10.0, wave=(45.0, 65.0), thick=0.85,
                     bridges=(0.3, 0.7)),
    "athlete": Layout("контраст: сзади густое плетение икры, спереди — пара мощных стоек",
                      branches={"gm": (4, 3.8), "gl": (3, 3.6), "sol": (4, 3.6), "ach": (3, 3.2), "ta": (1, 6.4),
                                "pl": (1, 4.5)},
                      thick_by={"tib": 1.05, "fib": 0.6}, bridges=(0.1, 0.28, 0.46, 0.64, 0.82),
                      sculpt=1.5, k_branch=6.0),
    "molten": Layout("литьё: связки оплавлены и слиты, как жидкий металл", meander=6.0,
                     sculpt=2.7, k_branch=10.0),
}


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


def egg_socket_sdf(P, D, rho, f_trim, n_trim, z, Xs, Ys, holes=True, hollow_egg=True):
    """Гильза: полость под культю, гладкий конус сверху и овальное «яйцо» снизу, как на референсе."""
    z_eq, r_eq, z_tip, z_t = egg(D)
    qx = np.maximum(rho - P.socket_flat_radius, 0.0)
    cav = gp.sd_round_cone_2d(qx, z - D.z_c1, D.r_b, D.r_top_rc, D.h_rc)
    d_os = D.ca * (rho - P.socket_flat_radius) + D.cb * (z - D.z_c1) - D.r_b - P.wall
    k0 = np.sqrt((rho / r_eq) ** 2 + ((z - z_eq) / EGG_C) ** 2)
    k1 = np.sqrt((rho / r_eq ** 2) ** 2 + ((z - z_eq) / EGG_C ** 2) ** 2) + 1e-9
    d_egg = k0 * (k0 - 1.0) / k1
    outer = np.minimum(np.maximum(d_os, z_t - z), d_egg)          # овал касается конуса — без шва
    shell = np.maximum(outer, -cav)
    if hollow_egg:                                                # пустота в «яйце» (если в нём нет окон)
        hollow = gp.round_intersect(outer + P.skirt, (z - D.z_fu + rho) / gp.SQRT2, 6.0)
        shell = np.maximum(shell, -hollow)
    # отверстия: маленькое круглое спереди-изнутри (как на фото) и сток снизу — пустота не запечатана
    a = math.radians(HOLE[0])
    ux, uy = math.sin(a), math.cos(a)
    zh = z_tip + HOLE[1] * EGG_C
    vu = Xs * ux + Ys * uy
    hole = np.maximum(np.sqrt((Xs - vu * ux) ** 2 + (Ys - vu * uy) ** 2 + (z - zh) ** 2) - HOLE[2], -vu)
    drain = np.maximum(np.maximum(np.hypot(Xs, Ys) - 3.5, z - (z_tip + 30.0)), (z_tip - 8.0) - z)
    if not hollow_egg:
        drain = np.full_like(drain, gp.BIG)                       # пустоты нет — и сток не нужен
    if holes:
        shell = np.maximum(shell, -np.minimum(hole, drain))
    trim = (z - D.z_rim - f_trim) * n_trim
    out = gp.round_intersect(shell, trim, 0.45 * P.wall)
    return (out, np.minimum(hole, drain), cav) if not holes else out


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
def shape_branch(path, r0, P, D, zs, tab, z_root_end, rng=None, layout=LAYOUTS["anatomy"], fan=None, spin=1.0,
                 climb=0.0):
    """Стройнее, «извилины», обжатие вокруг гильзы, «корни» в овал, схождение в горловину.
    climb > 0: от места посадки на гильзу (z_root_end) связка поднимается по ней рельефом
    на эту долю высоты до края и постепенно тает в стенке."""
    from scipy.ndimage import gaussian_filter1d, map_coordinates

    path = path[path[:, 2] <= z_root_end].copy()
    if len(path) < 4:
        return None, None
    z_tip = egg(D)[2]
    z_top = z_root_end
    if climb > 0 and path[-1, 2] >= z_root_end - 6.0:
        # продолжение вверх по гильзе: в том же направлении, что и у мышцы, с затуханием
        a, b = path[-1], path[max(len(path) - 5, 0)]
        v = (a[:2] - b[:2]) / max(a[2] - b[2], 1e-6)
        ph = math.atan2(a[0] - SOCKET_AXIS[0], a[1] - SOCKET_AXIS[1])
        z_rim = D.z_rim + D.c0 + D.c1 * math.cos(ph) + D.c2 * math.cos(2 * ph)
        z_top = z_root_end + climb * max(z_rim - RIM_MARGIN - z_root_end, 0.0)
        ze = np.arange(a[2] + 3.0, z_top, 3.0)
        sl = 30.0 * (1.0 - np.exp(-(ze - a[2]) / 30.0))
        path = np.vstack([path, np.column_stack([a[0] + v[0] * sl, a[1] + v[1] * sl, ze])])
    z = path[:, 2]
    # голень стройнее: к оси «щиколотка -> гильза»
    axis = np.outer(gp.smoothstep(40.0, z_tip, z), SOCKET_AXIS)
    path[:, :2] = axis + (path[:, :2] - axis) * SLIM
    if fan is not None and layout.fan:
        # «веер»: на середине голени связки ровнее по кругу; у горловины и у гильзы — как в анатомии
        phis, dphi = fan
        d = path[:, :2] - axis
        rr, ph = np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 0], d[:, 1]) % (2 * np.pi)
        fi = np.clip(np.interp(z, zs, np.arange(len(zs))), 0, len(zs) - 1)
        fj = ph / (2 * np.pi) * len(phis)
        dp = map_coordinates(np.concatenate([dphi, dphi[:, :1]], 1), [fi, fj], order=1, mode="nearest")
        w = layout.fan * gp.smoothstep(NECK_Z[0], NECK_Z[1], z) * (1.0 - gp.smoothstep(z_tip - 30.0, z_tip, z))
        ph = ph + w * dp
        path[:, :2] = axis + rr[:, None] * np.column_stack([np.sin(ph), np.cos(ph)])
    if layout.spiral and spin:
        # винт вокруг оси голени: у гильзы связки на своих местах, к горловине повёрнуты на spiral
        th = spin * np.radians(layout.spiral) * (gp.smoothstep(NECK_Z[0], z_tip, z) - 1.0)
        d = path[:, :2] - axis
        ct, st = np.cos(th), np.sin(th)
        path[:, :2] = axis + np.column_stack([d[:, 0] * ct - d[:, 1] * st, d[:, 0] * st + d[:, 1] * ct])
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
    root = gp.smoothstep(z_root_end - ROOT_LEN, z_root_end, z)       # 0 -> 1 к посадке на гильзу

    def sink(zv):
        """0 -> 1: связка уходит в стенку гильзы (сразу после посадки или в конце рельефа)."""
        if z_top > z_root_end:
            return gp.smoothstep(z_root_end + 0.25 * (z_top - z_root_end), z_top, zv)
        return gp.smoothstep(z_root_end - ROOT_LEN, z_root_end, zv)

    lift = 0.6 - 0.5 * root - 0.9 * sink(z) if z_top > z_root_end else 0.6 - 1.5 * root
    t_layer = s_sock + r0 * lift + np.clip((t - ri) / (ro - ri), 0.0, 1.0) * thick * (1.0 - root)
    w = gp.smoothstep(z_tip - 20.0, z_tip + 25.0, z)
    t_f = t + w * (t_layer - t)
    out = path.copy()
    out[:, :2] = c + u * t_f[:, None]
    # «извилины»: волна поперёк ветви, гаснет у горловины и у корня
    if rng is not None and layout.meander > 0:
        lam, ph0 = rng.uniform(*layout.wave), rng.uniform(0, 2 * np.pi)
        amp = layout.meander * gp.smoothstep(NECK_Z[0], NECK_Z[1], z) * (1.0 - gp.smoothstep(z_root_end - ROOT_LEN, z_root_end, z))
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
    rad = r0 * (1.0 - (0.7 if z_top > z_root_end else 0.45) * sink(zz))   # рельеф сходит на нет острым кончиком
    rad = rad * (1.0 + 0.3 * gp.smoothstep(NECK_Z[1], NECK_Z[0], zz))
    if layout.taper:
        rad = rad * (1.0 + layout.taper * (1.0 - 2.0 * gp.smoothstep(NECK_Z[0], z_tip, zz)))
    return out, rad


def cable(paths, off, turn, rng):
    """Связки одной мышцы собираются в «кабель»: нити ровным шагом по кругу радиуса off
    вокруг оси мышцы и свиты на turn градусов по всей длине."""
    P = np.stack([p[:, :2] for p in paths], axis=1)          # (k, n, 2) — все пути на одних и тех же z
    z = paths[0][:, 2]
    n = len(paths)
    c = P.mean(axis=1)
    d0 = P[len(z) // 2, 0] - c[len(z) // 2]
    a = (math.atan2(d0[1], d0[0]) + 2 * np.pi * np.arange(n)[None, :] / n
         + np.radians(turn) * rng.choice((-1.0, 1.0)) * ((z - z[0]) / max(z[-1] - z[0], 1.0))[:, None])
    P = c[:, None, :] + off * np.stack([np.cos(a), np.sin(a)], axis=-1)
    return [np.column_stack([P[:, i, 0], P[:, i, 1], z]) for i in range(n)]


def fan_table(paths, z_tip, zs, n_phi=120):
    """Таблица «веера»: на каждой высоте угловые положения связок вокруг оси голени
    разводятся к ровному шагу (порядок связок по кругу сохраняется). Возвращает
    (сетка углов, поправка угла [высота, угол]) — гладкую по высоте и по углу."""
    from scipy.ndimage import gaussian_filter
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    A = np.full((len(zs), len(paths)), np.nan)
    for j, p in enumerate(paths):
        ax = np.outer(gp.smoothstep(40.0, z_tip, p[:, 2]), SOCKET_AXIS)
        ang = np.unwrap(np.arctan2(p[:, 0] - ax[:, 0], p[:, 1] - ax[:, 1]))
        on = (zs >= p[0, 2]) & (zs <= p[-1, 2])
        A[on, j] = np.interp(zs[on], p[:, 2], ang) % (2 * np.pi)
    dphi = np.zeros((len(zs), n_phi))
    for i in range(len(zs)):
        a = np.sort(A[i][~np.isnan(A[i])])
        m = len(a)
        if m < 3:
            continue
        step = 2 * np.pi * np.arange(m) / m
        t = step + np.mean(a - step)                              # ровный шаг с тем же «средним поворотом»
        d = t - a
        # поправка для любого угла — линейно между соседними связками, по кругу
        x = np.r_[a[-1] - 2 * np.pi, a, a[0] + 2 * np.pi]
        y = np.r_[d[-1], d, d[0]]
        dphi[i] = np.interp(phis, x, y)
    dphi = gaussian_filter(dphi, (3.0, 2.0), mode=("nearest", "wrap"))
    return phis, dphi


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
        out.append((np.vstack(segs), np.concatenate(rads), key))
    return out


def _tangent(path, k):
    a, b = path[max(k - 2, 0)], path[min(k + 2, len(path) - 1)]
    t = b - a
    return t / max(np.linalg.norm(t), 1e-6)


def make_bridges(branches, D, rng, levels=BRIDGE_LEVELS, max_gap=36.0):
    """Плавные развилки между соседними ветвями разных мышц: перемычка уходит от одной ветви
    по касательной и по касательной же вливается в соседнюю (кубическая кривая Безье) —
    получается Y-образное ветвление, как на референсе, без узлов «под прямым углом»."""
    out = []
    z_lo, z_hi = D.z_n1 + 18.0, egg(D)[2] - 4.0
    for li, fr in enumerate(levels):
        z = z_lo + fr * (z_hi - z_lo) + rng.uniform(-6.0, 6.0)
        pts = []
        for bi, (pth, rad, key) in enumerate(branches):
            if pth[0, 2] + 15 <= z <= pth[-1, 2] - 15:
                k = int(np.searchsorted(pth[:, 2], z))
                pts.append((bi, k, key))
        if len(pts) < 3:
            continue
        c = np.mean([branches[bi][0][k][:2] for bi, k, _ in pts], axis=0)
        ang = lambda e: math.atan2(branches[e[0]][0][e[1]][0] - c[0], branches[e[0]][0][e[1]][1] - c[1])
        pts.sort(key=ang)
        for ia, ib in zip(pts, pts[1:] + pts[:1]):
            (ba, ka, keya), (bb, kb, keyb) = ia, ib
            pa, ra = branches[ba][0], branches[ba][1]
            pb, rb = branches[bb][0], branches[bb][1]
            gap = np.linalg.norm(pa[ka][:2] - pb[kb][:2])
            if keya == keyb or gap > max_gap or gap < ra[ka] + rb[kb] + 2.0:
                continue
            up = (li + len(out)) % 2 == 0                  # чередуем: развилка вверх / вниз
            dz = rng.uniform(26.0, 40.0) * (1.0 if up else -1.0)
            k2 = int(np.clip(np.searchsorted(pb[:, 2], pa[ka][2] + dz), 1, len(pb) - 2))
            p0, p3 = pa[ka], pb[k2]
            t0, t3 = _tangent(pa, ka) * np.sign(dz), _tangent(pb, k2) * np.sign(dz)
            L = np.linalg.norm(p3 - p0)
            p1, p2 = p0 + t0 * L * 0.45, p3 - t3 * L * 0.45
            tt = np.linspace(0, 1, max(8, int(L / 3.0)))[:, None]
            curve = (1 - tt) ** 3 * p0 + 3 * (1 - tt) ** 2 * tt * p1 + 3 * (1 - tt) * tt ** 2 * p2 + tt ** 3 * p3
            r_end = 0.8 * min(ra[ka], rb[k2])
            rr = r_end * (1.0 - 0.22 * np.sin(np.pi * tt[:, 0]))      # тоньше в середине — изящнее
            out.append((curve, rr, "bridge"))
    return out


def bone_path(mesh, z_lo, z_hi, step=8.0):
    from scipy.ndimage import gaussian_filter1d
    zs = np.arange(z_lo, z_hi, step)
    secs = slice_polygons(mesh, zs)
    pts = np.array([(g.centroid.x, g.centroid.y, z) for z, g in zip(zs, secs) if g is not None])
    pts[:, :2] = gaussian_filter1d(pts[:, :2], 1.0, axis=0, mode="nearest")
    return pts


def sculpt(field, grid, z_keep, sigma=SCULPT, log=print):
    """Скульптурная доводка решётки: размытие поля расстояний работает как сглаживающий
    «мастихин»: узкие щели между почти касающимися ветвями заплывают, мелкие бугорки и узелки
    сглаживаются, во внутренних углах стыков появляются ровные галтели; крупные окна остаются.
    Тонкая стенка гильзы (выше z_keep) не трогается — размытие её бы съело."""
    from scipy.ndimage import gaussian_filter
    t0 = time.time()
    k1 = int(np.searchsorted(grid.z, z_keep + 6.0))
    sub = field[:, :, :k1]
    blurred = gaussian_filter(sub, sigma / grid.v) - np.float32(SCULPT_GROW * (sigma / SCULPT) ** 2)
    w = gp.smoothstep(z_keep, z_keep - 12.0, grid.z[:k1])[None, None, :].astype(np.float32)
    field[:, :, :k1] = w * blurred + (1.0 - w) * sub
    log(f"  скульптурная доводка решётки ({time.time() - t0:.0f} с)")
    return field


# ----------------------------------------------------------------------
def build(anatomy_dir, out_dir, seed=7, voxel=0.45, max_faces=700_000, layout="anatomy", log=print):
    from scipy.spatial import cKDTree  # noqa: F401  (проверка зависимостей)

    t0 = time.time()
    lay = LAYOUTS[layout]
    rng = np.random.default_rng(seed)
    lrng = np.random.default_rng(seed + 1000)      # разброс высоты рельефа — своя случайность
    P = socket_params()
    D = gp.derive(P)
    meshes = {s.key: load_atlas_mesh(anatomy_dir, s.fma) for s in MUSCLES}
    z_eq, r_eq, z_tip, _ = egg(D)
    zs = np.arange(z_tip - 44.0, D.z_top + 5.0, 3.0)
    log(f"• раскладка «{layout}» — {lay.title}")
    log(f"• гильза: овал R={r_eq:.0f} мм, низ овала z={z_tip:.0f}, дно полости z={D.z_d:.0f}, край z={D.z_rim:.0f}")
    tab = deform_table(P, D, list(meshes.values()), zs)

    tubes = {}                                     # пути связок по мышцам (до стилизации)
    for i, spec in enumerate(MUSCLES):
        if spec.key in lay.branches:
            n, r0 = lay.branches[spec.key]
            tw = spec.twist * 0.5 if lay.twist is None else math.copysign(lay.twist, spec.twist or 1.0)
            sp = replace(spec, tube_r=r0 * lay.thick * lay.thick_by.get(spec.key, 1.0), twist=tw)
            tubes[spec.key] = muscle_tubes(meshes[spec.key], sp, z_lo=20.0, z_hi=z_tip + ROOT_RANGE[1] + 30.0,
                                           seed=seed + i, n=n, smooth_sigma=5.0)
    fan = fan_table([p for paths, _ in tubes.values() for p in paths], z_tip, zs) if lay.fan else None

    branches = []                                  # (путь, радиусы, группа)
    for spec in MUSCLES:
        if spec.key not in lay.branches:
            continue
        paths, radii = tubes[spec.key]
        if len(paths) >= 2:
            paths = stylize_cables(paths, rng, group=2, twist_range=(150.0, 300.0), shrink=1.0)
            if lay.cable:
                paths = cable(paths, *lay.cable, rng)
        spin = SPIN[spec.key] if lay.counter else 1.0
        for p, r in zip(paths, radii):
            z_end = z_tip + rng.uniform(*ROOT_RANGE)
            climb = min(lay.climb * CLIMB[spec.key] * lrng.uniform(0.8, 1.15), 1.0)
            q, rad = shape_branch(p, float(r), P, D, zs, tab, z_end, rng, lay, fan, spin, climb)
            if q is not None:
                branches.append((q, rad, spec.key))
    branches = tie_loose_ends(branches, z_tip)
    branches += make_bridges(branches, D, rng, lay.bridges)
    for k, f in BONES.items() if lay.bones else ():
        # кости-стойки: по реальной оси кости, вверху так же вливаются в гильзу, как мышцы
        b = load_atlas_mesh(anatomy_dir, f)
        q = bone_path(b, 14.0, z_tip + 80.0)
        z_end = z_tip + (40.0 if k == "tib" else 26.0)
        r_b = BONE_R[k] * lay.thick * lay.thick_by.get(k, 1.0)
        q, rad = shape_branch(q, r_b, P, D, zs, tab, z_end, rng, lay, fan, SPIN[k] if lay.counter else 1.0,
                              min(lay.climb * CLIMB[k], 1.0))
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
    lo[:2] -= 3.0 * lay.sculpt
    hi[:2] += 3.0 * lay.sculpt
    lo[2], hi[2] = -3.0 * lay.sculpt - 2.0, zt + 2.0             # запас под размытие: форма не упирается в край
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
    holes = np.full(grid.shape, gp.BIG, np.float32)
    cavity = np.full(grid.shape, gp.BIG, np.float32)
    for k0 in range(0, grid.shape[2], 24):
        z = grid.z[k0:k0 + 24][None, None, :]
        res = np.full((grid.shape[0], grid.shape[1], z.shape[2]), gp.BIG)
        if z.max() >= z_tip - 10.0:
            sock, hl, cav = egg_socket_sdf(P, D, rho_e, f_trim, n_trim, z, Xs3, Ys3, holes=False,
                                           hollow_egg=not lay.carve)
            res = np.minimum(res, sock)
            holes[:, :, k0:k0 + 24] = hl
            cavity[:, :, k0:k0 + 24] = cav
        if z.min() <= STEM[2] + 12.0:
            ank = gp.ankle_sdf(P, D, rho_c, z)
            stem = gp.sd_round_cone_2d(rho_c, z - STEM[0], STEM[1], STEM[3], STEM[2] - STEM[0])
            res = np.minimum(res, gp.smin(ank, stem, 8.0))
        base[:, :, k0:k0 + 24] = res

    lat = np.full(grid.shape, gp.BIG, np.float32)
    # поле связки считаем не ближе радиуса слияния с гильзой: иначе у тонких связок
    # на краю расчётной коробки остаётся ступенька
    reach = max(lay.k_branch, K_BODY) + 2 * voxel
    for pts, rad, _ in branches:
        pad = float(rad.max()) + reach
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
            ps = max(rad[j], rad[j + 1]) + reach
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
        lat[sl] = gp.smin(lat[sl], local.astype(np.float32), np.float32(lay.k_branch))
    if lay.carve:
        # окна в «яйце»: стенка остаётся только ближе w(z) к ветвям; w растёт кверху и уходит
        # в бесконечность к дну полости — арки окон закругляются сами, полость под культю цела.
        # Низ гильзы при этом сплошной: снизу получается свод из ветвей, а не голая пустота
        from scipy.ndimage import distance_transform_edt, gaussian_filter
        z_full = D.z_d - CARVE_TOP
        k_lo, k_hi = np.searchsorted(grid.z, z_tip - 15.0), np.searchsorted(grid.z, z_full)
        d_lat = distance_transform_edt(lat[:, :, k_lo:k_hi] > 0).astype(np.float32) * np.float32(voxel)
        d_lat = gaussian_filter(d_lat, 1.0 / voxel)                # расстояние до ветвей — и вдали от них
        sv = gp.smoothstep(z_tip - 5.0, z_full, grid.z[k_lo:k_hi])
        wz = (lay.carve * CARVE_W * sv ** 1.5 / np.maximum(1.0 - sv, 1e-3)).astype(np.float32)[None, None, :]
        base[:, :, k_lo:k_hi] = -gp.smin(-base[:, :, k_lo:k_hi], wz - d_lat, np.float32(CARVE_ROUND))
        del d_lat
    # слияние с гильзой: у «яйца» мягкое (K_BODY), выше, где связки идут рельефом, — чётче
    k_body = (K_BODY - (K_BODY - K_RELIEF) * gp.smoothstep(z_tip + 5.0, z_tip + 40.0, grid.z) * (lay.climb > 0))
    k_body = k_body.astype(np.float32)[None, None, :]
    for k0 in range(0, grid.shape[2], 64):
        base[:, :, k0:k0 + 64] = gp.smin(lat[:, :, k0:k0 + 64], base[:, :, k0:k0 + 64], k_body[:, :, k0:k0 + 64])
    del lat
    base = sculpt(base, grid, D.z_d - 6.0, lay.sculpt, log)
    base = np.maximum(base, -holes)                         # отверстия в овале — уже после доводки
    base = np.maximum(base, -cavity)                        # рельеф не заходит в полость под культю
    del holes, cavity
    sections = gp.section_profile(grid, base, P.ankle_height, D.z_d, 2.0)
    weakest = min(sections, key=lambda r: r["W_min_mm3"])
    log(f"  поле готово ({time.time() - t0:.0f} с); самое слабое сечение: z={weakest['z']} мм, "
        f"W={weakest['W_min_mm3']} мм³; сетка…")

    import trimesh
    base += np.float32(1e-4)                                 # без точных нулей — без «перешейков» в сетке
    body = gp.field_to_mesh(grid, base, process=False)
    del base
    trimesh.smoothing.filter_taubin(body, lamb=0.5, nu=0.53, iterations=6)   # мягкое сглаживание без усадки
    man = gp.to_manifold(body)
    tol = 0.02                                               # прореживание без потери замкнутости
    while man.num_tri() > max_faces and tol < 0.5:
        man = man.simplify(tol)
        tol *= 1.6
    log(f"  сетка: {man.num_tri():,} треугольников (допуск упрощения {tol / 1.6:.2f} мм)")
    # запечатанные пустоты (компоненты с отрицательным объёмом) заполняем
    man = gp.m3d_union([c for c in man.decompose() if c.volume() > 0]) - gp.ankle_cut(P)
    for cpt in sorted(man.decompose(), key=lambda m: -m.volume())[1:]:
        bb = cpt.bounding_box()
        log(f"  ! отдельный кусок {cpt.volume() / 1000:.2f} см³: x {bb[0]:.0f}..{bb[3]:.0f} y {bb[1]:.0f}..{bb[4]:.0f} z {bb[2]:.0f}..{bb[5]:.0f}")
    full = gp.from_manifold(man)
    out_dir.mkdir(parents=True, exist_ok=True)
    full.export(out_dir / "organic_leg_full.stl")
    ext = full.bounds[1] - full.bounds[0]
    info = {"layout": layout, "branches": len(branches), "size_mm": [round(float(e), 1) for e in ext],
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
    ap.add_argument("--voxel", type=float, default=0.45)
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="anatomy", help="раскладка связок")
    a = ap.parse_args()
    build(a.anatomy, a.out, a.seed, a.voxel, layout=a.layout)


if __name__ == "__main__":
    main()
