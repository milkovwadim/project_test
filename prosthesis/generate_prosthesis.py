#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор 3D-модели протеза голени (транстибиального) с «органической»
ажурной голенью — по мотивам фотографии.

Как устроено
------------
Модель описывается полем расстояний (SDF, signed distance field):

  1. Гильза — приёмная «чашка» под культю: конус с плоским скруглённым дном,
     анатомической линией края (бока выше, сзади выемка под колено)
     и сплошной «юбкой» снизу, которая продолжается в голень.
  2. Голень — оболочка формы ноги (сужается к щиколотке, сзади «икра»),
     в которой вырезаны случайные вытянутые окна (ячейки Вороного вокруг
     случайных центров). Перемычки между окнами — мягкие скруглённые ленты
     разной ширины, перетекающие друг в друга без «узлов».
  3. Щиколотка — цилиндр с плоским низом под стандартный 4-дырочный адаптер
     (female pyramid, «приёмник пирамидки»), на который ставится серийная стопа.

Поле превращается в сетку (marching cubes), сетка упрощается, после чего
крепёжные отверстия, гнёзда под гайки и разрез на две детали делаются точными
булевыми операциями (manifold3d) — так отверстия получаются ровными.

Установка и запуск
------------------
    pip install numpy scipy scikit-image trimesh fast-simplification manifold3d
    python generate_prosthesis.py                  # все варианты, шаг сетки 0.6 мм
    python generate_prosthesis.py --voxel 1.2      # быстрый черновик
    python generate_prosthesis.py --help           # все параметры

Все размеры — в миллиметрах. Ось Z — вверх, Y — вперёд (к носку), X — вбок.
z = 0 — нижняя плоскость щиколотки (туда прикручивается адаптер стопы).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BIG = 1.0e4  # «очень далеко» для поля расстояний
SQRT2 = math.sqrt(2.0)


def _p(default, help_text):
    return field(default=default, metadata={"help": help_text})


@dataclass
class Params:
    # ---------------- Главные размеры: подгоните под себя! ----------------
    pylon_length: float = _p(245.0, "высота от низа щиколотки до дна полости гильзы")
    socket_depth: float = _p(180.0, "глубина гильзы по передней стенке (от дна полости до края)")
    socket_top_radius: float = _p(55.0, "внутренний радиус гильзы у края: обхват/(2*pi)")
    socket_distal_radius: float = _p(41.0, "внутренний радиус гильзы у дна")
    socket_flat_radius: float = _p(22.0, "радиус плоского участка дна полости")
    socket_ml_scale: float = _p(1.05, "овальность гильзы: растяжение вбок")
    socket_ap_scale: float = _p(0.95, "овальность гильзы: растяжение вперёд-назад")
    trim_side: float = _p(12.0, "насколько боковые края гильзы выше переднего")
    trim_back: float = _p(25.0, "насколько задний край ниже переднего (выемка под колено)")
    wall: float = _p(5.0, "толщина стенки гильзы")
    floor: float = _p(14.0, "толщина дна гильзы в центре")
    skirt: float = _p(10.0, "толщина «юбки» под гильзой, к которой прирастают ветки")
    # ---------------- Голень: оболочка с «живыми» окнами ----------------
    shell_thickness: float = _p(10.0, "толщина стенки голени (вверху)")
    shell_grow: float = _p(0.5, "насколько стенка толще внизу, у щиколотки (доля)")
    hole_size: float = _p(28.0, "средний шаг окон (окно + перемычка), мм")
    hole_stretch: float = _p(2.0, "вытянутость окон по вертикали")
    band_width: float = _p(6.5, "средняя ширина перемычек между окнами, мм")
    band_grow: float = _p(1.8, "насколько перемычки шире внизу, у щиколотки (доля)")
    band_variation: float = _p(0.4, "разброс ширины перемычек 0…0.6 (больше = «рукотворнее»)")
    hole_round: float = _p(4.0, "скругление углов окон, мм")
    rim_round: float = _p(4.5, "скругление кромок окон, мм (не больше половины толщины стенки)")
    top_slots: float = _p(0.25, "насколько мельче окна у гильзы (0 = как везде)")
    lattice_bottom_radius: float = _p(23.0, "наружный радиус голени внизу, у щиколотки")
    lattice_taper: float = _p(0.8, "профиль сужения голени книзу (1 = прямой конус)")
    calf_bulge: float = _p(0.18, "«икра»: выпуклость голени сзади (доля радиуса)")
    seed: int = _p(5, "зерно случайности: другое число = другой узор окон")
    blend_body: float = _p(6.0, "радиус плавного перехода голени в щиколотку")
    # ---------------- Щиколотка и крепление стопы ----------------
    ankle_radius: float = _p(25.0, "радиус цилиндра щиколотки")
    ankle_height: float = _p(34.0, "высота цилиндра щиколотки")
    adapter_hole_spacing: float = _p(26.0, "4-дырочный адаптер: расстояние между центрами СОСЕДНИХ отверстий")
    adapter_recess_d: float = _p(0.0, "диаметр центрального углубления под выступ адаптера (0 = нет)")
    adapter_recess_h: float = _p(0.0, "глубина центрального углубления под адаптер")
    bolt_hole_d: float = _p(6.6, "диаметр отверстий под болты M6")
    nut_af: float = _p(10.3, "гнездо гайки M6: размер под ключ + зазор")
    nut_h: float = _p(5.6, "гнездо гайки M6: высота")
    nut_z: float = _p(14.0, "высота низа гаек над нижней плоскостью щиколотки")
    ankle_hole_depth: float = _p(30.0, "глубина отверстий под болты в щиколотке")
    # ---------------- Разъём для варианта из двух деталей ----------------
    split_gap: float = _p(9.0, "на сколько ниже дна полости проходит разъём")
    joint_bolt_radius: float = _p(15.0, "радиус окружности 4 болтов разъёма")
    insert_d: float = _p(8.0, "отверстие под термовставку M6 (диаметр)")
    insert_depth: float = _p(14.0, "отверстие под термовставку M6 (глубина)")
    countersink_d: float = _p(12.6, "диаметр зенковки под потайную головку M6")
    spigot_d: float = _p(20.0, "центрирующий выступ: диаметр")
    spigot_h: float = _p(3.0, "центрирующий выступ: высота")
    fit_clearance: float = _p(0.3, "зазор посадки центрирующего выступа")


# ======================================================================
#                       Примитивы поля расстояний
# ======================================================================

def smin(a, b, k):
    """Плавный минимум (кубический): объединяет две формы с галтелью радиусом ~k."""
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * h * k * (1.0 / 6.0)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def round_intersect(a, b, r):
    """Пересечение двух форм со скруглением общего ребра радиусом r."""
    ua = np.maximum(a + r, 0.0)
    ub = np.maximum(b + r, 0.0)
    return np.minimum(-r, np.maximum(a, b)) + np.sqrt(ua * ua + ub * ub)


def sd_round_cone_2d(qx, qy, r1, r2, h):
    """Тело вращения «шар r1 внизу — шар r2 на высоте h» в профиле (радиус, высота)."""
    b = (r1 - r2) / h
    a = math.sqrt(1.0 - b * b)
    k = -b * qx + a * qy
    d_side = a * qx + b * qy - r1
    d_bot = np.sqrt(qx * qx + qy * qy) - r1
    d_top = np.sqrt(qx * qx + (qy - h) ** 2) - r2
    return np.where(k < 0.0, d_bot, np.where(k > a * h, d_top, d_side))


# ======================================================================
#                          Производные размеры
# ======================================================================

def derive(P: Params) -> SimpleNamespace:
    D = SimpleNamespace()
    D.z_d = P.pylon_length                     # дно полости гильзы
    D.z_fu = D.z_d - P.floor                   # вершина «купола» под дном (свод 45°)
    D.r_b = P.socket_distal_radius - P.socket_flat_radius  # радиус скругления дна
    if D.r_b < 3.0:
        raise SystemExit("socket_distal_radius должен быть больше socket_flat_radius хотя бы на 3 мм")
    D.z_c1 = D.z_d + D.r_b
    top_eff = P.socket_top_radius - P.socket_flat_radius
    slope = (top_eff - D.r_b) / (P.socket_depth - D.r_b)
    D.h_rc = P.socket_depth + 250.0            # верхний шар полости — далеко над краем
    D.r_top_rc = D.r_b + slope * D.h_rc
    D.cb = (D.r_b - D.r_top_rc) / D.h_rc
    D.ca = math.sqrt(1.0 - D.cb * D.cb)
    D.z_rim = D.z_d + P.socket_depth           # передний край гильзы

    def rho_out(z):
        """Наружный «эллиптический» радиус гильзы на высоте z."""
        return P.socket_flat_radius + (D.r_b + P.wall - D.cb * (np.asarray(z) - D.z_c1)) / D.ca

    D.rho_out = rho_out
    # Низ юбки: свод 45° под дном встречается со внутренней стенкой юбки
    z = D.z_fu - 30.0
    for _ in range(60):
        z = D.z_fu - (float(rho_out(z)) - P.skirt / D.ca)
    D.z_junction = z
    D.z_sb = z - 3.0
    # Линия края: f(phi) = c0 + c1 cos(phi) + c2 cos(2 phi); phi = 0 — спереди
    S, B = P.trim_side, P.trim_back
    D.c1 = B / 2.0
    D.c0 = (S - B / 2.0) / 2.0
    D.c2 = (-B / 2.0 - S) / 2.0
    ph = np.linspace(0, 2 * np.pi, 721)
    D.z_top = D.z_rim + float(np.max(D.c0 + D.c1 * np.cos(ph) + D.c2 * np.cos(2 * ph)))
    # Щиколотка, шейка и диапазон решётки
    D.z_lb = P.ankle_height + 10.0             # низ решётчатой голени
    D.z_lt = D.z_sb                            # верх голени = низ юбки гильзы
    # «горлышко» над щиколоткой: сплошной цилиндр внутри низа голени
    D.z_n0, D.z_n1, D.r_n = P.ankle_height - 6.0, D.z_lb + 4.0, P.lattice_bottom_radius - 1.0
    D.z_split = D.z_d - P.split_gap
    if D.z_lt - D.z_lb < 60:
        raise SystemExit("Слишком короткая голень: увеличьте pylon_length или уменьшите floor/ankle_height")
    return D


def envelope(P: Params, D, z, phi):
    """Наружный радиус «голени» (огибающей решётки) на высоте z и угле phi."""
    z = np.asarray(z, float)
    phi = np.asarray(phi, float)
    t = (z - D.z_lb) / (D.z_lt - D.z_lb)
    tc = np.clip(t, 0.0, 1.0)
    r_top = float(D.rho_out(D.z_lt))
    base = P.lattice_bottom_radius + (r_top - P.lattice_bottom_radius) * tc ** P.lattice_taper
    base = base + (D.rho_out(z) - base) * smoothstep(0.8, 1.0, t)   # по касательной переходит в конус гильзы
    ell = 1.0 / np.sqrt(np.sin(phi) ** 2 / P.socket_ml_scale ** 2 + np.cos(phi) ** 2 / P.socket_ap_scale ** 2)
    s = tc * tc * (3.0 - 2.0 * tc)
    ell = 1.0 + (ell - 1.0) * s                       # снизу круг, сверху овал гильзы
    bulge = 1.0 + P.calf_bulge * np.sin(np.pi * tc) ** 2 * ((1.0 - np.cos(phi)) / 2.0) ** 2
    return base * ell * bulge


# ======================================================================
#          Голень: оболочка формы ноги со случайными «живыми» окнами
# ======================================================================
#
# Центры окон случайно разбрасываются по поверхности голени (Poisson-disk:
# не ближе заданного шага), окно = ячейка Вороного вокруг центра. Материал
# остаётся полосами вдоль границ ячеек; ширина каждой полосы своя (случайная),
# углы окон и кромки скруглены. Получаются плоские мягкие перемычки,
# перетекающие друг в друга без «узлов», — как у скульптурной работы.

def make_seeds(P: Params, D):
    rng = np.random.default_rng(P.seed)
    H = D.z_lt - D.z_lb
    span = 0.7 * P.hole_size * P.hole_stretch
    z0, z1 = D.z_lb - span, D.z_lt + span
    phis = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    zt = np.linspace(z0, z1, 240)
    r_mid = np.array([np.mean(envelope(P, D, np.full_like(phis, z), phis)) for z in zt]) - 0.5 * P.shell_thickness

    def spacing(z):
        return P.hole_size * (1.0 - P.top_slots * smoothstep(0.55, 1.0, (z - D.z_lb) / H))

    sp, sz = np.empty(0), np.empty(0)
    fails = 0
    while fails < 4000:
        ph, z = rng.uniform(0.0, 2 * np.pi), rng.uniform(z0, z1)
        if len(sp):
            zm = 0.5 * (sz + z)
            ds = ((sp - ph + np.pi) % (2 * np.pi) - np.pi) * np.interp(zm, zt, r_mid)
            if np.any(np.hypot(ds, (sz - z) / P.hole_stretch) < spacing(zm)):
                fails += 1
                continue
        sp, sz = np.append(sp, ph), np.append(sz, z)
        fails = 0
    R = envelope(P, D, sz, sp) - 0.5 * P.shell_thickness
    seeds = np.stack([R * np.sin(sp), R * np.cos(sp), sz], axis=-1)
    visible = int(np.sum((sz > D.z_lb + 10) & (sz < D.z_lt - 10)))
    return seeds, {"window_centers": len(seeds), "windows_visible_approx": visible}


def band_sdf(P: Params, D, pts, seeds_t, tree, k=12):
    """Расстояние до «перемычки» (<0 — материал, >0 — окно) для точек pts (x, y, z)."""
    s = P.hole_stretch
    pt = pts * (1.0, 1.0, 1.0 / s)                   # вытянутая метрика -> вытянутые окна
    _, nn = tree.query(pt, k=k)
    a = seeds_t[nn[:, 0]]
    t = (pts[:, 2] - D.z_lb) / (D.z_lt - D.z_lb)
    # к низу перемычки шире (там нагрузка больше); у гильзы окна плавно мельчают и закрываются
    grow = 1.0 + P.band_grow * (1.0 - np.clip(t, 0.0, 1.0)) ** 2
    closing = 1.5 * P.hole_size * smoothstep(0.86, 1.04, t)
    q = np.empty((len(pts), k + 1))
    q[:, k - 1] = pts[:, 2] - (D.z_n1 + 2.0)          # окна не опускаются в горловину над щиколоткой
    q[:, k] = (D.z_lt + 2.0) - pts[:, 2]              # и не заходят в юбку гильзы
    for j in range(1, k):
        b = seeds_t[nn[:, j]]
        n = b - a
        n /= np.linalg.norm(n, axis=1)[:, None]
        d = ((0.5 * (a + b) - pt) * n).sum(1) / np.sqrt(n[:, 0] ** 2 + n[:, 1] ** 2 + (n[:, 2] / s) ** 2)
        lo, hi = np.minimum(nn[:, 0], nn[:, j]), np.maximum(nn[:, 0], nn[:, j])
        h = np.modf(np.abs(np.sin(lo * 12.9898 + hi * 78.233)) * 43758.5453)[0]   # своя ширина у каждой перемычки
        w = P.band_width * grow * (1.0 + P.band_variation * (2.0 * h - 1.0)) + closing
        q[:, j - 1] = d - 0.5 * w
    m = q.min(1)
    r = P.hole_round                                  # мягкий минимум = скруглённые углы окон
    return m - r * np.log(np.exp(-(q - m[:, None]) / r).sum(1))


# ======================================================================
#                            Сборка поля
# ======================================================================

class Grid:
    def __init__(self, lo, hi, voxel):
        self.v = float(voxel)
        self.lo = np.asarray(lo, float)
        n = np.ceil((np.asarray(hi, float) - self.lo) / self.v).astype(int) + 1
        self.shape = tuple(int(x) for x in n)
        self.x = self.lo[0] + np.arange(n[0]) * self.v
        self.y = self.lo[1] + np.arange(n[1]) * self.v
        self.z = self.lo[2] + np.arange(n[2]) * self.v


def socket_sdf(P, D, rho, f_trim, n_trim, z):
    qx = np.maximum(rho - P.socket_flat_radius, 0.0)
    cav = sd_round_cone_2d(qx, z - D.z_c1, D.r_b, D.r_top_rc, D.h_rc)          # полость
    d_os = D.ca * (rho - P.socket_flat_radius) + D.cb * (z - D.z_c1) - D.r_b - P.wall
    outer = np.maximum(d_os, D.z_sb - z)                                          # наружный конус + юбка
    hollow = round_intersect(d_os + P.skirt, (z - D.z_fu + rho) / SQRT2, 6.0)     # пустота под дном, свод 45°
    shell = np.maximum(np.maximum(outer, -cav), -hollow)
    trim = (z - D.z_rim - f_trim) * n_trim                                         # анатомический край
    return round_intersect(shell, trim, 0.45 * P.wall)


def sd_rounded_cylinder(rho, z, r, z0, z1, re):
    """Цилиндр радиуса r от z0 до z1 со скруглёнными рёбрами re."""
    hz = 0.5 * (z1 - z0)
    dx = rho - (r - re)
    dz = np.abs(z - 0.5 * (z0 + z1)) - (hz - re)
    return np.minimum(np.maximum(dx, dz), 0.0) + np.sqrt(np.maximum(dx, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2) - re


def ankle_sdf(P, D, rho_c, z):
    cyl = sd_rounded_cylinder(rho_c, z, P.ankle_radius, 0.0, P.ankle_height, 2.0)
    neck = sd_rounded_cylinder(rho_c, z, D.r_n, D.z_n0, D.z_n1, 3.0)
    return smin(cyl, neck, 8.0)


def build_field(P, D, seeds, voxel, log=print):
    from scipy.spatial import cKDTree

    zt = D.z_top + 2.0
    phis = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    lat_max = max(float(np.max(envelope(P, D, np.full_like(phis, z), phis)))
                  for z in np.linspace(D.z_lb, D.z_lt, 60))
    half_x = max(float(D.rho_out(zt)) * P.socket_ml_scale, lat_max) + 3.0
    half_y = max(float(D.rho_out(zt)) * P.socket_ap_scale, lat_max) + 3.0
    grid = Grid((-half_x, -half_y, -2.0), (half_x, half_y, zt + 2.0), voxel)
    log(f"  сетка {grid.shape[0]}×{grid.shape[1]}×{grid.shape[2]} = {np.prod(grid.shape) / 1e6:.1f} млн вокселей")

    X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
    rho_c = np.hypot(X, Y)[..., None]
    rho_e = np.hypot(X / P.socket_ml_scale, Y / P.socket_ap_scale)[..., None]
    phi = np.arctan2(X, Y)
    f_trim = (D.c0 + D.c1 * np.cos(phi) + D.c2 * np.cos(2 * phi))[..., None]
    fp = -D.c1 * np.sin(phi) - 2.0 * D.c2 * np.sin(2 * phi)
    n_trim = (1.0 / np.sqrt(1.0 + (fp / np.maximum(np.hypot(X, Y), 20.0)) ** 2))[..., None]

    field_ = np.full(grid.shape, BIG, np.float32)
    z_ankle_top = D.z_n1 + 10.0
    slab = 24
    for k0 in range(0, grid.shape[2], slab):
        k1 = min(k0 + slab, grid.shape[2])
        zs = grid.z[k0:k1]
        z = zs[None, None, :]
        res = np.full((grid.shape[0], grid.shape[1], k1 - k0), BIG)
        if zs[-1] >= D.z_sb - 10.0:
            res = np.minimum(res, socket_sdf(P, D, rho_e, f_trim, n_trim, z))
        if zs[0] <= z_ankle_top:
            res = np.minimum(res, ankle_sdf(P, D, rho_c, z))
        field_[:, :, k0:k1] = res

    # голень: оболочка по форме ноги, в которой вырезаны окна
    seeds_t = seeds * (1.0, 1.0, 1.0 / P.hole_stretch)
    tree = cKDTree(seeds_t)
    z_lo, z_hi = D.z_n0, D.z_lt + 16.0
    rho2, phi2 = rho_c[..., 0], phi
    for k in np.nonzero((grid.z > z_lo - 4.0) & (grid.z < z_hi + 4.0))[0]:
        z = float(grid.z[k])
        tz = min(max((z - D.z_lb) / (D.z_lt - D.z_lb), 0.0), 1.0)
        th = 0.5 * P.shell_thickness * (1.0 + P.shell_grow * (1.0 - tz) ** 2)   # к щиколотке стенка толще
        sink = float(smoothstep(D.z_lt, D.z_lt + 8.0, z))           # прячем край оболочки в стенку гильзы
        rm = envelope(P, D, z, phi2) - th - sink
        slope = (envelope(P, D, z + 0.5, phi2) - th - rm - sink) / 0.5
        shell = np.abs(rho2 - rm) / np.sqrt(1.0 + slope ** 2) - th
        shell = np.maximum(shell, max(z_lo - z, z - z_hi))
        band = np.full(shell.shape, -BIG)
        near = shell < P.rim_round + 2.0 * grid.v
        if near.any():
            pts = np.stack([X[near], Y[near], np.full(int(near.sum()), z)], axis=-1)
            band[near] = band_sdf(P, D, pts, seeds_t, tree)
        lat = round_intersect(shell, band, P.rim_round)
        if z < D.z_n1 + 25.0:      # к щиколотке — плавная галтель
            field_[:, :, k] = smin(lat, field_[:, :, k], P.blend_body)
        else:                      # к гильзе — встык: поверхности и так продолжают друг друга
            field_[:, :, k] = np.minimum(lat, field_[:, :, k])
    return grid, field_


def section_profile(grid, field_, z_from, z_to, step=2.0):
    """Площадь и наименьший момент сопротивления изгибу горизонтальных сечений
    (считается по полю, без крепёжных отверстий) — чтобы найти самое слабое место."""
    X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
    dA = grid.v ** 2
    out = []
    for z in np.arange(z_from, z_to, step):
        m = field_[:, :, int(round((z - grid.lo[2]) / grid.v))] < 0
        if not m.any():
            continue
        dx = X[m] - X[m].mean()
        dy = Y[m] - Y[m].mean()
        w = min((dy ** 2).sum() * dA / np.abs(dy).max(), (dx ** 2).sum() * dA / np.abs(dx).max())
        out.append({"z": round(float(z), 1), "area_mm2": round(float(m.sum() * dA)), "W_min_mm3": round(float(w))})
    return out


# ======================================================================
#                      Сетка, упрощение, крепёж
# ======================================================================

def field_to_mesh(grid, field_):
    import trimesh
    from skimage.measure import marching_cubes

    verts, faces, _, _ = marching_cubes(field_, level=0.0, spacing=(grid.v,) * 3, allow_degenerate=False)
    verts += grid.lo
    mesh = trimesh.Trimesh(verts, faces, process=True)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def decimate(mesh, max_faces):
    import fast_simplification
    import trimesh

    if len(mesh.faces) <= max_faces:
        return mesh
    reduction = 1.0 - max_faces / len(mesh.faces)
    v, f = fast_simplification.simplify(np.asarray(mesh.vertices), np.asarray(mesh.faces),
                                        target_reduction=reduction, agg=5)
    return trimesh.Trimesh(v, f, process=True)


def to_manifold(mesh):
    import manifold3d as m3d

    mm = m3d.Mesh(vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
                  tri_verts=np.ascontiguousarray(mesh.faces, dtype=np.uint32))
    man = m3d.Manifold(mm)
    if man.status() != m3d.Error.NoError:
        raise RuntimeError(f"сетка не замкнута/не многообразие: {man.status()}")
    return man


def from_manifold(man):
    import trimesh

    mm = man.to_mesh()
    v = np.asarray(mm.vert_properties, dtype=np.float64)[:, :3]
    f = np.asarray(mm.tri_verts, dtype=np.int64)
    return trimesh.Trimesh(v, f, process=True)


def hex_slot(P, length):
    """Гнездо под шестигранную гайку с пазом-входом вдоль +X."""
    from manifold3d import Manifold

    hexp = Manifold.cylinder(P.nut_h, P.nut_af / math.sqrt(3.0), circular_segments=6)
    slot = Manifold.cube((length, P.nut_af, P.nut_h)).translate((0.0, -P.nut_af / 2.0, 0.0))
    return hexp + slot


def ankle_cut(P):
    """4 отверстия под болты адаптера стопы + боковые пазы для гаек."""
    from manifold3d import Manifold

    r_bc = P.adapter_hole_spacing / math.sqrt(2.0)
    parts = []
    for i in range(4):
        th = math.radians(45.0 + 90.0 * i)
        cx, cy = r_bc * math.cos(th), r_bc * math.sin(th)
        parts.append(Manifold.cylinder(P.ankle_hole_depth + 1.0, P.bolt_hole_d / 2.0, circular_segments=48)
                     .translate((cx, cy, -1.0)))
        parts.append(hex_slot(P, P.ankle_radius + 10.0).rotate((0.0, 0.0, math.degrees(th)))
                     .translate((cx, cy, P.nut_z)))
    if P.adapter_recess_d > 0 and P.adapter_recess_h > 0:
        parts.append(Manifold.cylinder(P.adapter_recess_h + 1.0, P.adapter_recess_d / 2.0, circular_segments=96)
                     .translate((0.0, 0.0, -1.0)))
    return Manifold.batch_boolean(parts, _op_add())


def _op_add():
    from manifold3d import OpType
    return OpType.Add


def joint_parts(P, D):
    """Разъём гильза/голень: зенкованные отверстия в дне гильзы, термовставки в голени, центрирующий выступ."""
    from manifold3d import Manifold

    zs, zd = D.z_split, D.z_d
    s_cut, p_cut = [], []
    h_cs = (P.countersink_d - P.bolt_hole_d) / 2.0
    for i in range(4):
        th = math.radians(45.0 + 90.0 * i)
        c = (P.joint_bolt_radius * math.cos(th), P.joint_bolt_radius * math.sin(th))
        s_cut.append(Manifold.cylinder(zd - zs + 2.0, P.bolt_hole_d / 2.0, circular_segments=48)
                     .translate((c[0], c[1], zs - 1.0)))
        s_cut.append(Manifold.cylinder(h_cs, P.bolt_hole_d / 2.0, P.countersink_d / 2.0, circular_segments=48)
                     .translate((c[0], c[1], zd - 0.5 - h_cs)))
        s_cut.append(Manifold.cylinder(15.0, P.countersink_d / 2.0, circular_segments=48)
                     .translate((c[0], c[1], zd - 0.5 - 1e-3)))
        p_cut.append(Manifold.cylinder(P.insert_depth + 1.0, P.insert_d / 2.0, circular_segments=48)
                     .translate((c[0], c[1], zs - P.insert_depth)))
        p_cut.append(Manifold.cylinder(10.0, P.bolt_hole_d / 2.0 + 0.2, circular_segments=48)
                     .translate((c[0], c[1], zs - P.insert_depth - 9.0)))
    rc = P.spigot_d / 2.0 + P.fit_clearance
    s_cut.append(Manifold.cylinder(P.spigot_h + P.fit_clearance + 1.0, rc, circular_segments=128)
                 .translate((0.0, 0.0, zs - 1.0)))
    p_add = Manifold.cylinder(P.spigot_h + 0.02, P.spigot_d / 2.0, circular_segments=128).translate((0.0, 0.0, zs - 0.02))
    return (Manifold.batch_boolean(s_cut, _op_add()), Manifold.batch_boolean(p_cut, _op_add()), p_add)


# ======================================================================
#                                main
# ======================================================================

def build(P: Params, voxel: float, max_faces: int, variants: str, out_dir: Path, mirror: bool, log=print):
    import trimesh

    t0 = time.time()
    D = derive(P)
    seeds, stats = make_seeds(P, D)
    log(f"• голень: {stats['window_centers']} центров окон (видимых ≈ {stats['windows_visible_approx']})")
    log("• считаю поле расстояний…")
    grid, field_ = build_field(P, D, seeds, voxel, log)
    sections = section_profile(grid, field_, P.ankle_height, D.z_d, 2.0)
    weakest = min(sections, key=lambda r: r["W_min_mm3"])
    log(f"  самое слабое сечение голени: z = {weakest['z']} мм, площадь {weakest['area_mm2']} мм², "
        f"момент сопротивления {weakest['W_min_mm3']} мм³")
    log(f"  готово за {time.time() - t0:.0f} с; marching cubes…")
    body = field_to_mesh(grid, field_)
    del field_
    log(f"  сетка: {len(body.faces):,} треугольников, замкнута: {body.is_watertight}")
    body = decimate(body, max_faces)
    log(f"  после упрощения: {len(body.faces):,} треугольников, замкнута: {body.is_watertight}")
    man = to_manifold(body) - ankle_cut(P)

    outputs = {}
    if variants in ("all", "full"):
        outputs["prosthesis_full"] = man
    if variants in ("all", "split"):
        s_cut, p_cut, p_add = joint_parts(P, D)
        socket_part = man.trim_by_plane((0.0, 0.0, 1.0), D.z_split) - s_cut
        pylon_part = (man.trim_by_plane((0.0, 0.0, -1.0), -D.z_split) + p_add) - p_cut
        outputs["prosthesis_part1_socket"] = socket_part.translate((0.0, 0.0, -D.z_split))
        outputs["prosthesis_part2_pylon"] = pylon_part

    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"params": asdict(P), "windows": stats, "parts": {},
            "weakest_section": weakest, "sections_every_2mm": sections}
    for name, m in outputs.items():
        if mirror:
            m = m.mirror((1.0, 0.0, 0.0))
            name += "_mirror"
        mesh = from_manifold(m)
        path = out_dir / f"{name}.stl"
        mesh.export(path)
        ext = mesh.bounds[1] - mesh.bounds[0]
        vol = mesh.volume / 1000.0
        info["parts"][name] = {
            "file": path.name,
            "size_mm": [round(float(e), 1) for e in ext],
            "volume_cm3": round(vol, 1),
            "mass_g_PETG_100pct": round(vol * 1.27),
            "mass_g_PA12_SLS_MJF": round(vol * 1.01),
            "triangles": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
        }
        log(f"• {path.name}: {ext[0]:.0f}×{ext[1]:.0f}×{ext[2]:.0f} мм, {vol:.0f} см³, "
            f"{len(mesh.faces):,} треуг., замкнута: {mesh.is_watertight}")
    info["key_heights_mm"] = {
        "socket_floor": round(D.z_d, 1),
        "socket_front_rim": round(D.z_rim, 1),
        "total_height": round(D.z_top, 1),
        "windows_zone_from_to": [round(D.z_lb, 1), round(D.z_lt, 1)],
        "split_plane": round(D.z_split, 1),
    }
    with open(out_dir / "info.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=2)
    log(f"Готово за {time.time() - t0:.0f} с → {out_dir}")
    return info


def main():
    ap = argparse.ArgumentParser(description="Генератор 3D-модели протеза голени с органической ажурной голенью",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--voxel", type=float, default=0.6, help="шаг сетки, мм (меньше = глаже, но дольше)")
    ap.add_argument("--max-faces", type=int, default=350_000, help="макс. треугольников после упрощения")
    ap.add_argument("--variants", choices=("all", "full", "split"), default="all",
                    help="full — цельная модель, split — гильза и голень отдельно")
    ap.add_argument("--mirror", action="store_true", help="зеркальная копия (узор окон для другой ноги)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "stl", help="папка для STL")
    for f in fields(Params):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default,
                        help=f.metadata.get("help", ""))
    args = ap.parse_args()
    P = Params(**{f.name: getattr(args, f.name) for f in fields(Params)})
    build(P, args.voxel, args.max_faces, args.variants, args.out, args.mirror)


if __name__ == "__main__":
    main()
