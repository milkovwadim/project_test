#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Киберпанк-макет протеза правой голени: реальные мышцы -> пучки трубок.

Сборка:
  * гильза (гладкая, как на фото) под культю;
  * мышцы реальной правой голени (BodyParts3D) превращены в пучки трубок; пучки свиты
    в «кабели», выше низа гильзы мышцы обнимают её снаружи;
  * «кости»-стойки — реальные большеберцовая и малоберцовая кости ниже гильзы;
  * узел щиколотки: сухожилия входят в муфты, снизу — 4 отверстия M6 под адаптер стопы;
  * хомуты на пучках и манжета под краем гильзы, куда уходят концы трубок.

Запуск:
    python build_cyber_leg.py --anatomy <папка со STL BodyParts3D> [--out stl]
Анатомия: BodyParts3D, (c) The Database Center for Life Science, CC BY-SA 2.1 Japan.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from anatomy_tubes import BONES, MUSCLES, atlas_to_model, load_atlas_mesh, muscle_tubes, tube_mesh  # noqa: E402
import generate_prosthesis as gp  # noqa: E402

# ---------------- основные размеры (мм, система модели) ----------------
SOCKET_AXIS = np.array([7.0, -22.0])   # ось гильзы — центр сечения голени в верхней трети
HOUSING_TOP = 32.0                      # верх узла щиколотки
TUBE_TOP_GAP = 22.0                     # трубки кончаются на столько ниже края гильзы
CUFF = (34.0, 22.0)                     # манжета: от края гильзы вниз, мм (низ, верх)
LAYER_MIN = 11.0                        # мин. толщина слоя мышц-трубок поверх гильзы
CUFF_LAYER = 5.0                        # толщина слоя трубок там, где они входят в манжету


def socket_params():
    """Гильза под атрофированную культю (~14 см от плато большеберцовой кости)."""
    return gp.Params(pylon_length=235.0, socket_depth=110.0, socket_top_radius=47.0,
                     socket_distal_radius=38.0, socket_flat_radius=20.0,
                     socket_ml_scale=0.95, socket_ap_scale=1.06, wall=4.0, floor=12.0, skirt=12.0,
                     trim_side=10.0, trim_back=24.0, lattice_bottom_radius=25.0)


# ======================================================================
#                              гильза
# ======================================================================

def socket_outer_radius(P, D, z, phi):
    """Наружный радиус гильзы (от её оси) на высоте z в направлении phi (0 — вперёд)."""
    ell = 1.0 / np.sqrt(np.sin(phi) ** 2 / P.socket_ml_scale ** 2 + np.cos(phi) ** 2 / P.socket_ap_scale ** 2)
    return D.rho_out(z) * ell


def rim_height(D, phi):
    return D.z_rim + D.c0 + D.c1 * np.cos(phi) + D.c2 * np.cos(2 * phi)


def socket_mesh(P, D, voxel=0.6):
    """Гильза из генератора прошлой версии (поле расстояний), сдвинутая на ось SOCKET_AXIS."""
    import trimesh
    from skimage.measure import marching_cubes

    zt = D.z_top + 3.0
    half = float(D.rho_out(zt)) * max(P.socket_ml_scale, P.socket_ap_scale) + 4.0
    lo = np.array([-half, -half, D.z_sb - 4.0])
    n = np.ceil((np.array([half, half, zt]) - lo) / voxel).astype(int) + 1
    x = lo[0] + np.arange(n[0]) * voxel
    y = lo[1] + np.arange(n[1]) * voxel
    zs = lo[2] + np.arange(n[2]) * voxel
    X, Y = np.meshgrid(x, y, indexing="ij")
    rho = np.hypot(X / P.socket_ml_scale, Y / P.socket_ap_scale)[..., None]
    phi = np.arctan2(X, Y)
    f_trim = (D.c0 + D.c1 * np.cos(phi) + D.c2 * np.cos(2 * phi))[..., None]
    fp = -D.c1 * np.sin(phi) - 2.0 * D.c2 * np.sin(2 * phi)
    n_trim = (1.0 / np.sqrt(1.0 + (fp / np.maximum(np.hypot(X, Y), 20.0)) ** 2))[..., None]
    field = np.empty(tuple(n), np.float32)
    for k0 in range(0, n[2], 32):
        z = zs[k0:k0 + 32][None, None, :]
        field[:, :, k0:k0 + 32] = gp.socket_sdf(P, D, rho, f_trim, n_trim, z)
    field[:, :, 0] = field[:, :, -1] = 1.0
    field[0], field[-1], field[:, 0], field[:, -1] = 1.0, 1.0, 1.0, 1.0
    v, f, _, _ = marching_cubes(field, 0.0, spacing=(voxel,) * 3)
    v += lo
    v[:, :2] += SOCKET_AXIS
    m = trimesh.Trimesh(v, f, process=True)
    if m.volume < 0:
        m.invert()
    return m


# ======================================================================
#              мышцы -> трубки, стилизация, обжатие вокруг гильзы
# ======================================================================

def ray_socket(P, D, c, u, z):
    """Расстояние от точки c вдоль единичного направления u до наружной поверхности гильзы."""
    r = D.rho_out(z)
    dx, dy = (c[..., 0] - SOCKET_AXIS[0]) / P.socket_ml_scale, (c[..., 1] - SOCKET_AXIS[1]) / P.socket_ap_scale
    ux, uy = u[..., 0] / P.socket_ml_scale, u[..., 1] / P.socket_ap_scale
    a = ux * ux + uy * uy
    b = dx * ux + dy * uy
    cc = dx * dx + dy * dy - r * r
    return (-b + np.sqrt(np.maximum(b * b - a * cc, 0.0))) / a


def deform_table(P, D, muscle_meshes, zs, n_phi=180):
    """Для каждой высоты: центр обжатия (самая «пустая» точка внутри гильзы, где у живой ноги
    глубокие мышцы и кости) и вдоль каждого луча из него — где начинаются и кончаются мышцы."""
    from shapely.geometry import LineString, Point
    from shapely.ops import polylabel, unary_union
    from anatomy_tubes import slice_polygons

    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    per = [slice_polygons(m, zs) for m in muscle_meshes]
    C = np.zeros((len(zs), 2))
    RI = np.zeros((len(zs), n_phi))
    RO = np.zeros((len(zs), n_phi))
    for i, z in enumerate(zs):
        geoms = [p[i] for p in per if p[i] is not None]
        U = unary_union(geoms).buffer(1.5) if geoms else None
        zz = max(z, D.z_sb + 2.0)
        ell = Point(0, 0).buffer(1.0, quad_segs=32)
        from shapely import affinity
        sock = affinity.translate(affinity.scale(ell, D.rho_out(zz) * P.socket_ml_scale * 0.92,
                                                 D.rho_out(zz) * P.socket_ap_scale * 0.92), *SOCKET_AXIS)
        free = sock.difference(U) if U is not None else sock
        if free.is_empty:
            C[i] = SOCKET_AXIS
        else:
            big = max(free.geoms, key=lambda g: g.area) if hasattr(free, "geoms") else free
            pl = polylabel(big, tolerance=0.5)
            C[i] = (pl.x, pl.y)
        if U is None:
            continue
        for j, ph in enumerate(phis):
            u = np.array([math.sin(ph), math.cos(ph)])
            hit = U.intersection(LineString([C[i], C[i] + 220.0 * u]))
            if hit.is_empty:
                continue
            pts = np.array([c for g in getattr(hit, "geoms", [hit]) for c in g.coords])
            d = (pts - C[i]) @ u
            RI[i, j], RO[i, j] = d.min(), d.max()
    from scipy.ndimage import gaussian_filter, gaussian_filter1d
    C = gaussian_filter1d(C, 6.0, axis=0, mode="nearest")
    RI = gaussian_filter(RI, (2.5, 2.5), mode=("nearest", "wrap"))
    RO = gaussian_filter(RO, (2.5, 2.5), mode=("nearest", "wrap"))
    return phis, C, RI, RO


def stylize_cables(paths, rng, group=7, twist_range=(120.0, 260.0), shrink=0.8):
    """Трубки мышцы -> несколько свитых «кабелей» с просветами между ними."""
    if len(paths) < 4:
        return paths
    P = np.stack([p[:, :2] for p in paths], axis=1)          # (k, n, 2)
    z = paths[0][:, 2]
    k_mid = int(np.argmax(np.ptp(P[:, :, 0], axis=1) * np.ptp(P[:, :, 1], axis=1)))
    g = max(1, int(round(len(paths) / group)))
    from scipy.cluster.vq import kmeans2
    _, lab = kmeans2(P[k_mid], g, seed=int(rng.integers(1 << 30)), minit="++")
    out = P.copy()
    L = max(z[-1] - z[0], 1.0)
    for gi in range(g):
        idx = np.nonzero(lab == gi)[0]
        if len(idx) == 0:
            continue
        c = P[:, idx, :].mean(axis=1, keepdims=True)
        tw = np.radians(rng.uniform(*twist_range)) * rng.choice((-1.0, 1.0))
        a = tw * (z - z[0]) / L
        ca, sa = np.cos(a)[:, None], np.sin(a)[:, None]
        d = (P[:, idx, :] - c) * shrink
        out[:, idx, :] = c + np.stack([d[..., 0] * ca - d[..., 1] * sa, d[..., 0] * sa + d[..., 1] * ca], axis=-1)
    return [np.column_stack([out[:, i, 0], out[:, i, 1], z]) for i in range(len(paths))]


def push_around_socket(path, P, D, zs, tab):
    """Выше низа гильзы мышцы «обжимаются» в слой поверх её наружной поверхности:
    вдоль луча из центра обжатия отрезок [начало мышц, конец мышц] переносится наружу гильзы,
    толщина слоя ~ 0.55 от настоящей толщины мышц (выпуклость икры сохраняется)."""
    from scipy.ndimage import gaussian_filter1d, map_coordinates

    phis, C, RI, RO = tab
    z = path[:, 2]
    fi = np.clip(np.interp(z, zs, np.arange(len(zs))), 0, len(zs) - 1)
    c = np.column_stack([np.interp(fi, np.arange(len(zs)), C[:, 0]), np.interp(fi, np.arange(len(zs)), C[:, 1])])
    d = path[:, :2] - c
    t = np.hypot(d[:, 0], d[:, 1])
    u = d / np.maximum(t, 1e-6)[:, None]
    ph = np.arctan2(u[:, 0], u[:, 1]) % (2 * np.pi)
    fj = ph / (2 * np.pi) * len(phis)
    RIw, ROw = np.concatenate([RI, RI[:, :1]], 1), np.concatenate([RO, RO[:, :1]], 1)
    ri = map_coordinates(RIw, [fi, fj], order=1, mode="nearest")
    ro = np.maximum(map_coordinates(ROw, [fi, fj], order=1, mode="nearest"), ri + 1.0)
    s_sock = ray_socket(P, D, c, u, np.maximum(z, D.z_sb + 2.0))
    thick = np.clip(0.55 * (ro - ri), LAYER_MIN, 30.0)
    rim = rim_height(D, np.arctan2(u[:, 0], u[:, 1]))     # приблизительно: направление от центра обжатия
    conv = gp.smoothstep(rim - 95.0, rim - CUFF[0] + 2.0, z)
    thick = thick + conv * (CUFF_LAYER - thick)
    t_new = s_sock + np.clip((t - ri) / (ro - ri), 0.0, 1.2) * thick
    w = gp.smoothstep(D.z_sb - 40.0, D.z_sb + 6.0, z)
    t_f = t + w * (t_new - t)
    out = path.copy()
    out[:, :2] = c + u * t_f[:, None]
    out[:, :2] = gaussian_filter1d(out[:, :2], 2.0, axis=0, mode="nearest")
    return out


def clip_top(path, D, phi_axis_ref=None):
    """Обрезать трубку под краем гильзы (край выше по бокам, ниже сзади)."""
    d = path[:, :2] - SOCKET_AXIS
    phi = np.arctan2(d[:, 0], d[:, 1])
    lim = rim_height(D, phi) - TUBE_TOP_GAP
    keep = path[:, 2] <= lim
    if keep.all():
        return path
    last = np.argmax(~keep)
    return path[:max(last, 2)]


# ======================================================================
#                          щиколотка и хомуты
# ======================================================================

def extrude(poly, z0, z1):
    import trimesh
    m = trimesh.creation.extrude_polygon(poly, z1 - z0)
    m.apply_translation((0, 0, z0))
    return m


def housing_meshes(tendon_feet, bone_feet):
    """Узел щиколотки: обтекаемый корпус по контуру сухожилий и костей, фаски, канавка."""
    import manifold3d as m3d
    import trimesh
    from shapely.geometry import MultiPoint, Point
    from shapely.ops import unary_union

    foot = unary_union([MultiPoint(p).convex_hull.buffer(r + 3.0) for p, r in tendon_feet] + bone_feet)
    base = unary_union([foot, Point(0, 0).buffer(26.0, quad_segs=32)]).convex_hull.buffer(2.0, quad_segs=16)

    from shapely.geometry.polygon import orient

    def cs(poly):
        return m3d.CrossSection([np.asarray(orient(poly, 1.0).exterior.coords)[:-1]])

    def layer(poly, z0, z1, inset_bottom=0.0, inset_top=0.0):
        lo, hi = poly.buffer(-inset_bottom, quad_segs=16), poly.buffer(-inset_top, quad_segs=16)
        # фаска: выпуклая оболочка двух тонких слоёв (контур выпуклый)
        a = cs(lo).extrude(0.01).translate((0, 0, z0))
        b = cs(hi).extrude(0.01).translate((0, 0, z1 - 0.01))
        return m3d.Manifold.batch_hull([a, b])

    groove = cs(base.buffer(0.5)).extrude(2.0).translate((0, 0, 15.0)) - cs(base.buffer(-1.3)).extrude(2.0).translate((0, 0, 15.0))
    body = m3d.Manifold.batch_boolean([layer(base, 0.0, 3.0, 2.5, 0.0), cs(base).extrude(HOUSING_TOP - 6.0).translate((0, 0, 3.0)),
                                       layer(base, HOUSING_TOP - 3.0, HOUSING_TOP, 0.0, 2.5)], m3d.OpType.Add) - groove
    mm = body.to_mesh()
    return [trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=True)], base


def collar_meshes(tendon_feet):
    """Муфты на входе сухожилий в узел щиколотки."""
    from shapely.geometry import MultiPoint
    out = []
    for pts, r in tendon_feet:
        h = MultiPoint(pts).convex_hull.buffer(r + 1.8, quad_segs=12)
        out.append(extrude(h, HOUSING_TOP - 1.0, HOUSING_TOP + 7.0))
        out.append(extrude(h.buffer(0.8, quad_segs=12), HOUSING_TOP + 2.5, HOUSING_TOP + 4.5))
    return out


def clamp_mesh(paths, r, zc, height=5.0):
    """Хомут вокруг пучка на высоте zc."""
    from shapely.geometry import MultiPoint
    pts = []
    for p in paths:
        if p[0, 2] <= zc <= p[-1, 2]:
            k = np.searchsorted(p[:, 2], zc)
            pts.append(p[min(k, len(p) - 1), :2])
    if len(pts) < 3:
        return None
    h = MultiPoint(pts).convex_hull
    ring = h.buffer(r + 1.8, quad_segs=12).difference(h.buffer(max(r - 1.0, 0.6), quad_segs=12))
    if ring.is_empty:
        return None
    return extrude(ring, zc - height / 2, zc + height / 2)


def cuff_mesh(P, D, tube_tops, n=240):
    """Манжета под краем гильзы — дугой там, где в неё входят трубки."""
    import trimesh
    ang = np.sort(np.array([math.atan2(x - SOCKET_AXIS[0], y - SOCKET_AXIS[1]) % (2 * np.pi) for x, y in tube_tops]))
    gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi]))
    k = int(np.argmax(gaps))                          # самый большой промежуток без трубок — разрыв дуги
    a0 = ang[(k + 1) % len(ang)] - math.radians(10.0)
    a1 = a0 + (2 * np.pi - gaps[k]) + math.radians(20.0)
    ph = np.linspace(a0, a1, n)
    zt = rim_height(D, ph)
    z0, z1 = zt - CUFF[0], zt - CUFF[1]
    r_in0 = socket_outer_radius(P, D, z0, ph) - 1.5
    r_in1 = socket_outer_radius(P, D, z1, ph) - 1.5
    th = CUFF_LAYER + 6.0
    s_, c_ = np.sin(ph), np.cos(ph)
    corners = [(r_in0, z0), (r_in0 + th, z0), (r_in1 + th, z1), (r_in1, z1)]
    V = np.stack([np.column_stack([SOCKET_AXIS[0] + rr * s_, SOCKET_AXIS[1] + rr * c_, zz]) for rr, zz in corners], axis=1)
    V = V.reshape(-1, 3)
    F = []
    for i in range(n - 1):
        for a in range(4):
            b = (a + 1) % 4
            p0, p1, q0, q1 = i * 4 + a, i * 4 + b, (i + 1) * 4 + a, (i + 1) * 4 + b
            F += [(p0, q0, q1), (p0, q1, p1)]
    F += [(0, 1, 2), (0, 2, 3)]                       # торцы дуги
    e = (n - 1) * 4
    F += [(e, e + 2, e + 1), (e, e + 3, e + 2)]
    m = trimesh.Trimesh(V, np.array(F), process=True)
    m.fix_normals()
    return m


# ======================================================================
#                                main
# ======================================================================

def to_manifold(m):
    import manifold3d as m3d
    man = m3d.Manifold(m3d.Mesh(vert_properties=np.ascontiguousarray(m.vertices, np.float32),
                                tri_verts=np.ascontiguousarray(m.faces, np.uint32)))
    if man.status() != m3d.Error.NoError:
        raise RuntimeError(f"не многообразие: {man.status()}")
    return man


def build(anatomy_dir, out_dir, seed=7, voxel=0.6, export_groups=False, log=print):
    import manifold3d as m3d
    import trimesh

    t0 = time.time()
    rng = np.random.default_rng(seed)
    P = socket_params()
    D = gp.derive(P)
    log(f"• гильза: дно полости z={D.z_d:.0f}, передний край z={D.z_rim:.0f}, низ юбки z={D.z_sb:.0f}")

    meshes = {s.key: load_atlas_mesh(anatomy_dir, s.fma) for s in MUSCLES}
    bones = {k: load_atlas_mesh(anatomy_dir, f) for k, f in BONES.items()}
    zt = np.arange(D.z_sb - 44.0, D.z_top + 5.0, 3.0)
    tab = deform_table(P, D, list(meshes.values()), zt)
    log(f"  сечения голени готовы ({time.time() - t0:.0f} с)")

    groups = {}          # имя группы -> список trimesh
    tendon_feet = []
    n_tubes = 0
    for i, spec in enumerate(MUSCLES):
        paths, radii = muscle_tubes(meshes[spec.key], spec, z_lo=HOUSING_TOP - 6.0, z_hi=D.z_top, seed=seed + i)
        if not paths:
            continue
        paths = stylize_cables(paths, rng)
        paths = [clip_top(push_around_socket(p, P, D, zt, tab), D) for p in paths]
        g = groups.setdefault(spec.key, [])
        for p, r in zip(paths, radii):
            if len(p) >= 3:
                V, F = tube_mesh(p, r, sides=10)
                g.append(trimesh.Trimesh(V, F, process=False))
                n_tubes += 1
        # сухожилие входит в узел щиколотки
        foot = [p[0, :2] for p in paths if p[0, 2] < HOUSING_TOP + 2.0]
        if len(foot) >= 1:
            pts = np.array(foot)
            if len(pts) < 3:
                pts = np.vstack([pts, pts + 0.5, pts - 0.5])
            tendon_feet.append((pts, float(np.max(radii))))
        # хомуты: два по длине видимой части пучка (ниже гильзы)
        z_a = max(min(p[0, 2] for p in paths), HOUSING_TOP + 12.0)
        z_b = min(max(p[-1, 2] for p in paths), D.z_sb - 8.0)
        if z_b - z_a > 50:
            for fr in (0.3, 0.68):
                cm = clamp_mesh(paths, float(np.max(radii)), z_a + fr * (z_b - z_a))
                if cm is not None:
                    groups.setdefault("clamps", []).append(cm)
    log(f"• трубок: {n_tubes} ({time.time() - t0:.0f} с)")

    # кости-стойки: от узла щиколотки до низа гильзы, сглаженные
    from shapely.ops import unary_union  # noqa: F401
    from anatomy_tubes import slice_polygons
    bone_feet = []
    for k, b in bones.items():
        b = b.subdivide_loop(2)
        b = b.slice_plane([0, 0, 4.0], [0, 0, 1], cap=True).slice_plane([0, 0, D.z_sb + 14.0], [0, 0, -1], cap=True)
        groups.setdefault("bones", []).append(b)
        s = slice_polygons(b, [HOUSING_TOP - 2.0])[0]
        if s is not None:
            bone_feet.append(s.buffer(3.0))

    hp, base = housing_meshes(tendon_feet, bone_feet)
    groups["housing"] = hp + collar_meshes(tendon_feet)
    tube_tops = [m.vertices[-1, :2] for k in [sp.key for sp in MUSCLES] for m in groups.get(k, [])
                 if m.vertices[-1, 2] > D.z_sb]
    groups["socket"] = [socket_mesh(P, D, voxel), cuff_mesh(P, D, tube_tops)]
    log(f"• гильза, кости, щиколотка, хомуты ({time.time() - t0:.0f} с); объединяю…")

    out_dir.mkdir(parents=True, exist_ok=True)
    if export_groups:
        for name, ms in groups.items():
            trimesh.util.concatenate(ms).export(out_dir / f"group_{name}.stl")
    mans = []
    for ms in groups.values():
        for m in ms:
            if m.volume < 0:
                m.invert()
            mans.append(to_manifold(m))
    body = m3d.Manifold.batch_boolean(mans, m3d.OpType.Add)
    body = body - gp.ankle_cut(gp.Params(ankle_radius=45.0))
    # пылинки нулевого объёма (артефакты слияния пересекающихся трубок) — выбрасываем
    comps = [c for c in sorted(body.decompose(), key=lambda m: -m.volume()) if c.volume() > 50.0]
    for cpt in comps[1:12]:
        bb = cpt.bounding_box()
        log(f"  ! отдельный кусок: {cpt.volume() / 1000:.2f} см³, z {bb[2]:.0f}..{bb[5]:.0f}, x {bb[0]:.0f}..{bb[3]:.0f}, y {bb[1]:.0f}..{bb[4]:.0f}")
    body = m3d.Manifold.batch_boolean(comps, m3d.OpType.Add)
    mm = body.to_mesh()
    full = trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)
    full.export(out_dir / "cyber_leg_full.stl")
    ext = full.bounds[1] - full.bounds[0]
    info = {"tubes": n_tubes, "size_mm": [round(float(e), 1) for e in ext], "volume_cm3": round(full.volume / 1000, 1),
            "triangles": len(full.faces), "watertight": bool(full.is_watertight),
            "bodies": len(comps)}
    json.dump(info, open(out_dir / "cyber_info.json", "w"), ensure_ascii=False, indent=2)
    log(f"• готово за {time.time() - t0:.0f} с: {ext[0]:.0f}×{ext[1]:.0f}×{ext[2]:.0f} мм, "
        f"{info['volume_cm3']} см³, {info['triangles']:,} треуг., замкнута: {info['watertight']}, тел: {info['bodies']}")
    return info


def main():
    ap = argparse.ArgumentParser(description="Киберпанк-макет протеза голени из мышц-трубок")
    ap.add_argument("--anatomy", required=True, help="папка со STL BodyParts3D (FMA*.stl)")
    ap.add_argument("--out", type=Path, default=HERE / "stl")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--voxel", type=float, default=0.6)
    ap.add_argument("--groups", action="store_true", help="дополнительно выгрузить части по группам")
    a = ap.parse_args()
    build(a.anatomy, a.out, a.seed, a.voxel, a.groups)


if __name__ == "__main__":
    main()
