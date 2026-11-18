import math
import numbers

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, SphericalVoronoi


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _relax_interior(xyz, equator_count):
    """Three area-centroid Lloyd steps; pole and equator stay fixed.

    Mirrored sites restrict the Voronoi cells to the hemisphere. The vector
    surface integral of each cell is obtained from its great-circle edges.
    This is construction work only and adds no simulated orientations.
    """
    xyz = xyz.copy()
    for _ in range(3):
        reflected = xyz[xyz[:, 2] > 1e-12] * [1, 1, -1]
        voronoi = SphericalVoronoi(np.vstack((xyz, reflected)))
        voronoi.sort_vertices_of_regions()
        for index in range(1, len(xyz) - equator_count):
            vertices = voronoi.vertices[voronoi.regions[index]]
            following = np.roll(vertices, -1, axis=0)
            cross = np.cross(vertices, following)
            length = np.linalg.norm(cross, axis=1)
            valid = length > 1e-14
            angle = np.arctan2(length[valid], (vertices[valid] * following[valid]).sum(axis=1))
            centroid = (cross[valid] * (angle / length[valid])[:, None]).sum(axis=0)
            if centroid @ xyz[index] < 0:
                centroid = -centroid
            xyz[index] = centroid / np.linalg.norm(centroid)
    return xyz


def _constrain_edges(points, triangles, constraints):
    """Recover sector sides, then restore constrained Delaunay legality.

    Full-circle meshes do not use this path. Stereographic projection preserves
    circles, so planar empty-circle tests also give spherical Delaunay edges.
    """
    triangles = triangles.copy()
    edges = {}

    def triangle_edges(t):
        return [tuple(sorted((int(t[i]), int(t[(i + 1) % 3])))) for i in range(3)]

    def store(index, t):
        if index < len(triangles):
            for edge in triangle_edges(triangles[index]):
                edges[edge].discard(index)
                if not edges[edge]:
                    del edges[edge]
        a, b, c = points[t]
        if _cross(b - a, c - a) < 0:
            t = [t[0], t[2], t[1]]
        triangles[index] = t
        for edge in triangle_edges(t):
            edges.setdefault(edge, set()).add(index)

    for index, t in enumerate(triangles):
        for edge in triangle_edges(t):
            edges.setdefault(edge, set()).add(index)

    def opposite(edge):
        adjacent = edges.get(edge, ())
        if len(adjacent) != 2:
            return None
        i, j = sorted(adjacent)
        c = next(int(v) for v in triangles[i] if v not in edge)
        d = next(int(v) for v in triangles[j] if v not in edge)
        a, b = edge
        if (_cross(points[d] - points[c], points[a] - points[c]) *
                _cross(points[d] - points[c], points[b] - points[c])) >= 0:
            return None
        return i, j, c, d

    def flip(edge, data):
        i, j, c, d = data
        a, b = edge
        store(i, [c, d, a])
        store(j, [d, c, b])
        return triangle_edges(triangles[i]) + triangle_edges(triangles[j])

    fixed = set()
    for a, b in constraints:
        target = tuple(sorted((a, b)))
        for _ in range(10 * len(triangles)):
            if target in edges:
                break
            pa, pb = points[[a, b]]
            changed = False
            for edge in list(edges):
                if edge in fixed or a in edge or b in edge:
                    continue
                pc, pd = points[list(edge)]
                if (_cross(pb - pa, pc - pa) * _cross(pb - pa, pd - pa) < 0 and
                        _cross(pd - pc, pa - pc) * _cross(pd - pc, pb - pc) < 0):
                    data = opposite(edge)
                    if data is not None:
                        flip(edge, data)
                        changed = True
                        break
            if not changed:
                raise ValueError("Could not triangulate the azimuthal sector boundary")
        else:
            raise ValueError("Azimuthal boundary triangulation did not converge")
        fixed.add(target)

    pending = list(edges)
    while pending:
        edge = pending.pop()
        if edge in fixed:
            continue
        data = opposite(edge)
        if data is None:
            continue
        i, j, c, d = data
        a, b = edge
        u, v, w = points[[a, b, c]] - points[d]
        determinant = (np.dot(u, u) * _cross(v, w) -
                       np.dot(v, v) * _cross(u, w) +
                       np.dot(w, w) * _cross(u, v))
        orientation = _cross(points[b] - points[a], points[c] - points[a])
        if determinant * np.sign(orientation) > 1e-13:
            pending.extend(flip(edge, data))
    return triangles


def hemisphere_mesh(frequency, phi_limits):
    """Return evaluation vertices, geometric vertices, triangles and gather map.

    Frequency F keeps both legacy budgets: (F-1)**2 triangles and
    F*(F-1)//2 + 2 evaluated vertices. A reduced azimuth is a periodic symmetry
    sector: its geometric end meridian reuses values from its start meridian.
    """
    if isinstance(frequency, bool) or not isinstance(frequency, numbers.Integral) or frequency < 4:
        raise ValueError("grid frequency must be an integer >= 4")
    limits = np.asarray(phi_limits, dtype=np.float64)
    if limits.shape != (2,) or not np.isfinite(limits).all():
        raise ValueError("phi_limits must contain two finite angles")
    start, end = limits
    span = end - start
    if not 0 < span <= 2 * math.pi:
        raise ValueError("phi_limits must span more than 0 and at most 2*pi")
    periodic_circle = span == 2 * math.pi
    count = frequency * (frequency - 1) // 2 + 2
    equator_count = frequency + 1
    side_count = 0 if periodic_circle else min(
        max(1, round((math.pi / 2) / math.sqrt(span / count)) - 1),
        count - equator_count - 2,
    )
    interior_count = count - equator_count - side_count - 1
    i = np.arange(interior_count, dtype=np.float64)
    z_min = 0.5 * math.sqrt(span / count)
    z = z_min + (1 - z_min) * (i + 0.5) / interior_count
    golden_angle = math.pi * (3 - math.sqrt(5))
    phase = np.remainder((i + 0.5) * golden_angle, 2 * math.pi)
    interior = np.column_stack((start + phase * span / (2 * math.pi), np.arccos(z)))
    equator = np.column_stack((start + np.arange(equator_count) * span / equator_count,
                               np.full(equator_count, math.pi / 2)))
    side_theta = np.arange(1, side_count + 1) * (math.pi / 2) / (side_count + 1)
    side = np.column_stack((np.full(side_count, start), side_theta))
    initial = np.vstack(([start, 0.0], interior, equator, side))
    final = initial
    gather = np.arange(count, dtype=np.int64)
    constraints = []
    if not periodic_circle:
        end_side = np.column_stack((np.full(side_count + 1, end),
                                    np.r_[side_theta, math.pi / 2]))
        final = np.vstack((initial, end_side))
        gather = np.r_[gather, np.arange(count - side_count, count), 1 + interior_count]
        for meridian in ([0] + list(range(count - side_count, count)) + [1 + interior_count],
                         [0] + list(range(count, count + side_count + 1))):
            constraints.extend(zip(meridian, meridian[1:]))

    phi = final[:, 0] - start
    if periodic_circle or span <= math.pi:
        theta = final[:, 1]
        xyz = np.column_stack((np.sin(theta) * np.cos(phi),
                               np.sin(theta) * np.sin(phi), np.cos(theta)))
        if periodic_circle:
            xyz = _relax_interior(xyz, equator_count)
            initial = np.column_stack((start + np.remainder(np.arctan2(xyz[:, 1], xyz[:, 0]), 2 * math.pi),
                                       np.arccos(np.clip(xyz[:, 2], -1, 1))))
            initial[0] = [start, 0.0]
            initial[-equator_count:] = equator
            final = initial
        hull = ConvexHull(np.vstack((xyz, np.zeros(3))))
        triangles = hull.simplices[hull.equations[:, 3] < -1e-12].copy()
        u, v, w = np.moveaxis(xyz[triangles], 1, 0)
        reverse = np.einsum("ij,ij->i", u, np.cross(v, w)) < 0
        triangles[reverse] = triangles[reverse][:, [0, 2, 1]]
    else:
        radius = np.tan(final[:, 1] / 2)
        xy = np.column_stack((radius * np.cos(phi), radius * np.sin(phi)))
        triangles = _constrain_edges(xy, Delaunay(xy).simplices, constraints)
        center = xy[triangles].mean(axis=1)
        angle = np.remainder(np.arctan2(center[:, 1], center[:, 0]), 2 * math.pi)
        triangles = triangles[angle < span]
    offset = np.argmin(triangles, axis=1)[:, None]
    triangles = triangles[np.arange(len(triangles))[:, None], (np.arange(3) + offset) % 3]
    triangles = triangles[np.lexsort(triangles.T[::-1])]
    return initial, final, triangles.astype(np.int64, copy=False), gather
