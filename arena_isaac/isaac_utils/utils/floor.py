"""Floor height under a point, for pedestrians in worlds whose floor is not at z 0."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pxr import Usd

# Scan worlds keep their ramps and levels. The floor comes from the stage's static collision meshes, read once into a grid
# of the heights of their flat, upward facing triangles, so a lookup is numpy only: PhysX ray casts from the physics tick
# returned nothing at times and crashed Isaac, and from a spawn service call they stalled it past the caller's timeout.
# A walker stands on the highest surface up to STEP above its last floor and climbs at most SLOPE per meter walked, so it
# takes stairs and ramps but not a table, a ledge or a wall's bumps. Robots are not ground-followed (one spawned over a
# lower floor drops onto it). ARENA_ISAAC_GROUND_FOLLOW=0 turns it off.

# The safety ground plane run_isaacsim adds at z -1 is an infinite PhysX plane: a scan whose floors go below it gets an
# invisible floor across its ramps that robots drive on (smu_link_full's ramp drops to -2.35). keep_ground_below moves
# it under the lowest static collider of every spawned asset.

ENABLED = os.environ.get("ARENA_ISAAC_GROUND_FOLLOW", "1") != "0"
GROUND_PATH = "/World/groundPlane"
GROUND_BELOW = 1.0  # m the safety ground plane keeps under the lowest static collider
CELL = 0.2  # m, grid cell
FLAT = 0.85  # least upward share of a triangle's normal for it to be floor: slopes up to ~30 deg
STEP = 0.25  # m above the last floor a walker may find the next one: a stair riser or a stretch of ramp
SLOPE = 1.0  # m a walker may rise per m walked: stairs (~0.6) and ramps pass
CHECK = 10.0  # s between checks whether the stage's collision meshes changed
MAX_SUB = 400  # barycentric subdivisions of one triangle at most: a 40 m floor triangle


class FloorGrid:
    """Heights of the flat, upward facing surfaces of world-frame triangles (n, 3, 3), per CELL square."""

    def __init__(self, triangles: np.ndarray) -> None:
        t = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
        normal = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        length = np.linalg.norm(normal, axis=1)
        t = t[(length > 1e-12) & (normal[:, 2] >= FLAT * length)]
        # a barycentric lattice no coarser than half a cell over every triangle, so each cell it covers gets a sample
        edge = np.linalg.norm(t - np.roll(t, 1, axis=1), axis=2).max(axis=1)
        subdivisions = np.clip(np.ceil(edge / (CELL / 2)), 1, MAX_SUB).astype(int)
        parts = [np.zeros((0, 3))]
        for n in np.unique(subdivisions):
            i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
            on = i + j <= n
            weights = np.stack([1 - (i[on] + j[on]) / n, i[on] / n, j[on] / n], axis=1)
            parts.append(np.einsum("kv,tvd->tkd", weights, t[subdivisions == n]).reshape(-1, 3))
        samples = np.concatenate(parts)
        keys = self._key(samples[:, 0], samples[:, 1])
        pairs = np.unique(np.stack([keys.astype(float), np.round(samples[:, 2], 2)], axis=1), axis=0)
        self._keys, self._z = pairs[:, 0].astype(np.int64), pairs[:, 1]

    @staticmethod
    def _key(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        return np.floor(np.asarray(x) / CELL).astype(np.int64) * 1_000_003 + np.floor(np.asarray(y) / CELL).astype(np.int64)

    def heights(self, x: float, y: float) -> np.ndarray:
        """Surface heights in the cell holding (x, y), ascending."""
        key = int(self._key(x, y))
        return self._z[np.searchsorted(self._keys, key, "left") : np.searchsorted(self._keys, key, "right")]


def floor_from(heights: np.ndarray, z: float, first: bool) -> float | None:
    """The highest of a cell's surface heights up to STEP above z, else for a first placement the lowest (a raised floor)."""
    below = heights[heights <= z + STEP]
    if len(below):
        return float(below[-1])
    return float(heights[0]) if first and len(heights) else None


def climb(floor: float, found: float, moved: float) -> float:
    """The floor a walker steps onto: found, at most SLOPE times the distance moved above its last floor."""
    return min(found, floor + SLOPE * moved)


_grid: FloorGrid | None = None
_fingerprint: tuple[str, ...] | None = None
_checked = -np.inf


def floor_under(x: float, y: float, z: float, *, first: bool = False) -> float | None:
    """Floor height under (x, y) for a walker last on floor z, None to keep its height."""
    if not ENABLED:
        return None
    global _grid, _fingerprint, _checked
    if time.monotonic() - _checked > CHECK:
        _checked = time.monotonic()
        prims = _static_collision_meshes()
        fingerprint = tuple(str(prim.GetPath()) for prim in prims)
        if fingerprint != _fingerprint:
            triangles = [_triangles(prim) for prim in prims]
            _grid, _fingerprint = FloorGrid(np.concatenate(triangles) if triangles else np.zeros((0, 3, 3))), fingerprint
    return floor_from(_grid.heights(x, y), z, first) if _grid is not None else None


def ground_z(current: float, lowest: float | None) -> float:
    """The safety ground plane's height: where it is, or GROUND_BELOW under the lowest static collider when that is lower."""
    return current if lowest is None else min(current, lowest - GROUND_BELOW)


def keep_ground_below(root: Usd.Prim) -> None:
    """Lower the safety ground plane under the static collision meshes of a spawned asset, if they reach below it."""
    import omni.usd
    from pxr import UsdGeom

    from isaac_utils.utils import geom

    ground = omni.usd.get_context().get_stage().GetPrimAtPath(GROUND_PATH)
    meshes = _static_collision_meshes(root)
    if not ground.IsValid() or not meshes:
        return
    cache = UsdGeom.XformCache()

    def bottom(prim: Usd.Prim) -> float | None:
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [], float).reshape(-1, 3)
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim))  # row vectors: world = point @ matrix
        return float((points @ matrix[:3, 2] + matrix[3, 2]).min()) if len(points) else None

    bottoms = [z for z in map(bottom, meshes) if z is not None]
    current = float(cache.GetLocalToWorldTransform(ground).ExtractTranslation()[2])
    target = ground_z(current, min(bottoms) if bottoms else None)
    if target < current:
        geom.move(GROUND_PATH, translation=geom.Translation(0.0, 0.0, target))


def _static_collision_meshes(root: Usd.Prim | None = None) -> list[Usd.Prim]:
    """Every collision mesh prim (under root, else on the stage) that is not part of a rigid body or articulation (robots, dynamic obstacles)."""
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    prims = []
    for prim in Usd.PrimRange(root) if root is not None else omni.usd.get_context().get_stage().Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        parent = prim
        while parent.IsValid() and not (parent.HasAPI(UsdPhysics.RigidBodyAPI) or parent.HasAPI(UsdPhysics.ArticulationRootAPI)):
            parent = parent.GetParent()
        if not parent.IsValid():
            prims.append(prim)
    return prims


def _triangles(prim: Usd.Prim) -> np.ndarray:
    """World-frame triangles (n, 3, 3) of a mesh prim, polygons fanned."""
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    points, counts, indices = (np.asarray(attr.Get()) for attr in (mesh.GetPointsAttr(), mesh.GetFaceVertexCountsAttr(), mesh.GetFaceVertexIndicesAttr()))
    if not len(points) or not len(counts) or counts.max() < 3:
        return np.zeros((0, 3, 3))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    fans = [np.stack([indices[starts[counts > k + 1]], indices[starts[counts > k + 1] + k], indices[starts[counts > k + 1] + k + 1]], axis=1) for k in range(1, int(counts.max()) - 1)]
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
    return (points.astype(float) @ matrix[:3, :3].T + matrix[:3, 3])[np.concatenate(fans)]
