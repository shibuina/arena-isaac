"""The floor under a point: a grid of flat surface heights from collision triangles, the walker's floor among them."""

import numpy as np
import pytest

from isaac_utils.utils.floor import SLOPE, FloorGrid, climb, floor_from


def _square(x0: float, y0: float, x1: float, y1: float, z0: float, z1: float | None = None) -> list[list[list[float]]]:
    """Two upward facing triangles over a rectangle, rising from z0 at y0 to z1 at y1 (a ramp)."""
    z1 = z0 if z1 is None else z1
    a, b, c, d = [x0, y0, z0], [x1, y0, z0], [x1, y1, z1], [x0, y1, z1]
    return [[a, b, c], [a, c, d]]


def _grid() -> FloorGrid:
    lower = _square(0, 0, 4, 4, -2.3)
    ramp = _square(0, 4, 4, 6, -2.3, -1.3)
    table = _square(1, 1, 2, 2, -1.55)
    ceiling = [[t[0], t[2], t[1]] for t in _square(0, 0, 4, 4, 0.1)]  # faces down
    wall = [[[3.9, 0, -2.3], [3.9, 4, -2.3], [3.9, 4, 0.0]]]
    return FloorGrid(np.array(lower + ramp + table + ceiling + wall))


def test_the_grid_keeps_flat_upward_surfaces_only() -> None:
    grid = _grid()
    assert grid.heights(0.5, 0.5) == pytest.approx([-2.3])
    assert grid.heights(1.5, 1.5) == pytest.approx([-2.3, -1.55])
    assert len(grid.heights(9.0, 9.0)) == 0


def test_a_walker_stands_on_the_highest_surface_within_a_step() -> None:
    grid = _grid()
    assert floor_from(grid.heights(1.5, 1.5), -2.3, first=False) == pytest.approx(-2.3)  # not on the table
    assert floor_from(grid.heights(1.5, 5.0), -1.9, first=False) == pytest.approx(-1.8, abs=0.11)  # up the ramp
    assert floor_from(grid.heights(0.5, 0.5), 0.0, first=True) == pytest.approx(-2.3)  # a first placement drops to the floor


def test_a_raised_floor_is_found_on_a_first_placement_only() -> None:
    raised = FloorGrid(np.array(_square(0, 0, 2, 2, 1.8)))
    assert floor_from(raised.heights(1.0, 1.0), 0.0, first=True) == pytest.approx(1.8)
    assert floor_from(raised.heights(1.0, 1.0), 0.0, first=False) is None


def test_a_placement_keeps_the_floor_of_its_stacked_level() -> None:
    stacked = FloorGrid(np.array(_square(0, 0, 2, 2, 0.0) + _square(0, 0, 2, 2, 3.0)))
    assert floor_from(stacked.heights(1.0, 1.0), 3.0, first=True) == pytest.approx(3.0)
    assert floor_from(stacked.heights(1.0, 1.0), 0.0, first=True) == pytest.approx(0.0)


def test_a_walker_climbs_at_most_slope_times_its_walk() -> None:
    assert climb(0.0, 0.17, 0.3) == pytest.approx(0.17)  # a stair riser after a tread
    assert climb(0.0, 0.4, 0.1) == pytest.approx(SLOPE * 0.1)  # a ledge beside the path
    assert climb(0.0, 0.4, 0.0) == pytest.approx(0.0)  # standing still never climbs
    assert climb(0.0, -2.3, 0.05) == pytest.approx(-2.3)  # stepping down is immediate
