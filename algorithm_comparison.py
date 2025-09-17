import csv
import math
import time
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

from pathfinding3d.core.grid import Grid
from pathfinding3d.core.diagonal_movement import DiagonalMovement
from pathfinding3d.finder.a_star import AStarFinder
from pathfinding3d.finder.dijkstra import DijkstraFinder
from pathfinding3d.finder.breadth_first import BreadthFirstFinder
from pathfinding3d.finder.theta_star import ThetaStarFinder

# We take inspiration from maze_runner, but build our own pipeline here
from maze_runner import _generate_spread_maze as generate_maze
from maze_runner import _find_nearest_walkable as nearest_walkable


@dataclass
class Category:
    name: str
    dmin: float
    dmax: float
    length: int


def euclid(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    ax, ay, az = a; bx, by, bz = b
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)


def path_length(coords: List[Tuple[int, int, int]]) -> float:
    if not coords or len(coords) < 2:
        return 0.0
    total = 0.0
    for (x1, y1, z1), (x2, y2, z2) in zip(coords, coords[1:]):
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


def coord_from_node_or_tuple(p) -> Tuple[int, int, int]:
    if hasattr(p, 'x') and hasattr(p, 'y') and hasattr(p, 'z'):
        return (p.x, p.y, p.z)
    if hasattr(p, 'identifier'):
        return tuple(p.identifier)
    return (int(p[0]), int(p[1]), int(p[2]))


def sample_points_for_distance(matrix: np.ndarray,
                               dmin: float,
                               dmax: float,
                               rng: np.random.Generator,
                               max_tries: int = 200) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], float]:
    """Pick start/end so that euclidean distance in [dmin, dmax] after walkable snapping.
    Retries up to max_tries times; last attempt may be out-of-range if necessary.
    """
    W, H, L = matrix.shape
    last = None
    for _ in range(max_tries):
        sx = int(rng.integers(0, W))
        sy = int(rng.integers(0, H))
        sz = int(rng.integers(0, L))

        # random direction in 3D
        v = rng.normal(0, 1, size=3)
        norm = np.linalg.norm(v)
        if norm == 0:
            continue
        v = v / norm
        r = float(rng.uniform(dmin, dmax))
        exf = np.array([sx, sy, sz], dtype=float) + v * r
        ex, ey, ez = int(round(exf[0])), int(round(exf[1])), int(round(exf[2]))
        if not (0 <= ex < W and 0 <= ey < H and 0 <= ez < L):
            continue

        s0 = (sx, sy, sz)
        e0 = (ex, ey, ez)
        s = nearest_walkable(matrix, s0, max_r=max(10, L // 4))
        e = nearest_walkable(matrix, e0, max_r=max(10, L // 4))
        d = euclid(s, e)
        last = (s, e, d)
        if dmin <= d <= dmax:
            return s, e, d

    # fallback to last valid pair (may be out of range)
    if last is None:
        # extreme fallback: opposite Z ends in corridor-ish center
        s = nearest_walkable(matrix, (W // 2, H // 2, 1), max_r=max(10, L // 4))
        e = nearest_walkable(matrix, (W // 2 - 1, H // 2 - 1, max(0, L - 2)), max_r=max(10, L // 4))
        return s, e, euclid(s, e)
    return last


def run_one(grid: Grid, finder, s: Tuple[int, int, int], e: Tuple[int, int, int]) -> Tuple[float, int, float, bool]:
    grid.cleanup()
    sn = grid.node(*s)
    en = grid.node(*e)
    t0 = time.time()
    path, ops = finder.find_path(sn, en, grid)
    dt = time.time() - t0
    coords = [coord_from_node_or_tuple(p) for p in (path or [])]
    plen = path_length(coords)
    success = bool(coords) and coords[-1] == e
    return dt, ops, plen, success


def main():
    # Distance categories and corresponding maze lengths
    categories = [
        Category('10-50', 10.0, 50.0, 50),
        Category('50-100', 50.0, 100.0, 100),
        Category('100-200', 100.0, 200.0, 200),
        Category('200-300', 200.0, 300.0, 300),
    ]

    runs_per_category = 100
    density = 0.25  # obstacle density inspiration from maze_runner
    csv_filename = 'pathfinding_comparison_results.csv'

    algos = [
        ('A*', AStarFinder(diagonal_movement=DiagonalMovement.always)),
        ('Dijkstra', DijkstraFinder(diagonal_movement=DiagonalMovement.always)),
        ('Breadth-First', BreadthFirstFinder(diagonal_movement=DiagonalMovement.always)),
        ('Theta*', ThetaStarFinder(diagonal_movement=DiagonalMovement.always)),
    ]

    with open(csv_filename, 'w', newline='') as f:
        fieldnames = [
            'distance_category', 'length', 'seed', 'algorithm',
            'direct_distance', 'execution_time', 'operations', 'success', 'path_length'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ci, cat in enumerate(categories):
            print(f"\nCategory {cat.name} • length={cat.length} • {runs_per_category} runs")
            for i in range(runs_per_category):
                # Unique seed per maze per category
                seed = 500000 + ci * 10000 + i
                rng = np.random.default_rng(seed)

                # Maze generation (excluded from algorithm timing)
                matrix = generate_maze(length=cat.length, density=density, seed=seed)
                grid = Grid(matrix=matrix)

                # Sample start/end for the requested distance range
                s, e, d = sample_points_for_distance(matrix, cat.dmin, cat.dmax, rng)
                print(f"  run {i+1:03d}: seed={seed} direct_d={d:.2f} s={s} e={e}")

                # Run all algorithms on the same maze and start/end
                for name, finder in algos:
                    dt, ops, plen, success = run_one(grid, finder, s, e)
                    writer.writerow({
                        'distance_category': cat.name,
                        'length': cat.length,
                        'seed': seed,
                        'algorithm': name,
                        'direct_distance': round(d, 4),
                        'execution_time': round(dt, 6),
                        'operations': int(ops),
                        'success': bool(success),
                        'path_length': round(plen, 4),
                    })
                f.flush()

    print(f"\nComparison complete! Results saved to {csv_filename}")


if __name__ == "__main__":
    main()
