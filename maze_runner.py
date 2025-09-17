# maze_runner.py
import math
import time
from typing import Dict, Tuple, Any
import numpy as np

from pathfinding3d.core.grid import Grid
from pathfinding3d.core.diagonal_movement import DiagonalMovement
from pathfinding3d.finder.a_star import AStarFinder
from pathfinding3d.finder.dijkstra import DijkstraFinder
from pathfinding3d.finder.breadth_first import BreadthFirstFinder
from pathfinding3d.finder.theta_star import ThetaStarFinder

# ----------------- helpers -----------------
def _coord_from_node_or_tuple(p):
    if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
        return (p.x, p.y, p.z)
    if hasattr(p, "identifier"):
        return tuple(p.identifier)
    return (p[0], p[1], p[2])

def _path_len_steps(coords):
    if len(coords) < 2:
        return 0.0, 0
    d = 0.0
    for (x1, y1, z1), (x2, y2, z2) in zip(coords, coords[1:]):
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        d += math.sqrt(dx*dx + dy*dy + dz*dz)
    return d, len(coords) - 1

def _euclid(a, b):
    ax, ay, az = a; bx, by, bz = b
    return math.sqrt((bx-ax)**2 + (by-ay)**2 + (bz-az)**2)

def _find_nearest_walkable(matrix: np.ndarray, p: Tuple[int,int,int], max_r: int = 10):
    x, y, z = p
    sx, sy, sz = matrix.shape
    if 0 <= x < sx and 0 <= y < sy and 0 <= z < sz and matrix[x, y, z] == 1:
        return p
    for r in range(1, max_r + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if max(abs(dx), abs(dy), abs(dz)) == r:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if 0 <= nx < sx and 0 <= ny < sy and 0 <= nz < sz and matrix[nx, ny, nz] == 1:
                            return (nx, ny, nz)
    return p

def _carve(matrix: np.ndarray, x0, x1, y0, y1, z0, z1):
    sx, sy, sz = matrix.shape
    x0, x1 = max(0, x0), min(sx, x1)
    y0, y1 = max(0, y0), min(sy, y1)
    z0, z1 = max(0, z0), min(sz, z1)
    if x1 > x0 and y1 > y0 and z1 > z0:
        matrix[x0:x1, y0:y1, z0:z1] = 0

# ----------------- maze generation -----------------
def _generate_spread_maze(length: int, density: float, seed: int) -> np.ndarray:
    """
    Create a 50 x 50 x length "asteroid field" with varied, round-ish obstacles.

    Design goals for a space-flight feel:
    - Asteroids are ellipsoids with rough edges (size variety: small/medium/large).
    - They appear in loose clusters with some randomness, like belts/clouds.
    - A subtle meandering corridor remains open so a valid path exists.
    - No artificial flat walls; the space is organic and 3D.

    Note: The final obstacle fraction is approximate (prioritizes structure first,
    then tops up towards the requested density).
    """
    W = H = 50               # X, Y fixed
    L = int(length)          # Z = length (maze "length")
    assert L >= 4, "length must be >= 4"

    rng = np.random.default_rng(seed)
    m = np.ones((W, H, L), dtype=np.int8)

    total_cells = m.size
    target_obstacles = int(total_cells * density)

    # --- meandering open corridor (a thin tube through space) ---
    wp_count = max(6, min(18, L // 6 + 6))
    wp_zs = np.linspace(1, max(1, L - 2), wp_count, dtype=int)
    pos = np.array([W // 2, H // 2], dtype=int)
    corridor_points = []  # (x,y,z)
    for i, zc in enumerate(wp_zs):
        drift = rng.integers(-4, 5, size=2)
        # bias alternates to encourage weaving
        bias = np.array([1 if i % 2 == 0 else -1, -1 if i % 2 == 0 else 1]) * rng.integers(0, 3, size=2)
        pos = np.clip(pos + drift + bias, [3, 3], [W - 4, H - 4])
        corridor_points.append((int(pos[0]), int(pos[1]), int(zc)))

    corridor_mask = np.zeros_like(m, dtype=bool)
    def _mark_corridor_segment(p0, p1, radius: int = 2):
        x0, y0, z0 = p0; x1, y1, z1 = p1
        if z0 == z1:
            zs = [z0]
        else:
            step = 1 if z1 >= z0 else -1
            zs = list(range(z0, z1 + step, step))
        for z in zs:
            if z1 == z0:
                tx = x0; ty = y0
            else:
                t = (z - z0) / (z1 - z0)
                tx = int(round(x0 + t * (x1 - x0)))
                ty = int(round(y0 + t * (y1 - y0)))
            xa0, xa1 = max(0, tx - radius), min(W, tx + radius + 1)
            ya0, ya1 = max(0, ty - radius), min(H, ty + radius + 1)
            corridor_mask[xa0:xa1, ya0:ya1, z] = True

    for a, b in zip(corridor_points, corridor_points[1:]):
        _mark_corridor_segment(a, b, radius=2)

    # keep corridor open
    m[corridor_mask] = 1

    # --- asteroid clusters (evenly distributed across Z via bands) ---
    # We split the length into bands and fill each band up to its share of obstacles.
    band_count = max(6, min(24, L // 25))
    band_edges = np.linspace(0, L, band_count + 1, dtype=int)

    # Ellipsoid size sampler: small/medium/large radii
    def _rand_radii():
        typ = rng.choice(["S", "M", "L"], p=[0.6, 0.3, 0.1])
        if typ == "S":
            rx = int(rng.integers(2, 4)); ry = int(rng.integers(2, 4)); rz = int(rng.integers(1, 3))
        elif typ == "M":
            rx = int(rng.integers(3, 7)); ry = int(rng.integers(3, 7)); rz = int(rng.integers(2, 5))
        else:  # L
            rx = int(rng.integers(6, 11)); ry = int(rng.integers(6, 11)); rz = int(rng.integers(3, 8))
        return rx, ry, rz

    def _place_ellipsoid(center, radii, roughness=0.2, z_limits=None) -> int:
        cx, cy, cz = center
        rx, ry, rz = radii
        # bounding box
        x0, x1 = max(0, cx - rx - 1), min(W, cx + rx + 2)
        y0, y1 = max(0, cy - ry - 1), min(H, cy + ry + 2)
        z0, z1 = max(0, cz - rz - 1), min(L, cz + rz + 2)
        if z_limits is not None:
            zl0, zl1 = z_limits
            z0 = max(z0, zl0)
            z1 = min(z1, zl1)
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            return 0
        xs = np.arange(x0, x1)[:, None, None]
        ys = np.arange(y0, y1)[None, :, None]
        zs = np.arange(z0, z1)[None, None, :]
        # normalized squared distance in ellipsoid metric
        fx = ((xs - cx) / max(1, rx)) ** 2
        fy = ((ys - cy) / max(1, ry)) ** 2
        fz = ((zs - cz) / max(1, rz)) ** 2
        field = fx + fy + fz
        if roughness > 0:
            jitter = roughness * rng.random(size=field.shape)
            mask = field <= (1.0 - jitter)
        else:
            mask = field <= 1.0
        sub = m[x0:x1, y0:y1, z0:z1]
        before = int(sub.sum())
        # mark asteroid voxels as obstacles (0)
        sub[mask] = 0
        after = int(sub.sum())
        # Preserve corridor if touched
        sub_corr = corridor_mask[x0:x1, y0:y1, z0:z1]
        if sub_corr.any():
            sub[sub_corr] = 1
        return before - int(sub.sum())

    # populate field towards desired density, band by band to balance along Z
    current_obstacles = int(total_cells - m.sum())
    total_remaining = max(0, target_obstacles - current_obstacles)
    for bi in range(len(band_edges) - 1):
        z0b, z1b = int(band_edges[bi]), int(band_edges[bi + 1])
        if z1b <= z0b:
            continue
        band_depth = z1b - z0b
        band_volume = W * H * band_depth
        # aimed obstacles for this band (approximate)
        band_target = int(band_volume * density)
        # already in band (mostly zero here but safe)
        band_obs_now = int((m[:, :, z0b:z1b] == 0).sum())
        band_remaining = max(0, min(total_remaining, band_target - band_obs_now))

        # local clusters inside band
        clusters_in_band = max(1, min(4, band_depth // 25 + 1))
        band_centers = []
        for _ in range(clusters_in_band):
            cx0 = int(rng.integers(6, W - 6))
            cy0 = int(rng.integers(6, H - 6))
            # Safe Z range for narrow bands: prefer (z0b+1, z1b-1), fall back to (z0b, z1b)
            z_low = max(z0b + 1, 0)
            z_high = min(z1b - 1, L - 1)
            if z_high <= z_low:
                z_low = max(z0b, 0)
                z_high = min(z1b, L)
            if z_high <= z_low:
                cz0 = int(z_low)
            else:
                cz0 = int(rng.integers(z_low, z_high))
            band_centers.append((cx0, cy0, cz0))

        attempts = 0
        max_attempts = 1200
        while band_remaining > 0 and total_remaining > 0 and attempts < max_attempts:
            attempts += 1
            if rng.random() < 0.75 and band_centers:
                cc = band_centers[int(rng.integers(0, len(band_centers)))]
                cx = int(np.clip(rng.normal(cc[0], 5), 1, W - 2))
                cy = int(np.clip(rng.normal(cc[1], 5), 1, H - 2))
                # tighter around band in Z; clip with safe bounds even for thin bands
                lo = max(z0b + 1, 0)
                hi = min(z1b - 1, L - 1)
                if hi <= lo:
                    lo = max(z0b, 0)
                    hi = min(z1b, L - 1)
                if hi <= lo:
                    hi = min(lo + 1, L - 1)
                cz = int(np.clip(rng.normal(cc[2], max(3, band_depth / 6)), lo, hi))
            else:
                cx = int(rng.integers(1, W - 1))
                cy = int(rng.integers(1, H - 1))
                lo = max(z0b + 1, 0)
                hi = min(z1b - 1, L - 1)
                if hi <= lo:
                    hi = lo + 1
                cz = int(rng.integers(lo, hi))

            rx, ry, rz = _rand_radii()
            delta = _place_ellipsoid((cx, cy, cz), (rx, ry, rz), roughness=0.22, z_limits=(z0b, z1b))
            if delta <= 0:
                continue
            band_remaining -= delta
            total_remaining -= delta
            # Keep corridor open
            m[corridor_mask] = 1

        if total_remaining <= 0:
            break

    # global top-up if under target (uniform across full Z)
    attempts = 0
    max_attempts = 2000
    while total_remaining > 0 and attempts < max_attempts:
        attempts += 1
        cx = int(rng.integers(1, W - 1))
        cy = int(rng.integers(1, H - 1))
        cz = int(rng.integers(1, L - 1))
        rx, ry, rz = _rand_radii()
        delta = _place_ellipsoid((cx, cy, cz), (rx, ry, rz), roughness=0.2)
        if delta <= 0:
            continue
        total_remaining -= delta
        m[corridor_mask] = 1

    # ensure corridor stays open
    m[corridor_mask] = 1

    return m

# ----------------- main API -----------------
def build_maze_and_run(length: int,
                       density: float = 0.4,   # denser default
                       seed: int = 42) -> Dict[str, Any]:
    """
    Build a 50 x 50 x <length> maze (Z dimension = length).
    Start is snapped near z≈0; End near z≈length-1 (opposite ends along Z).
    Runs A*, Dijkstra, BFS, Theta* once each and returns matrix + metrics.
    """
    matrix = _generate_spread_maze(length, density, seed)

    # Opposite ends along Z (length), center-ish in X/Y
    start_guess = (25, 25, 1)
    end_guess   = (24, 24, max(0, length - 2))  # slight offset to reduce symmetry

    start = _find_nearest_walkable(matrix, start_guess, max_r=max(10, length // 4))
    end   = _find_nearest_walkable(matrix, end_guess,   max_r=max(10, length // 4))
    matrix[start] = 1
    matrix[end] = 1

    dist = _euclid(start, end)

    grid = Grid(matrix=matrix)
    algos = [
        ('A*', AStarFinder(diagonal_movement=DiagonalMovement.always)),
        ('Dijkstra', DijkstraFinder(diagonal_movement=DiagonalMovement.always)),
        ('Breadth-First', BreadthFirstFinder(diagonal_movement=DiagonalMovement.always)),
        ('Theta*', ThetaStarFinder(diagonal_movement=DiagonalMovement.always)),
    ]

    results: Dict[str, Dict[str, Any]] = {}
    for name, finder in algos:
        grid.cleanup()
        s = grid.node(*start)
        e = grid.node(*end)

        t0 = time.time()
        path_nodes, ops = finder.find_path(s, e, grid)
        elapsed = time.time() - t0

        if path_nodes:
            coords = [_coord_from_node_or_tuple(p) for p in path_nodes]
            plen, steps = _path_len_steps(coords)
            success = True
        else:
            coords = []
            plen, steps = 0.0, 0
            success = False

        eff = (dist / plen) if (success and plen > 0) else 0.0

        results[name] = {
            "success": success,
            "execution_time": elapsed,
            "operations": ops,
            "path_length": plen,
            "steps": steps,
            "path_efficiency": eff,
            "path": coords
        }

    return {
        "matrix": matrix,
        "start": start,
        "end": end,
        "euclidean_distance": dist,
        "results": results
    }
