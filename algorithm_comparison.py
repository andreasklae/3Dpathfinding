import numpy as np
import csv
import time
import math
from pathfinding3d.core.diagonal_movement import DiagonalMovement
from pathfinding3d.core.grid import Grid
from pathfinding3d.finder.a_star import AStarFinder
from pathfinding3d.finder.dijkstra import DijkstraFinder
from pathfinding3d.finder.breadth_first import BreadthFirstFinder
from pathfinding3d.finder.theta_star import ThetaStarFinder

def create_maze():
    """Create the same complex 3D maze used in the original test"""
    matrix = np.ones((40, 40, 40), dtype=np.int8)

    # Extra large boxes (8x8x8)
    matrix[16:24, 16:24, 16:24] = 0
    matrix[2:10, 2:10, 2:10] = 0
    matrix[30:38, 30:38, 30:38] = 0

    # Large boxes
    matrix[5:9, 5:9, 5:9] = 0
    matrix[31:35, 31:35, 31:35] = 0
    matrix[10:14, 25:29, 1:5] = 0
    matrix[25:29, 10:14, 35:39] = 0
    matrix[15:19, 15:19, 15:19] = 0
    matrix[20:24, 20:24, 20:24] = 0

    # Medium boxes
    matrix[1:4, 1:4, 1:4] = 0
    matrix[36:39, 36:39, 36:39] = 0
    matrix[8:11, 8:11, 8:11] = 0
    matrix[28:31, 28:31, 28:31] = 0
    matrix[12:15, 12:15, 12:15] = 0
    matrix[25:28, 25:28, 25:28] = 0

    # Random small/variable boxes
    np.random.seed(42)
    for _ in range(80):
        x = np.random.randint(1, 37)
        y = np.random.randint(1, 37)
        z = np.random.randint(1, 37)
        size_x = np.random.randint(1, 8)
        size_y = np.random.randint(1, 8)
        size_z = np.random.randint(1, 8)
        if x + size_x <= 40 and y + size_y <= 40 and z + size_z <= 40:
            matrix[x:x+size_x, y:y+size_y, z:z+size_z] = 0

    # Irregular long/tall/flat
    matrix[5:15, 5:7, 5:7] = 0
    matrix[25:35, 25:27, 25:27] = 0
    matrix[5:7, 5:15, 5:7] = 0
    matrix[25:27, 25:35, 25:27] = 0
    matrix[10:12, 10:12, 5:25] = 0
    matrix[28:30, 28:30, 15:35] = 0
    matrix[8:18, 8:18, 5:7] = 0
    matrix[22:32, 22:32, 33:35] = 0

    # More structured obstacles
    matrix[6:10, 20:24, 6:10] = 0
    matrix[30:34, 16:20, 30:34] = 0
    matrix[12:16, 12:16, 25:29] = 0
    matrix[3:6, 3:6, 3:6] = 0
    matrix[34:37, 34:37, 34:37] = 0
    matrix[15:18, 15:18, 15:18] = 0
    matrix[22:25, 22:25, 22:25] = 0
    matrix[34:37, 34:37, 34:37] = 0

    # More irregular shapes
    matrix[1:5, 1:5, 15:17] = 0
    matrix[35:39, 35:39, 23:25] = 0
    matrix[18:20, 18:20, 1:15] = 0
    matrix[20:22, 20:22, 25:39] = 0

    # Clusters
    matrix[5:8, 5:8, 5:8] = 0
    matrix[6:9, 6:9, 6:9] = 0
    matrix[35:38, 5:8, 5:8] = 0
    matrix[34:37, 6:9, 6:9] = 0
    matrix[5:8, 35:38, 5:8] = 0
    matrix[6:9, 34:37, 6:9] = 0
    matrix[35:38, 35:38, 5:8] = 0
    matrix[34:37, 34:37, 6:9] = 0

    # Near start/end but not blocking endpoints
    matrix[1:4, 1:4, 1:4] = 0
    matrix[2:5, 2:5, 2:5] = 0
    matrix[36:39, 36:39, 36:39] = 0
    matrix[35:38, 35:38, 35:38] = 0

    # Middle area
    matrix[18:21, 1:4, 18:21] = 0
    matrix[18:21, 36:39, 18:21] = 0

    # Block typical corridor & force detours
    matrix[0:5, 8:12, 2:8] = 0
    matrix[28:35, 35:40, 30:36] = 0
    matrix[15:25, 15:25, 15:25] = 0

    return matrix

def find_nearest_walkable_point(matrix, point, max_radius=10):
    """Find the nearest walkable point to the given point within max_radius"""
    x, y, z = point
    if 0 <= x < matrix.shape[0] and 0 <= y < matrix.shape[1] and 0 <= z < matrix.shape[2]:
        if matrix[x, y, z] == 1:
            return point
    for radius in range(1, max_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy), abs(dz)) == radius:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if (0 <= nx < matrix.shape[0] and
                            0 <= ny < matrix.shape[1] and
                            0 <= nz < matrix.shape[2] and
                            matrix[nx, ny, nz] == 1):
                            return (nx, ny, nz)
    return point

def generate_start_end_pairs(matrix):
    """Generate 100 different start and end point pairs with varying distances"""
    np.random.seed(123)
    pairs = []

    # Short distances 25 pairs
    for _ in range(250):
        start = (np.random.randint(0, 20), np.random.randint(0, 20), np.random.randint(0, 20))
        end = (start[0] + np.random.randint(5, 16),
               start[1] + np.random.randint(5, 16),
               start[2] + np.random.randint(5, 16))
        end = (min(end[0], 39), min(end[1], 39), min(end[2], 39))
        start = find_nearest_walkable_point(matrix, start)
        end = find_nearest_walkable_point(matrix, end)
        pairs.append((start, end))

    # Medium distances 25 pairs
    for _ in range(250):
        start = (np.random.randint(0, 25), np.random.randint(0, 25), np.random.randint(0, 25))
        end = (start[0] + np.random.randint(15, 31),
               start[1] + np.random.randint(15, 31),
               start[2] + np.random.randint(15, 31))
        end = (min(end[0], 39), min(end[1], 39), min(end[2], 39))
        start = find_nearest_walkable_point(matrix, start)
        end = find_nearest_walkable_point(matrix, end)
        pairs.append((start, end))

    # Long distances - 25 pairs
    for _ in range(250):
        start = (np.random.randint(0, 15), np.random.randint(0, 15), np.random.randint(0, 15))
        end = (start[0] + np.random.randint(30, 41),
               start[1] + np.random.randint(30, 41),
               start[2] + np.random.randint(30, 41))
        end = (min(end[0], 39), min(end[1], 39), min(end[2], 39))
        start = find_nearest_walkable_point(matrix, start)
        end = find_nearest_walkable_point(matrix, end)
        pairs.append((start, end))

    # Very long distances (diagonal) - 25 pairs
    for _ in range(250):
        start = (np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10))
        end = (np.random.randint(30, 40), np.random.randint(30, 40), np.random.randint(30, 40))
        start = find_nearest_walkable_point(matrix, start)
        end = find_nearest_walkable_point(matrix, end)
        pairs.append((start, end))

    return pairs

def calculate_euclidean_distance(a, b):
    """Euclidean distance between two 3D points"""
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2 + (b[2] - a[2])**2)

def _coord_from_node_or_tuple(p):
    """Return (x,y,z) regardless if p is a Node with attributes or a tuple."""
    if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z"):
        return (p.x, p.y, p.z)
    if hasattr(p, "identifier"):
        return tuple(p.identifier)  # many libs store (x,y,z) here
    # assume tuple-like
    return (p[0], p[1], p[2])

def _path_length_and_steps(coords):
    """Sum Euclidean segment lengths and count steps between consecutive waypoints."""
    if len(coords) < 2:
        return 0.0, 0
    total = 0.0
    for (x1, y1, z1), (x2, y2, z2) in zip(coords, coords[1:]):
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        total += math.sqrt(dx*dx + dy*dy + dz*dz)
    return total, len(coords) - 1

def run_algorithm(finder, start_node, end_node, grid):
    """Run a pathfinding algorithm and return metrics"""
    start_time = time.time()
    path, runs = finder.find_path(start_node, end_node, grid)
    execution_time = time.time() - start_time

    if path and len(path) > 0:
        path_coords = [_coord_from_node_or_tuple(p) for p in path]
        path_length, steps = _path_length_and_steps(path_coords)
        success = True
    else:
        path_coords = []
        path_length = 0.0
        steps = 0
        success = False

    return {
        'execution_time': execution_time,
        'operations': runs,
        'path_length': path_length,  # geometric length
        'steps': steps,              # waypoint transitions
        'success': success,
        'path': path_coords
    }

def main():
    print("Starting pathfinding algorithm comparison...")

    matrix = create_maze()
    start_end_pairs = generate_start_end_pairs(matrix)

    algorithms = [
        ('A*', AStarFinder(diagonal_movement=DiagonalMovement.always)),
        ('Dijkstra', DijkstraFinder(diagonal_movement=DiagonalMovement.always)),
        ('Breadth-First', BreadthFirstFinder(diagonal_movement=DiagonalMovement.always)),
        ('Theta*', ThetaStarFinder(diagonal_movement=DiagonalMovement.always))
    ]

    csv_filename = 'pathfinding_comparison_results.csv'

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = [
            'test_case', 'start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z',
            'euclidean_distance', 'algorithm', 'execution_time', 'operations',
            'path_length', 'steps', 'success', 'path_efficiency'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for test_case, (start_coords, end_coords) in enumerate(start_end_pairs, 1):
            print(f"\nRunning test case {test_case}/100: Start {start_coords} -> End {end_coords}")

            grid = Grid(matrix=matrix)
            start_node = grid.node(start_coords[0], start_coords[1], start_coords[2])
            end_node = grid.node(end_coords[0], end_coords[1], end_coords[2])

            if not start_node.walkable:
                print(f"ERROR: Start point {start_coords} is still blocked after correction!")
                start_coords = find_nearest_walkable_point(matrix, start_coords, max_radius=20)
                matrix[start_coords[0], start_coords[1], start_coords[2]] = 1
                grid = Grid(matrix=matrix)
                start_node = grid.node(start_coords[0], start_coords[1], start_coords[2])
                print(f"  Corrected to: {start_coords}")

            if not end_node.walkable:
                print(f"ERROR: End point {end_coords} is still blocked after correction!")
                end_coords = find_nearest_walkable_point(matrix, end_coords, max_radius=20)
                matrix[end_coords[0], end_coords[1], end_coords[2]] = 1
                grid = Grid(matrix=matrix)
                end_node = grid.node(end_coords[0], end_coords[1], end_coords[2])
                print(f"  Corrected to: {end_coords}")

            distance = calculate_euclidean_distance(start_coords, end_coords)
            print(f"Euclidean distance: {distance:.2f}")

            for algorithm_name, finder in algorithms:
                print(f"  Running {algorithm_name}...")
                grid.cleanup()  # reset between runs

                results = run_algorithm(finder, start_node, end_node, grid)

                # Path efficiency = shortest straight-line distance / actual geometric path length
                if results['success'] and results['path_length'] > 0:
                    path_efficiency = distance / results['path_length']  # (0,1], 1.0 is optimal
                else:
                    path_efficiency = 0.0

                writer.writerow({
                    'test_case': test_case,
                    'start_x': start_coords[0],
                    'start_y': start_coords[1],
                    'start_z': start_coords[2],
                    'end_x': end_coords[0],
                    'end_y': end_coords[1],
                    'end_z': end_coords[2],
                    'euclidean_distance': round(distance, 4),
                    'algorithm': algorithm_name,
                    'execution_time': round(results['execution_time'], 6),
                    'operations': results['operations'],
                    'path_length': round(results['path_length'], 4),
                    'steps': results['steps'],
                    'success': results['success'],
                    'path_efficiency': round(path_efficiency, 6)
                })

                print(f"    Success: {results['success']}, Time: {results['execution_time']:.4f}s, "
                      f"Operations: {results['operations']}, Path Len: {results['path_length']:.2f}, "
                      f"Steps: {results['steps']}")

    print(f"\nComparison complete! Results saved to {csv_filename}")

    print("\nSummary Statistics:")
    print("=" * 50)
    try:
        import pandas as pd
        import pickle
        df = pd.read_csv(csv_filename)

        for algorithm in df['algorithm'].unique():
            alg = df[df['algorithm'] == algorithm]
            success_rate = (alg['success'].sum() / len(alg)) * 100 if len(alg) else 0.0
            avg_time = alg['execution_time'].mean() if len(alg) else float('nan')
            avg_ops = alg['operations'].mean() if len(alg) else float('nan')
            avg_len = alg[alg['success'] == True]['path_length'].mean() if (alg['success'] == True).any() else float('nan')
            avg_eff = alg[alg['success'] == True]['path_efficiency'].mean() if (alg['success'] == True).any() else float('nan')

            print(f"\n{algorithm}:")
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Average Execution Time: {avg_time:.4f}s")
            print(f"  Average Operations: {avg_ops:.0f}")
            print(f"  Average Path Length: {avg_len:.2f}")
            print(f"  Average Path Efficiency: {avg_eff:.3f}")

    except (ImportError, FileNotFoundError) as e:
        print(f"Could not generate detailed summary: {e}")
        print("Basic summary: finished; see CSV for details.")

if __name__ == "__main__":
    main()