# 3D Pathfinding Maze and Algorithm Comparison

This repository generates and visualizes 3D mazes, runs several pathfinding algorithms, and analyzes their performance. You can:

- Create and view an interactive HTML visualization of a single maze and the algorithm paths.
- Run batched comparisons across multiple mazes to produce a CSV of results.
- Generate charts and summaries from the CSV in a notebook.

## Requirements

- Python 3.10 or newer
- Python packages:
  - numpy
  - plotly
  - pandas
  - matplotlib
  - seaborn
  - scipy
  - pathfinding3d

Install with:

```
pip install numpy plotly pandas matplotlib seaborn scipy pathfinding3d
```

To run the analysis notebook, also install Jupyter or use an editor that supports notebooks.

## Quick Start

### 1) Generate and view a single 3D maze

Runs a 50x50xL maze, computes paths using A*, Dijkstra, Breadth First, and Theta*, and opens an interactive HTML if Plotly is available.

```
python main.py            # default length L=100
python main.py 300        # optional length argument in [10, 1000]
```

Artifacts produced in the project folder:

- `maze_visual_L{length}_d{density}_s{seed}.html` interactive 3D view
- `maze_run_L{length}_d{density}_s{seed}.pkl` run payload with matrix, start, end, and metrics

Notes:

- `main.py` uses density 0.25 and seed 41 by default and will auto open the HTML in your browser. If Plotly is not installed the HTML export is skipped and only console output is shown.
- The maze grid is 50 by 50 by L. Start and end are placed near opposite ends along the Z axis.

### 2) Run algorithm comparison and export CSV

Benchmarks all four algorithms across distance categories and multiple seeds. Produces `pathfinding_comparison_results.csv`.

```
python algorithm_comparison.py
```

Details:

- Categories cover different direct distance ranges and example lengths L in {50, 100, 200, 300}.
- Default runs per category is 100, so expect a longer runtime.
- The output CSV is written to the repository root. Running again will create a new file and overwrite the previous CSV.

CSV columns:

- `distance_category`, `length`, `seed`, `algorithm`, `direct_distance`, `execution_time`, `operations`, `success`, `path_length`

An example CSV is already included as `pathfinding_comparison_results.csv`. You can rerun the comparison at any time to regenerate it.

### 3) Analyze results and create charts

Open and run the notebook to produce plots and summaries from the CSV.

```
algorithm_analysis.ipynb
```

Behavior:

- If `pathfinding_comparison_results.csv` is missing the notebook will run `algorithm_comparison.py` first, then load the fresh CSV and continue with the analysis.
- The notebook uses pandas, matplotlib, seaborn, and scipy for the visualizations and tests.

## Project Structure

- `main.py` entry point for single maze generation and HTML visualization
- `run_and_export.py` helper that runs one maze, saves a pickle, and writes an HTML if Plotly is available
- `maze_runner.py` maze generation and pathfinding execution logic used by both single runs and comparisons
- `algorithm_comparison.py` batch runner that generates many mazes and writes `pathfinding_comparison_results.csv`
- `algorithm_analysis.ipynb` analysis notebook that reads the CSV and creates charts
- `pathfinding_comparison_results.csv` example results CSV already included
- `index.html` legacy or sample HTML artifact kept in the repository

## Troubleshooting

- Import error for `pathfinding3d`: install the package with pip or ensure the module is on `PYTHONPATH`.
- HTML does not open: ensure `plotly` is installed. Without Plotly, the run still completes but the HTML export is skipped.
- Long runtimes for comparisons: reduce `runs_per_category` in `algorithm_comparison.py` if needed.

