# run_and_export.py
import pickle
from datetime import datetime

from maze_runner import build_maze_and_run

# Plotly is optional; we try to import it for HTML output
try:
    import numpy as np
    import plotly.graph_objects as go
    from plotly.io import write_html
    _PLOTLY = True
except Exception:
    _PLOTLY = False


def _make_html(matrix, start, end, results, outfile: str, auto_open: bool = False):
    if not _PLOTLY:
        print("Plotly not available; skipping HTML export.")
        return None

    obs_vol = (matrix == 0).astype(np.int8)
    fig = go.Figure()

    # Obstacles as isosurface (blocky, fast)
    if obs_vol.any():
        x, y, z = np.mgrid[0:matrix.shape[0], 0:matrix.shape[1], 0:matrix.shape[2]]
        fig.add_trace(go.Isosurface(
            x=x.flatten(), y=y.flatten(), z=z.flatten(),
            value=obs_vol.flatten(),
            isomin=0.5, isomax=1,
            surface_count=1,
            opacity=1,
            colorscale=[[0, "gray"], [1, "gray"]],
            caps=dict(x_show=False, y_show=False, z_show=False),
            name="obstacles"
        ))

    # Paths (one per algorithm)
    for algo, res in results.items():
        coords = res.get("path", [])
        if coords:
            px, py, pz = zip(*coords)
        else:
            px, py, pz = [], [], []
        fig.add_trace(go.Scatter3d(
            x=px, y=py, z=pz, mode="lines",
            line=dict(width=6),
            name=f"path • {algo}"
        ))

    # Start/End
    fig.add_trace(go.Scatter3d(
        x=[start[0]], y=[start[1]], z=[start[2]],
        mode="markers+text", text=["start"], textposition="top center",
        marker=dict(size=8, symbol="circle"), name="start"
    ))
    fig.add_trace(go.Scatter3d(
        x=[end[0]], y=[end[1]], z=[end[2]],
        mode="markers+text", text=["end"], textposition="top center",
        marker=dict(size=8, symbol="x"), name="end"
    ))

    fig.update_layout(
        title="3D Maze • Obstacles and Algorithm Paths",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    write_html(fig, file=outfile, auto_open=auto_open, include_plotlyjs="cdn")
    return outfile


def run_and_export(length: int,
                   density: float = 0.12,
                   seed: int = 42,
                   tag: str | None = None,
                   auto_open_html: bool = False) -> tuple[str, str | None]:
    """
    Runs one maze instance (50x50xlength), exports a pickle and one HTML.
    Returns (pickle_path, html_path_or_None).
    """
    out = build_maze_and_run(length=length, density=density, seed=seed)

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "params": {"width": 50, "height": 50, "length": length, "density": density, "seed": seed},
        **out
    }

    tag = tag or f"L{length}_d{str(density).replace('.','p')}_s{seed}"
    pkl_name = f"maze_run_{tag}.pkl"
    with open(pkl_name, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved pickle: {pkl_name}")

    html_name = f"maze_visual_{tag}.html"
    html_path = _make_html(out["matrix"], out["start"], out["end"], out["results"], html_name, auto_open=auto_open_html)
    if html_path:
        print(f"Saved HTML: {html_path}")

    # brief stdout summary
    print(f"Start: {payload['start']}  End: {payload['end']}  Direct distance: {payload['euclidean_distance']:.2f}")
    for algo, res in payload["results"].items():
        print(f"{algo:13s} | success={res['success']}  time={res['execution_time']:.4f}s  "
              f"ops={res['operations']}  len={res['path_length']:.2f}  eff={res['path_efficiency']:.3f}")

    return pkl_name, html_path