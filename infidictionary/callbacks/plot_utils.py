import numpy as np
import plotly.graph_objects as go


def make_trace(
    x_np: np.ndarray,    # (N,)
    y_np: np.ndarray,    # (N,)
    data_np: np.ndarray, # (N, C)
    z_min: float | None = None,
    z_max: float | None = None,
):
    """Return a Plotly trace for a single (N, C) signal field.

    C=1 → Histogram2dContour coloured by scalar value.
    C=3 → Scattergl with each point coloured by its RGB value.
    Anything else raises ValueError.
    """
    C = data_np.shape[1]
    if C == 1:
        return go.Histogram2dContour(
            x=x_np, y=y_np, z=data_np[:, 0],
            histfunc="avg", colorscale="Viridis",
            zmin=z_min, zmax=z_max,
            showscale=False,
            nbinsx=50, nbinsy=50,
        )
    elif C == 3:
        rgb = np.clip(data_np, 0.0, 1.0)
        colors = [
            f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
            for r, g, b in rgb
        ]
        return go.Scattergl(
            x=x_np, y=y_np,
            mode="markers",
            marker=dict(color=colors, size=1),
            showlegend=False,
        )
    else:
        raise ValueError(
            f"make_trace only supports C=1 or C=3, got C={C}"
        )
