import colorsys

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
    C=2 → Scattergl with hue = vector direction, brightness = magnitude
           (complex-amplitude / phase-wheel colormap).
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
    elif C == 2:
        # Phase-wheel: hue encodes direction, brightness encodes magnitude.
        # atan2 maps the 2D vector to [-π, π]; we wrap to [0, 1] for HSV hue.
        hue = (np.arctan2(data_np[:, 1], data_np[:, 0]) / (2.0 * np.pi) + 0.5) % 1.0
        mag = np.sqrt(data_np[:, 0] ** 2 + data_np[:, 1] ** 2)
        mag_max = mag.max()
        value = mag / mag_max if mag_max > 0 else np.zeros_like(mag)
        colors = [
            "rgb({},{},{})".format(
                int(r * 255), int(g * 255), int(b * 255)
            )
            for r, g, b in (colorsys.hsv_to_rgb(h, 1.0, v) for h, v in zip(hue, value))
        ]
        return go.Scattergl(
            x=x_np, y=y_np,
            mode="markers",
            marker=dict(color=colors, size=2),
            showlegend=False,
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
            f"make_trace only supports C=1, C=2, or C=3, got C={C}"
        )
