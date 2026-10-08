"""Plotly figures with a shared style, exported as interactive HTML and vector PDF.

PDF export goes through Kaleido, which drives a headless Chrome. If Chrome is
missing, run ``pixi run setup-chrome`` once; HTML is written regardless.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

FONT = dict(family="Helvetica, Arial, sans-serif", size=13, color="#222")
METHOD_COLORS = {"UKF": "#1f77b4", "EKF": "#ff7f0e", "BO": "#2ca02c"}
CONTROLLER_COLORS = {
    "state_feedback": "#1f77b4", "optimal_control": "#ff7f0e", "pid": "#2ca02c",
    "h_infinity": "#d62728", "sliding_mode": "#9467bd", "dynamic_output_feedback": "#8c564b",
    "neural_network": "#e377c2",
}
LABELS = {
    "state_feedback": "State feedback", "optimal_control": "Optimal control", "pid": "PID",
    "h_infinity": "H∞", "sliding_mode": "Sliding mode",
    "dynamic_output_feedback": "Dynamic output feedback", "neural_network": "Neural network",
}


def style(fig: go.Figure, title: str, width: int, height: int) -> go.Figure:
    fig.update_layout(template="plotly_white", font=FONT, width=width, height=height,
                      title=dict(text=title, x=0.5, xanchor="center"),
                      margin=dict(l=70, r=20, t=60, b=55),
                      legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#ccc", borderwidth=1))
    fig.update_xaxes(showline=True, linecolor="#444", mirror=True, ticks="outside", gridcolor="#eee")
    fig.update_yaxes(showline=True, linecolor="#444", mirror=True, ticks="outside", gridcolor="#eee")
    return fig


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def add_band(fig: go.Figure, x, lo, mid, hi, name: str, color: str, row=None, col=None,
             showlegend: bool = True, dash: str | None = None) -> None:
    """Median line with a shaded 25-75 % band."""
    x = np.asarray(x)
    kw = {} if row is None else dict(row=row, col=col)
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([hi, lo[::-1]]),
                             fill="toself", fillcolor=rgba(color, 0.2), line=dict(width=0),
                             hoverinfo="skip", showlegend=False, legendgroup=name), **kw)
    fig.add_trace(go.Scatter(x=x, y=mid, name=name, line=dict(color=color, width=2, dash=dash),
                             legendgroup=name, showlegend=showlegend), **kw)


_pdf_warned = False


def save(fig: go.Figure, stem: Path) -> list[Path]:
    """Write <stem>.html (plotly.js shared next to it, works offline) and <stem>.pdf."""
    global _pdf_warned
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    html = stem.with_suffix(".html")
    fig.write_html(html, include_plotlyjs="directory")
    out = [html]
    try:
        fig.write_image(stem.with_suffix(".pdf"))
        out.append(stem.with_suffix(".pdf"))
    except Exception as exc:  # noqa: BLE001 - Kaleido raises various errors without Chrome
        if not _pdf_warned:
            warnings.warn(f"PDF export failed ({exc}); run `pixi run setup-chrome` once. HTML was written.")
            _pdf_warned = True
    return out
