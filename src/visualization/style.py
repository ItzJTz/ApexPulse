"""Shared Matplotlib styling for standalone ApexPulse plots."""

import matplotlib.pyplot as plt


def apply_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#0e1117",
            "axes.facecolor": "#161b22",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#c9d1d9",
            "text.color": "#c9d1d9",
            "xtick.color": "#c9d1d9",
            "ytick.color": "#c9d1d9",
            "grid.color": "#21262d",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
        }
    )
