"""Matplotlib styling and figure helpers (static, print-oriented figures)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical slots in fixed order (reference palette); colour follows the entity, never its rank.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
METHOD_STYLE = {
    "full": {"color": SLOTS[0], "marker": "o", "label": "Full (abstraction refinement)"},
    "fixed": {"color": SLOTS[1], "marker": "s", "label": "Fixed abstraction"},
    "oracle": {"color": SLOTS[2], "marker": "^", "label": "Oracle abstraction"},
    "mlci": {"color": SLOTS[3], "marker": "D", "label": "MLCI"},
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setup():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 6,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_layout(ax, layout, title: str = "", show_aisles: bool = True):
    """Grid figure: shelves dark, aisles (ground truth, evaluation only) light blue, labels for specials."""
    W, H = layout.width, layout.height
    for (x, y) in layout.shelves:
        ax.add_patch(plt.Rectangle((x, H - 1 - y), 1, 1, color="#6b6a66"))
    if show_aisles:
        for name, cells in layout.aisles.items():
            for (x, y) in cells:
                ax.add_patch(plt.Rectangle((x, H - 1 - y), 1, 1, color="#cfe2f7"))
            x, y = cells[len(cells) // 2]
            ax.text(x + 0.5, H - 1 - y + 0.5, name, ha="center", va="center", fontsize=8, color=INK, weight="bold")
    specials = [("E", layout.entrance), ("X", layout.exit), ("C", layout.cart_station)]
    if layout.return_area:
        specials.append(("R", layout.return_area))
    for lab, (x, y) in specials:
        ax.text(x + 0.5, H - 1 - y + 0.5, lab, ha="center", va="center", fontsize=10, weight="bold", color=INK)
    for name, (x, y) in layout.items.items():
        ax.text(x + 0.5, H - 1 - y + 0.5, name, ha="center", va="center", fontsize=7, color="white")
    for i in range(W + 1):
        ax.plot([i, i], [0, H], color="#bdbcb7", lw=0.5)
    for j in range(H + 1):
        ax.plot([0, W], [j, j], color="#bdbcb7", lw=0.5)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(W) + 0.5)
    ax.set_xticklabels(range(W), fontsize=6)
    ax.set_yticks(np.arange(H) + 0.5)
    ax.set_yticklabels(range(H - 1, -1, -1), fontsize=6)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title)


def draw_path(ax, domain, traj, color=SLOTS[0], offset=0.0, label=None):
    H = domain.layout.height
    st = traj.states(domain)
    pts = [domain.cell(s[0]) for s in st]
    xs = [p[0] + 0.5 + offset for p in pts]
    ys = [H - 1 - p[1] + 0.5 + offset for p in pts]
    ax.plot(xs, ys, color=color, lw=1.5, alpha=0.9, label=label)
    for k, a in enumerate(traj.actions):
        n = domain.action_name(a)
        x, y = xs[k], ys[k]
        if n == "PICKUP_CART":
            ax.plot(x, y, marker="^", color=color, ms=7, mec="white")
        elif n == "LEAVE_CART":
            ax.plot(x, y, marker="v", color=color, ms=7, mec="white")
        elif n.startswith("PICKUP_ITEM"):
            ax.plot(x, y, marker="o", color=color, ms=5, mec="white")


def mean_ci_line(ax, x, means, lo, hi, style, label=None):
    ax.plot(x, means, color=style["color"], marker=style["marker"], label=label or style["label"], ms=5)
    ax.fill_between(x, lo, hi, color=style["color"], alpha=0.15, lw=0)
