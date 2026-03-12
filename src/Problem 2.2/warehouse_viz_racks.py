import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, List, Tuple

from warehouse_env_racks import GRID_SIZE, DEPOT, CONGESTION_RADIUS


def plot_convergence(history: Dict[str, List[float]]) -> None:
    """Plot objective-value convergence curves for each algorithm."""
    plt.figure(figsize=(10, 6))
    for name, values in history.items():
        plt.plot(values, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value f(s)")
    plt.title("Convergence of Local Search Algorithms")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergence.png", dpi=150)
    plt.show()


def plot_layout(state: List[Tuple[int, int]], title: str = "Warehouse Layout") -> None:
    """Plot the warehouse grid showing depot, racks, and congestion radius."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw grid
    for x in range(GRID_SIZE + 1):
        ax.axhline(y=x, color="lightgrey", linewidth=0.5)
        ax.axvline(x=x, color="lightgrey", linewidth=0.5)

    # Draw congestion radius circle
    circle = patches.Circle(
        (DEPOT[0] + 0.5, DEPOT[1] + 0.5),
        CONGESTION_RADIUS,
        linewidth=1.5,
        edgecolor="orange",
        facecolor="orange",
        alpha=0.1,
        label=f"Congestion zone (r={CONGESTION_RADIUS})",
    )
    ax.add_patch(circle)

    # Draw depot
    depot_rect = patches.Rectangle(
        (DEPOT[0], DEPOT[1]), 1, 1,
        linewidth=2, edgecolor="black", facecolor="green",
    )
    ax.add_patch(depot_rect)
    ax.text(DEPOT[0] + 0.5, DEPOT[1] + 0.5, "D", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")

    # Draw racks
    for x, y in state:
        color = "red" if abs(x - DEPOT[0]) + abs(y - DEPOT[1]) <= CONGESTION_RADIUS else "steelblue"
        rect = patches.Rectangle(
            (x, y), 1, 1,
            linewidth=1, edgecolor="black", facecolor=color, alpha=0.8,
        )
        ax.add_patch(rect)

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)

    # Legend
    legend_elements = [
        patches.Patch(facecolor="green", edgecolor="black", label="Depot"),
        patches.Patch(facecolor="red", edgecolor="black", label="Rack (congested)"),
        patches.Patch(facecolor="steelblue", edgecolor="black", label="Rack (normal)"),
        patches.Patch(facecolor="orange", alpha=0.2, edgecolor="orange", label="Congestion zone"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png", dpi=150)
    plt.show()
