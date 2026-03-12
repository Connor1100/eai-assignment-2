import random
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from local_search import genetic_algorithm, hill_climbing, simulated_annealing
from warehouse_env_racks import objective
from warehouse_viz_racks import plot_convergence, plot_layout

NUM_RUNS = 20


def run_experiments() -> None:
    """Run all three algorithms 20 times and report results."""
    algorithms = {
        "Hill Climbing": hill_climbing,
        "Simulated Annealing": simulated_annealing,
        "Genetic Algorithm": genetic_algorithm,
    }

    results: Dict[str, List[float]] = {name: [] for name in algorithms}
    best_states: Dict[str, Tuple[list, float]] = {}
    all_histories: Dict[str, List[List[float]]] = {name: [] for name in algorithms}
    timings: Dict[str, List[float]] = {name: [] for name in algorithms}

    for name, algo in algorithms.items():
        print(f"\n{'=' * 60}")
        print(f"Running {name} ({NUM_RUNS} trials)...")
        print(f"{'=' * 60}")

        best_overall_cost = float("inf")

        for i in range(NUM_RUNS):
            random.seed(i)
            np.random.seed(i)

            start = time.perf_counter()
            state, cost, history = algo()
            elapsed = time.perf_counter() - start

            results[name].append(cost)
            all_histories[name].append(history)
            timings[name].append(elapsed)

            if cost < best_overall_cost:
                best_overall_cost = cost
                best_states[name] = (state, cost)

            print(f"  Run {i + 1:2d}: f(s) = {cost:.4f}  ({elapsed:.3f}s)")

    # ---- Summary table ----
    print(f"\n{'=' * 60}")
    print("Summary of Results")
    print(f"{'=' * 60}")
    print(f"{'Algorithm':<25} {'Best':>8} {'Mean':>8} {'Std':>8} {'Avg Time':>10}")
    print("-" * 63)
    for name in algorithms:
        vals = results[name]
        times = timings[name]
        print(
            f"{name:<25} {min(vals):8.4f} {np.mean(vals):8.4f} "
            f"{np.std(vals):8.4f} {np.mean(times):10.3f}s"
        )

    # ---- Convergence plot (average across runs) ----
    avg_histories: Dict[str, List[float]] = {}
    for name in algorithms:
        max_len = max(len(h) for h in all_histories[name])
        padded = []
        for h in all_histories[name]:
            padded.append(h + [h[-1]] * (max_len - len(h)))
        avg_histories[name] = np.mean(padded, axis=0).tolist()

    plot_convergence(avg_histories)

    # ---- Best layout per algorithm ----
    for name in algorithms:
        state, cost = best_states[name]
        plot_layout(state, title=f"{name} (f={cost:.4f})")


if __name__ == "__main__":
    run_experiments()
