"""
ucs_pathfinder.py
=================
Uniform Cost Search over the WarehouseEnv state space.

The robot must:
  1. Navigate from its start position to the Pickup cell (P).
  2. Execute the 'PICK' action to collect the item.
  3. Navigate to the Dropoff cell (D) while carrying the item (goal).

All move actions cost 1 step, so g(n) equals the number of steps taken and
UCS is guaranteed to return the optimal (shortest) path.

State representation
--------------------
A state is a 3-tuple: (row: int, col: int, has_item: bool)
"""

from __future__ import annotations

import heapq
import sys
import time
from itertools import count

from warehouse_env import WarehouseEnv
from warehouse_viz import replay_animation


# ---------------------------------------------------------------------------
# Environment helpers
# These bridge the WarehouseEnv API to the needs of UCS without modifying
# warehouse_env.py.
# ---------------------------------------------------------------------------

def _initial_state(env: WarehouseEnv) -> tuple:
    """Return the initial state (row, col, has_item=False)."""
    r, c = env.start_pos
    return (r, c, False)


def _successors(
    env: WarehouseEnv,
    state: tuple,
    pickup_pos: tuple,
) -> list[tuple[str, tuple, int]]:
    """Return reachable (action, next_state, step_cost) triples."""
    r, c, has_item = state
    results = []

    # Movement: N / E / S / W, each costs 1 step.
    for action, (dr, dc) in env.MOVE_DELTAS.items():
        nr, nc = r + dr, c + dc
        if not env._is_wall(nr, nc):
            results.append((action, (nr, nc, has_item), 1))

    # Pickup: valid only on the P tile when the robot is not already loaded.
    if (r, c) == pickup_pos and not has_item:
        results.append(("PICK", (r, c, True), 1))

    return results


def _render_state(env: WarehouseEnv, state: tuple) -> list[list[str]]:
    """Overlay the robot on env.grid and return a 2-D char frame."""
    r, c, has_item = state
    frame = [list(row) for row in env.grid]
    frame[r][c] = "r" if has_item else "R"
    return frame


def _get_frames(
    env: WarehouseEnv,
    path: list,
    initial: tuple,
) -> list[list[list[str]]]:
    """Convert a solution path into a list of rendered grid frames."""
    frames = [_render_state(env, initial)]
    for _, state in path:
        frames.append(_render_state(env, state))
    return frames


# ---------------------------------------------------------------------------
# Uniform Cost Search
# ---------------------------------------------------------------------------

def uniform_cost_search(
    env: WarehouseEnv,
) -> tuple[list[tuple[str, tuple]] | None, dict]:
    """
    Search for the optimal (minimum-step) path through the warehouse.

    Parameters
    ----------
    env : WarehouseEnv

    Returns
    -------
    path : list of (action, resulting_state) pairs, or None if unsolvable.
    stats : dict
        nodes_expanded   - number of states popped and fully processed
        path_length      - number of steps in the solution (0 if unsolved)
        computation_time - wall-clock seconds elapsed
    """
    t_start = time.perf_counter()
    _tie = count()                      # monotonic counter; breaks g(n) ties

    pickup_pos = env._find_tile("P")
    dropoff_pos = env._find_tile("D")
    initial = _initial_state(env)

    # -- Frontier ----------------------------------------------------------
    # Min-heap ordered by g(n).  Each entry:
    #   (g, tie_breaker, state, path)
    frontier: list = [(0, next(_tie), initial, [])]

    # Best g(n) currently in the heap for each state.
    frontier_costs: dict[tuple, int | float] = {initial: 0}

    # -- Explored ----------------------------------------------------------
    # Every state that has been fully expanded, mapped to its optimal g(n).
    explored: dict[tuple, int] = {}

    nodes_expanded = 0
    max_frontier_size = 1           # initial state is already in frontier

    # -- Main loop ---------------------------------------------------------
    while frontier:
        g, _, state, path = heapq.heappop(frontier)

        # Lazy deletion: skip if a cheaper path already expanded this state.
        if state in explored:
            continue

        frontier_costs.pop(state, None)
        explored[state] = g
        nodes_expanded += 1

        # Goal test at expansion — guarantees optimality.
        r, c, has_item = state
        if has_item and (r, c) == dropoff_pos:
            return path, {
                "nodes_expanded":   nodes_expanded,
                "path_length":      len(path),
                "computation_time": time.perf_counter() - t_start,
                "max_frontier_size": max_frontier_size,
            }

        # Expand successors.
        for action, next_state, step_cost in _successors(env, state, pickup_pos):
            if next_state in explored:
                continue
            new_g = g + step_cost
            if new_g < frontier_costs.get(next_state, float("inf")):
                frontier_costs[next_state] = new_g
                heapq.heappush(
                    frontier,
                    (new_g, next(_tie), next_state, path + [(action, next_state)]),
                )
                if len(frontier_costs) > max_frontier_size:
                    max_frontier_size = len(frontier_costs)

    # Frontier exhausted — no solution exists.
    return None, {
        "nodes_expanded":   nodes_expanded,
        "path_length":      0,
        "computation_time": time.perf_counter() - t_start,
        "max_frontier_size": max_frontier_size,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_grid(grid: list[list[str]], title: str = "") -> None:
    if title:
        print(title)
    for row in grid:
        print("  " + "".join(row))


def _print_stats(stats: dict) -> None:
    print("+------------------------------------------+")
    print("|         UCS Search Statistics            |")
    print("+------------------------------------------+")
    print(f"|  Nodes expanded   : {stats['nodes_expanded']:<22}|")
    print(f"|  Path length      : {stats['path_length']:<22}|")
    print(f"|  Computation time : {stats['computation_time']:.6f} s             |")
    print("+------------------------------------------+")


def _print_path(path: list[tuple[str, tuple]] | None) -> None:
    if path is None:
        print("No solution found.")
        return
    print(f"\nOptimal path  ({len(path)} steps):")
    print(f"  {'Step':>4}  {'Action':<7}  {'(row,col)':<10}  Status")
    print("  " + "-" * 44)
    for i, (action, state) in enumerate(path, 1):
        row, col, has_item = state
        status = "loaded  " if has_item else "unloaded"
        note = "  <- item collected!" if action == "PICK" else ""
        print(f"  {i:>4}  {action:<7}  ({row},{col})        {status}{note}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("  Warehouse  -  Uniform Cost Search")
    print("=" * 50)

    env = WarehouseEnv()

    # Show the layout as defined in warehouse_env.py.
    print("\nWarehouse layout  (from warehouse_env.py):")
    for line in env.grid:
        print("  " + line)

    pickup_pos = env._find_tile("P")
    dropoff_pos = env._find_tile("D")

    print(f"\n  Start   : row {env.start_pos[0]}, col {env.start_pos[1]}")
    print(f"  Pickup  : row {pickup_pos[0]}, col {pickup_pos[1]}"
          f"  ->  'P' in layout")
    print(f"  Dropoff : row {dropoff_pos[0]}, col {dropoff_pos[1]}"
          f"  ->  'D' in layout")

    # Run UCS.
    print("\nRunning UCS...")
    path, stats = uniform_cost_search(env)
    print()
    _print_stats(stats)
    _print_path(path)

    if path is None:
        sys.exit(1)

    # Render frames and show initial / goal state.
    initial = _initial_state(env)
    frames = _get_frames(env, path, initial)
    print()
    _print_grid(frames[0], "Initial state  (frame 0):")
    print()
    _print_grid(frames[-1], f"Goal state  (frame {len(path)}):")

    # Launch the interactive warehouse animation.
    print("\nLaunching animation  (close window to exit)...")
    replay_animation(frames, interval_ms=300)
