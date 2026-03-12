import random
from typing import List, Tuple

import numpy as np

GRID_SIZE = 20
NUM_RACKS = 20
DEPOT = (10, 10)
CONGESTION_RADIUS = 5
CONGESTION_WEIGHT = 2.0


State = List[Tuple[int, int]]


def random_state() -> State:
    """Generate a random initial state of 20 unique rack positions (excluding the depot)."""
    all_cells = [
        (x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE) if (x, y) != DEPOT
    ]
    return random.sample(all_cells, NUM_RACKS)


def objective(state: State) -> float:
    """Compute f(s) = (1/20) * sum(manhattan distances from depot) + 2.0 * congestion_count."""
    total_dist = sum(abs(x - DEPOT[0]) + abs(y - DEPOT[1]) for x, y in state)
    avg_dist = total_dist / NUM_RACKS

    congestion_count = sum(
        1
        for x, y in state
        if abs(x - DEPOT[0]) + abs(y - DEPOT[1]) <= CONGESTION_RADIUS
    )

    return avg_dist + CONGESTION_WEIGHT * congestion_count


def get_neighbors(state: State) -> List[State]:
    """Generate all neighbors by moving one rack by ±1 in x or y, keeping positions unique."""
    occupied = set(state)
    neighbors = []
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i, (x, y) in enumerate(state):
        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < GRID_SIZE
                and 0 <= ny < GRID_SIZE
                and (nx, ny) != DEPOT
                and (nx, ny) not in occupied
            ):
                neighbor = list(state)
                neighbor[i] = (nx, ny)
                neighbors.append(neighbor)

    return neighbors
