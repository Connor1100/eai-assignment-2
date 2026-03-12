import math
import random
from typing import List, Tuple

import numpy as np

from warehouse_env_racks import (
    DEPOT,
    GRID_SIZE,
    NUM_RACKS,
    State,
    get_neighbors,
    objective,
    random_state,
)


# ---------------------------------------------------------------------------
# 1. Steepest-Ascent Hill Climbing (minimise objective via best-improving move)
# ---------------------------------------------------------------------------


def hill_climbing(max_iterations: int = 1000) -> Tuple[State, float, List[float]]:
    """Steepest-ascent hill climbing that minimises the objective function.

    At each step, evaluate *all* neighbours and move to the one with the
    lowest objective value.  Stop when no neighbour improves on the current
    state or when `max_iterations` is reached.
    """
    current = random_state()
    current_cost = objective(current)
    history: List[float] = [current_cost]

    for _ in range(max_iterations):
        neighbors = get_neighbors(current)
        if not neighbors:
            break

        best_neighbor = min(neighbors, key=objective)
        best_cost = objective(best_neighbor)

        if best_cost >= current_cost:
            break  # no improvement – local minimum reached

        current = best_neighbor
        current_cost = best_cost
        history.append(current_cost)

    return current, current_cost, history


# ---------------------------------------------------------------------------
# 2. Simulated Annealing with exponential cooling
# ---------------------------------------------------------------------------


def simulated_annealing(
    max_iterations: int = 5000,
    initial_temp: float = 50.0,
    cooling_rate: float = 0.995,
    min_temp: float = 1e-3,
) -> Tuple[State, float, List[float]]:
    """Simulated annealing with an exponential cooling schedule.

    Accepts worse solutions with probability exp(-delta / T) where delta is
    the increase in objective value and T is the current temperature.
    """
    current = random_state()
    current_cost = objective(current)

    best = list(current)
    best_cost = current_cost

    temp = initial_temp
    history: List[float] = [current_cost]

    for _ in range(max_iterations):
        if temp < min_temp:
            break

        neighbors = get_neighbors(current)
        if not neighbors:
            break

        neighbor = random.choice(neighbors)
        neighbor_cost = objective(neighbor)
        delta = neighbor_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = neighbor
            current_cost = neighbor_cost

        if current_cost < best_cost:
            best = list(current)
            best_cost = current_cost

        temp *= cooling_rate
        history.append(best_cost)

    return best, best_cost, history


# ---------------------------------------------------------------------------
# 3. Genetic Algorithm
# ---------------------------------------------------------------------------


def _tournament_select(
    population: List[State], fitnesses: List[float], k: int = 3
) -> State:
    """Select an individual via tournament selection (minimisation)."""
    indices = random.sample(range(len(population)), k)
    winner = min(indices, key=lambda i: fitnesses[i])
    return list(population[winner])


def _crossover(parent1: State, parent2: State) -> State:
    """Order-based crossover that preserves uniqueness of rack positions."""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child_positions = set(parent1[start:end])
    child: List[Tuple[int, int] | None] = [None] * size
    child[start:end] = parent1[start:end]

    fill = [pos for pos in parent2 if pos not in child_positions]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1

    return child  # type: ignore[return-value]


def _mutate(state: State, mutation_rate: float = 0.1) -> State:
    """Mutate by moving a random rack to a random valid neighbouring cell."""
    if random.random() > mutation_rate:
        return state

    state = list(state)
    occupied = set(state)
    idx = random.randrange(len(state))
    x, y = state[idx]

    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(deltas)
    for dx, dy in deltas:
        nx, ny = x + dx, y + dy
        if (
            0 <= nx < GRID_SIZE
            and 0 <= ny < GRID_SIZE
            and (nx, ny) != DEPOT
            and (nx, ny) not in occupied
        ):
            state[idx] = (nx, ny)
            return state

    return state  # no valid move found, return unchanged


def genetic_algorithm(
    pop_size: int = 50,
    max_generations: int = 500,
    mutation_rate: float = 0.15,
    tournament_k: int = 3,
) -> Tuple[State, float, List[float]]:
    """Genetic algorithm with tournament selection, crossover, and mutation."""
    # Initialise population
    population = [random_state() for _ in range(pop_size)]
    fitnesses = [objective(ind) for ind in population]

    best_idx = int(np.argmin(fitnesses))
    best = list(population[best_idx])
    best_cost = fitnesses[best_idx]
    history: List[float] = [best_cost]

    for _ in range(max_generations):
        new_population: List[State] = []

        # Elitism: carry over the best individual
        new_population.append(list(best))

        while len(new_population) < pop_size:
            parent1 = _tournament_select(population, fitnesses, tournament_k)
            parent2 = _tournament_select(population, fitnesses, tournament_k)
            child = _crossover(parent1, parent2)
            child = _mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        fitnesses = [objective(ind) for ind in population]

        gen_best_idx = int(np.argmin(fitnesses))
        if fitnesses[gen_best_idx] < best_cost:
            best = list(population[gen_best_idx])
            best_cost = fitnesses[gen_best_idx]

        history.append(best_cost)

    return best, best_cost, history
