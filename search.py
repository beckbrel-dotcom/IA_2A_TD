"""Cost-based search algorithms for the Pokémon environment."""

from __future__ import annotations

import heapq
import random
from itertools import count

from pokemon_game import (
    GameMap,
    State,
    movement_cost,
    successors,
)


def reconstruct_path(
    parents: dict[State, State | None],
    goal: State,
) -> list[State]:
    """Reconstruct a path from a parent dictionary."""
    path: list[State] = []
    current: State | None = goal

    while current is not None:
        path.append(current)
        current = parents[current]

    path.reverse()
    return path


def path_cost(
    game_map: GameMap,
    path: list[State],
) -> int:
    """Return the total cost of a path."""
    return sum(
        movement_cost(game_map, state)
        for state in path[1:]
    )




def manhattan_distance(
    state: State,
    goal: State,
) -> int:
    """Return the Manhattan distance between two states."""
    return (
        abs(state[0] - goal[0])
        + abs(state[1] - goal[1])
    )


def generate_random_path(
    game_map: GameMap,
    start: State,
    goal: State,
    seed: int | None = None,
) -> list[State]:
    """Generate a random valid path from start to goal.

    This function is provided only to demonstrate the visualization.
    It is not guaranteed to return a minimum-cost or shortest path.

    The function performs a randomized depth-first exploration: the valid
    successor states are shuffled before being added to the frontier.

    Parameters
    ----------
    game_map : GameMap
        Weighted Pokémon map.
    start : State
        Initial trainer position.
    goal : State
        Pokémon Center position.
    seed : int or None, default=None
        Random seed used to make the result reproducible.

    Returns
    -------
    list[State]
        A valid path beginning at ``start`` and ending at ``goal``.

    Raises
    ------
    RuntimeError
        If no path exists between the start and the goal.
    """
    rng = random.Random(seed)

    frontier = [start]

    parents: dict[State, State | None] = {
        start: None,
    }

    while frontier:
        current = frontier.pop()

        if current == goal:
            return reconstruct_path(parents, goal)

        next_states = successors(game_map, current)
        rng.shuffle(next_states)

        for next_state in next_states:
            if next_state in parents:
                continue

            parents[next_state] = current
            frontier.append(next_state)

    raise RuntimeError(
        "No path exists between the start and the goal."
    )

def dijkstra(
    game_map: GameMap,
    start: State,
    goal: State,
) -> tuple[list[State], int]:
    """Return a minimum-cost path using Dijkstra's algorithm.

    Returns
    -------
    path : list[State]
        Minimum-cost path from start to goal.
    considered_states : int
        Number of states removed from the priority queue.
    """
    if start == goal:
        return [start], 0

    rows = len(game_map)
    cols = len(game_map[0])

    # 1. Initialisation de la matrice des coûts avec l'infini
    costs_matrix = [[float("inf") for _ in range(cols)] for _ in range(rows)]
    costs_matrix[start[0]][start[1]] = 0

    # 2. File de priorité (min-heap) stockant les tuples : (coût_cumulé, état)
    pq = [(0, start)]
    
    # Dictionnaire pour mémoriser le parent de chaque état et reconstruire le chemin
    came_from = {start: None}
    
    considered_states = 0

    while pq:
        current_cost, current_state = heapq.heappop(pq)

        # Si nous avons atteint le centre Pokémon (goal)
        if current_state == goal:
            break

        # Si nous avons déjà trouvé un chemin plus court vers cet état, on l'ignore
        if current_cost > costs_matrix[current_state[0]][current_state[1]]:
            continue

        considered_states += 1

        # 3. Exploration des voisins valides grâce à la fonction du framework
        for neighbor in successors(game_map, current_state):
            # Calcul du coût pour entrer dans le voisin (herbe vs rocher)
            move_cost = movement_cost(game_map, neighbor)
            new_cost = current_cost + move_cost

            r, c = neighbor
            # Si on trouve un chemin moins coûteux vers le voisin
            if new_cost < costs_matrix[r][c]:
                costs_matrix[r][c] = new_cost
                came_from[neighbor] = current_state
                heapq.heappush(pq, (new_cost, neighbor))

    # Si aucun chemin n'a été trouvé vers le but
    if goal not in came_from:
        return [], considered_states

    # 4. Reconstruction du chemin final de l'arrivée vers le départ
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()  # Remet le chemin dans l'ordre du départ vers l'arrivée

    return path, considered_states


def a_star(
    game_map: GameMap,
    start: State,
    goal: State,
) -> tuple[list[State], int]:
    """Return a minimum-cost path using A* search.

    The Manhattan distance is used as the heuristic.

    Returns
    -------
    path : list[State]
        Minimum-cost path from start to goal.
    considered_states : int
        Number of states removed from the priority queue.
    """
    if start == goal:
        return [start], 0

    # Fonction heuristique : distance de Manhattan
    def heuristic(state: State, target: State) -> int:
        return abs(state[0] - target[0]) + abs(state[1] - target[1])

    # File de priorité stockant les tuples : (f_cost, g_cost, state)
    # f_cost = g_cost (coût réel depuis le départ) + h_cost (estimation vers l'objectif)
    start_h = heuristic(start, goal)
    pq = [(start_h, 0, start)]

    # Dictionnaires pour les coûts réels (g_costs) et la reconstruction du chemin
    g_costs = {start: 0}
    came_from = {start: None}
    
    considered_states = 0

    while pq:
        f_cost, current_g, current_state = heapq.heappop(pq)

        # Si on atteint l'objectif
        if current_state == goal:
            break

        # Si on a déjà trouvé un meilleur chemin pour cet état, on l'ignore
        if current_g > g_costs.get(current_state, float("inf")):
            continue

        considered_states += 1

        # Exploration des voisins valides en utilisant le framework du jeu
        for neighbor in successors(game_map, current_state):
            move_cost = movement_cost(game_map, neighbor) #
            new_g = current_g + move_cost

            # Si ce nouveau chemin vers le voisin est meilleur
            if new_g < g_costs.get(neighbor, float("inf")):
                g_costs[neighbor] = new_g
                came_from[neighbor] = current_state
                
                # Calcul de f = g + h
                h_cost = heuristic(neighbor, goal)
                new_f = new_g + h_cost
                
                heapq.heappush(pq, (new_f, new_g, neighbor))

    # Si aucun chemin n'a été trouvé
    if goal not in came_from:
        return [], considered_states

    # Reconstruction du chemin de l'arrivée vers le départ
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()

    return path, considered_states