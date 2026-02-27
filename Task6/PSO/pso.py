"""
Subject:        PSO algorithm
Inheritance:    Particle_Swarm_Optimization.py from Task 3
@author:        Clémence-Philomène HINOT, John HOARAU, Copilot AI
@date:          10/02/2026
"""

# ********** IMPORTATION **********

import numpy as np
from typing import Callable, Tuple, List

# ********** PARTICLE SWARM OPTIMISATION **********

"""
Particle Swarm Optimisation for single-objective minimisation.

Velocity update rule (standard PSO):
    v  <-  w*v  +  c1*r1*(pBest - x)  +  c2*r2*(gBest - x)
    x  <-  x + v

Boundary handling: reflecting walls — particles that overshoot a bound are
placed on the boundary and their velocity component is reversed.

Parameters
----------
objective_func    : Callable  — f(x: np.ndarray) -> float, function to minimise.
bounds            : list      — [(min, max), ...] one tuple per decision variable.
num_particles     : int       — swarm size (default 30, solver.py uses 30).
max_iter          : int       — maximum number of iterations / generations.
w                 : float     — inertia weight        (0 <= w <= 1).
c1                : float     — cognitive coefficient (personal best pull).
c2                : float     — social coefficient   (global best pull).
max_velocity_rate : float     — v_max = max_velocity_rate * (ub - lb).
seed              : int|None  — random seed for reproducibility.
verbose           : bool      — print progress every 10 % of iterations.
**kwargs          : ignored   — absorbs extra keyword arguments from solver.py.

Returns
-------
best_position : np.ndarray  — best parameter vector found.
best_cost     : float       — objective value at best_position.
history       : List[float] — global best cost after each iteration.
n_fevals      : int         — total objective-function evaluations.
"""
def particle_swarm(
    objective_func: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    num_particles: int = 50,
    max_iter: int = 200,
    w: float = 0.9,
    c1: float = 2.0,
    c2: float = 2.0,
    max_velocity_rate: float = 0.2,
    seed: int = None,
    verbose: bool = False,
    **kwargs
) -> Tuple[np.ndarray, float, List[float], int]:
    
    if seed is not None:
        np.random.seed(seed)

    n_vars = len(bounds)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    rng = ub - lb  # search-space width per dimension

    # ----- Velocity limits -----
    v_max = max_velocity_rate * rng
    v_min = -v_max

    # ----- Initialisation -----
    # Positions: uniform random within bounds
    positions  = lb + np.random.rand(num_particles, n_vars) * rng
    # Velocities: uniform random within [v_min, v_max]
    velocities = v_min + np.random.rand(num_particles, n_vars) * (v_max - v_min)

    # Evaluate initial swarm
    costs    = np.array([objective_func(positions[i]) for i in range(num_particles)])
    n_fevals = num_particles

    # Personal bests
    p_best_pos  = positions.copy()
    p_best_cost = costs.copy()

    # Global best
    g_best_idx  = int(np.argmin(p_best_cost))
    g_best_pos  = p_best_pos[g_best_idx].copy()
    g_best_cost = p_best_cost[g_best_idx]

    history: List[float] = []
    log_every = max(1, max_iter // 10)

    # ----- Main optimisation loop -----
    for iteration in range(max_iter):

        r1 = np.random.rand(num_particles, n_vars)
        r2 = np.random.rand(num_particles, n_vars)

        # Velocity update
        cognitive  = c1 * r1 * (p_best_pos - positions)
        social     = c2 * r2 * (g_best_pos - positions)
        velocities = w * velocities + cognitive + social

        # Clamp velocities to [v_min, v_max]
        velocities = np.clip(velocities, v_min, v_max)

        # Position update
        positions = positions + velocities

        # Reflecting boundary: project back and reverse velocity component
        for d in range(n_vars):
            too_low  = positions[:, d] < lb[d]
            too_high = positions[:, d] > ub[d]
            positions[too_low,  d] = lb[d]
            positions[too_high, d] = ub[d]
            velocities[too_low,  d] *= -1.0
            velocities[too_high, d] *= -1.0

        # Evaluate new positions
        new_costs = np.array([objective_func(positions[i]) for i in range(num_particles)])
        n_fevals += num_particles

        # Update personal bests
        improved = new_costs < p_best_cost
        p_best_pos[improved]  = positions[improved]
        p_best_cost[improved] = new_costs[improved]

        # Update global best
        g_best_idx_new = int(np.argmin(p_best_cost))
        if p_best_cost[g_best_idx_new] < g_best_cost:
            g_best_cost = p_best_cost[g_best_idx_new]
            g_best_pos  = p_best_pos[g_best_idx_new].copy()

        history.append(g_best_cost)

        if verbose and (iteration + 1) % log_every == 0:
            print(f"  [PSO] Iter {iteration + 1:>4d}/{max_iter} | "
                  f"Best cost: {g_best_cost:.6f}")

    return g_best_pos, g_best_cost, history, n_fevals