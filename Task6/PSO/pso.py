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

Inertia weight (adaptive=True):
    w decreases linearly from w_start (0.9) to w_end (0.4) over iterations,
    balancing exploration early on and exploitation later — consistent with
    pymoo's adaptive=True behaviour in Particle_Swarm_Optimization.py.

Perturbation (pertube_best=True):
    At each iteration, the global best position is slightly perturbed by
    Gaussian noise to prevent premature convergence — consistent with
    pymoo's pertube_best=True in PSO_single_objective().

Boundary handling: absorbing walls — particles that overshoot a bound are
    placed on the boundary and their velocity component is zeroed, consistent
    with pymoo's default boundary handling in Particle_Swarm_Optimization.py.

Initial velocity: "random" — velocities uniformly sampled within [v_min, v_max],
    consistent with initial_velocity="random" in Particle_Swarm_Optimization.py.

Parameters
----------
objective_func    : Callable  — f(x: np.ndarray) -> float, function to minimise.
bounds            : list      — [(min, max), ...] one tuple per decision variable.
num_particles     : int       — swarm size (default 50).
max_iter          : int       — maximum number of iterations / generations.
w                 : float     — initial inertia weight (0 <= w <= 1).
                                If adaptive=True, this is the starting value (w_start).
c1                : float     — cognitive coefficient (personal best pull).
c2                : float     — social coefficient   (global best pull).
max_velocity_rate : float     — v_max = max_velocity_rate * (ub - lb).
adaptive          : bool      — linearly decay w from w (0.9) to 0.4 over iterations.
pertube_best      : bool      — apply Gaussian perturbation to global best each iteration.
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
    num_particles: int = 250,
    max_iter: int = 500,
    w: float = 0.9,
    c1: float = 2.0,
    c2: float = 2.0,
    max_velocity_rate: float = 0.2,
    adaptive: bool = True,
    pertube_best: bool = True,
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

    # ----- Adaptive inertia weight schedule -----
    # Linearly decays w from w_start to w_end over iterations,
    # matching pymoo's adaptive=True behaviour.
    w_start = w        # initial inertia (exploration-heavy)
    w_end   = 0.4      # final inertia   (exploitation-heavy)

    # ----- Initialisation -----
    # Positions: uniform random within bounds
    positions = lb + np.random.rand(num_particles, n_vars) * rng

    # Velocities: "random" — uniform within [v_min, v_max],
    # consistent with initial_velocity="random" in Particle_Swarm_Optimization.py
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

        # -- Adaptive inertia weight: linear decay from w_start to w_end --
        # Consistent with pymoo's adaptive=True in PSO_single_objective()
        if adaptive:
            w_current = w_start - (w_start - w_end) * (iteration / max(1, max_iter - 1))
        else:
            w_current = w_start

        # -- Perturbation of global best --
        # Adds small Gaussian noise to gBest to escape local optima,
        # consistent with pymoo's pertube_best=True in PSO_single_objective().
        # Noise amplitude decays over time as the search focuses.
        if pertube_best:
            sigma = 0.1 * rng * (1.0 - iteration / max_iter)
            g_best_perturbed = g_best_pos + np.random.randn(n_vars) * sigma
            g_best_perturbed = np.clip(g_best_perturbed, lb, ub)
        else:
            g_best_perturbed = g_best_pos

        r1 = np.random.rand(num_particles, n_vars)
        r2 = np.random.rand(num_particles, n_vars)

        # Velocity update (using perturbed gBest for social component)
        cognitive  = c1 * r1 * (p_best_pos - positions)
        social     = c2 * r2 * (g_best_perturbed - positions)
        velocities = w_current * velocities + cognitive + social

        # Clamp velocities to [v_min, v_max]
        velocities = np.clip(velocities, v_min, v_max)

        # Position update
        positions = positions + velocities

        # Absorbing boundary: project back and zero the velocity component —
        # consistent with pymoo's default boundary handling
        for d in range(n_vars):
            too_low  = positions[:, d] < lb[d]
            too_high = positions[:, d] > ub[d]
            positions[too_low,  d] = lb[d]
            positions[too_high, d] = ub[d]
            velocities[too_low,  d] = 0.0    # absorb (zero), not reflect
            velocities[too_high, d] = 0.0

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
                  f"w: {w_current:.3f} | Best cost: {g_best_cost:.6f}")

    return g_best_pos, g_best_cost, history, n_fevals
