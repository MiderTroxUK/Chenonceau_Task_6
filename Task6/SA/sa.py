"""
Subject:        SA algorithm (simple version)
@author:        Pierre Bédrune, John HOARAU
@date:          04/03/2026

Simulated Annealing basique. Les contraintes géométriques sont gérées par le solver.
"""

import numpy as np
from typing import Callable, Tuple, List


def simulated_annealing(
    objective_func: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    **kwargs
) -> Tuple[np.ndarray, float, List[float], int]:
    """
    Simulated Annealing for minimization.
    
    Parameters via **kwargs:
        bounds: List of (min, max) tuples
        T_initial: Initial temperature (default 5000)
        T_final: Final temperature (default 1e-8)
        alpha: Cooling rate (default 0.995)
        max_iter: Max iterations (default 5000)
        n_neighbors: Neighbors per iteration (default 3)
        step_size: Perturbation size (default 0.1)
        seed: Random seed
        verbose: Print progress
        
    Returns:
        best_pos, best_cost, history, n_fevals
    """
    
    # Extract parameters
    bounds = kwargs.get('bounds', None)
    t_initial = kwargs.get('T_initial', 5000.0)
    t_final = kwargs.get('T_final', 1e-8)
    alpha = kwargs.get('alpha', 0.995)
    max_iter = kwargs.get('max_iter', 5000)
    n_neighbors = kwargs.get('n_neighbors', 3)
    step_size = kwargs.get('step_size', 0.1)
    adaptive_step = kwargs.get('adaptive_step', True)
    seed = kwargs.get('seed', None)
    verbose = kwargs.get('verbose', False)

    if seed is not None:
        np.random.seed(seed)

    # Initialize
    x = np.array(initial_guess, dtype=float)
    n_vars = len(x)
    
    if bounds is not None:
        lb = np.array([b[0] for b in bounds], dtype=float)
        ub = np.array([b[1] for b in bounds], dtype=float)
    else:
        lb = np.full(n_vars, 5.0)
        ub = np.full(n_vars, 80.0)
    rng = ub - lb
    
    x = np.clip(x, lb, ub)
    current_cost = objective_func(x)
    n_fevals = 1

    best_pos = x.copy()
    best_cost = current_cost
    history: List[float] = [best_cost]
    
    t = t_initial
    log_every = max(1, max_iter // 10)

    # Main loop
    for iteration in range(max_iter):
        
        # Adaptive step size
        if adaptive_step:
            t_ratio = np.clip(np.sqrt(t / t_initial), 0.01, 1.0)
            step_current = step_size * t_ratio
        else:
            step_current = step_size

        # Generate neighbors
        best_neighbor_x = None
        best_neighbor_cost = np.inf

        for _ in range(n_neighbors):
            perturbation = np.random.randn(n_vars) * step_current * rng
            neighbor_x = np.clip(x + perturbation, lb, ub)

            neighbor_cost = objective_func(neighbor_x)
            n_fevals += 1

            if neighbor_cost < best_neighbor_cost:
                best_neighbor_cost = neighbor_cost
                best_neighbor_x = neighbor_x.copy()

        # Metropolis criterion
        delta_e = best_neighbor_cost - current_cost
        accept = delta_e < 0 or np.random.rand() < np.exp(-delta_e / t)

        if accept:
            x = best_neighbor_x
            current_cost = best_neighbor_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_pos = x.copy()

        history.append(best_cost)
        t = alpha * t

        if t < t_final:
            break

        if verbose and (iteration + 1) % log_every == 0:
            print(f"  [SA] Iter {iteration+1:>5d}/{max_iter} | "
                  f"T: {t:.2e} | Best: {best_cost:.2f}")

    return best_pos, best_cost, history, n_fevals