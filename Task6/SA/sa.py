
import numpy as np
from typing import Callable, Tuple, List


def simulated_annealing(
    objective_func: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    **kwargs
) -> Tuple[np.ndarray, float, List[float], int]:
    """
    Pierre's Simulated Annealing implementation.

    Args:
        objective_func: The function to minimize. Takes an array of variables and returns the cost.
        initial_guess: The starting point for the algorithm.

    Returns:
        best_position (np.ndarray): The best combination of parameters found.
        best_cost (float): The cost evaluated at the best_position.
        history (List[float]): The history of the best cost over iterations.
        n_fevals (int): Total number of times the objective function was evaluated.
    """

    # ----- Extract parameters from kwargs -----
    bounds = kwargs.get('bounds', None)
    T_initial = kwargs.get('T_initial', 5000.0)
    T_final = kwargs.get('T_final', 1e-8)
    alpha = kwargs.get('alpha', 0.995)
    max_iter = kwargs.get('max_iter', 5000)
    n_neighbors = kwargs.get('n_neighbors', 3)
    step_size = kwargs.get('step_size', 0.15)
    adaptive_step = kwargs.get('adaptive_step', True)
    seed = kwargs.get('seed', None)
    verbose = kwargs.get('verbose', False)

    if seed is not None:
        np.random.seed(seed)

    x = np.array(initial_guess, dtype=float)
    n_vars = len(x)

    # ----- Bounds handling -----
    if bounds is not None:
        lb = np.array([b[0] for b in bounds], dtype=float)
        ub = np.array([b[1] for b in bounds], dtype=float)
        rng = ub - lb  # search-space width per dimension
    else:
        lb = None
        ub = None
        rng = np.abs(x) + 1.0  # fallback scaling

    # ----- Initial evaluation -----
    current_cost = objective_func(x)
    n_fevals = 1

    best_pos = x.copy()
    best_cost = current_cost

    history: List[float] = [best_cost]
    log_every = max(1, max_iter // 10)

    # ----- Temperature schedule -----
    T = T_initial

    # ----- Main optimisation loop -----
    for iteration in range(max_iter):

        # -- Adaptive step size: decreases with temperature --
        # Allows large exploration at high T, fine-tuning at low T
        if adaptive_step:
            t_ratio = np.sqrt(T / T_initial)
            t_ratio = np.clip(t_ratio, 0.01, 1.0)
            step_current = step_size * t_ratio
        else:
            step_current = step_size

        # -- Generate and evaluate neighbors --
        # Sample n_neighbors candidates, keep the best for acceptance test
        best_neighbor_x = None
        best_neighbor_cost = np.inf

        for _ in range(n_neighbors):
            # Gaussian perturbation scaled by search range
            perturbation = np.random.randn(n_vars) * step_current * rng
            neighbor_x = x + perturbation

            # Clipping boundary: project back onto bounds
            if lb is not None and ub is not None:
                neighbor_x = np.clip(neighbor_x, lb, ub)

            neighbor_cost = objective_func(neighbor_x)
            n_fevals += 1

            if neighbor_cost < best_neighbor_cost:
                best_neighbor_cost = neighbor_cost
                best_neighbor_x = neighbor_x.copy()

        # -- Metropolis acceptance criterion --
        delta_E = best_neighbor_cost - current_cost

        if delta_E < 0:
            # Better solution: always accept
            accept = True
        else:
            # Worse solution: accept with probability exp(-ΔE/T)
            acceptance_prob = np.exp(-delta_E / T)
            accept = np.random.rand() < acceptance_prob

        if accept and best_neighbor_x is not None:
            x = best_neighbor_x
            current_cost = best_neighbor_cost

            # Update global best
            if current_cost < best_cost:
                best_cost = current_cost
                best_pos = x.copy()

        history.append(best_cost)

        # -- Cooling: geometric decay --
        T = alpha * T

        # -- Early stopping --
        if T < T_final:
            if verbose:
                print(f"  [SA] Early stop at iter {iteration + 1}: T={T:.2e} < T_final={T_final:.2e}")
            break

        # -- Verbose output --
        if verbose and (iteration + 1) % log_every == 0:
            print(f"  [SA] Iter {iteration + 1:>5d}/{max_iter} | "
                  f"T: {T:.2e} | Best cost: {best_cost:.6f}")

    return best_pos, best_cost, history, n_fevals
