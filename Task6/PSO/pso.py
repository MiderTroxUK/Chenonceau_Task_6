import numpy as np
from typing import Callable, Tuple, List

def particle_swarm(
    objective_func: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    num_particles: int = 50,
    max_iter: int = 200,
    **kwargs
) -> Tuple[np.ndarray, float, List[float], int]:
    """
    Clémence's Particle Swarm Optimization implementation.
    
    Args:
        objective_func: The function to minimize. Takes an array of variables and returns the cost.
        bounds: List of tuples representing the (min, max) range for each parameter.
        num_particles: Number of particles in the swarm.
        max_iter: Maximum number of iterations.
        
    Returns:
        best_position (np.ndarray): The best combination of parameters found.
        best_cost (float): The cost evaluated at the best_position.
        history (List[float]): The history of the global best cost over iterations.
        n_fevals (int): Total number of times the objective function was evaluated.
    """
    
    # TODO (Clémence): Implement your PSO here!
    
    # These return values are dummy values (dummy_x, 0.0, [0.0], 1)
    # They are here temporarily so that solver.py won't crash during testing.
    # Replace these with your actual best position, cost, history, and eval count.
    dummy_x = np.array([b[0] for b in bounds])
    return dummy_x, 0.0, [0.0], 1

