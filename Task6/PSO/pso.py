"""
Subject:        PSO algorithm
Inheritance:    Particle_Swarm_Optimization.py from Task 3
@author:        Clémence-Philomène HINOT, John HOARAU, Copilot AI
@date:          10/02/2026
"""

# ********** IMPORTATION **********

import numpy as np
from typing import Callable, Tuple, List

# pymoo engine — mirrors Particle_Swarm_Optimization.py
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.core.termination import Termination
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize

# ********** CONVERGENCE CALLBACK **********

class MyCallback(Callback):
    """
    Tracks the global best cost after each generation.
    Mirrors MyCallback in Particle_Swarm_Optimization_multi_obj.py.

    data["best_f1"] : global best cost per generation — returned as history
                      so solver.py can plot the convergence curve.
    data["F"]       : all particle costs per generation.
    data["n_nds"]   : number of particles that matched the global best.
    """

    def __init__(self):
        super().__init__()
        self.data["best_f1"] = []
        self.data["F"]       = []
        self.data["n_nds"]   = []

    def notify(self, algorithm):
        F = algorithm.pop.get("F")          # shape (pop_size, 1)
        best = float(F.min())
        self.data["F"].append(F.copy())
        self.data["best_f1"].append(best)
        self.data["n_nds"].append(int((F == best).sum()))

# ********** PYMOO PROBLEM WRAPPER **********

class _FuncProblem(Problem):
    """
    Wraps a plain callable f(x) -> float into a pymoo Problem so that
    pymoo's PSO engine can be used without changing solver.py's interface.
    """

    def __init__(self, objective_func: Callable, bounds: list):
        lb = np.array([b[0] for b in bounds], dtype=float)
        ub = np.array([b[1] for b in bounds], dtype=float)
        super().__init__(
            n_var=len(bounds),
            n_obj=1,
            xl=lb,
            xu=ub,
        )
        self._func = objective_func
        self.n_fevals = 0

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate all particles in the current generation."""
        F = np.array([self._func(X[i]) for i in range(len(X))], dtype=float)
        self.n_fevals += len(X)
        out["F"] = F.reshape(-1, 1)

# ********** STAGNATION-AWARE TERMINATION **********

class StagnationTermination(Termination):
    """
    Stops when EITHER:
      - max_gen generations are reached, OR
      - the global best has not improved by more than `tol` over the
        last `stagnation_window` generations (early stopping).
    """

    def __init__(self, max_gen: int, stagnation_window: int = 50, tol: float = 1e-6):
        super().__init__()
        self.max_gen           = max_gen
        self.stagnation_window = stagnation_window
        self.tol               = tol
        self._history          = []

    def _update(self, algorithm):
        F = algorithm.pop.get("F")
        self._history.append(float(F.min()))

        if algorithm.n_gen >= self.max_gen:
            return 1.0

        if len(self._history) >= self.stagnation_window:
            window      = self._history[-self.stagnation_window:]
            improvement = window[0] - window[-1]
            if improvement < self.tol:
                return 1.0

        return algorithm.n_gen / self.max_gen

# ********** PARTICLE SWARM OPTIMISATION **********

"""
Interface expected by solver.py (PsoOptimizer):

    best_x, best_cost, history, n_fevals = particle_swarm(
        objective_func = self._objective_func,
        bounds         = self.problem.get_bounds(self.case_num),
        num_particles  = 200,
        max_iter       = 750,
        ...
    )

history is returned as an empty list — only the final result is needed.

Parameters
----------
objective_func    : Callable  — f(x: np.ndarray) -> float, function to minimise.
bounds            : list      — [(min, max), ...] one tuple per decision variable.
num_particles     : int       — swarm size.
max_iter          : int       — number of generations.
w                 : float     — inertia weight (pymoo adaptive=True decays this).
c1                : float     — cognitive coefficient.
c2                : float     — social coefficient.
max_velocity_rate : float     — v_max = max_velocity_rate * (ub - lb).
seed              : int|None  — random seed for reproducibility.
verbose           : bool      — print pymoo progress each generation.
**kwargs          : ignored   — absorbs extra keys from solver.py.

Returns
-------
best_position : np.ndarray  — best parameter vector found.
best_cost     : float       — objective value at best_position.
history       : []          — empty list (history not tracked).
n_fevals      : int         — total objective-function evaluations.
"""

def particle_swarm(
    objective_func: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    num_particles: int = 200,
    max_iter: int = 50000,
    stagnation_window: int = 50,
    tol: float = 1e-6,
    w: float = 0.9,
    c1: float = 2.0,
    c2: float = 2.0,
    max_velocity_rate: float = 0.3,
    seed: int = 10,
    verbose: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, float, List[float], int]:

    try:
        # ----- Wrap objective into a pymoo Problem -----
        problem = _FuncProblem(objective_func, bounds)

        # ----- Configure PSO — mirrors PSO_single_objective() -----
        algorithm = PSO(
            pop_size          = num_particles,
            w                 = w,
            c1                = c1,
            c2                = c2,
            max_velocity_rate = max_velocity_rate,
            adaptive          = True,
            initial_velocity  = "random",
            pertube_best      = True,
        )

        # ----- Convergence callback -----
        callback = MyCallback()

        # ----- Stagnation-aware termination -----
        termination = StagnationTermination(
            max_gen           = max_iter,
            stagnation_window = stagnation_window,
            tol               = tol,
        )

        # ----- Run optimisation -----
        result = minimize(
            problem,
            algorithm,
            termination = termination,
            seed        = seed,
            verbose     = verbose,
            callback    = callback,
        )

        # Store callback on result for external access
        result.callback = callback

        history = callback.data["best_f1"]  # one float per generation

        return result.X.flatten(), float(result.F.flatten()[0]), history, problem.n_fevals

    except Exception as e:
        print(f"Error during PSO optimisation: {e}")
        raise