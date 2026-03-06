import numpy as np
from typing import Callable, Tuple, List, Optional, Sequence

def shape_guidance_penalty(
    objective_func, x, smooth_weight=0.0, slope_weight=0.0,
    slope_min=0.20, slope_max=10.0, active_cases=None
):
    """Fallback dummy function if true shapes guidance is missing."""
    return 0.0


def bfgs_optimize(
    objective_func: Callable[[np.ndarray], float],
    initial_guess: np.ndarray,
    **kwargs
) -> Tuple[np.ndarray, float, List[float], int]:
    """
    Martin's BFGS implementation.

    Args:
        objective_func: The function to minimize. Takes an array of variables and returns the cost.
        initial_guess: The starting point for the algorithm.

    Returns:
        best_position (np.ndarray): The best combination of parameters found.
        best_cost (float): The cost evaluated at the best_position.
        history (List[float]): The history of the best cost over iterations.
        n_fevals (int): Total number of times the objective function was evaluated.
    """
    max_iter = int(kwargs.get("max_iter", 300))
    tol_grad = float(kwargs.get("tol_grad", kwargs.get("gtol", 1e-6)))
    tol_step = float(kwargs.get("tol_step", 1e-10))
    fd_eps = float(kwargs.get("fd_eps", kwargs.get("epsilon", 1e-6)))
    c1 = float(kwargs.get("c1", 1e-4))
    backtrack_beta = float(kwargs.get("backtrack_beta", 0.5))
    min_step = float(kwargs.get("min_step", 1e-12))
    verbose = bool(kwargs.get("verbose", False))
    reset_hessian_on_bad_curvature = bool(kwargs.get("reset_hessian_on_bad_curvature", True))
    use_shape_guidance = bool(kwargs.get("use_shape_guidance", True))
    # Stronger shape guidance defaults to discourage flat vertical stacks
    # while keeping flexibility at the fixed-end transitions.
    shape_smooth_weight = float(kwargs.get("shape_smooth_weight", 0.0))
    shape_slope_weight = float(kwargs.get("shape_slope_weight", 5e4))
    shape_slope_min = float(kwargs.get("shape_slope_min", 0.20))
    shape_slope_max = float(kwargs.get("shape_slope_max", 10.0))
    shape_active_cases = kwargs.get("shape_active_cases", (1, 2, 3, 4, 5, 6, 7))

    x = np.asarray(initial_guess, dtype=float).reshape(-1)
    n_vars = x.size

    bounds = kwargs.get("bounds")
    if bounds is None and kwargs.get("use_problem_bounds", True):
        bounds = _infer_bounds_from_objective(objective_func)
    lb, ub = _parse_bounds(bounds, n_vars)
    has_bounds = lb is not None

    if has_bounds:
        x = _project_box(x, lb, ub)

    n_fevals = 0
    best_x = x.copy()
    best_cost = np.inf

    def evaluate(z: np.ndarray) -> float:
        nonlocal n_fevals, best_cost, best_x
        z = np.asarray(z, dtype=float)
        if has_bounds:
            z = _project_box(z, lb, ub)

        cost = float(objective_func(z))
        if use_shape_guidance:
            cost += shape_guidance_penalty(
                objective_func=objective_func,
                x=z,
                smooth_weight=shape_smooth_weight,
                slope_weight=shape_slope_weight,
                slope_min=shape_slope_min,
                slope_max=shape_slope_max,
                active_cases=shape_active_cases
            )

        if not np.isfinite(cost):
            cost = np.finfo(float).max
        n_fevals += 1
        if cost < best_cost:
            best_cost = cost
            best_x = z.copy()
        return cost

    def numerical_gradient(z: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(z, dtype=float)
        for i in range(n_vars):
            step = fd_eps * max(1.0, abs(z[i]))
            z_plus = z.copy()
            z_minus = z.copy()
            z_plus[i] += step
            z_minus[i] -= step

            if has_bounds:
                z_plus = _project_box(z_plus, lb, ub)
                z_minus = _project_box(z_minus, lb, ub)

            denom = z_plus[i] - z_minus[i]
            if abs(denom) < 1e-15:
                grad[i] = 0.0
                continue

            f_plus = evaluate(z_plus)
            f_minus = evaluate(z_minus)
            grad[i] = (f_plus - f_minus) / denom

        return grad

    def armijo_backtracking(z: np.ndarray, p: np.ndarray, fz: float, g: np.ndarray):
        alpha = 1.0
        g_tp = float(np.dot(g, p))

        # Ensure descent direction.
        if g_tp >= 0.0:
            p = -g
            g_tp = -float(np.dot(g, g))

        while alpha >= min_step:
            z_new = z + alpha * p
            if has_bounds:
                z_new = _project_box(z_new, lb, ub)
            f_new = evaluate(z_new)
            if f_new <= fz + c1 * alpha * g_tp:
                return alpha, z_new, f_new
            alpha *= backtrack_beta

        z_new = z + min_step * p
        if has_bounds:
            z_new = _project_box(z_new, lb, ub)
        return min_step, z_new, evaluate(z_new)

    f_x = evaluate(x)
    g = numerical_gradient(x)
    inv_hessian = np.eye(n_vars, dtype=float)

    history: List[float] = [float(best_cost)]

    for k in range(1, max_iter + 1):
        grad_norm = float(np.linalg.norm(g))
        if grad_norm < tol_grad:
            break

        if not np.all(np.isfinite(inv_hessian)) or np.linalg.norm(inv_hessian, ord=np.inf) > 1e10:
            inv_hessian = np.eye(n_vars, dtype=float)
        direction = -inv_hessian @ g
        alpha, x_new, f_new = armijo_backtracking(x, direction, f_x, g)
        s = x_new - x

        if float(np.linalg.norm(s)) < tol_step:
            x, f_x = x_new, f_new
            history.append(float(best_cost))
            break

        g_new = numerical_gradient(x_new)
        y = g_new - g
        y_s = float(np.dot(y, s))
        curvature_scale = float(np.linalg.norm(y) * np.linalg.norm(s))
        strong_curvature = y_s > max(1e-8, 1e-6 * curvature_scale)

        if strong_curvature and np.isfinite(y_s):
            rho = 1.0 / y_s
            if not np.isfinite(rho) or rho > 1e10:
                if reset_hessian_on_bad_curvature:
                    inv_hessian = np.eye(n_vars, dtype=float)
            else:
                eye = np.eye(n_vars, dtype=float)
                v = eye - rho * np.outer(s, y)
                try:
                    with np.errstate(over="raise", divide="raise", invalid="raise"):
                        h_candidate = v @ inv_hessian @ v.T + rho * np.outer(s, s)
                    if np.all(np.isfinite(h_candidate)):
                        inv_hessian = h_candidate
                    elif reset_hessian_on_bad_curvature:
                        inv_hessian = np.eye(n_vars, dtype=float)
                except FloatingPointError:
                    if reset_hessian_on_bad_curvature:
                        inv_hessian = np.eye(n_vars, dtype=float)
        elif reset_hessian_on_bad_curvature:
            inv_hessian = np.eye(n_vars, dtype=float)

        if not np.all(np.isfinite(inv_hessian)) and reset_hessian_on_bad_curvature:
            inv_hessian = np.eye(n_vars, dtype=float)

        x, f_x, g = x_new, f_new, g_new
        history.append(float(best_cost))

        if verbose and (k == 1 or k % 25 == 0):
            print(
                f"[BFGS] Iter {k:4d}/{max_iter} | Best cost: {best_cost:.6f} | "
                f"||g||: {grad_norm:.3e} | alpha: {alpha:.3e}"
            )

    return best_x.copy(), float(best_cost), history, int(n_fevals)


def _project_box(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lb), ub)


def _parse_bounds(
    bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]],
    n_vars: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if bounds is None:
        return None, None
    if len(bounds) != n_vars:
        raise ValueError(f"Expected {n_vars} bounds, got {len(bounds)}")

    lb = np.array(
        [-np.inf if b[0] is None else float(b[0]) for b in bounds],
        dtype=float
    )
    ub = np.array(
        [np.inf if b[1] is None else float(b[1]) for b in bounds],
        dtype=float
    )
    if np.any(lb > ub):
        raise ValueError("Invalid bounds: lower bound greater than upper bound.")
    return lb, ub


def _infer_bounds_from_objective(
    objective_func: Callable[[np.ndarray], float]
) -> Optional[Sequence[Tuple[float, float]]]:
    """
    Try to infer bounds from solver wrapper:
    objective_func -> BaseOptimizer._objective_func -> {problem, case_num}.
    """
    owner = getattr(objective_func, "__self__", None)
    if owner is None:
        return None

    problem = getattr(owner, "problem", None)
    case_num = getattr(owner, "case_num", None)
    if problem is None or case_num is None or not hasattr(problem, "get_bounds"):
        return None

    try:
        return problem.get_bounds(case_num)
    except Exception:
        return None
