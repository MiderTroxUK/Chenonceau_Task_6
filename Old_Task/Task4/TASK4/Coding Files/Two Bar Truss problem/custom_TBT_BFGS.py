"""
Subject:        Custom BFGS Implementation for Two Bar Truss
@author:        Clémence-Philomène HINOT + Claude AI
@date:          17/01/2026
@Description:   This module implements BFGS optimization from scratch.
"""

# ********** IMPORTATION **********

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np

# ********** CUSTOM BFGS **********

@dataclass
class BFGSOptions:
    """Configuration options for BFGS optimizer"""
    max_iter: int = 300              # Maximum iterations
    tol_grad: float = 1e-8           # Gradient norm tolerance for convergence
    tol_step: float = 1e-12          # Step size tolerance
    fd_eps: float = 1e-6             # Finite difference step size
    c1: float = 1e-4                 # Armijo condition parameter
    backtrack_beta: float = 0.5      # Backtracking reduction factor
    min_step: float = 1e-12          # Minimum step size
    verbose: bool = False            # Print iteration details


def project_box(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """
    Project point x onto box constraints [lb, ub].
    
    For each component: x_i = max(lb_i, min(x_i, ub_i))
    """
    return np.minimum(np.maximum(x, lb), ub)


def finite_difference_grad(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float
) -> np.ndarray:
    """
    Compute gradient using central finite differences.
    
    For each dimension i:
        ∂f/∂x_i ≈ [f(x + ε·e_i) - f(x - ε·e_i)] / (2ε)
    
    Cost: 2n function evaluations (where n = len(x))
    
    Args:
        f: Objective function
        x: Current point
        eps: Finite difference step size
    
    Returns:
        Gradient vector ∇f(x)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    g = np.zeros(n, dtype=float)
    
    for i in range(n):
        # Perturb in positive direction
        x_plus = x.copy()
        x_plus[i] += eps
        f_plus = f(x_plus)
        
        # Perturb in negative direction
        x_minus = x.copy()
        x_minus[i] -= eps
        f_minus = f(x_minus)
        
        # Central difference
        g[i] = (f_plus - f_minus) / (2.0 * eps)
    
    return g


def armijo_backtracking(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    p: np.ndarray,
    fx: float,
    g: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    c1: float,
    beta: float,
    min_step: float,
) -> Tuple[float, np.ndarray, float]:
    """
    Armijo backtracking line search with box constraint projection.
    
    Finds step size α such that the Armijo condition is satisfied:
        f(x + α·p) ≤ f(x) + c1·α·∇f(x)ᵀp  (sufficient decrease)
    
    Algorithm:
        1. Start with α = 1
        2. While Armijo condition not satisfied:
           - Reduce α ← β·α
           - Project x + α·p onto [lb, ub]
        3. Return α, new point, new function value
    
    Args:
        f: Objective function
        x: Current point
        p: Search direction
        fx: f(x), function value at current point
        g: ∇f(x), gradient at current point
        lb, ub: Box constraints
        c1: Armijo parameter (typically 1e-4)
        beta: Backtracking factor (typically 0.5)
        min_step: Minimum allowed step size
    
    Returns:
        (alpha, x_new, f_new): Step size, new point, new function value
    """
    alpha = 1.0
    gTp = float(np.dot(g, p))
    
    # Safety check: if p is not a descent direction, use steepest descent
    if gTp >= 0.0:
        p = -g
        gTp = float(np.dot(g, p))
    
    # Backtracking loop
    while alpha >= min_step:
        # Proposed step with projection
        x_new = project_box(x + alpha * p, lb, ub)
        f_new = f(x_new)
        
        # Check Armijo condition
        if f_new <= fx + c1 * alpha * gTp:
            return alpha, x_new, f_new
        
        # Reduce step size
        alpha *= beta
    
    # If no acceptable step found, take minimum step
    x_new = project_box(x + min_step * p, lb, ub)
    return min_step, x_new, f(x_new)


def bfgs_update(
    H: np.ndarray,
    s: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    BFGS update formula for inverse Hessian approximation.
    
    Standard BFGS formula:
        ρ = 1 / (yᵀs)
        V = I - ρ·s·yᵀ
        H_{k+1} = V·H_k·Vᵀ + ρ·s·sᵀ
    
    Where:
        s = x_{k+1} - x_k  (step)
        y = ∇f_{k+1} - ∇f_k  (gradient difference)
        H ≈ [∇²f]⁻¹  (inverse Hessian approximation)
    
    Curvature condition: yᵀs > 0 must hold for positive definiteness.
    If violated, reset H to identity matrix.
    
    Args:
        H: Current inverse Hessian approximation
        s: Step vector
        y: Gradient difference
    
    Returns:
        Updated inverse Hessian approximation
    """
    n = len(s)
    ys = float(np.dot(y, s))
    
    # Check curvature condition
    if ys > 1e-12:
        # BFGS update
        rho = 1.0 / ys
        I = np.eye(n)
        V = I - rho * np.outer(s, y)
        H_new = V @ H @ V.T + rho * np.outer(s, s)
        return H_new
    else:
        # Curvature condition violated - reset to identity
        return np.eye(n)


def bfgs_minimize(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    opts: BFGSOptions | None = None,
) -> Dict[str, object]:
    """
    BFGS minimization with box constraints (via projection).
    
    Algorithm:
        1. Initialize: H = I (identity matrix), x = project(x0)
        2. For k = 0, 1, 2, ...
           a. Compute gradient: g = ∇f(x)
           b. Check convergence: ||g|| < tol_grad
           c. Search direction: p = -H·g
           d. Line search: find α satisfying Armijo condition
           e. Update: x ← x + α·p, project onto bounds
           f. BFGS update: H ← BFGS(H, s, y)
        3. Return solution
    
    Args:
        f: Objective function to minimize
        x0: Initial guess
        lb: Lower bounds
        ub: Upper bounds
        opts: BFGS options (use defaults if None)
    
    Returns:
        Dictionary containing:
            - x_best: Optimal solution found
            - f_best: Objective value at solution
            - history_f: Function values at each iteration
            - history_grad_norm: Gradient norms at each iteration
            - history_alpha: Step sizes at each iteration
            - iters: Number of iterations performed
            - converged: Boolean convergence flag
    """
    if opts is None:
        opts = BFGSOptions()
    
    # Initialize
    x = project_box(np.asarray(x0, dtype=float), lb, ub)
    n = len(x)
    H = np.eye(n)  # Initial inverse Hessian approximation
    
    # Evaluate initial point
    fx = f(x)
    g = finite_difference_grad(f, x, opts.fd_eps)
    
    # History tracking
    history_f: List[float] = [fx]
    history_grad_norm: List[float] = [float(np.linalg.norm(g))]
    history_alpha: List[float] = []
    
    if opts.verbose:
        print(f"[BFGS] Iteration 0:")
        print(f"  f(x) = {fx:.10e}")
        print(f"  ||∇f|| = {history_grad_norm[-1]:.6e}")
        print(f"  x = {x}")
    
    # Main BFGS loop
    for k in range(1, opts.max_iter + 1):
        grad_norm = float(np.linalg.norm(g))
        
        # Convergence check: gradient norm
        if grad_norm < opts.tol_grad:
            if opts.verbose:
                print(f"\n[BFGS] Converged: ||∇f|| = {grad_norm:.6e} < {opts.tol_grad:.6e}")
            break
        
        # Search direction: p = -H·g (quasi-Newton direction)
        p = -H @ g
        
        # Line search with Armijo backtracking
        alpha, x_new, f_new = armijo_backtracking(
            f=f,
            x=x,
            p=p,
            fx=fx,
            g=g,
            lb=lb,
            ub=ub,
            c1=opts.c1,
            beta=opts.backtrack_beta,
            min_step=opts.min_step,
        )
        
        # Compute step
        s = x_new - x
        
        # Convergence check: step size
        if float(np.linalg.norm(s)) < opts.tol_step:
            x, fx = x_new, f_new
            history_f.append(fx)
            history_grad_norm.append(float(np.linalg.norm(g)))
            history_alpha.append(alpha)
            if opts.verbose:
                print(f"\n[BFGS] Converged: ||step|| = {np.linalg.norm(s):.6e} < {opts.tol_step:.6e}")
            break
        
        # Gradient at new point
        g_new = finite_difference_grad(f, x_new, opts.fd_eps)
        
        # Gradient difference
        y = g_new - g
        
        # BFGS update of inverse Hessian approximation
        H = bfgs_update(H, s, y)
        
        # Update state
        x, fx, g = x_new, f_new, g_new
        
        # Record history
        history_f.append(fx)
        history_grad_norm.append(float(np.linalg.norm(g)))
        history_alpha.append(alpha)
        
        # Verbose output
        if opts.verbose and (k % 10 == 0 or k == 1):
            print(f"\n[BFGS] Iteration {k}:")
            print(f"  f(x) = {fx:.10e}")
            print(f"  ||∇f|| = {history_grad_norm[-1]:.6e}")
            print(f"  α = {alpha:.6e}")
            print(f"  x = {x}")
    
    # Check gradient convergence
    grad_converged = history_grad_norm[-1] < opts.tol_grad
    
    # Check step convergence (practical convergence)
    step_converged = False
    if len(history_alpha) > 0 and history_alpha[-1] < opts.min_step * 10:
        step_converged = True
    
    # Combined: converged if EITHER criterion is met
    converged = grad_converged or step_converged
    
    # Determine termination reason
    if grad_converged:
        termination = "Gradient norm below tolerance"
    elif step_converged:
        termination = "Step size below threshold (practical convergence)"
    else:
        termination = "Maximum iterations reached"
    
    return {
        "x_best": x,
        "f_best": fx,
        "history_f": np.array(history_f, dtype=float),
        "history_grad_norm": np.array(history_grad_norm, dtype=float),
        "history_alpha": np.array(history_alpha, dtype=float),
        "iters": len(history_f) - 1,
        "converged": bool(converged),
        "grad_converged": bool(grad_converged),
        "step_converged": bool(step_converged),
        "termination": termination,
    }