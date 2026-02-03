from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np


@dataclass
class BFGSOptions:
    max_iter: int = 300
    tol_grad: float = 1e-8
    tol_step: float = 1e-10
    fd_eps: float = 1e-6  # finite difference step
    c1: float = 1e-4      # Armijo parameter
    backtrack_beta: float = 0.5
    min_step: float = 1e-12
    verbose: bool = False

    # Noise injection (Task 4 robustness experiment)
    noise_sigma: float = 0.0      # std-dev of additive Gaussian noise on objective evaluations
    noise_seed: int = 0           # RNG seed to make noisy runs reproducible
    noise_avg: int = 1            # number of noisy samples averaged per evaluation


def gear_ratio(x: np.ndarray) -> float:
    """Gear train ratio: ratio = (N1*N3)/(N2*N4)."""
    x = np.asarray(x, dtype=float)
    return (x[0] * x[2]) / (x[1] * x[3])


def continuous_objective(x: np.ndarray, target_ratio: float) -> float:
    """Smooth objective for BFGS: f(x) = (ratio(x) - target_ratio)^2."""
    r = gear_ratio(x)
    e = r - target_ratio
    return float(e * e)


def make_noisy_objective(
    f_clean: Callable[[np.ndarray], float],
    sigma: float,
    seed: int,
    n_avg: int = 1,
) -> Callable[[np.ndarray], float]:
    """Return a callable that adds controlled Gaussian noise to objective evaluations.

    f_noisy(x) = mean_{j=1..n_avg}( f_clean(x) + sigma * N(0,1) )

    Notes:
    - For sigma=0, the function reduces to f_clean.
    - A fixed seed makes the whole optimisation run reproducible.
    - Averaging (n_avg>1) reduces noise variance but increases evaluation cost.
    """
    if sigma <= 0.0:
        return f_clean

    rng = np.random.default_rng(seed)
    n_avg = max(1, int(n_avg))

    def f(x: np.ndarray) -> float:
        acc = 0.0
        for _ in range(n_avg):
            acc += float(f_clean(x) + sigma * rng.normal())
        return acc / n_avg

    return f


def project_box(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Projection onto box constraints [lb, ub]."""
    return np.minimum(np.maximum(x, lb), ub)


def round_and_clip(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Round to nearest integer then clip into bounds."""
    xr = np.rint(x).astype(int)
    xr = np.minimum(np.maximum(xr, lb.astype(int)), ub.astype(int))
    return xr


def integer_error(x_int: np.ndarray, target_ratio: float) -> float:
    """Absolute error in the true discrete problem."""
    r = gear_ratio(x_int.astype(float))
    return float(abs(r - target_ratio))


def discrete_repair_search(
    x_int0: np.ndarray,
    target_ratio: float,
    lb: np.ndarray,
    ub: np.ndarray,
    radius: int = 2,
) -> Tuple[np.ndarray, float]:
    """Local discrete search around an initial integer solution.

    Evaluates all integer combinations in the hypercube [x_int0-radius, x_int0+radius]
    (clipped to [lb, ub]) and returns the best integer design and its absolute error.
    """
    x0 = np.asarray(x_int0, dtype=int)
    lb_i = lb.astype(int)
    ub_i = ub.astype(int)

    r = int(max(0, radius))
    best_x = x0.copy()
    best_err = integer_error(best_x, target_ratio)

    ranges = []
    for i in range(len(x0)):
        lo = max(lb_i[i], x0[i] - r)
        hi = min(ub_i[i], x0[i] + r)
        ranges.append(range(lo, hi + 1))

    for n1 in ranges[0]:
        for n2 in ranges[1]:
            for n3 in ranges[2]:
                for n4 in ranges[3]:
                    cand = np.array([n1, n2, n3, n4], dtype=int)
                    err = integer_error(cand, target_ratio)
                    if err < best_err:
                        best_err = err
                        best_x = cand

    return best_x, float(best_err)


def finite_difference_grad(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float) -> np.ndarray:
    """Central finite difference gradient."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2.0 * eps)
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
    """Armijo backtracking line search on projected steps."""
    alpha = 1.0
    gTp = float(np.dot(g, p))

    # If direction is not a descent direction, fall back to steepest descent
    if gTp >= 0.0:
        p = -g
        gTp = float(np.dot(g, p))

    while alpha >= min_step:
        x_new = project_box(x + alpha * p, lb, ub)
        f_new = f(x_new)
        if f_new <= fx + c1 * alpha * gTp:
            return alpha, x_new, f_new
        alpha *= beta

    x_new = project_box(x + min_step * p, lb, ub)
    return min_step, x_new, f(x_new)


def bfgs_minimise_box_projected(
    f_raw: Callable[[np.ndarray], float],
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    opts: BFGSOptions,
) -> Dict[str, object]:
    """BFGS on a box-constrained domain via projection.

    Note: With noise_sigma>0, objective evaluations are noisy, which intentionally
    violates deterministic assumptions and can cause loss of convergence.
    """
    x = project_box(np.asarray(x0, dtype=float), lb, ub)

    f_clean = lambda z: f_raw(project_box(z, lb, ub))
    f = make_noisy_objective(
        f_clean,
        sigma=opts.noise_sigma,
        seed=opts.noise_seed,
        n_avg=opts.noise_avg,
    )

    n = len(x)
    H = np.eye(n)

    fx = f(x)
    g = finite_difference_grad(f, x, opts.fd_eps)

    history_f: List[float] = [fx]
    history_grad_norm: List[float] = [float(np.linalg.norm(g))]
    history_alpha: List[float] = []

    if opts.verbose:
        print(f"[BFGS] iter=0 f={fx:.6e} ||g||={history_grad_norm[-1]:.6e} x={x}")

    for k in range(1, opts.max_iter + 1):
        grad_norm = float(np.linalg.norm(g))
        if grad_norm < opts.tol_grad:
            break

        p = -H @ g

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

        s = x_new - x
        if float(np.linalg.norm(s)) < opts.tol_step:
            x, fx = x_new, f_new
            history_f.append(fx)
            history_grad_norm.append(float(np.linalg.norm(g)))
            history_alpha.append(alpha)
            break

        g_new = finite_difference_grad(f, x_new, opts.fd_eps)
        y = g_new - g

        ys = float(np.dot(y, s))
        if ys > 1e-12:
            rho = 1.0 / ys
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)
        else:
            H = np.eye(n)

        x, fx, g = x_new, f_new, g_new

        history_f.append(fx)
        history_grad_norm.append(float(np.linalg.norm(g)))
        history_alpha.append(alpha)

        if opts.verbose and (k % 10 == 0 or k == 1):
            print(
                f"[BFGS] iter={k} f={fx:.6e} ||g||={history_grad_norm[-1]:.6e} "
                f"alpha={alpha:.2e} x={x}"
            )

    # Conservative convergence flag under noise:
    converged = (history_grad_norm[-1] < opts.tol_grad) or (
        len(history_f) >= 2 and float(abs(history_f[-1] - history_f[-2])) < 1e-30
    )

    return {
        "x_best": x,
        "f_best": fx,
        "history_f": np.array(history_f, dtype=float),
        "history_grad_norm": np.array(history_grad_norm, dtype=float),
        "history_alpha": np.array(history_alpha, dtype=float),
        "iters": len(history_f) - 1,
        "converged": bool(converged),
    }


def multi_start_bfgs(
    target_ratio: float,
    lb: np.ndarray,
    ub: np.ndarray,
    n_starts: int = 500,
    seed: int = 42,
    opts: BFGSOptions | None = None,
    repair_radius: int = 2,
) -> Dict[str, object]:
    """Multi-start BFGS.

    IMPORTANT:
    We SELECT the best candidate primarily by integer error (after rounding+repair),
    and use continuous objective as a tie-breaker.
    """
    if opts is None:
        opts = BFGSOptions()

    rng = np.random.default_rng(seed)
    best = None

    def f_raw(x: np.ndarray) -> float:
        return continuous_objective(x, target_ratio=target_ratio)

    for i in range(n_starts):
        x0 = rng.uniform(lb, ub)
        out = bfgs_minimise_box_projected(
            f_raw=f_raw,
            x0=x0,
            lb=lb,
            ub=ub,
            opts=opts,
        )

        x_cont = out["x_best"]
        f_cont = float(out["f_best"])

        x_int0 = round_and_clip(x_cont, lb, ub)
        x_int, err_int = discrete_repair_search(
            x_int0=x_int0,
            target_ratio=target_ratio,
            lb=lb,
            ub=ub,
            radius=repair_radius,
        )

        record = {
            **out,
            "start_index": i,
            "x0": x0,
            "x_int0": x_int0,
            "x_int": x_int,
            "err_int": float(err_int),
            "ratio_cont": gear_ratio(x_cont),
            "ratio_int": gear_ratio(x_int.astype(float)),
        }

        if best is None:
            best = record
            continue

        # Primary criterion: best integer error
        if record["err_int"] < float(best["err_int"]):
            best = record
            continue

        # Tie-breaker: best continuous objective
        if record["err_int"] == float(best["err_int"]) and f_cont < float(best["f_best"]):
            best = record

    assert best is not None
    return best