"""
Subject:        Main Script
@author:        Pierre Bédrune + Claude AI 
@date:          15/01/2026
@Description:   Custom BFGS implementation for the Two Bar Truss problem.
                Adapted from the Gear Train BFGS approach.
"""

# ********** IMPORTATION **********

# mathematics
import numpy as np

# visualisation
import matplotlib.pyplot as plt

# store data
import os

# custom BFGS algorithm
from custom_TBT_BFGS import bfgs_minimize, BFGSOptions

# Two Bar Truss problem
from two_bar_plane_truss_problem import Two_Bar_Plane_Truss_Problem

# ********** NOISE WRAPPER **********

def make_noisy_objective(f_clean, sigma, seed=0):
    """
    Build a noisy objective:
        f_noisy(x) = f_clean(x) + sigma * N(0,1)
    """
    rng = np.random.default_rng(seed)

    if sigma <= 0.0:
        return f_clean

    def f_noisy(x):
        return float(f_clean(x)) + float(sigma) * float(rng.normal())

    return f_noisy


# ********** TWO BAR TRUSS PROBLEM **********

# create an instance of the truss problem
truss_problem = Two_Bar_Plane_Truss_Problem()

# define objective function and keep only the sum value
def objective_clean(x):
    sum_value, objectives, constraints = truss_problem.evaluate(x)
    return sum_value

# problem bounds
lb = np.array([0.1, 0.5])   # lower bounds: [x1_min, x2_min]
ub = np.array([2.25, 2.5])  # upper bounds: [x1_max, x2_max]

# ********** OPTIMISATION **********

# initial guess
x0 = np.array([1.0, 1.5])

# BFGS options
opts = BFGSOptions(
    max_iter=1000,           # Maximum iterations
    tol_grad=1e-5,          # Gradient norm tolerance
    tol_step=1e-40,         # Step size tolerance
    fd_eps=1e-6,            # Finite difference step
    c1=1e-4,                # Armijo parameter
    backtrack_beta=0.5,     # Backtracking factor
    min_step=1e-12,         # Minimum step size
    verbose=False           # Set to True to see iteration details
)

# ********** VISUALISATION **********

# Create plots directory
plot_dir = "TBT_PLOT"
os.makedirs(plot_dir, exist_ok=True)

# ********** RUN WITH 3 NOISE LEVELS **********

sigmas = [0.0, 1e-4, 1e-2, 1]   # <- change ici si tu veux d’autres niveaux
base_seed = 42              # seed de base (reproductible)

for sigma in sigmas:

    # objective = noisy(objective_clean)
    objective = make_noisy_objective(objective_clean, sigma=sigma, seed=base_seed)

    # BFGS optimisation
    result = bfgs_minimize(
        f=objective,           # function to minimise
        x0=x0,                 # initial guess
        lb=lb,                 # lower bound
        ub=ub,                 # upper bound
        opts=opts              # options
    )

    # evaluate final solution (sur la fonction "clean" physique, sans bruit)
    final_sum, final_obj, final_con = truss_problem.evaluate(result["x_best"])

    # calculate physical values
    position = result["x_best"][0] * truss_problem.h
    area = result["x_best"][1] * truss_problem.A_min

    # check constraint satisfaction
    constraints_satisfied = (final_con[0] <= 0) and (final_con[1] <= 0)
    in_bounds = ((lb[0] <= result["x_best"][0] <= ub[0]) and (lb[1] <= result["x_best"][1] <= ub[1]))

    # ********** DISPLAY RESULTS **********

    print("\n" + "="*70)
    print(f"RUN WITH NOISE: sigma = {sigma:.2e}")
    print("="*70)

    print(f"\nConvergence Status:")
    print(f"  S: {result['converged']}")
    print(f"  Iterations: {result['iters']}")
    print(f"  Function evaluations: {result['iters'] * 2 * len(x0) + 1}")
    print(f"    (1 initial + {result['iters']} iterations × {2*len(x0)} for finite differences)")

    print(f"\nOptimal Solution (normalized variables):")
    print(f"  x1 (position factor) = {result['x_best'][0]:.10f}")
    print(f"  x2 (area factor) = {result['x_best'][1]:.10f}")
    print(f"  Within bounds: {'✓ YES' if in_bounds else '✗ NO'}")

    print(f"\nPhysical Solution:")
    print(f"  Position = {position:.8f} m")
    print(f"  Area = {area:.8e} m²")

    print(f"\nObjective Function:")
    print(f"  Combined objective (noisy during optimisation) = {result['f_best']:.12e}")
    print(f"  Combined objective (clean re-eval) = {final_sum:.12e}")
    print(f"  f1 (weight) = {final_obj[0]:.8f} kg")
    print(f"  f2 (displacement) = {final_obj[1]:.8e} m")

    print(f"\nConstraints:")
    print(f"  Normalized (for optimization):")
    print(f"    g1 = {final_con[0]:+.8e}  {'✓' if final_con[0] <= 0 else '✗'}")
    print(f"    g2 = {final_con[1]:+.8e}  {'✓' if final_con[1] <= 0 else '✗'}")

    # Get actual constraint values using normalized=False
    g1_actual = truss_problem.g1(normalised=False)
    g2_actual = truss_problem.g2(normalised=False)

    print(f"  Actual (Pa):")
    print(f"    g1 (σ_tension - σ₀) = {g1_actual:+.6e} Pa  {'✓' if g1_actual <= 0 else '✗'}")
    print(f"    g2 (σ_compression - σ₀) = {g2_actual:+.6e} Pa  {'✓' if g2_actual <= 0 else '✗'}")

    print(f"\nConvergence Metrics:")
    print(f"  Final gradient norm: {result['history_grad_norm'][-1]:.8e}")
    if len(result['history_alpha']) > 0:
        print(f"  Final step size α: {result['history_alpha'][-1]:.8e}")

    # ********** VISUALISATION **********

    print("\nGenerating convergence plots...")

    tag = f"sigma_{sigma:.0e}".replace("+", "").replace("-", "m")  # filename-safe

    # Plot 1: Objective function convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(result["history_f"], 'b-o', linewidth=2, markersize=5, label='Objective f(x)')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Objective Function Value', fontsize=12)
    plt.title(f'BFGS Convergence: Objective Function (sigma={sigma:.0e})', fontsize=14, fontweight='bold')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'bfgs_convergence_objective_{tag}.png'), dpi=300)
    print(f"  Saved: {plot_dir}/bfgs_convergence_objective_{tag}.png")

    # Plot 2: Gradient norm convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(result["history_grad_norm"], 'r-o', linewidth=2, markersize=5, label='Gradient Norm')
    plt.axhline(y=opts.tol_grad, color='g', linestyle='--', linewidth=2,
                label=f'Tolerance = {opts.tol_grad:.0e}')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel(r'Gradient Norm $\|\nabla f(x)\|$', fontsize=12)
    plt.title(f'BFGS Convergence: Gradient Norm (sigma={sigma:.0e})', fontsize=14, fontweight='bold')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'bfgs_convergence_gradient_{tag}.png'), dpi=300)
    print(f"  Saved: {plot_dir}/bfgs_convergence_gradient_{tag}.png")

    # Plot 3: Step sizes (line search)
    if len(result["history_alpha"]) > 0:
        plt.figure(figsize=(10, 6))
        plt.semilogy(range(1, len(result["history_alpha"]) + 1),
                     result["history_alpha"], 'g-o', linewidth=2, markersize=5,
                     label='Step Size α')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel(r'Step Size $\alpha$ (from line search)', fontsize=12)
        plt.title(f'BFGS: Line Search Step Sizes (sigma={sigma:.0e})', fontsize=14, fontweight='bold')
        plt.grid(True, which='both', alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'bfgs_step_sizes_{tag}.png'), dpi=300)
        print(f"  Saved: {plot_dir}/bfgs_step_sizes_{tag}.png")

    plt.close('all')
