"""
Subject:        BFGS Limits Testing with Noise
@author:        Pierre Bédrune + Claude AI
@date:          17/01/2026
@Description:   Systematic testing of BFGS algorithm limits with various noise levels
                using CURRENT parameters to identify breaking points
"""

# ********** IMPORTATION **********

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, '/mnt/project')

from custom_TBT_BFGS import bfgs_minimize, BFGSOptions
from two_bar_plane_truss_problem import Two_Bar_Plane_Truss_Problem

# ********** NOISE WRAPPER **********

def make_noisy_objective(f_clean, sigma, seed=0):
    """
    Create noisy version of objective function
    f_noisy(x) = f_clean(x) + σ * N(0,1)
    """
    rng = np.random.default_rng(seed)
    
    if sigma <= 0.0:
        return f_clean
    
    def f_noisy(x):
        return float(f_clean(x)) + float(sigma) * float(rng.normal())
    
    return f_noisy


# ********** EVALUATION FUNCTION **********

def evaluate_bfgs_performance(
    objective_clean,
    objective_noisy,
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    opts: BFGSOptions,
    sigma: float,
    problem_instance,
    run_id: int
) -> Dict:
    """
    Evaluate BFGS performance for a single run
    
    Returns:
        Dictionary with performance metrics
    """
    
    # Run optimization
    start_time = time.time()
    result = bfgs_minimize(
        f=objective_noisy,
        x0=x0,
        lb=lb,
        ub=ub,
        opts=opts
    )
    elapsed_time = time.time() - start_time
    
    # Evaluate on CLEAN objective (ground truth)
    final_sum_clean, final_obj, final_con = problem_instance.evaluate(result["x_best"])
    
    # Evaluate on NOISY objective (what optimizer saw)
    final_sum_noisy = objective_noisy(result["x_best"])
    
    # Constraint satisfaction
    constraints_ok = (final_con[0] <= 0) and (final_con[1] <= 0)
    bounds_ok = np.all((lb <= result["x_best"]) & (result["x_best"] <= ub))
    
    # Feasibility
    feasible = constraints_ok and bounds_ok
    
    # Convergence quality
    if len(result["history_f"]) > 1:
        # Relative improvement between first and last iteration
        improvement = abs(result["history_f"][0] - result["history_f"][-1]) / (abs(result["history_f"][0]) + 1e-10)
    else:
        improvement = 0.0
    
    # Oscillation measure (variance in last 20% of iterations)
    n_tail = max(1, int(0.2 * len(result["history_f"])))
    tail_values = result["history_f"][-n_tail:]
    oscillation = float(np.std(tail_values)) if len(tail_values) > 1 else 0.0
    
    return {
        'run_id': run_id,
        'sigma': sigma,
        'converged': result['converged'],
        'grad_converged': result['grad_converged'],
        'step_converged': result['step_converged'],
        'termination': result['termination'],
        'iterations': result['iters'],
        'time_sec': elapsed_time,
        'f_final_noisy': result['f_best'],
        'f_final_clean': final_sum_clean,
        'f_initial': result["history_f"][0],
        'improvement_pct': improvement * 100,
        'oscillation': oscillation,
        'grad_norm_final': result['history_grad_norm'][-1],
        'step_size_final': result['history_alpha'][-1] if len(result['history_alpha']) > 0 else np.nan,
        'x1_final': result['x_best'][0],
        'x2_final': result['x_best'][1],
        'f1_weight': final_obj[0],
        'f2_displacement': final_obj[1],
        'g1_constraint': final_con[0],
        'g2_constraint': final_con[1],
        'constraints_satisfied': constraints_ok,
        'bounds_satisfied': bounds_ok,
        'feasible': feasible,
        'history_f': result["history_f"],
        'history_grad': result["history_grad_norm"],
        'history_alpha': result["history_alpha"]
    }


# ********** MAIN TEST FUNCTION **********

def test_bfgs_limits(
    noise_levels: List[float],
    n_runs_per_noise: int = 10,
    output_dir: str = "BFGS_LIMITS_TEST"
):
    """
    Systematic testing of BFGS with different noise levels
    
    Args:
        noise_levels: List of sigma values to test
        n_runs_per_noise: Number of independent runs per noise level
        output_dir: Directory to save results
    """
    
    print("="*80)
    print("BFGS LIMITS TESTING - SYSTEMATIC NOISE ANALYSIS")
    print("="*80)
    print(f"\nTest Configuration:")
    print(f"  Noise levels: {noise_levels}")
    print(f"  Runs per noise level: {n_runs_per_noise}")
    print(f"  Total runs: {len(noise_levels) * n_runs_per_noise}")
    print(f"  Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Problem setup
    truss = Two_Bar_Plane_Truss_Problem()
    
    def objective_clean(x):
        sum_val, _, _ = truss.evaluate(x)
        return sum_val
    
    x0 = np.array([1.0, 1.5])
    lb = np.array([0.1, 0.5])
    ub = np.array([2.25, 2.5])
    
    # CURRENT BFGS parameters (from your main_TBT_BFGS_noise.py)
    opts = BFGSOptions(
        max_iter=1000,
        tol_grad=1e-5,
        tol_step=1e-12,
        fd_eps=1e-6,
        c1=1e-4,
        backtrack_beta=0.5,
        min_step=1e-12,
        verbose=False
    )
    
    print(f"\nBFGS Parameters (CURRENT/FIXED):")
    print(f"  max_iter = {opts.max_iter}")
    print(f"  tol_grad = {opts.tol_grad:.2e}")
    print(f"  tol_step = {opts.tol_step:.2e}")
    print(f"  fd_eps = {opts.fd_eps:.2e}")
    print(f"  c1 = {opts.c1:.2e}")
    print(f"  backtrack_beta = {opts.backtrack_beta}")
    print(f"  min_step = {opts.min_step:.2e}")
    
    # Store all results
    all_results = []
    
    # Run tests
    total_runs = len(noise_levels) * n_runs_per_noise
    current_run = 0
    
    print(f"\n{'='*80}")
    print("STARTING TESTS")
    print(f"{'='*80}\n")
    
    for sigma in noise_levels:
        print(f"\n{'─'*80}")
        print(f"NOISE LEVEL: σ = {sigma:.2e}")
        print(f"{'─'*80}")
        
        for run in range(n_runs_per_noise):
            current_run += 1
            
            # Create noisy objective with different seed for each run
            seed = 42 + run  # Reproducible but different per run
            objective_noisy = make_noisy_objective(objective_clean, sigma, seed=seed)
            
            # Run optimization
            print(f"  Run {run+1}/{n_runs_per_noise} (Global: {current_run}/{total_runs})...", end=" ")
            
            result_dict = evaluate_bfgs_performance(
                objective_clean=objective_clean,
                objective_noisy=objective_noisy,
                x0=x0,
                lb=lb,
                ub=ub,
                opts=opts,
                sigma=sigma,
                problem_instance=truss,
                run_id=run + 1
            )
            
            all_results.append(result_dict)
            
            # Quick summary
            status = "✓" if result_dict['converged'] else "✗"
            feasible = "✓" if result_dict['feasible'] else "✗"
            print(f"{status} Conv | {feasible} Feas | {result_dict['iterations']:3d} iter | "
                  f"f={result_dict['f_final_clean']:.4e}")
    
    # Convert to DataFrame (excluding history arrays)
    results_summary = []
    for r in all_results:
        r_copy = r.copy()
        # Remove history arrays for CSV
        r_copy.pop('history_f', None)
        r_copy.pop('history_grad', None)
        r_copy.pop('history_alpha', None)
        results_summary.append(r_copy)
    
    df = pd.DataFrame(results_summary)
    
    # Save raw results
    csv_path = os.path.join(output_dir, "bfgs_limits_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    return df, all_results


# ********** ANALYSIS FUNCTIONS **********

def analyze_results(df: pd.DataFrame, output_dir: str):
    """
    Analyze and visualize results
    """
    
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}\n")
    
    # Group by noise level
    grouped = df.groupby('sigma')
    
    # Summary statistics
    summary = grouped.agg({
        'converged': ['sum', 'mean'],
        'feasible': ['sum', 'mean'],
        'iterations': ['mean', 'std', 'min', 'max'],
        'time_sec': ['mean', 'std'],
        'f_final_clean': ['mean', 'std', 'min', 'max'],
        'improvement_pct': ['mean', 'std'],
        'oscillation': ['mean', 'std'],
        'grad_norm_final': ['mean', 'std'],
    }).round(6)
    
    print("Summary Statistics by Noise Level:")
    print(summary)
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary_statistics.csv")
    summary.to_csv(summary_path)
    print(f"\n✓ Summary saved to: {summary_path}")
    
    # Identify breaking point
    print(f"\n{'─'*80}")
    print("BREAKING POINT ANALYSIS")
    print(f"{'─'*80}\n")
    
    for sigma in df['sigma'].unique():
        subset = df[df['sigma'] == sigma]
        conv_rate = subset['converged'].mean() * 100
        feas_rate = subset['feasible'].mean() * 100
        
        print(f"σ = {sigma:.2e}:")
        print(f"  Convergence rate: {conv_rate:.1f}%")
        print(f"  Feasibility rate: {feas_rate:.1f}%")
        
        if conv_rate < 50:
            print(f"  ⚠️  WARNING: Less than 50% convergence rate!")
        if feas_rate < 80:
            print(f"  ⚠️  WARNING: Less than 80% feasible solutions!")
        print()


def plot_results(df: pd.DataFrame, all_results: List[Dict], output_dir: str):
    """
    Create comprehensive visualization
    """
    
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    noise_levels = sorted(df['sigma'].unique())
    
    # ========== PLOT 1: Success Rates ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Convergence rate
    conv_rates = [df[df['sigma'] == s]['converged'].mean() * 100 for s in noise_levels]
    axes[0].plot(noise_levels, conv_rates, 'o-', linewidth=2, markersize=8, color='blue')
    axes[0].axhline(y=50, color='red', linestyle='--', label='50% threshold')
    axes[0].set_xlabel('Noise Level (σ)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Convergence Rate (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('BFGS Convergence vs Noise', fontsize=14, fontweight='bold')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Feasibility rate
    feas_rates = [df[df['sigma'] == s]['feasible'].mean() * 100 for s in noise_levels]
    axes[1].plot(noise_levels, feas_rates, 'o-', linewidth=2, markersize=8, color='green')
    axes[1].axhline(y=80, color='orange', linestyle='--', label='80% threshold')
    axes[1].set_xlabel('Noise Level (σ)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Feasibility Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Solution Feasibility vs Noise', fontsize=14, fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Average iterations
    avg_iters = [df[df['sigma'] == s]['iterations'].mean() for s in noise_levels]
    axes[2].plot(noise_levels, avg_iters, 'o-', linewidth=2, markersize=8, color='purple')
    axes[2].set_xlabel('Noise Level (σ)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Average Iterations', fontsize=12, fontweight='bold')
    axes[2].set_title('Convergence Speed vs Noise', fontsize=14, fontweight='bold')
    axes[2].set_xscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_success_rates.png'), dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 1_success_rates.png")
    plt.close()
    
    # ========== PLOT 2: Convergence Metrics ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Final objective value
    for sigma in noise_levels:
        subset = df[df['sigma'] == sigma]
        axes[0, 0].scatter([sigma]*len(subset), subset['f_final_clean'], 
                          alpha=0.6, s=50, label=f'σ={sigma:.0e}')
    axes[0, 0].set_xlabel('Noise Level (σ)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Final Objective (Clean)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Solution Quality vs Noise', fontsize=12, fontweight='bold')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    
    # Final gradient norm
    for sigma in noise_levels:
        subset = df[df['sigma'] == sigma]
        axes[0, 1].scatter([sigma]*len(subset), subset['grad_norm_final'], 
                          alpha=0.6, s=50, label=f'σ={sigma:.0e}')
    axes[0, 1].axhline(y=1e-5, color='red', linestyle='--', label='tol_grad=1e-5')
    axes[0, 1].set_xlabel('Noise Level (σ)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Final Gradient Norm', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Gradient Convergence vs Noise', fontsize=12, fontweight='bold')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)
    
    # Oscillation
    for sigma in noise_levels:
        subset = df[df['sigma'] == sigma]
        axes[1, 0].scatter([sigma]*len(subset), subset['oscillation'], 
                          alpha=0.6, s=50, label=f'σ={sigma:.0e}')
    axes[1, 0].set_xlabel('Noise Level (σ)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Oscillation (Std of last 20%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Solution Stability vs Noise', fontsize=12, fontweight='bold')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)
    
    # Improvement percentage
    for sigma in noise_levels:
        subset = df[df['sigma'] == sigma]
        axes[1, 1].scatter([sigma]*len(subset), subset['improvement_pct'], 
                          alpha=0.6, s=50, label=f'σ={sigma:.0e}')
    axes[1, 1].set_xlabel('Noise Level (σ)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Optimization Progress vs Noise', fontsize=12, fontweight='bold')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_convergence_metrics.png'), dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 2_convergence_metrics.png")
    plt.close()
    
    # ========== PLOT 3: Convergence Curves ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, sigma in enumerate(noise_levels[:6]):  # Max 6 plots
        row = idx // 3
        col = idx % 3
        
        # Get all runs for this noise level
        runs_data = [r for r in all_results if r['sigma'] == sigma]
        
        # Plot each run
        for run_data in runs_data:
            axes[row, col].semilogy(run_data['history_f'], alpha=0.5, linewidth=1)
        
        axes[row, col].set_xlabel('Iteration', fontsize=10)
        axes[row, col].set_ylabel('Objective Value', fontsize=10)
        axes[row, col].set_title(f'σ = {sigma:.2e}', fontsize=11, fontweight='bold')
        axes[row, col].grid(True, alpha=0.3)
    
    plt.suptitle('Convergence Curves for Different Noise Levels', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_convergence_curves.png'), dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 3_convergence_curves.png")
    plt.close()
    
    # ========== PLOT 4: Boxplots ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Objective value boxplot
    data_obj = [df[df['sigma'] == s]['f_final_clean'].values for s in noise_levels]
    bp1 = axes[0, 0].boxplot(data_obj, labels=[f'{s:.0e}' for s in noise_levels])
    axes[0, 0].set_xlabel('Noise Level (σ)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Final Objective (Clean)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Objective Value Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Iterations boxplot
    data_iter = [df[df['sigma'] == s]['iterations'].values for s in noise_levels]
    bp2 = axes[0, 1].boxplot(data_iter, labels=[f'{s:.0e}' for s in noise_levels])
    axes[0, 1].set_xlabel('Noise Level (σ)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Iterations', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Iteration Count Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Gradient norm boxplot
    data_grad = [df[df['sigma'] == s]['grad_norm_final'].values for s in noise_levels]
    bp3 = axes[1, 0].boxplot(data_grad, labels=[f'{s:.0e}' for s in noise_levels])
    axes[1, 0].axhline(y=1e-5, color='red', linestyle='--', label='tol_grad')
    axes[1, 0].set_xlabel('Noise Level (σ)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Final Gradient Norm', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Gradient Norm Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Time boxplot
    data_time = [df[df['sigma'] == s]['time_sec'].values for s in noise_levels]
    bp4 = axes[1, 1].boxplot(data_time, labels=[f'{s:.0e}' for s in noise_levels])
    axes[1, 1].set_xlabel('Noise Level (σ)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Computation Time Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_distributions.png'), dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 4_distributions.png")
    plt.close()
    
    print("\n✓ All visualizations generated!")


# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    
    # Define noise levels to test (logarithmic scale)
    noise_levels = [
        0.0,      # No noise (baseline)
        1e-6,     # Tiny noise
        1e-5,     # Very small
        1e-4,     # Small
        5e-4,     # Medium-small
        1e-3,     # Medium
        5e-3,     # Medium-large
        1e-2,     # Large
        5e-2,     # Very large
        1e-1,     # Huge
        5e-1,     # Extreme
        1.0,      # Maximum tested
    ]
    
    # Run tests
    df, all_results = test_bfgs_limits(
        noise_levels=noise_levels,
        n_runs_per_noise=15,  # 15 runs per noise level for statistical significance
        output_dir="BFGS_LIMITS_TEST"
    )
    
    # Analyze results
    analyze_results(df, "BFGS_LIMITS_TEST")
    
    # Plot results
    plot_results(df, all_results, "BFGS_LIMITS_TEST")
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nAll results saved in: BFGS_LIMITS_TEST/")
    print(f"  - bfgs_limits_results.csv (raw data)")
    print(f"  - summary_statistics.csv (statistics)")
    print(f"  - 1_success_rates.png")
    print(f"  - 2_convergence_metrics.png")
    print(f"  - 3_convergence_curves.png")
    print(f"  - 4_distributions.png")