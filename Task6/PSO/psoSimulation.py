"""
Subject:        Cooling Tower optimisation with PSO algorithm
Inheritance:    PSO_simulation.py from Task 3
@author:        Clémence-Philomène HINOT
@date:          10/02/2026
"""

# ********** IMPORTATION **********

import numpy as np
import matplotlib
import csv
import os

# PSO algorithms
from psoAlgorithm import ParticleSwarm

# Import the cooling problem
# TO DO

# ========== MAIN OPTIMISATION RUNNER ==========

def run_optimisation(problem, pso_params, output_dir='results', 
                    var_names=None, obj_names=None, 
                    var_labels=None, obj_labels=None):
    """
    Run PSO optimisation on a given problem
    
    :param problem: Pymoo Problem instance
    :param pso_params: Dictionary of PSO parameters
    :param output_dir: Directory to save results
    :param var_names: List of variable names for CSV
    :param obj_names: List of objective names for CSV
    :param var_labels: List of variable labels for plots
    :param obj_labels: List of objective labels for plots
    :return: Optimisation result
    """
    
    print("="*70)
    print("COOLING TOWER OPTIMISATION USING PSO")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize PSO
    print("\n[1/3] Initialising PSO optimiser...")
    pso = ParticleSwarm(problem, pso_params)
    print("✓ PSO initialised")
    
    # Run optimization
    print("\n[2/3] Running optimization...")
    result = pso.PSO_multi_objective(Verbose=True)
    print(f"✓ Optimization complete - Found {len(result.F)} Pareto solutions")
    
    # Save results
    print("\n[3/3] Saving results...")
    
    # CSV files
    csv_path = os.path.join(output_dir, 'pareto_solutions.csv')
    save_results_to_csv(result, csv_path, var_names, obj_names)
    
    summary_path = os.path.join(output_dir, 'summary.csv')
    save_summary(result, summary_path, var_names, obj_names)
    
    # Plots
    pareto_path = os.path.join(output_dir, 'pareto_front.png')
    plot_pareto_front(result, pareto_path, obj_labels)
    
    decision_path = os.path.join(output_dir, 'decision_space.png')
    plot_decision_space(result, decision_path, var_labels)
    
    print("\n" + "="*70)
    print("OPTIMIsATION COMPLETE!")
    print(f"All results saved in: {output_dir}")
    print("="*70)
    
    return result

# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    
    # Create problem
    # TO DO
    
    # Configure PSO parameters
    pso_params = {
        'pop_size': 100,
        'w': 0.9,
        'c1': 2.0,
        'c2': 2.0,
        'max_velocity_rate': 0.2,
        'n_gen': 200,
        'seed': 42
    }
    
    # Define labels for visualization and export
    # TO DO
    # var_names = ['x1', 'x2']
    # obj_names = ['Weight', 'Displacement']
    # var_labels = ['x1 (normalized position)', 'x2 (normalized area)']
    # obj_labels = ['Weight (kg)', 'Displacement (m)']
    
    # Run optimization
    result = run_optimisation(
        problem=problem,
        pso_params=pso_params,
        output_dir='results_truss',
        var_names=var_names,
        obj_names=obj_names,
        var_labels=var_labels,
        obj_labels=obj_labels
    )