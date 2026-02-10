"""
Subject:        Visualisation functions
Inheritance:    PSO_simulation.py from Task 3
@author:        Clémence-Philomène HINOT
@date:          10/02/2026
"""

# ********** IMPORTATION **********

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
import os

# ********** Visualisation functions **********

def plot_pareto_front(result, save_path, obj_labels=None):
    """
    Plot the Pareto front for bi-objective optimization
    
    :param result: Optimization result from pymoo
    :param save_path: Path to save the figure
    :param obj_labels: List of [xlabel, ylabel] for objectives
    """
    if obj_labels is None:
        obj_labels = ['Objective 1', 'Objective 2']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(result.F[:, 0], result.F[:, 1], color='blue', s=50, alpha=0.6, edgecolors='black')
    ax.set_xlabel(obj_labels[0], fontsize=12)
    ax.set_ylabel(obj_labels[1], fontsize=12)
    ax.set_title('Pareto Front', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Pareto front saved to {save_path}")


def plot_decision_space(result, save_path, var_labels=None):
    """
    Plot the decision space colored by first objective
    
    :param result: Optimization result from pymoo
    :param save_path: Path to save the figure
    :param var_labels: List of [xlabel, ylabel] for decision variables
    """
    if var_labels is None:
        var_labels = ['Variable 1', 'Variable 2']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(result.X[:, 0], result.X[:, 1], c=result.F[:, 0], 
                        cmap='viridis', s=50, alpha=0.6, edgecolors='black')
    ax.set_xlabel(var_labels[0], fontsize=12)
    ax.set_ylabel(var_labels[1], fontsize=12)
    ax.set_title('Decision Space', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.colorbar(scatter, label='First Objective')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Decision space saved to {save_path}")


def plot_convergence(result, save_path):
    """
    Plot convergence history if available
    
    :param result: Optimization result from pymoo
    :param save_path: Path to save the figure
    """
    if hasattr(result.algorithm, 'callback') and hasattr(result.algorithm.callback, 'data'):
        # Extract convergence data if available
        pass
    else:
        print("Convergence data not available")


# ********** Export functions **********

def save_results_to_csv(result, save_path, var_names=None, obj_names=None):
    """
    Save optimization results to CSV
    
    :param result: Optimization result from pymoo
    :param save_path: Path to save CSV file
    :param var_names: List of variable names
    :param obj_names: List of objective names
    """
    n_var = result.X.shape[1]
    n_obj = result.F.shape[1]
    
    if var_names is None:
        var_names = [f'x{i+1}' for i in range(n_var)]
    if obj_names is None:
        obj_names = [f'f{i+1}' for i in range(n_obj)]
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = var_names + obj_names
        if hasattr(result, 'G') and result.G is not None:
            n_constr = result.G.shape[1]
            header += [f'g{i+1}' for i in range(n_constr)]
        writer.writerow(header)
        
        # Data
        for i in range(len(result.F)):
            row = list(result.X[i, :]) + list(result.F[i, :])
            if hasattr(result, 'G') and result.G is not None:
                row += list(result.G[i, :])
            writer.writerow(row)
    
    print(f"Results saved to {save_path}")


def save_summary(result, save_path, var_names=None, obj_names=None):
    """
    Save summary of best solutions to CSV
    
    :param result: Optimization result from pymoo
    :param save_path: Path to save summary CSV
    :param var_names: List of variable names
    :param obj_names: List of objective names
    """
    n_var = result.X.shape[1]
    n_obj = result.F.shape[1]
    
    if var_names is None:
        var_names = [f'x{i+1}' for i in range(n_var)]
    if obj_names is None:
        obj_names = [f'f{i+1}' for i in range(n_obj)]
    
    # Find key solutions
    min_indices = [np.argmin(result.F[:, i]) for i in range(n_obj)]
    middle_idx = len(result.F) // 2
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Summary of Key Solutions'])
        writer.writerow(['Solution Type'] + var_names + obj_names)
        
        # Min solutions for each objective
        for i, idx in enumerate(min_indices):
            row = [f'Min {obj_names[i]}'] + list(result.X[idx, :]) + list(result.F[idx, :])
            writer.writerow(row)
        
        # Compromise solution
        row = ['Compromise'] + list(result.X[middle_idx, :]) + list(result.F[middle_idx, :])
        writer.writerow(row)
    
    print(f"Summary saved to {save_path}")