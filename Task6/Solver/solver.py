import time
import numpy as np
from typing import Tuple, Dict, Any, List

# Add the parent directory to the path so we can import the problem
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Problem.cooling_tower import CoolingTowerProblem
# These will be the team member's actual implementations
# As long as they expose an `optimize(cost_func, x0, **kwargs)` interface or similar
try:
    from SA.sa import simulated_annealing
except ImportError:
    simulated_annealing = None

try:
    from BFGS.bfgs import bfgs_optimize
except ImportError:
    bfgs_optimize = None

try:
    from PSO.pso import particle_swarm
except ImportError:
    particle_swarm = None

class BaseOptimizer:
    """
    Base class for all optimization algorithms (BFGS, SA, PSO).
    Provides access to the problem instance and cost function wrapper.
    """
    def __init__(self, problem: CoolingTowerProblem, case_num: int = 1, fixed_radii: np.ndarray = None, **kwargs):
        self.problem = problem
        self.case_num = case_num
        self.fixed_radii = fixed_radii
        self.options = kwargs
        
    def _objective_func(self, x):
        """Wrapper for the objective function for the custom solvers to use."""
        return self.problem.cost_function(x, self.case_num, fixed_radii=self.fixed_radii)

    def optimize(self, initial_guess: np.ndarray) -> Tuple[np.ndarray, float, List[float], Dict[str, Any]]:
        raise NotImplementedError("Each team member must implement this method for their specific algorithm.")


class BfgsOptimizer(BaseOptimizer):
    """
    Martin's BFGS Implementation Wrapper.
    """
    def optimize(self, initial_guess: np.ndarray):
        start_time = time.time()
        
        if bfgs_optimize is None:
            raise NotImplementedError("bfgs.py module not found or missing bfgs_optimize function.")
            
        # Call Martin's actual function
        best_x, best_cost, history, n_fevals = bfgs_optimize(
            objective_func=self._objective_func, 
            initial_guess=initial_guess, 
            **self.options
        )
        
        time_taken = time.time() - start_time
        metrics = {"n_fevals": n_fevals, "time_taken": time_taken}
        return best_x, best_cost, history, metrics


class SaOptimizer(BaseOptimizer):
    """
    Pierre's Simulated Annealing Implementation Wrapper.
    """
    def optimize(self, initial_guess: np.ndarray):
        start_time = time.time()
        
        if simulated_annealing is None:
            raise NotImplementedError("sa.py module not found or missing simulated_annealing function.")
            
        # Call Pierre's actual function
        best_x, best_cost, history, n_fevals = simulated_annealing(
            objective_func=self._objective_func,
            initial_guess=initial_guess,
            **self.options
        )
        
        time_taken = time.time() - start_time
        metrics = {"n_fevals": n_fevals, "time_taken": time_taken}
        return best_x, best_cost, history, metrics


class PsoOptimizer(BaseOptimizer):
    """
    Clémence's Particle Swarm Optimization Implementation Wrapper.
    """
    def optimize(self, initial_guess: np.ndarray):
        start_time = time.time()
        
        if particle_swarm is None:
            raise NotImplementedError("pso.py module not found or missing particle_swarm function.")
            
        # Call Clémence's actual function (PSO often takes bounds rather than purely an initial guess)
        bounds = self.problem.get_bounds(self.case_num)
        
        best_x, best_cost, history, n_fevals = particle_swarm(
            objective_func=self._objective_func,
            bounds=bounds,
            **self.options
        )
        
        time_taken = time.time() - start_time
        metrics = {"n_fevals": n_fevals, "time_taken": time_taken}
        return best_x, best_cost, history, metrics

def run_scenario(optimizer_class, problem: CoolingTowerProblem, case_num: int, scenario_name: str, fixed_radii: np.ndarray = None, **solver_kwargs):
    """
    Helper function to run a specific scenario and display the results.
    """
    print(f"\n{'='*50}")
    print(f"Running Scenario {case_num}: {scenario_name}")
    print(f"Algorithm: {optimizer_class.__name__}")
    print(f"{'='*50}")
    
    optimizer = optimizer_class(problem, case_num=case_num, fixed_radii=fixed_radii, **solver_kwargs)
    
    initial_guess = problem.get_initial_guess(case_num)
    
    # Compute the initial guess profile (for plotting comparison later)
    init_radii, init_heights = problem.unpack_decision_variables(initial_guess, case_num, fixed_radii=fixed_radii)
    
    # Run optimization
    try:
        best_x, best_cost, history, metrics = optimizer.optimize(initial_guess)
    except NotImplementedError as e:
        print(f"Skipping: {e}")
        return None, None, None, None
    
    best_radii, best_heights = problem.unpack_decision_variables(best_x, case_num, fixed_radii=fixed_radii)
    
    from Problem.cooling_tower import total_surface_area, total_volume
    final_area = total_surface_area(best_radii, best_heights)
    final_vol = total_volume(best_radii, best_heights)
    
    target_vol = problem.v_target * 1.5 if case_num == 6 else problem.v_target
    vol_error = abs(final_vol - target_vol)
    
    # --- Data Collection and Visualization ---
    import os
    import csv
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create output directory
    algo_name = optimizer_class.__name__.replace('Optimizer', '')
    base_out_dir = os.path.join(os.path.dirname(__file__), "output")
    out_dir = os.path.join(base_out_dir, algo_name)
    os.makedirs(out_dir, exist_ok=True)
    
    base_filename = f"Case{case_num}_{algo_name}"
    
    # 1. Plot Convergence History
    plt.figure()
    plt.plot(history, label='Cost')
    plt.title(f"Convergence History: Case {case_num} ({algo_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(x)")
    plt.yscale('log') # Log scale is usually better for penalty functions
    plt.grid(True)
    plt.legend()
    conv_plot_path = os.path.join(out_dir, f"{base_filename}_convergence.png")
    plt.savefig(conv_plot_path)
    plt.close()
    
    # 2. Plot Cooling Tower Profile (Optimized vs Initial Guess)
    z_coords = np.concatenate(([0], np.cumsum(best_heights)))
    z_coords_init = np.concatenate(([0], np.cumsum(init_heights)))
    plt.figure(figsize=(6, 8))
    # Initial guess (dashed red)
    plt.plot(init_radii, z_coords_init, 'r--s', alpha=0.5, label='Initial Guess')
    plt.plot(-init_radii, z_coords_init, 'r--s', alpha=0.5)
    # Optimized (solid blue)
    plt.plot(best_radii, z_coords, 'b-o', label='Optimized')
    plt.plot(-best_radii, z_coords, 'b-o')
    plt.fill_betweenx(z_coords, -best_radii, best_radii, alpha=0.15, color='blue')
    plt.title(f"Profile: Case {case_num} ({algo_name})\nArea: {final_area:.2f} | Vol Error: {vol_error:.2f}")
    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    prof_plot_path = os.path.join(out_dir, f"{base_filename}_profile.png")
    plt.savefig(prof_plot_path)
    plt.close()
    
    # 3. Plot 3D Wireframe
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    theta = np.linspace(0, 2*np.pi, 30)
    
    # Create the 3D surface grid
    for i in range(len(best_heights)):
        z_bot = z_coords[i]
        z_top = z_coords[i+1]
        r_bot = best_radii[i]
        r_top = best_radii[i+1]
        
        # Simple frustum side
        z_grid = np.linspace(z_bot, z_top, 5)
        r_grid = np.linspace(r_bot, r_top, 5)
        
        Z, Theta = np.meshgrid(z_grid, theta)
        R, _ = np.meshgrid(r_grid, theta)
        
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        
        ax.plot_wireframe(X, Y, Z, color='b', alpha=0.5)

    ax.set_title(f"3D Wireframe: Case {case_num} ({algo_name})")
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Fix aspect ratio manually
    max_radius = np.max(best_radii)
    max_height = np.max(z_coords)
    ax.set_xlim([-max_radius, max_radius])
    ax.set_ylim([-max_radius, max_radius])
    ax.set_zlim([0, max_height])
    ax.set_box_aspect([1, 1, max_height / (2*max_radius)]) # Set 3D aspect ratio
    
    wireframe_plot_path = os.path.join(out_dir, f"{base_filename}_wireframe.png")
    plt.savefig(wireframe_plot_path)
    plt.close()
    
    # 4. Save Metrics to CSV
    csv_path = os.path.join(base_out_dir, "summary_metrics.csv")
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Scenario", "Algorithm", "TimeTaken_sec", "FunctionEvals", 
                "FinalArea", "FinalVolume", "TargetVolume", "VolumeError", 
                "FinalCost", "BestRadii", "BestHeights"
            ])
        writer.writerow([
            scenario_name,
            algo_name,
            metrics.get('time_taken', 0.0),
            metrics.get('n_fevals', 0),
            final_area,
            final_vol,
            target_vol,
            vol_error,
            best_cost,
            str(best_radii.tolist()),
            str(best_heights.tolist())
        ])
        
    print(f"\nSaved artifacts to {out_dir}:")
    print(f" - {base_filename}_convergence.png")
    print(f" - {base_filename}_profile.png")
    print(f" - {base_filename}_wireframe.png")
    print(f" - summary_metrics.csv (Appended)")
    
    print("\n--- Results ---")
    print(f"Time Taken:      {metrics.get('time_taken', 0.0):.4f} seconds")
    print(f"Function Evals:  {metrics.get('n_fevals', 0)}")
    print(f"Final Area:      {final_area:.2f} m^2")
    print(f"Final Volume:    {final_vol:.2f} m^3 (Target: {target_vol:.2f})")
    print(f"Volume Error:    {vol_error:.2f} m^3")
    print(f"Final Cost:      {best_cost:.2f}")
    
    return best_radii, best_heights, history, metrics

def generate_comparison_convergence(case_num, histories_dict, out_dir):
    """
    Plots convergence histories for all algorithms on a single figure for a given case.
    
    Args:
        case_num: The scenario number.
        histories_dict: dict of {algo_name: history_list}.
        out_dir: Output directory path.
    """
    import matplotlib.pyplot as plt
    
    if not histories_dict:
        return
        
    plt.figure(figsize=(10, 6))
    colors = {'Sa': 'red', 'Bfgs': 'blue', 'Pso': 'green'}
    for algo_name, history in histories_dict.items():
        if history is not None and len(history) > 1:
            plt.plot(history, label=algo_name, color=colors.get(algo_name, 'black'))
    
    plt.title(f"Convergence Comparison: Case {case_num}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(x)")
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    path = os.path.join(out_dir, f"Case{case_num}_comparison_convergence.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


def generate_comparison_table(out_dir):
    """
    Reads summary_metrics.csv and generates a formatted comparison table as a PNG image.
    """
    import csv
    import matplotlib.pyplot as plt
    
    csv_path = os.path.join(out_dir, "summary_metrics.csv")
    if not os.path.isfile(csv_path):
        print("No summary_metrics.csv found.")
        return
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if len(rows) < 2:
        print("Not enough data for comparison table.")
        return
    
    header = rows[0]
    data = rows[1:]
    
    # Build table with key columns only
    table_header = ["Scenario", "Algorithm", "Time (s)", "F. Evals", "Final Area", "Vol Error", "Final Cost"]
    table_data = []
    for row in data:
        table_data.append([
            row[0],                             # Scenario
            row[1],                             # Algorithm
            f"{float(row[2]):.4f}",             # TimeTaken_sec
            row[3],                             # FunctionEvals
            f"{float(row[4]):.2f}",             # FinalArea
            f"{float(row[7]):.2f}",             # VolumeError
            f"{float(row[8]):.2f}"              # FinalCost
        ])
    
    fig, ax = plt.subplots(figsize=(16, max(4, len(table_data) * 0.4 + 1)))
    ax.axis('off')
    ax.set_title("Optimization Results Comparison Table", fontsize=14, fontweight='bold', pad=20)
    
    table = ax.table(
        cellText=table_data,
        colLabels=table_header,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    
    # Color the header row
    for j in range(len(table_header)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row shading
    for i in range(1, len(table_data) + 1):
        color = '#D9E2F3' if i % 2 == 0 else '#FFFFFF'
        for j in range(len(table_header)):
            table[(i, j)].set_facecolor(color)
    
    table_path = os.path.join(out_dir, "comparison_table.png")
    plt.savefig(table_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved comparison table: {table_path}")


if __name__ == "__main__":
    import os
    
    problem = CoolingTowerProblem()
    
    # Create output directory and clear old CSV to avoid stale data
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "summary_metrics.csv")
    if os.path.isfile(csv_path):
        os.remove(csv_path)
    
    print("Running all 8 cases across all 3 algorithms...")
    
    # List of all available optimizer classes
    optimizers = [SaOptimizer, BfgsOptimizer, PsoOptimizer]
    
    # Scenarios
    scenarios = [
        (1, "Mandatory Ref Case"),
        (2, "Variable Heights"),
        (3, "Full Freedom"),
        (4, "Tight Waist"),
        (5, "Cylindrical Start"),
        (6, "High Volume"),
        (7, "Restricted Bounds"),
        (8, "Hyperbolic Fit")
    ]
    
    # Collect convergence histories per case for comparison plots
    # Structure: {case_num: {algo_name: history}}
    all_histories = {case_num: {} for case_num, _ in scenarios}
    
    for opt_class in optimizers:
        print(f"\n\n{'*'*60}")
        print(f"*** Starting Runs for {opt_class.__name__} ***")
        print(f"{'*'*60}\n")
        
        algo_name = opt_class.__name__.replace('Optimizer', '')
        case_1_best_radii = None
        
        for case_num, scenario_name in scenarios:
            if case_num == 2:
                if case_1_best_radii is None:
                    print("Skipping Case 2: Requires radii from Case 1.")
                    continue
                best_radii, best_heights, history, metrics = run_scenario(
                    opt_class, problem, case_num, scenario_name, fixed_radii=case_1_best_radii
                )
            else:
                best_radii, best_heights, history, metrics = run_scenario(
                    opt_class, problem, case_num, scenario_name
                )
                
            if case_num == 1 and best_radii is not None:
                case_1_best_radii = best_radii
            
            # Store history for comparison convergence plots
            all_histories[case_num][algo_name] = history
    
    # Generate comparison convergence plots (all 3 algos on one figure per case)
    print("\n\nGenerating comparison convergence plots...")
    for case_num, _ in scenarios:
        generate_comparison_convergence(case_num, all_histories[case_num], out_dir)
    
    # Generate comparison table
    print("\nGenerating comparison table...")
    generate_comparison_table(out_dir)
    
    print("\n\nAll runs complete! Check the 'output' folder for all artifacts.")
