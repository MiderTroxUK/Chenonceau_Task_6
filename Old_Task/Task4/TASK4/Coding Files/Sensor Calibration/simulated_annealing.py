# Objective: Simulated Annealing Optimization
# Author: John Hoarau
# Date: 2025-12-03


import math
import random
import matplotlib.pyplot as plt
import csv

class SimulatedAnnealing:
    def __init__(self, problem, initial_temp=1000, cooling_rate=0.9, min_temp=0.1, metropolis_steps=100, seed=None):
        """
        :param seed: Integer to fix randomness for reproducibility.
        :param metropolis_steps: Number of iterations (Markov Chain) at each temperature.
        """
        self.problem = problem
        self.initial_temp = initial_temp
        self.temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.metropolis_steps = metropolis_steps
        
        # 1. Set the Seed
        if seed is not None:
            random.seed(seed)
            print(f"Random Seed set to: {seed}")

        # Data Logging
        self.history = {
            "iteration": [],
            "temperature": [],
            "energy": [],
            "best_energy": [],
            "accepted": [] # Ratio for this temperature step
        }
        self.best_solution = None

    def _update_temperature(self):
        """
        Implements the Cooling Schedule.
        Currently: Geometric cooling.
        """
        self.temp *= self.cooling_rate

    def solve(self, verbose=True):
        current_solution = self.problem.generate_initial_solution()
        current_energy = self.problem.evaluate(current_solution)
        
        best_solution = current_solution
        best_energy = current_energy
        self.best_solution = best_solution
        
        iteration = 0
        
        print("--- Starting Optimization ---")

        while self.temp > self.min_temp:
            accepted_this_step = 0
            
            # Inner Loop: Equilibration (Markov Chain)
            for _ in range(self.metropolis_steps):
                # Create neighbor
                new_solution = self.problem.get_neighbor(current_solution)
                new_energy = self.problem.evaluate(new_solution)
                
                energy_diff = new_energy - current_energy
                accepted = False

                # Metropolis Criterion
                if new_energy < current_energy:
                    accepted = True
                else:
                    try:
                        probability = math.exp(-energy_diff / self.temp)
                    except OverflowError:
                        probability = 0.0
                        
                    if random.random() < probability:
                        accepted = True
                
                # Update State
                if accepted:
                    current_solution = new_solution
                    current_energy = new_energy
                    accepted_this_step += 1
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_solution = current_solution
                        self.best_solution = best_solution

            # 2. Collect Metrics (Once per Temperature Step)
            self.history["iteration"].append(iteration)
            self.history["temperature"].append(self.temp)
            self.history["energy"].append(current_energy)
            self.history["best_energy"].append(best_energy)
            
            # Log ratio for this step
            step_ratio = accepted_this_step / self.metropolis_steps if self.metropolis_steps > 0 else 0
            self.history["accepted"].append(step_ratio)

            # Cool down
            self._update_temperature()
            iteration += 1

        if verbose:
            print(f"--- Finished in {iteration} temperature steps ---")
            print(f"Final Energy: {best_energy:.4f}")
            
            avg_ratio = sum(self.history["accepted"]) / len(self.history["accepted"]) if self.history["accepted"] else 0
            print(f"Avg Acceptance Ratio: {avg_ratio:.2%}")

        return self.best_solution

    def plot_statistics(self, save_path=None):

        """
        Generates 2 graphs:
        1. Energy & Temperature over time.
        2. Acceptance Ratio over time.
        """
        iterations = self.history["iteration"]
        
        # Setup the plot
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # --- Graph 1: Energy vs Temperature ---
        color = 'tab:red'
        ax1.set_ylabel('Energy (Cost)', color=color)
        ax1.plot(iterations, self.history["energy"], color=color, alpha=0.3, label="Current Energy")
        ax1.plot(iterations, self.history["best_energy"], color='black', linewidth=2, label="Best Energy")
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper right')
        
        # Create a second y-axis for Temperature sharing the same x-axis
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Temperature', color=color)  
        ax2.plot(iterations, self.history["temperature"], color=color, linestyle='--', label="Temperature")
        ax2.tick_params(axis='y', labelcolor=color)
        ax1.set_title('Optimization History: Energy & Temperature')

        # --- Graph 2: Cumulative Acceptance Ratio ---
        # Calculate running average of acceptance
        cumulative_acceptance = []
        running_sum = 0
        for i, val in enumerate(self.history["accepted"]):
            running_sum += val
            cumulative_acceptance.append(running_sum / (i + 1))

        ax3.set_xlabel('Iterations (Temperature Steps)')
        ax3.set_ylabel('Acceptance Ratio')
        ax3.plot(iterations, cumulative_acceptance, color='green', label="Cumulative Acceptance Ratio")
        # Add instantaneous ratio too for debug
        ax3.plot(iterations, self.history["accepted"], color='lightgreen', alpha=0.5, label="Instantaneous Ratio")
        
        ax3.axhline(y=0.0, color='r', linestyle='-', alpha=0.3)
        ax3.set_title('Cooling Efficiency (Acceptance Ratio)')
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            
        plt.show()
    
    def plot_best_solution(self, save_path=None):
        """
        Plots the best solution found by the optimizer.
        """
        if self.best_solution is not None:
            self.problem.plot_model(self.best_solution, save_path=save_path)
        else:
            print("No solution found yet. Run solve() first.")

    def plot_model(self, solution, save_path=None):
        """
        Plots a specific solution.
        """
        self.problem.plot_model(solution, save_path=save_path)

    def save_to_csv(self, filename="annealing_results.csv"):
        """
        Saves the history of the run to a CSV file.
        Format: iteration, temperature, energy, best_energy, accepted
        """
               
        # 1. Get the column names (headers)
        keys = list(self.history.keys())
        
        # 2. Get the data columns
        values = list(self.history.values())
        
        # 3. Zip them together (this turns columns into rows)
        rows = zip(*values)

        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write the header
                writer.writerow(keys)
                
                # Write the data
                writer.writerows(rows)
                
            print(f"--> Data successfully saved to '{filename}'")
            
        except IOError as e:
            print(f"Error saving file: {e}")