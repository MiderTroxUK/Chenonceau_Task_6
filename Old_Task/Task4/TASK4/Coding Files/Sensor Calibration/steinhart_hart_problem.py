#@Author: John.H
#Date: 2025-12-03
#Version: 1.0
#Description: Steinhart-Hart Problem


import math
import random
import csv
from problem_interface import Problem

class SteinhartHartProblem(Problem):
    def __init__(self, data_file):
        self.data = []
        self._load_data(data_file)
        
    def _load_data(self, filename):
        """Loads thermistor data from CSV."""
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Store R and T (in Kelvin)
                # T_K = T_C + 273.15
                r = float(row['Resistance_Ohms'])
                t_c = float(row['Temperature_C'])
                t_k = t_c + 273.15
                self.data.append({'R': r, 'T_K': t_k})

    def generate_initial_solution(self):
        """
        Returns a random starting state [A, B, C].
        Typical values: A ~ 1e-3, B ~ 1e-4, C ~ 1e-7
        """
        return [
            random.uniform(0.0005, 0.002),   # A
            random.uniform(0.0001, 0.0005),  # B
            random.uniform(1e-8, 1e-6)       # C
        ]

    def evaluate(self, solution):
        """
        Calculates SSE: sum((T_measured - T_model)^2)
        Model: 1/T = A + B*ln(R) + C*(ln(R))^3
        => T_model = 1 / (A + B*ln(R) + C*(ln(R))^3)
        """
        A, B, C = solution
        sse = 0.0
        
        for point in self.data:
            R = point['R']
            T_measured = point['T_K']
            
            try:
                ln_R = math.log(R)
                denom = A + B * ln_R + C * (ln_R ** 3)
                
                if denom == 0:
                    return float('inf')
                
                T_model = 1.0 / denom
                
                error = T_measured - T_model
                sse += error ** 2
            except (ValueError, ZeroDivisionError):
                return float('inf')
                
        return sse

    def get_neighbor(self, solution):
        """
        Perturbs [A, B, C] slightly.
        """
        A, B, C = solution
        
        # Perturbation factors tailored to the scale of parameters
        new_A = A + random.gauss(0, 1e-5)
        new_B = B + random.gauss(0, 1e-6)
        new_C = C + random.gauss(0, 1e-9)
        
        return [new_A, new_B, new_C]

    def plot_model(self, solution, save_path=None):
        """
        Plots the measured data vs the model curve.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        A, B, C = solution
        
        # Extract data
        R_data = [d['R'] for d in self.data]
        T_data = [d['T_K'] - 273.15 for d in self.data] # Convert back to Celsius for plotting
        
        # Generate smooth curve
        min_R = min(R_data)
        max_R = max(R_data)
        R_smooth = np.linspace(min_R, max_R, 100)
        T_smooth = []
        
        for r in R_smooth:
            try:
                ln_R = math.log(r)
                denom = A + B * ln_R + C * (ln_R ** 3)
                if denom != 0:
                    t_k = 1.0 / denom
                    T_smooth.append(t_k - 273.15)
                else:
                    T_smooth.append(None)
            except ValueError:
                T_smooth.append(None)
                
        plt.figure(figsize=(10, 6))
        plt.scatter(R_data, T_data, color='red', label='Measured Data')
        plt.plot(R_smooth, T_smooth, color='blue', label='Steinhart-Hart Model')
        
        plt.xlabel('Resistance (Ohms)')
        plt.ylabel('Temperature (Celsius)')
        plt.title('Thermistor Calibration: Measured vs Model')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            
        plt.show()
