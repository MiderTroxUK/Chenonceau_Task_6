"""
Subject:        PSO algorithm
Inheritance:    Particle_Swarm_Optimization.py from Task 3
@author:        Clémence-Philomène HINOT
@date:          10/02/2026
"""

# ********** IMPORTATION **********

# library for mutli-objective optimization
import pymoo

# single-objective optimization
from pymoo.algorithms.soo.nonconvex.pso import PSO

# minimize function
from pymoo.optimize import minimize

# ********** CLASS DEFINITION **********

class ParticleSwarm:

    # ---------- INITIALIZATION ----------

    def __init__(self, problem, param):
        """
        Initialize Particle Swarm Optimization
        
        :param problem: Optimization problem
        :param param: Dictionary with parameters:
            - pop_size (int): Population size
            - w (float): Inertia weight
            - c1 (float): Cognitive parameter
            - c2 (float): Social parameter
            - max_velocity_rate (float): Maximum velocity rate
            - archive_size (int): Archive size for MOPSO
            - n_gen (int): Number of generations
            - seed (int): Random seed for reproducibility
        """

        # Extraction of parameters
        self.pop_size = param.get('pop_size')
        self.w = param.get('w')
        self.c1 = param.get('c1')
        self.c2 = param.get('c2')
        self.max_velocity_rate = param.get('max_velocity_rate')
        self.archive_size = param.get('archive_size')
        self.n_gen = param.get('n_gen')
        self.seed = param.get('seed')

        # validate parameters
        if self.pop_size <= 0:
            raise ValueError("pop_size must be positive")
        if not (0 <= self.w <= 1):
            raise ValueError("w (inertia weight) should be in [0, 1]")
        if self.c1 <= 0 or self.c2 <= 0:
            raise ValueError("c1 and c2 must be positive")

        # param verbose: Whether to print optimization progress
        Verbose = False

        # Best solution
        self.best_solution = minimize(
            problem,
            PSO,
            termination = ('n_gen', self.n_gen), 
            seed = self.seed, 
            verbose = Verbose
        )