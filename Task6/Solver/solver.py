"""
Subject:        Solver with Geometric Constraints for Cooling Tower
@author:        Pierre Bédrune, John HOARAU
@date:          04/03/2026

ARCHITECTURE:
    - Les algorithmes (SA, PSO, BFGS) optimisent librement
    - Le solver vérifie et corrige chaque solution selon les règles géométriques
    - Les contraintes sont appliquées APRÈS chaque itération ou à la fin

RÈGLES GÉOMÉTRIQUES (construction hyperboloïde):

    Points fixes:
        - P0 (bas): r0, z=0
        - P10 (haut): rm, z=H
    
    Construction depuis le HAUT (P10 → P5):
        - Chaque point doit être EN DESSOUS et À DROITE (r plus petit)
        - La dérivée doit rester à gauche de la droite précédente (convexité)
    
    Construction depuis le BAS (P0 → P5):
        - Chaque point doit être AU DESSUS et À DROITE (r plus petit)
        - La dérivée doit rester à gauche de la droite précédente (convexité)
    
    Point WAIST (P5):
        - Doit satisfaire les contraintes des deux côtés
        - C'est le minimum local
"""

import time
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Problem.cooling_tower import CoolingTowerProblem

# Import algorithms
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


# =============================================================================
# GEOMETRIC CONSTRAINTS
# =============================================================================

class GeometricConstraints:
    """
    Vérifie et corrige les solutions pour garantir une forme hyperboloïde.
    
    Règles:
        1. Monotonie: rayons décroissent de r0 vers waist, puis croissent vers rm
        2. Convexité: courbure vers l'extérieur (dérivée seconde > 0)
        3. Waist: point minimum, satisfait les contraintes des deux côtés
    """
    
    def __init__(self, r0: float, rm: float, n_points: int = 11):
        """
        Args:
            r0: Rayon fixe en bas (z=0)
            rm: Rayon fixe en haut (z=H)
            n_points: Nombre total de points (11 = r0 + 9 intérieurs + rm)
        """
        self.r0 = r0
        self.rm = rm
        self.n_points = n_points
        self.n_interior = n_points - 2  # Points à optimiser
        self.waist_idx = n_points // 2  # Index du waist dans le tableau complet
    
    def get_full_radii(self, interior_radii: np.ndarray) -> np.ndarray:
        """Reconstruit le tableau complet [r0, r1, ..., r9, rm]"""
        return np.concatenate([[self.r0], interior_radii, [self.rm]])
    
    def check_point_from_bottom(self, radii: np.ndarray, idx: int) -> Dict[str, Any]:
        """
        Vérifie les contraintes pour un point construit depuis le bas.
        
        Args:
            radii: Tableau complet des rayons
            idx: Index du point à vérifier (1 à waist_idx)
            
        Returns:
            Dict avec 'valid', 'violations', 'suggested_r'
        """
        result = {
            'valid': True,
            'violations': [],
            'suggested_r': radii[idx]
        }
        
        # Contrainte 1: r[idx] < r[idx-1] (doit être plus petit que le précédent)
        if radii[idx] >= radii[idx - 1]:
            result['valid'] = False
            result['violations'].append(f"P{idx}: r={radii[idx]:.2f} >= r_prev={radii[idx-1]:.2f}")
            result['suggested_r'] = radii[idx - 1] * 0.95
        
        # Contrainte 2: Convexité (dérivée seconde > 0)
        # Contrainte 2: Convexité (d2r/dz2 approximée par différences centrées)
        # Utiliser la différence centrée au point idx: r[idx-1] - 2*r[idx] + r[idx+1]
        if idx >= 1 and idx <= self.n_points - 2:
            d2r = radii[idx - 1] - 2 * radii[idx] + radii[idx + 1]
            if d2r < 0:
                result['valid'] = False
                result['violations'].append(f"P{idx}: non convexe (d2r={d2r:.2f})")
                # Correction: proposer une valeur légèrement plus petite que la moyenne des voisins
                target = 0.9 * ((radii[idx - 1] + radii[idx + 1]) / 2.0)
                result['suggested_r'] = min(result['suggested_r'], target)
        
        return result
    
    def check_point_from_top(self, radii: np.ndarray, idx: int) -> Dict[str, Any]:
        """
        Vérifie les contraintes pour un point construit depuis le haut.
        
        Args:
            radii: Tableau complet des rayons
            idx: Index du point à vérifier (waist_idx à n_points-2)
            
        Returns:
            Dict avec 'valid', 'violations', 'suggested_r'
        """
        result = {
            'valid': True,
            'violations': [],
            'suggested_r': radii[idx]
        }
        
        # Contrainte 1: r[idx] < r[idx+1] (doit être plus petit que le suivant)
        if radii[idx] >= radii[idx + 1]:
            result['valid'] = False
            result['violations'].append(f"P{idx}: r={radii[idx]:.2f} >= r_next={radii[idx+1]:.2f}")
            result['suggested_r'] = radii[idx + 1] * 0.95
        
        # Contrainte 2: Convexité (dérivée seconde > 0)
        # Contrainte 2: Convexité (différence centrée au point idx)
        if idx >= 1 and idx <= self.n_points - 2:
            d2r = radii[idx - 1] - 2 * radii[idx] + radii[idx + 1]
            if d2r < 0:
                result['valid'] = False
                result['violations'].append(f"P{idx}: non convexe depuis haut (d2r={d2r:.2f})")
                target = 0.9 * ((radii[idx - 1] + radii[idx + 1]) / 2.0)
                result['suggested_r'] = min(result['suggested_r'], target)
        
        return result
    
    def check_waist_point(self, radii: np.ndarray) -> Dict[str, Any]:
        """
        Vérifie les contraintes pour le point waist (P5 = 11ème point).
        
        DOIT SATISFAIRE LES CONDITIONS DES DEUX CÔTÉS:
            - Depuis le bas: r[5] < r[4] ET convexité (d2r avec P3, P4, P5)
            - Depuis le haut: r[5] < r[6] ET convexité (d2r avec P5, P6, P7)
        
        Returns:
            Dict avec 'valid', 'violations', 'suggested_r'
        """
        idx = self.waist_idx  # = 5
        result = {
            'valid': True,
            'violations': [],
            'suggested_r': radii[idx]
        }
        
        # === CONTRAINTES DEPUIS LE BAS ===
        # 1. Monotonie: P5 < P4
        if radii[idx] >= radii[idx - 1]:
            result['valid'] = False
            result['violations'].append(f"Waist P{idx}: r={radii[idx]:.2f} >= r_gauche={radii[idx-1]:.2f}")
        
        # 2. Convexité bas: d2r(P3, P4, P5) > 0
        if idx >= 1 and idx <= self.n_points - 2:
            d2r_bottom = radii[idx - 1] - 2 * radii[idx] + radii[idx + 1]
            if d2r_bottom < 0:
                result['valid'] = False
                result['violations'].append(f"Waist P{idx}: non convexe depuis bas (d2r={d2r_bottom:.2f})")
        
        # === CONTRAINTES DEPUIS LE HAUT ===
        # 3. Monotonie: P5 < P6
        if radii[idx] >= radii[idx + 1]:
            result['valid'] = False
            result['violations'].append(f"Waist P{idx}: r={radii[idx]:.2f} >= r_droite={radii[idx+1]:.2f}")
        
        # 4. Convexité haut: d2r(P5, P6, P7) > 0
        if idx >= 1 and idx <= self.n_points - 2:
            d2r_top = radii[idx - 1] - 2 * radii[idx] + radii[idx + 1]
            if d2r_top < 0:
                result['valid'] = False
                result['violations'].append(f"Waist P{idx}: non convexe depuis haut (d2r={d2r_top:.2f})")
        
        # Suggestion: si invalide, prendre le minimum des voisins
        if not result['valid']:
            result['suggested_r'] = min(radii[idx - 1], radii[idx + 1]) * 0.9
        
        return result
    
    def check_all_points(self, interior_radii: np.ndarray, verbose: bool = False) -> Dict[str, Any]:
        """
        Vérifie tous les points et retourne un rapport complet.
        
        Returns:
            Dict avec 'all_valid', 'violations', 'point_reports'
        """
        radii = self.get_full_radii(interior_radii)
        
        report = {
            'all_valid': True,
            'violations': [],
            'point_reports': {}
        }
        
        # Points depuis le bas (P1 à P_waist-1)
        for idx in range(1, self.waist_idx):
            pt_report = self.check_point_from_bottom(radii, idx)
            report['point_reports'][f'P{idx}_bottom'] = pt_report
            if not pt_report['valid']:
                report['all_valid'] = False
                report['violations'].extend(pt_report['violations'])
        
        # Point waist
        waist_report = self.check_waist_point(radii)
        report['point_reports']['waist'] = waist_report
        if not waist_report['valid']:
            report['all_valid'] = False
            report['violations'].extend(waist_report['violations'])
        
        # Points depuis le haut (P_waist+1 à P9)
        for idx in range(self.waist_idx + 1, self.n_points - 1):
            pt_report = self.check_point_from_top(radii, idx)
            report['point_reports'][f'P{idx}_top'] = pt_report
            if not pt_report['valid']:
                report['all_valid'] = False
                report['violations'].extend(pt_report['violations'])
        
        if verbose and not report['all_valid']:
            print(f"  [GeoCheck] {len(report['violations'])} violations détectées")
            for v in report['violations'][:5]:  # Afficher max 5
                print(f"    - {v}")
        
        return report
    
    def correct_solution(self, interior_radii: np.ndarray, max_iterations: int = 10) -> np.ndarray:
        """
        Corrige une solution pour respecter toutes les contraintes géométriques.
        
        Algorithme:
            1. Construire depuis le bas (P0 → waist)
            2. Construire depuis le haut (Pm → waist)
            3. Ajuster le waist pour satisfaire les deux côtés
            4. Répéter si nécessaire
        
        Returns:
            Radii corrigés
        """
        result = interior_radii.copy()
        
        for iteration in range(max_iterations):
            radii = self.get_full_radii(result)
            changed = False
            
            # === Construction depuis le BAS ===
            for idx in range(1, self.waist_idx):
                # Monotonie: r[idx] < r[idx-1]
                if radii[idx] >= radii[idx - 1]:
                    radii[idx] = radii[idx - 1] * 0.95
                    changed = True
                
                # Convexité (différence centrée au point idx)
                if idx >= 1 and idx <= self.n_points - 2:
                    d2r = radii[idx - 1] - 2 * radii[idx] + radii[idx + 1]
                    if d2r < 0:
                        # Projeter vers une valeur inférieure à la moyenne des voisins
                        radii[idx] = 0.9 * ((radii[idx - 1] + radii[idx + 1]) / 2.0)
                        changed = True
            
            # === Construction depuis le HAUT ===
            for idx in range(self.n_points - 2, self.waist_idx, -1):
                # Monotonie: r[idx] < r[idx+1]
                if radii[idx] >= radii[idx + 1]:
                    radii[idx] = radii[idx + 1] * 0.95
                    changed = True
                
                # Convexité (différence centrée au point idx)
                if idx >= 1 and idx <= self.n_points - 2:
                    d2r = radii[idx - 1] - 2 * radii[idx] + radii[idx + 1]
                    if d2r < 0:
                        radii[idx] = 0.9 * ((radii[idx - 1] + radii[idx + 1]) / 2.0)
                        changed = True
            
            # === Waist (doit être minimum) ===
            idx = self.waist_idx
            max_allowed = min(radii[idx - 1], radii[idx + 1]) - 0.5
            if radii[idx] > max_allowed:
                radii[idx] = max(max_allowed, 5.0)
                changed = True
            
            # Mettre à jour result (sans r0 et rm)
            result = radii[1:-1]
            
            # Bornes min
            result = np.clip(result, 5.0, None)
            
            if not changed:
                break
        
        return result
    
    def compute_penalty(self, interior_radii: np.ndarray) -> float:
        """
        Calcule une pénalité pour les violations de contraintes.
        PÉNALITÉS TRÈS FORTES pour forcer la forme hyperboloïde.
        """
        interior_radii = np.asarray(interior_radii)
        # Accept either interior radii (length n_interior) or full radii (length n_points)
        if interior_radii.shape[0] == self.n_points:
            radii = interior_radii.copy()
        else:
            radii = self.get_full_radii(interior_radii)
        penalty = 0.0
        
        # Pénalité monotonie depuis le bas (r[i] doit être < r[i-1])
        for idx in range(1, self.waist_idx + 1):
            if radii[idx] >= radii[idx - 1]:
                diff = radii[idx] - radii[idx - 1] + 0.1
                penalty += diff ** 2 * 10000  # TRÈS FORT
        
        # Pénalité monotonie depuis le haut (r[i] doit être < r[i+1])
        for idx in range(self.waist_idx, self.n_points - 1):
            if radii[idx] >= radii[idx + 1]:
                diff = radii[idx] - radii[idx + 1] + 0.1
                penalty += diff ** 2 * 10000  # TRÈS FORT
        
        # Pénalité convexité bas (d2r > 0)
        for idx in range(1, self.waist_idx + 1):
            # centred second difference at idx
            if idx >= 1 and idx <= self.n_points - 2:
                d2r = radii[idx - 1] - 2 * radii[idx] + radii[idx + 1]
                if d2r < 0:
                    penalty += abs(d2r) ** 2 * 5000
        
        # Pénalité convexité haut (d2r > 0)
        for idx in range(self.waist_idx, self.n_points - 1):
            if idx >= 1 and idx <= self.n_points - 2:
                d2r = radii[idx - 1] - 2 * radii[idx] + radii[idx + 1]
                if d2r < 0:
                    penalty += abs(d2r) ** 2 * 5000
        
        return penalty


# =============================================================================
# OPTIMIZER CLASSES
# =============================================================================

class BaseOptimizer:
    """Base class for all optimization algorithms."""
    
    def __init__(self, problem: CoolingTowerProblem, case_num: int = 1,
                 geo_constraints: GeometricConstraints = None,
                 fixed_radii: np.ndarray = None, **kwargs):
        self.problem = problem
        self.case_num = case_num
        self.fixed_radii = fixed_radii
        self.options = kwargs
        self.geo_constraints = geo_constraints

    def _objective_func(self, x):
        """Objective function avec pénalité géométrique optionnelle."""
        # Try to unpack decision variables to get full radii/height representation
        try:
            radii_full, heights = self.problem.unpack_decision_variables(x, self.case_num, fixed_radii=self.fixed_radii)
        except Exception:
            radii_full = None
            heights = None

        # If no geometric constraints, evaluate normally
        if self.geo_constraints is None:
            return self.problem.cost_function(x, self.case_num, fixed_radii=self.fixed_radii)

        # When geometric constraints exist, enforce them strictly when possible
        # Cases where decision variables are interior radii: 1,4,5,6,7
        if radii_full is not None and self.case_num in [1, 4, 5, 6, 7, 3]:
            interior = np.asarray(radii_full)[1:-1]
            # Always project/correct interior to feasible set
            corrected = self.geo_constraints.correct_solution(interior, max_iterations=50)

            # Reconstruct decision vector according to case
            if self.case_num in [1, 4, 5, 6, 7]:
                x_eval = np.asarray(corrected)
            elif self.case_num == 3:
                x_eval = np.concatenate((np.asarray(corrected), np.asarray(heights)))

            # Evaluate cost on corrected (feasible) vector
            return self.problem.cost_function(x_eval, self.case_num, fixed_radii=self.fixed_radii)

        # Cases where radii are fixed inputs (Case 2) or generated from parameters (Case 8)
        # Verify geometry of the implied radii; if invalid, return very large cost to reject
        if radii_full is not None and self.case_num in [2, 8]:
            interior = np.asarray(radii_full)[1:-1]
            report = self.geo_constraints.check_all_points(interior, verbose=False)
            if not report['all_valid']:
                return 1e12
            # Valid -> evaluate normally
            return self.problem.cost_function(x, self.case_num, fixed_radii=self.fixed_radii)

        # Fallback: evaluate normally
        return self.problem.cost_function(x, self.case_num, fixed_radii=self.fixed_radii)

    def optimize(self, initial_guess: np.ndarray):
        raise NotImplementedError


class SaOptimizer(BaseOptimizer):
    """Simulated Annealing Wrapper."""
    
    def optimize(self, initial_guess: np.ndarray):
        start_time = time.time()

        if simulated_annealing is None:
            raise NotImplementedError("sa.py not found.")

        bounds = self.problem.get_bounds(self.case_num)

        best_x, best_cost, history, n_fevals = simulated_annealing(
            objective_func=self._objective_func,
            initial_guess=initial_guess,
            bounds=bounds,
            **self.options
        )

        # PAS de correction finale - les pénalités guident l'optimisation
        # La solution doit naturellement satisfaire les contraintes

        time_taken = time.time() - start_time
        metrics = {"n_fevals": n_fevals, "time_taken": time_taken}
        return best_x, best_cost, history, metrics


class BfgsOptimizer(BaseOptimizer):
    """BFGS Wrapper."""
    
    def optimize(self, initial_guess: np.ndarray):
        start_time = time.time()

        if bfgs_optimize is None:
            raise NotImplementedError("bfgs.py not found.")

        bounds = self.problem.get_bounds(self.case_num)

        best_x, best_cost, history, n_fevals = bfgs_optimize(
            objective_func=self._objective_func,
            initial_guess=initial_guess,
            bounds=bounds,
            **self.options
        )

        # PAS de correction finale - les pénalités guident l'optimisation

        time_taken = time.time() - start_time
        metrics = {"n_fevals": n_fevals, "time_taken": time_taken}
        return best_x, best_cost, history, metrics


class PsoOptimizer(BaseOptimizer):
    """Particle Swarm Wrapper."""
    
    def optimize(self, initial_guess: np.ndarray):
        start_time = time.time()

        if particle_swarm is None:
            raise NotImplementedError("pso.py not found.")

        bounds = self.problem.get_bounds(self.case_num)

        best_x, best_cost, history, n_fevals = particle_swarm(
            objective_func=self._objective_func,
            bounds=bounds,
            **self.options
        )

        # PAS de correction finale - les pénalités guident l'optimisation

        time_taken = time.time() - start_time
        metrics = {"n_fevals": n_fevals, "time_taken": time_taken}
        return best_x, best_cost, history, metrics


# =============================================================================
# SCENARIO RUNNER
# =============================================================================

def run_scenario(optimizer_class, problem: CoolingTowerProblem, case_num: int,
                 scenario_name: str, fixed_radii: np.ndarray = None, 
                 use_geo_constraints: bool = True, **extra_kwargs):
    """Run a specific scenario with geometric constraints."""
    
    print(f"\n{'='*60}")
    print(f"Scenario {case_num}: {scenario_name}")
    print(f"Algorithm: {optimizer_class.__name__}")
    print(f"Geometric Constraints: {'ON' if use_geo_constraints else 'OFF'}")
    print(f"{'='*60}")

    # Create geometric constraints
    geo_constraints = None
    if use_geo_constraints:
        geo_constraints = GeometricConstraints(
            r0=problem.r0,
            rm=problem.rm,
            n_points=problem.m + 1
        )

    # If fixed_radii provided for cases like Case 2, ensure they respect geometry (correct if needed)
    if use_geo_constraints and fixed_radii is not None:
        fr = np.asarray(fixed_radii)
        # Accept either interior or full radii
        if fr.shape[0] == problem.m + 1:
            interior_fr = fr[1:-1]
        else:
            interior_fr = fr

        report_fr = geo_constraints.check_all_points(interior_fr, verbose=False)
        if not report_fr['all_valid']:
            corrected = geo_constraints.correct_solution(interior_fr, max_iterations=50)
            fixed_radii = np.concatenate(([geo_constraints.r0], corrected, [geo_constraints.rm]))

    optimizer = optimizer_class(
        problem, 
        case_num=case_num,
        geo_constraints=geo_constraints,
        fixed_radii=fixed_radii, 
        **extra_kwargs
    )

    initial_guess = problem.get_initial_guess(case_num)
    init_radii, init_heights = problem.unpack_decision_variables(
        initial_guess, case_num, fixed_radii=fixed_radii
    )

    # Run optimization
    try:
        best_x, best_cost, history, metrics = optimizer.optimize(initial_guess)
    except NotImplementedError as e:
        print(f"Skipping: {e}")
        return None, None, None, None

    best_radii, best_heights = problem.unpack_decision_variables(
        best_x, case_num, fixed_radii=fixed_radii
    )

    # If geometric constraints are enabled, apply final correction to the returned solution
    if geo_constraints is not None:
        interior_best = np.asarray(best_radii)[1:-1]
        # Apply correction to ensure final geometry validity
        corrected_interior = geo_constraints.correct_solution(interior_best, max_iterations=50)
        # Rebuild full radii using corrected interior
        best_radii = np.concatenate(([geo_constraints.r0], corrected_interior, [geo_constraints.rm]))
        report = geo_constraints.check_all_points(corrected_interior, verbose=True)
        print(f"  [GeoCheck] Final: {'VALID' if report['all_valid'] else 'INVALID'}")

    # Compute metrics
    from Problem.cooling_tower import total_surface_area, total_volume
    final_area = total_surface_area(best_radii, best_heights)
    final_vol = total_volume(best_radii, best_heights)

    target_vol = problem.v_target * 1.5 if case_num == 6 else problem.v_target
    vol_error = abs(final_vol - target_vol)

    # === Save Results ===
    import csv
    import matplotlib.pyplot as plt

    algo_name = optimizer_class.__name__.replace('Optimizer', '')
    base_out_dir = os.path.join(os.path.dirname(__file__), "output")
    out_dir = os.path.join(base_out_dir, algo_name)
    os.makedirs(out_dir, exist_ok=True)

    base_filename = f"Case{case_num}_{algo_name}"

    # 1. Convergence Plot
    plt.figure()
    plt.plot(history, label='Cost')
    plt.title(f"Convergence: Case {case_num} ({algo_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(x)")
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{base_filename}_convergence.png"))
    plt.close()

    # 2. Profile Plot with Initial Guess
    z_coords = np.concatenate(([0], np.cumsum(best_heights)))
    z_coords_init = np.concatenate(([0], np.cumsum(init_heights)))
    
    plt.figure(figsize=(8, 10))
    
    # Initial Guess (ROUGE)
    plt.plot(init_radii, z_coords_init, 'r--s', alpha=0.6, markersize=8, label='Initial Guess')
    plt.plot(-init_radii, z_coords_init, 'r--s', alpha=0.6, markersize=8)
    
    # Optimized (BLEU)
    plt.plot(best_radii, z_coords, 'b-o', markersize=6, label='Optimized')
    plt.plot(-best_radii, z_coords, 'b-o', markersize=6)
    plt.fill_betweenx(z_coords, -best_radii, best_radii, alpha=0.15, color='blue')
    
    # Mark waist
    waist_idx = len(best_radii) // 2
    plt.plot([best_radii[waist_idx], -best_radii[waist_idx]], 
             [z_coords[waist_idx], z_coords[waist_idx]], 
             'g--', linewidth=2, label=f'Waist (r={best_radii[waist_idx]:.1f}m)')
    
    plt.title(f"Profile: Case {case_num} ({algo_name})\nArea: {final_area:.2f} | Vol Error: {vol_error:.2f}")
    plt.xlabel("Radius (m)")
    plt.ylabel("Height (m)")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(out_dir, f"{base_filename}_profile.png"), dpi=150)
    plt.close()

    print(f"\n--- Results ---")
    print(f"Time:        {metrics.get('time_taken', 0.0):.4f} s")
    print(f"Func Evals:  {metrics.get('n_fevals', 0)}")
    print(f"Final Area:  {final_area:.2f} m²")
    print(f"Final Vol:   {final_vol:.2f} m³ (Target: {target_vol:.2f})")
    print(f"Vol Error:   {vol_error:.2f} m³")

    return best_radii, best_heights, history, metrics


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    problem = CoolingTowerProblem()

    # Test Case 1 with SA
    print("\n" + "="*70)
    print("TEST SA")
    print("="*70)
    run_scenario(
        SaOptimizer, problem, case_num=1,
        scenario_name="Mandatory Ref Case",
        use_geo_constraints=True,
        T_initial=10000,
        alpha=0.997,
        max_iter=10000,
        seed=42,
        verbose=True
    )
    
    # Test Case 1 with PSO
    print("\n" + "="*70)
    print("TEST PSO")
    print("="*70)
    run_scenario(
        PsoOptimizer, problem, case_num=1,
        scenario_name="Mandatory Ref Case",
        use_geo_constraints=True,
        num_particles=100,
        max_iter=300,
        seed=42,
        verbose=True
    )