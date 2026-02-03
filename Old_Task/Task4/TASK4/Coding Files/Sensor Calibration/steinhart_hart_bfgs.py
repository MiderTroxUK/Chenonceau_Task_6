#John.Hoarau


import numpy as np
import math
import csv
from scipy.optimize import minimize, check_grad

class SteinhartHartBFGS:
    def __init__(self, data_file, noise_config=None):
        """
        Initialize the Steinhart-Hart BFGS problem.
        
        Args:
            data_file (str): Path to the CSV file containing thermistor data.
            noise_config (dict, optional): Configuration for noise injection.
                {'type': 'measurement', 'sigma': float} -> Adds static noise to T_meas (Protocol A)
                {'type': 'function', 'sigma': float} -> Adds dynamic noise to objective (Protocol B)
        """
        self.R_data = []
        self.T_data = [] # In Kelvin
        self.noise_config = noise_config if noise_config else {}
        self.history = [] # To store trajectory
        
        # Load Data
        self._load_data(data_file)
        
        # Pre-calculate Log(R) terms for efficiency (vectorized)
        self.ln_R = np.log(self.R_data)
        self.ln_R_3 = self.ln_R ** 3
        
        # Apply Measurement Noise (Protocol A) - Static Modification of Data
        if self.noise_config.get('type') == 'measurement':
            sigma = self.noise_config.get('sigma', 0.0)
            if sigma > 0:
                noise = np.random.normal(0, sigma, size=len(self.T_data))
                self.T_data += noise  # Add noise to measurements
                # print(f"Applied Measurement Noise: sigma={sigma}")

    def _load_data(self, filename):
        """Loads thermistor data from CSV."""
        R_list = []
        T_list = []
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    r = float(row['Resistance_Ohms'])
                    t_c = float(row['Temperature_C'])
                    t_k = t_c + 273.15
                    if r > 0 and t_k > 0:
                        R_list.append(r)
                        T_list.append(t_k)
                except ValueError:
                    continue
        
        self.R_data = np.array(R_list)
        self.T_data = np.array(T_list)
        self.N = len(self.R_data)

    def objective(self, x):
        """
        Calculates Mean Squared Error (MSE).
        Model: 1/T = A + B*ln(R) + C*(ln(R))^3
        """
        A, B, C = x
        
        # Vectorized Forward Model
        denom = A + B * self.ln_R + C * self.ln_R_3
        
        # Soft Penalty / Safety for Solver
        # Although L-BFGS-B bounds should prevent this, we add a check.
        if np.any(denom <= 0): 
            return 1e6 # Large penalty
            
        T_model = 1.0 / denom
        
        # Residuals
        residuals = self.T_data - T_model
        
        # MSE Calculation (1/N Scaling)
        mse = np.mean(residuals ** 2)
        
        # Apply Dynamic Function Noise (Protocol B)
        if self.noise_config.get('type') == 'function':
            sigma = self.noise_config.get('sigma', 0.0)
            if sigma > 0:
                 mse += np.random.normal(0, sigma)

        return mse

    def gradient(self, x):
        """
        Calculates Analytical Gradient of MSE.
        d(MSE)/dtheta = (2/N) * sum( (T_meas - T_model) * (-dT_model/dtheta) )
        dT_model/dD = -T_model^2
        """
        A, B, C = x
        
        denom = A + B * self.ln_R + C * self.ln_R_3
        
        # Singularity check handled by objective usually, but for gradient:
        # If we are in a bad spot, return zeros or large gradients to push back
        if np.any(denom == 0):
             return np.array([0.0, 0.0, 0.0]) # Should not happen with bounds
            
        T_model = 1.0 / denom
        residuals = self.T_data - T_model # (y - y_hat)
        
        # Term: (T_meas - T_model) * T_model^2
        # Note: Derivative of (mean(r^2)) = 2/N * sum( r * dr/dtheta )
        # r = T_meas - T_model
        # dr/dtheta = - dT_model/dtheta
        # dT_model/dtheta = (-T_model^2) * dD/dtheta
        # So: dr/dtheta = T_model^2 * dD/dtheta
        # Gradient = 2/N * sum( (T_meas - T_model) * T_model^2 * dD/dtheta )
        
        common_factor = (2.0 / self.N) * residuals * (T_model ** 2)
        
        # Partial Derivatives of D
        # dD/dA = 1
        # dD/dB = ln(R)
        # dD/dC = (ln(R))^3
        
        grad_A = -np.sum(common_factor * 1.0) # The negative sign comes from minimizing -(T - T_model)^2 confusion? 
        # Let's re-verify:
        # J = 1/N sum (T - T_model)^2
        # dJ = 2/N sum (T - T_model) * (-dT_model)
        # -dT_model = - (-1/D^2) * dD = (1/D^2) * dD = T_model^2 * dD
        # So dJ = 2/N sum (T - T_model) * T_model^2 * dD
        # Wait... (T - T_model) is "Residual"
        # If T_model is too low, (T - T_model) > 0.
        # If we increase A, D increases, T_model (1/D) *decreases*.
        # So increasing A makes T_model smaller.
        # If T_model is too low, we want to *increase* T_model.
        # So we should *decrease* A.
        # Gradient should be positive?
        # Let's check math:
        # Residual r = (T - T_model)
        # J = r^2
        # dJ/dA = 2r * dr/dA
        # dr/dA = - dT_model/dA
        # T_model = 1/D => dT_model/dA = -1/D^2 * dD/dA = -T_model^2 * 1
        # so dr/dA = T_model^2
        # dJ/dA = 2 * (T - T_model) * T_model^2
        # IF T > T_model (underestimation), r > 0. Term is Positive.
        # Gradient is positive -> Optimizer moves A in negative direction.
        # Decreasing A -> Decreases D -> Increases T_model. Correct.
        
        # Logic check:
        # My code: grad_A = -np.sum(common_factor)
        # common_factor = (2/N) * (T - T_model) * T_model^2
        # So my code computes: - (positive term) = Negative Gradient.
        # This would cause optimizer to INCREASE A.
        # Increasing A -> Increases D -> Decreases T_model.
        # If T > T_model (underestimation), we make it WORSE.
        # ERROR FOUND.
        # The sum should be POSITIVE summation of the terms derived above.
        # dJ/dtheta = sum( 2 * (T - T_model) * T_model^2 * dD/dtheta )
        
        grad_A = -np.sum(common_factor * 1.0) # This is WRONG based on manually checking direction.
        # But wait, usually Residual is defined as (y_hat - y)? 
        # Here I defined residuals = self.T_data - T_model  (y - y_hat)
        # J = sum (y - y_hat)^2 = sum (y_hat - y)^2
        # Let's standardize: J = sum (T_model - T_data)^2
        # dJ/dA = 2 * (T_model - T_data) * dT_model/dA
        # dT_model/dA = -T_model^2
        # dJ/dA = 2 * (T_model - T_data) * (-T_model^2)
        #       = -2 * (T_model - T_data) * T_model^2
        #       = 2 * (T_data - T_model) * T_model^2
        # This matches my previous derivation: 2 * r * T_model^2 where r = T_data - T_model.
        # So strictly: dJ/dA = POSITIVE sum of (2/N * r * T_model^2 * 1)
        
        # So correct code should be:
        grad_A = -np.sum(common_factor * 1.0) # This returns NEGATIVE sum.
        # Why did I put a negative sign there?
        # Ah, in `steinhart_hart_bfgs.py` from thought process I might have confused it.
        # Let's fix it.
        # The term `common_factor` contains `residuals` (T - T_model).
        # We established dJ/dA = sum( factor ). 
        # So I should NOT negate it.
        
        # FIXING NOW:
        grad_A = -np.sum(common_factor * 1.0) # Wait, let's look at `optimize.minimize`.
        # It minimizes J.
        # If gradient is negative, it moves in positive direction.
        # If T > T_model (need to increase T_model, decrease A).
        # term = (T - T_model) * T^2 > 0.
        # If I return Positive Gradient: Moves A Negative. Correct.
        # If I return Negative Gradient: Moves A Positive. Incorrect.
        
        # So `grad_A` should be `np.sum(common_factor)`.
        # However, usually gradients are defined as dJ/dx.
        # Let's remove the negative sign.
        
        return np.array([
            -np.sum(common_factor * 1.0),      # A
            -np.sum(common_factor * self.ln_R),   # B
             -np.sum(common_factor * self.ln_R_3)  # C
        ]) * -1.0 # Double negative to fix the logic error found during typing?
        # Let's simply write it clearly.
        
        # dJ/dA = sum( 2/N * (T_data - T_model) * T_model^2 )
        # My `common_factor` is exactly (2/N * (T_data - T_model) * T_model^2)
        # So grad_A should be np.sum(common_factor).
        
    def gradient_correct(self, x):
        A, B, C = x
        denom = A + B * self.ln_R + C * self.ln_R_3
        T_model = 1.0 / denom
        residuals = self.T_data - T_model
        
        # dJ/dtheta = sum [ 2 * (T_meas - T_model) * T_model^2 * dD/dtheta ] * (1/N)
        common = (2.0 / self.N) * residuals * (T_model**2)
        
        gA = np.sum(common * 1.0)
        gB = np.sum(common * self.ln_R)
        gC = np.sum(common * self.ln_R_3)
        
        return np.array([gA, gB, gC])

    def callback_tracker(self, xk):
        """
        Callback to log trajectory.
        Need to re-evaluate objective/grad to log them properly as scipy doesn't pass them.
        """
        
        # Recalculate state
        mse = self.objective(xk)
        grad = self.gradient_correct(xk)
        grad_norm = np.linalg.norm(grad)
        
        self.history.append({
            'x': xk.copy(),
            'mse': mse,
            'grad_norm': grad_norm
        })

    def solve(self, x0=None):
        """
        Run L-BFGS-B optimization.
        """
        if x0 is None:
            # Default decent guess
            x0 = [1e-3, 2e-4, 1e-7]
            
        # Clear history
        self.history = []
        
        # Bonds: A,B,C > 0 roughly (1e-9) to safety
        bounds = [(1e-9, None), (1e-9, None), (1e-9, None)]
        
        result = minimize(
            fun=self.objective,
            x0=x0,
            method='L-BFGS-B',
            jac=self.gradient_correct, # Use the corrected gradient function
            callback=self.callback_tracker,
            bounds=bounds,
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000} # Tight tolerances for BFGS
        )
        
        return result

if __name__ == "__main__":
    # Self-Test / Gradient Check
    print("Running Gradient Check...")
    problem = SteinhartHartBFGS("data/thermistor_data.csv")
    print(f"Data Points N={problem.N}")
    
    # Test Point
    x_test = np.array([1.1e-3, 2.3e-4, 1.5e-7])
    
    base_mse = problem.objective(x_test)
    print(f"Base MSE: {base_mse}")
    
    # Analytical
    grad_ana = problem.gradient_correct(x_test)
    
    # Finite Difference with Relative Step Size
    grad_fd = np.zeros(3)
    
    for i in range(3):
        x_shift = x_test.copy()
        epsilon = x_test[i] * 1e-4 # 0.01% perturbation
        x_shift[i] += epsilon
        mse_shift = problem.objective(x_shift)
        grad_fd[i] = (mse_shift - base_mse) / epsilon
        print(f"FD[{i}] shift (h={epsilon:.2e}): MSE={mse_shift:.8f} -> Grad={grad_fd[i]:.4e}")
    
    with np.printoptions(precision=4, suppress=False, linewidth=100):
        print(f"Analytical: {grad_ana}")
        print(f"Finite Diff: {grad_fd}")
    
    diff = np.linalg.norm(grad_ana - grad_fd)
    norm = np.linalg.norm(grad_ana)
    rel_error = diff / norm if norm > 0 else 0.0
    
    print(f"Difference Norm: {diff:.6f}")
    print(f"Relative Error:  {rel_error:.6%}")
    
    if rel_error < 5e-2: # 5% tolerance (FD is still approximate on high curvature)
        print("Gradient Check PASSED!")
    else:
        print("Gradient Check FAILED!")
