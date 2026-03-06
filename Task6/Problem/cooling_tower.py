#@author: John.H
import numpy as np

def slant_height(r_bottom, r_top, height):
    """
    Calculates the slant height of a frustum.
    s = sqrt((r_bot - r_top)^2 + h^2)
    """
    return np.sqrt((r_bottom - r_top)**2 + height**2)

def frustum_surface_area(r_bottom, r_top, height):
    """
    Calculates the lateral surface area of a conical frustum.
    A = pi * (r_bot + r_top) * s
    """
    s = slant_height(r_bottom, r_top, height)
    return np.pi * (r_bottom + r_top) * s

def frustum_volume(r_bottom, r_top, height):
    """
    Calculates the volume of a conical frustum.
    V = (pi * h / 3) * (r_bot^2 + r_bot*r_top + r_top^2)
    """
    return (np.pi * height / 3.0) * (r_bottom**2 + r_bottom * r_top + r_top**2)

def total_surface_area(radii, heights):
    """
    Calculates the total lateral surface area of the tower.
    
    Args:
        radii: array-like of shape (m+1,), [r0, r1, ..., rm]
        heights: array-like of shape (m,), [h1, h2, ..., hm]
    """
    radii = np.asarray(radii)
    heights = np.asarray(heights)
    m = len(heights)
    
    if len(radii) != m + 1:
        raise ValueError(f"Expected {m+1} radii for {m} heights, got {len(radii)}")
    
    total_area = 0.0
    for i in range(m):
        r_bot = radii[i]
        r_top = radii[i+1]
        h = heights[i]
        total_area += frustum_surface_area(r_bot, r_top, h)
        
    return total_area

def total_volume(radii, heights):
    """
    Calculates the total volume of the tower.
    
    Args:
        radii: array-like of shape (m+1,), [r0, r1, ..., rm]
        heights: array-like of shape (m,), [h1, h2, ..., hm]
    """
    radii = np.asarray(radii)
    heights = np.asarray(heights)
    m = len(heights)
    
    if len(radii) != m + 1:
        raise ValueError(f"Expected {m+1} radii for {m} heights, got {len(radii)}")
    
    tot_vol = 0.0
    for i in range(m):
        r_bot = radii[i]
        r_top = radii[i+1]
        h = heights[i]
        tot_vol += frustum_volume(r_bot, r_top, h)
        
    return tot_vol

class CoolingTowerProblem:
    """
    Defines the optimization problem for the Cooling Tower.
    """
    def __init__(self, m=10, r0=39.3, rm=27.4, total_height=36.5, v_target=70320.0):
        self.m = m
        self.r0 = r0
        self.rm = rm
        self.total_height = total_height
        self.v_target = v_target
        
        # Case 1 defaults: equal heights
        self.fixed_heights = np.full(m, total_height / m)

    def unpack_decision_variables(self, x, case_num, **kwargs):
        """
        Reconstructs the full radii and heights arrays from the decision variables based on the scenario.
        """
        x = np.asarray(x)
        if case_num in [1, 4, 5, 6, 7]:
            # x contains [r1, r2, ..., r_{m-1}]
            radii = np.zeros(self.m + 1)
            radii[0] = self.r0
            radii[-1] = self.rm
            radii[1:-1] = x
            heights = kwargs.get('fixed_heights', self.fixed_heights)
            return radii, heights
            
        elif case_num == 2:
            # x contains [h1, h2, ..., hm]
            radii = kwargs.get('fixed_radii')
            if radii is None:
                raise ValueError("Case 2 requires 'fixed_radii' kwarg.")
            return radii, x
            
        elif case_num == 3:
            # x contains [r1...r_{m-1}, h1...hm]
            radii = np.zeros(self.m + 1)
            radii[0] = self.r0
            radii[-1] = self.rm
            radii[1:-1] = x[:self.m - 1]
            heights = x[self.m - 1:]
            return radii, heights
            
        elif case_num == 8:
            # x contains [a, b, c] - hyperbola parameters
            a, b, c = x
            heights = self.fixed_heights
            # Calculate z heights (cumulative sum)
            z_levels = np.zeros(self.m + 1)
            for i in range(1, self.m + 1):
                z_levels[i] = z_levels[i-1] + heights[i-1]
                
            # r(z) = a * sqrt(1 + (z-c)^2 / b^2)
            radii = a * np.sqrt(1 + ((z_levels - c)**2) / (b**2))
            return radii, heights
            
        else:
            raise ValueError(f"Unknown case_num: {case_num}")

    def cost_function(self, x, case_num, penalty_weight=1e5, **kwargs):
        """
        Objective function for all cases.
        J(x) = A_total(x) + rho * (V_total(x) - V_target)^2 + custom_penalties
        """
        radii, heights = self.unpack_decision_variables(x, case_num, **kwargs)
        
        area = total_surface_area(radii, heights)
        vol = total_volume(radii, heights)
        
        # Case 6 has 50% more volume
        target_vol = self.v_target * 1.5 if case_num == 6 else self.v_target
        
        penalty = penalty_weight * (vol - target_vol)**2
        
        # Case 4: Tight Waist Constraint
        # r_mid < 0.7 * min(r0, rm)
        if case_num == 4:
            mid_idx = self.m // 2
            r_mid = radii[mid_idx]
            max_allowed = 0.7 * min(self.r0, self.rm)
            if r_mid > max_allowed:
                penalty += penalty_weight * (r_mid - max_allowed)**2
                
        # Optional: Add height positivity constraint if not handled by optimizer bounds (Cases 2, 3)
        if case_num in [2, 3]:
            # Penalize negative or zero heights
            neg_heights = heights[heights < 0.1]
            if len(neg_heights) > 0:
                penalty += penalty_weight * np.sum((0.1 - neg_heights)**2)
                
        return area + penalty
    
    def _hyperboloid_radii(self, r_waist):
        """
        Build a convex hyperboloid profile [r0, r1, ..., rm] that satisfies
        monotonicity (decreasing to waist, increasing to top) and convexity
        (positive second differences).

        Lower half:  r(t) = r0 - (r0 - r_waist) * sqrt(t)   [convex, fast drop]
        Upper half:  r(t) = r_waist + (rm - r_waist) * t^2   [convex, slow rise]
        """
        n = self.m + 1
        mid = n // 2
        radii = np.zeros(n)
        radii[0] = self.r0
        radii[-1] = self.rm
        radii[mid] = r_waist
        for i in range(1, mid):
            t = i / mid
            radii[i] = self.r0 - (self.r0 - r_waist) * np.sqrt(t)
        for i in range(mid + 1, n - 1):
            t = (i - mid) / (n - 1 - mid)
            radii[i] = r_waist + (self.rm - r_waist) * t**2
        return radii

    def get_initial_guess(self, case_num, **kwargs):
        """
        Returns a reasonable initial guess for the specified case.
        Uses a hyperboloid-shaped profile for radii cases to satisfy
        geometric constraints from the start.
        """
        if case_num in [1, 4, 7]:
            # Hyperboloid initial guess (waist ~19m gives V ≈ 70,320)
            radii = self._hyperboloid_radii(r_waist=19.0)
            return radii[1:-1]

        elif case_num == 6:
            # High Volume: wider waist for 1.5x volume target
            radii = self._hyperboloid_radii(r_waist=24.0)
            return radii[1:-1]

        elif case_num == 5:
            # Cylindrical Start (must start from cylinder per specification)
            r_avg = (self.r0 + self.rm) / 2.0
            return np.full(self.m - 1, r_avg)

        elif case_num == 2:
            # Heights guess: equal slices
            return self.fixed_heights.copy()

        elif case_num == 3:
            # Hyperboloid radii + equal heights
            radii = self._hyperboloid_radii(r_waist=19.0)
            heights_guess = self.fixed_heights.copy()
            return np.concatenate((radii[1:-1], heights_guess))

        elif case_num == 8:
            # a, b, c guess for hyperbola r(z) = a * sqrt(1 + (z-c)^2 / b^2)
            a_guess = min(self.r0, self.rm) * 0.8
            b_guess = self.total_height
            c_guess = self.total_height / 2.0
            return np.array([a_guess, b_guess, c_guess])

        else:
            raise ValueError(f"Unknown case_num: {case_num}")

    def get_bounds(self, case_num):
        """
        Returns the variable bounds [(lower, upper), ...] for each decision variable.
        Used by BFGS (L-BFGS-B) and PSO for box-constrained optimization.
        """
        if case_num == 7:
            # Restricted Bounds: r_i in [20, 60]
            return [(20.0, 60.0)] * (self.m - 1)
            
        elif case_num in [1, 4, 5, 6]:
            # General radii bounds: between some sensible physical limits
            r_min = 5.0
            r_max = max(self.r0, self.rm) * 2.0
            return [(r_min, r_max)] * (self.m - 1)
            
        elif case_num == 2:
            # Heights bounds: positive heights, max ~2x equal slice
            h_min = 0.5
            h_max = self.total_height * 0.5
            return [(h_min, h_max)] * self.m
            
        elif case_num == 3:
            # Radii bounds + Heights bounds
            r_min, r_max = 5.0, max(self.r0, self.rm) * 2.0
            h_min, h_max = 0.5, self.total_height * 0.5
            radii_bounds = [(r_min, r_max)] * (self.m - 1)
            height_bounds = [(h_min, h_max)] * self.m
            return radii_bounds + height_bounds
            
        elif case_num == 8:
            # Hyperbola parameters: a, b, c
            return [
                (5.0, max(self.r0, self.rm)),        # a: waist radius
                (5.0, self.total_height * 3.0),      # b: curvature
                (0.0, self.total_height)              # c: waist height
            ]
        else:
            raise ValueError(f"Unknown case_num: {case_num}")


def numerical_gradient(func, x, epsilon=1e-6):
    """
    Computes the gradient of a scalar function using central finite differences.
    
    grad_i = (f(x + e_i * eps) - f(x - e_i * eps)) / (2 * eps)
    
    Args:
        func: scalar-valued function f(x) -> float.
        x: current point (np.ndarray).
        epsilon: step size for finite differences.
        
    Returns:
        grad: np.ndarray of same shape as x.
    """
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        grad[i] = (func(x_plus) - func(x_minus)) / (2.0 * epsilon)
    return grad


def analytical_surface_area_hyperboloid(a, b, c, z_bottom, z_top, n_steps=1000):
    """
    Computes the exact lateral surface area of a Hyperboloid of One Sheet
    by numerical integration of the surface-of-revolution formula.
    
    r(z) = a * sqrt(1 + (z - c)^2 / b^2)
    
    A = 2*pi * integral_{z_bot}^{z_top} r(z) * sqrt(1 + (dr/dz)^2) dz
    
    This is useful for the report discussion comparing discrete (frustum)
    approximation vs the true curved surface.
    
    Args:
        a: waist radius parameter.
        b: curvature parameter.
        c: waist height parameter.
        z_bottom: lower integration limit.
        z_top: upper integration limit.
        n_steps: number of integration steps (trapezoidal rule).
        
    Returns:
        area: the analytically integrated surface area.
    """
    z = np.linspace(z_bottom, z_top, n_steps)
    
    r = a * np.sqrt(1.0 + ((z - c)**2) / (b**2))
    # dr/dz = a * (z - c) / (b^2 * sqrt(1 + (z-c)^2 / b^2))
    drdz = a * (z - c) / (b**2 * np.sqrt(1.0 + ((z - c)**2) / (b**2)))
    
    integrand = r * np.sqrt(1.0 + drdz**2)
    
    # Trapezoidal integration
    area = 2.0 * np.pi * np.trapz(integrand, z)
    return area
