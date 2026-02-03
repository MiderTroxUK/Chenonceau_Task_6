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

    def unpack_decision_variables_case1(self, x):
        """
        Reconstructs the full radii and heights arrays from the decision variables for Case 1.
        Case 1: Optimize inner radii r1...r(m-1). Fixed heights. Fixed r0, rm.
        
        Args:
            x: array of shape (m-1,), interior radii.
        
        Returns:
            radii: shape (m+1,)
            heights: shape (m,)
        """
        # x contains [r1, r2, ..., r_{m-1}]
        radii = np.zeros(self.m + 1)
        radii[0] = self.r0
        radii[-1] = self.rm
        radii[1:-1] = x
        
        return radii, self.fixed_heights

    def cost_function_case1(self, x, penalty_weight=1e3):
        """
        Objective function for Case 1 (Minimize Surface Area with Volume Penalty).
        J(x) = A_total(x) + rho * (V_total(x) - V_target)^2
        
        Args:
            x: decision variables (m-1 radii)
            penalty_weight: scalar weight for constraint violation
        """
        radii, heights = self.unpack_decision_variables_case1(x)
        
        area = total_surface_area(radii, heights)
        vol = total_volume(radii, heights)
        
        penalty = penalty_weight * (vol - self.v_target)**2
        return area + penalty
    
    def get_initial_guess_case1_linear(self):
        """
        Returns a linear interpolation between r0 and rm as an initial guess for x.
        """
        # Linspace from r0 to rm with m+1 points, take the inner m-1 points
        return np.linspace(self.r0, self.rm, self.m + 1)[1:-1]

    def get_initial_guess_case1_cylinder(self):
        """
        Returns a cylinder guess (average radius) for x.
        """
        r_avg = (self.r0 + self.rm) / 2.0
        return np.full(self.m - 1, r_avg)
