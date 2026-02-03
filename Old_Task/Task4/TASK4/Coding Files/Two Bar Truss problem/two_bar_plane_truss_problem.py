"""
Subject:    Two Bar Plane Truss Problem adapted for BFGS optimisation
Source:     https://www.sciencedirect.com/science/article/pii/S0898122111010406
@author:    Clemence-Philomene HINOT & Claude AI
@date:      12/01/2026
"""

# ********** IMPORTATION **********

import math

# ********** CLASS DEFINITION **********

class Two_Bar_Plane_Truss_Problem():

    # ---------- INITIALIZATION ----------

    def __init__(self):
        # constants
        self.rho = 7833             # unit: kg/m^3
        self.h = 2.54               # unit: m
        self.P = 44482.2            # unit: N
        self.E = 2.07e11            # unit: Pa
        self.sigma0 = 1.38e8        # unit: Pa
        self.A_min = 6.4516e-4      # unit: m^2

        # variables
        self.pos = 0.0
        self.area = 0.0

        # physical boundaries
        self.pos_min = 0.1 * self.h
        self.pos_max = 2.25 * self.h
        self.area_min = 0.5 * self.A_min
        self.area_max = 2.5 * self.A_min

    # ---------- CONSTITUTIVE EQUATIONS ----------

    # Weight objective
    def f1(self):
        x1 = self.pos/self.h # need to use normalised x1 in this function
        return 2 * self.rho * self.h * self.area * math.sqrt(1 + pow(x1, 2))

    # Displacement objective
    def f2(self):
        x1 = self.pos/self.h
        return (self.P * pow(self.h, 3) * pow((1 + pow(x1, 2)), 1.5)) / ( math.sqrt(1 + pow(x1, 4)) * 2 * math.sqrt(2) * self.E * x1 * self.area)
    
    # ---------- CONSTRAINT EQUATIONS ----------

    # Stress constraint (tension)
    def g1(self, normalised=True):
        x1 = self.pos/self.h
        stress = (self.P * (1 + x1) * math.sqrt(1 + pow(x1, 2)) / (2 * math.sqrt(2) * x1 * self.area))
        
        if normalised:
            return (stress / self.sigma0) - 1.0 # normalised result
        else:
            return stress - self.sigma0 # real result in Pa
        
    
    # Stress constraint (compression)
    def g2(self, normalised=True):
        x1 = self.pos/self.h
        stress = (self.P * (1 - x1) * math.sqrt(1 + pow(x1, 2)) / (2 * math.sqrt(2) * x1 * self.area))
        
        if normalised:
            return (stress / self.sigma0) - 1.0 # normalised result
        else:
            return stress - self.sigma0 # real result in Pa
    
    # ---------- EVALUATION FUNCTION ----------

    def evaluate(self, x):
        """
        Evaluate objectives and constraints
        
        :param x: [x1_normalized, x2_normalized]
        :return: (sum_value, [f1, f2], [g1, g2])
        """

        # physical position
        self.pos = x[0] * self.h

        # physical area
        self.area = x[1] * self.A_min

        # weight
        f1 = self.f1()

        # movement
        f2 = self.f2()

        # constraints
        g1 = self.g1()
        g2 = self.g2()

        # penalty to apply if the constraint g > 0
        penalty1 = max(0, g1)**2
        penalty2 = max(0, g2)**2
        
        # weight for the contraint functions
        weightConstraint = 10000.0

        # weight for f1 and f2
        #   they have the same
        weightFunction = 0.5

        # weighted sum
        sum = weightFunction * f1 + weightFunction * f2 + weightConstraint * (penalty1 + penalty2)
        
        # results
        return sum, [f1, f2], [g1, g2]