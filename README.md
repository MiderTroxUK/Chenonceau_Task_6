# Task 6: Hyperboloid Cooling Tower Optimization

**Project:** Group Project Spring 2026 - Task 6  
**Goal:** Minimize Construction Cost (Surface Area) subject to Fixed Volume using Stochastic and Deterministic Methods.

## 1. Project Overview & Team Roles

This document outlines the mathematical formulation, optimization strategy, and implementation plan for Task 6.

### Team Roles:
*   **Project Lead / Integration:** John Hoarau (Mathematical Decoupling, GitHub, Retro-planning)
*   **BFGS Implementation:** Martin Harambat
*   **Simulated Annealing (SA):** Pierre Bédrune
*   **Particle Swarm Optimization (PSO) & LaTeX:** Clémence-Philomène

**Deadlines:**
*   **Report Due:** 06/03/2026

## 2. Mathematical Formulation

### 2.1 Geometric Primitives (The "Physics")

The cooling tower is approximated as a stack of **Conical Frustums** (straight-sided cone slices).
Let there be $m$ frustums, meaning $m+1$ horizontal cross-sections (rings).

**Coordinate System:**
Let $z$ be the vertical axis. The base of the tower is at $z_0 = 0$.
The height of ring $i$ ($z_i$) is the cumulative sum of heights: $z_i = \sum_{k=1}^{i} h_k$.

Let each frustum $i$ (from $1$ to $m$) be defined by:
*   **Bottom radius:** $r_{i-1}$ at height $z_{i-1}$
*   **Top radius:** $r_i$ at height $z_i$
*   **Height of section:** $h_i = z_i - z_{i-1}$

#### **Derived Formulae**

For a single frustum $i$:

1.  **Slant Height ($s_i$):**
    $$ s_i = \sqrt{(r_{i-1} - r_{i})^2 + h_i^2} $$
    *(Note: This assumes straight edges, not curved)*

2.  **Lateral Surface Area ($A_i$):**
    $$ A_i = \pi (r_{i-1} + r_{i}) s_i = \pi (r_{i-1} + r_{i}) \sqrt{(r_{i-1} - r_{i})^2 + h_i^2} $$

3.  **Volume ($V_i$):**
    $$ V_i = \frac{\pi h_i}{3} (r_{i-1}^2 + r_{i-1}r_{i} + r_{i}^2) $$

#### **Global Objective Function**
We must **minimize** the Total Surface Area $A_{total}$:
$$ A_{total}(\mathbf{x}) = \sum_{i=1}^{m} A_i $$

#### **Global Constraint**
The total volume must equal the Target Volume $V_{target}$:
$$ V_{total}(\mathbf{x}) = \sum_{i=1}^{m} V_i = V_{target} $$

## 3. The Optimization Model

### 3.1 Design Variables ($\mathbf{x}$)
The vector $\mathbf{x}$ changes based on the scenario, but generally consists of:
*   **Radii:** $r_1, r_2, \dots, r_{m-1}$ (Note: $r_0$ and $r_m$ are usually fixed boundary conditions).
*   **Heights:** $h_1, h_2, \dots, h_m$ (In some scenarios, these are fixed equal slices).

### 3.2 Constraints Handling

#### **Equality Constraint (Volume)**
Since we are using PSO/SA (unconstrained naturals) and BFGS, we handle the fixed volume constraint via a **Quadratic Penalty Function**.

**Cost Function to Minimize:**
$$ J(\mathbf{x}) = A_{total}(\mathbf{x}) + \rho (V_{total}(\mathbf{x}) - V_{target})^2 $$
*Tip: Use a dynamic $\rho$ (start small, increase per iteration) to avoid getting stuck early.*

#### **Inequality Constraints (Bounds & Shape)**
1.  **Bounds:** $r_{min} \le r_i \le r_{max}$
2.  **Shape Constraint (Hyperboloid):**
    The reference shape is a Hyperboloid of One Sheet:
    $$ \frac{r^2}{a^2} - \frac{(z-c)^2}{b^2} = 1 \quad \Rightarrow \quad r(z) = a \sqrt{1 + \frac{(z-c)^2}{b^2}} $$
    where $c$ is the height of the waist (narrowest part), $a$ is the waist radius, and $b$ controls the curvature.

## 4. Simulation Scenarios (The 8 Cases)

We will compare the performance of **BFGS**, **PSO**, and **Simulated Annealing** across these 8 equivalent scenarios.

| Case | Name | Description & Variables | Constraints | Difficulty |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Mandatory Ref Case** | **Ref Case**: $V=7.032 \times 10^4$, $m=10$. Fixed Heights ($h_i \approx 3.65$). $r_0=39.3, r_{10}=27.4$. Vary $r_1 \dots r_9$. | Fixed Volume | Medium |
| **2** | **Variable Heights** | Fix Radii ($r_i$ from Case 1 result), Vary Heights $h_1 \dots h_{10}$. | Fixed Volume | Medium |
| **3** | **Full Freedom** | Vary **both** Radii ($r_1 \dots r_9$) and Heights ($h_1 \dots h_{10}$). | Fixed Volume | High |
| **4** | **Tight Waist** | Case 1 + Constraint: $r_{mid} < 0.7 \times \min(r_0, r_{10})$ (Force narrow throat). | Volume + Radius constraint | High |
| **5** | **Cylindrical Start** | Search starting from a cylinder ($r_i = r_0$ for all $i$). | Fixed Volume | Low |
| **6** | **High Volume** | Increase $V_{target}$ by 50%. | Fixed Volume | Medium |
| **7** | **Restricted Bounds** | Case 1 but with strict bounds on $r_i$ (e.g., $r_i \in [20, 60]$). | Volume + Box Constraints | Medium |
| **8** | **Hyperbolic Fit** | Instead of independent $r_i$, optimize parameters $a, b, c$ of the hyperbola equation $r(z)$. | Fixed Volume | High (Parametric) |

## 5. Report Deliverables Checklist

The final report must include:

1.  **Visualizations:**
    *   [ ] Profile plots ($r$ vs $z$) for the optimized towers compared to the initial guess.
    *   [ ] 3D Wireframe plots of the cooling towers.
    *   [ ] Convergence plots (Cost vs Iteration) for BFGS, SA, and PSO.

2.  **Tables:**
    *   [ ] Comparison Table: Final Area, Total Volume Error, Time Taken, Function Evaluations for each Algorithm & Scenario.

3.  **Analysis:**
    *   [ ] Commentary on which algorithm performed best (speed vs accuracy).
    *   [ ] Discussion on discrete approximation (frustums) vs true curved surface.

## 6. Implementation Plan

### **Phase 1: Setup & Primitives (John)**
- [ ] Create `cooling_tower.py` module.
- [ ] Implement `surface_area(radii, heights)` and `volume(radii, heights)`.
- [ ] Implement `cost_function(x, penalty_weight)`.

### **Phase 2: Algorithms (Team)**
- [ ] **Martin:** Adapt BFGS to use `cost_function`. Gradients via finite differences if analytical too complex.
- [ ] **Pierre:** Adapt SA (Simulated Annealing) for continuous variables.
- [ ] **Clémence:** Adapt PSO (Particle Swarm) with boundary handling.

### **Phase 3: Execution (All)**
- [ ] Verify Case 1 (Mandatory) first.
- [ ] Run the remaining 7 cases.
- [ ] Collect metrics: Function Evaluations, Time, Final Surface Area, Volume verification.

### **Phase 4: Reporting (Clémence/John)**
- [ ] Plot Cross-sections of towers.
- [ ] Compare convergence curves.
- [ ] Write Report.
