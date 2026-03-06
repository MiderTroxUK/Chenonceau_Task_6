import re

with open(r'c:\Cranfield\Chenonceau_Task_6\Task6_Latex\Sections\5_Discussion.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace SA max volume error
content = content.replace('volume errors remain below 0.5\\,m$^3$ in all cases', 'volume errors remain below 0.75\\,m$^3$ in all cases')

# Replace SA vs BFGS area ratio for Case 6
content = content.replace('4.7$\\times$ difference', '5.1$\\times$ difference')

# Replace PSO claims
old_pso_1 = r'''PSO, in its current configuration, is the least suitable method for this problem class. The severe volume constraint violations in five out of eight cases (Table~\ref{tab:comparison_all}) indicate a fundamental mismatch between the penalty-based constraint handling and the swarm-based search dynamics. Unlike SA, which evaluates perturbations of a single solution and can therefore be efficiently guided by the penalty gradient, PSO distributes its evaluations across an entire swarm, diluting the penalty signal across many particles. For PSO to become competitive, its constraint handling would need to be redesigned---for example, by implementing a feasibility-preserving velocity update rule or a repair operator that projects infeasible particles onto the constraint surface.'''

new_pso_1 = r'''PSO, in its current configuration, demonstrates excellent volume constraint satisfaction but struggles to find the lowest surface area. The volume constraint is accurately satisfied in all eight cases (Table~\ref{tab:comparison_all}), indicating that the penalty formulation is sufficient for navigating the swarm toward the feasible manifold. However, unlike BFGS, which exploits local curvature information to refine the solution precision, PSO distributes its evaluations across an entire swarm, rendering fine-tuning near the optimal boundary computationally expensive.'''

content = content.replace(old_pso_1, new_pso_1)

old_pso_2 = r'''\item \textbf{PSO is not recommended} for equality-constrained surface optimisation problems unless a problem-specific constraint-handling mechanism is implemented. The penalty-based approach is insufficient for the swarm dynamics.'''

new_pso_2 = r'''\item \textbf{PSO is capable of satisfying constraints reliably}, but its high computational cost and lack of fine local refinement make it less efficient than BFGS for precise area minimization.'''

content = content.replace(old_pso_2, new_pso_2)

content = content.replace('PSO was allocated 29$\\times$ more evaluations than BFGS', 'PSO was allocated 13$\\times$ more evaluations than BFGS')

# Insert BFGS robust safeguard context
bfgs_insert_marker = r'''This rapid convergence is characteristic of well-conditioned problems where the quadratic penalty term does not excessively distort the Hessian eigenspectrum.'''
bfgs_insert_text = bfgs_insert_marker + r''' It should be noted that the implemented BFGS algorithm includes bespoke robustness safeguards, such as \texttt{shape\_guidance\_penalty}, rigorous \texttt{armijo\_backtracking}, and a \texttt{reset\_hessian\_on\_bad\_curvature} toggle, which significantly enhance its stability compared to a vanilla quasi-Newton implementation.'''

content = content.replace(bfgs_insert_marker, bfgs_insert_text)

with open(r'c:\Cranfield\Chenonceau_Task_6\Task6_Latex\Sections\5_Discussion.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated 5_Discussion.tex successfully")
