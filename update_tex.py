import re

with open(r'c:\Cranfield\Chenonceau_Task_6\Task6_Latex\Sections\4_Results.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace tab:comparison_all
old_table = r'''\begin{tabular}{l | r r | r r | r r}
\hline
 & \multicolumn{2}{c|}{\textbf{BFGS}} & \multicolumn{2}{c|}{\textbf{SA}} & \multicolumn{2}{c}{\textbf{PSO}} \\
\textbf{Case} & Area & Vol Err & Area & Vol Err & Area & Vol Err \\
\hline
1 --- Reference          & \textbf{7\,769} & 0.00 & 7\,877 & 0.06 & 8\,021 & 36\,579\$^\dagger\$ \\
2 --- Variable Heights   & \textbf{7\,776} & 0.00 & 7\,871 & 0.01 & 11\,458 & 0.03 \\
3 --- Full Freedom       & \textbf{7\,775} & 0.00 & 10\,540 & 0.43 & 9\,909 & 14\,870\$^\dagger\$ \\
4 --- Tight Waist        & \textbf{7\,775} & 0.00 & 7\,779 & 0.03 & 8\,024 & 37\,601\$^\dagger\$ \\
5 --- Cylindrical Start  & \textbf{7\,793} & 0.00 & 8\,054 & 0.18 & 8\,073 & 32\,385\$^\dagger\$ \\
6 --- High Volume        & \textbf{7\,678} & 0.00 & 36\,042 & 0.02 & 39\,394 & 0.13 \\
7 --- Restricted Bounds  & 8\,046 & 0.00 & \textbf{7\,798} & 0.10 & 8\,042 & 26\,378\$^\dagger\$ \\
8 --- Hyperbolic Fit     & 5\,962 & 0.00 & 5\,766 & 0.00 & \textbf{5\,743} & 0.03 \\
\hline
\end{tabular}%'''

new_table = r'''\begin{tabular}{l | r r | r r | r r}
\hline
 & \multicolumn{2}{c|}{\textbf{BFGS}} & \multicolumn{2}{c|}{\textbf{SA}} & \multicolumn{2}{c}{\textbf{PSO}} \\
\textbf{Case} & Area & Vol Err & Area & Vol Err & Area & Vol Err \\
\hline
1 --- Reference          & \textbf{7\,776} & 0.00 & 7\,782 & 0.05 & 8\,054 & 0.00 \\
2 --- Variable Heights   & \textbf{6\,903} & 0.00 & 7\,855 & 0.01 & 7\,879 & 0.18 \\
3 --- Full Freedom       & 13\,300 & 8.96 & \textbf{8\,775} & 0.10 & 11\,467 & 1.09 \\
4 --- Tight Waist        & \textbf{7\,782} & 0.00 & 7\,795 & 0.21 & 8\,055 & 0.10 \\
5 --- Cylindrical Start  & \textbf{7\,790} & 0.00 & 7\,844 & 0.74 & 8\,054 & 0.00 \\
6 --- High Volume        & \textbf{7\,680} & 0.00 & 39\,172 & 0.02 & 15\,401 & 1.13 \\
7 --- Restricted Bounds  & 7\,914 & 0.00 & \textbf{7\,876} & 0.10 & 8\,056 & 0.05 \\
8 --- Hyperbolic Fit     & 5\,962 & 0.00 & 5\,733 & 0.02 & \textbf{5\,696} & 0.00 \\
\hline
\end{tabular}%'''

content = content.replace(old_table.replace('\\$', '$'), new_table)

# Replace table caption for comparison_all
content = content.replace(
    'Cells marked with $\dagger$ denote volume errors exceeding 1\\,000\\,m$^3$, indicating effective constraint failure.',
    'All algorithms effectively satisfied the volume constraints (except BFGS on Case 3).'
)

# Text replacements
content = content.replace('area 26\\% larger than SA', 'area 51\\% larger than SA')
content = content.replace('volume error remains below 0.5\\,m$^3$', 'volume error remains below 0.75\\,m$^3$')
content = content.replace('4.7$\\times$ larger', '5.1$\\times$ larger')

old_pso_text = r'''\textbf{PSO exhibits severe constraint failures in five out of eight cases.} Cases~1, 3, 4, 5, and~7 show volume errors exceeding 14\,000\,m$^3$---more than 20\% of the target volume. This indicates that the penalty weight $\rho = 10^5$, while sufficient for BFGS and SA, is inadequate for guiding the PSO swarm toward the feasible region within the allocated 150\,200 function evaluations. Only Cases~2, 6, and~8 achieve feasible solutions, suggesting that PSO requires either a substantially higher penalty weight, a problem-specific constraint-handling mechanism, or significantly more iterations to reliably satisfy the equality constraint.'''

new_pso_text = r'''\textbf{PSO exhibits excellent constraint satisfaction.} In all cases, volume errors remain below 1.2\,m$^3$, confirming that the penalty formulation is sufficient to guide the swarm toward the feasible manifold. However, it requires significantly more function evaluations and generally struggles to match BFGS in final surface area minimization. This is likely because standard PSO lacks a local refinement mechanism once navigating near the precise constraint boundary.'''

content = content.replace(old_pso_text, new_pso_text)

# Replace tab:perf_comparison
old_perf = r'''\begin{tabular}{l r r r}
\hline
\textbf{Metric} & \textbf{BFGS} & \textbf{SA} & \textbf{PSO} \\
\hline
Mean time per case (s) & 0.39 & 3.56 & 6.04 \\
Median time per case (s) & 0.23 & 3.49 & 6.30 \\
Mean function evaluations & 5\,256 & 16\,540 & 150\,200 \\
Fastest case (s) & 0.01 (Case 8) & 1.61 (Case 2) & 4.11 (Case 6) \\
Slowest case (s) & 1.47 (Case 5) & 6.16 (Case 3) & 7.34 (Case 3) \\
\hline
\end{tabular}'''

new_perf = r'''\begin{tabular}{l r r r}
\hline
\textbf{Metric} & \textbf{BFGS} & \textbf{SA} & \textbf{PSO} \\
\hline
Mean time per case (s) & 1.19 & 1.16 & 33.81 \\
Median time per case (s) & 0.49 & 1.33 & 21.94 \\
Mean function evaluations & 9\,709 & 16\,540 & 133\,800 \\
Fastest case (s) & 0.02 (Case 8) & 0.51 (Cases 2,6) & 2.17 (Case 6) \\
Slowest case (s) & 6.23 (Case 3) & 1.79 (Case 3) & 101.35 (Case 4) \\
\hline
\end{tabular}'''

content = content.replace(old_perf, new_perf)

old_perf_text_1 = r'''\textbf{BFGS is the fastest method by a wide margin}, averaging 0.39\,s per case compared to 3.56\,s for SA and 6.04\,s for PSO. This advantage stems from the superlinear convergence rate of quasi-Newton methods: BFGS requires on average only 5\,256 function evaluations to reach the optimum, 3$\times$ fewer than SA and 29$\times$ fewer than PSO. In Case~8 (Hyperbolic Fit), BFGS converges in only 294 evaluations (0.01\,s), demonstrating the exceptional efficiency of gradient-based methods on smooth, low-dimensional landscapes.'''

new_perf_text_1 = r'''\textbf{BFGS is highly efficient}, with a median time of 0.49\,s per case compared to 1.33\,s for SA and 21.94\,s for PSO. This advantage stems from the superlinear convergence rate of quasi-Newton methods: BFGS requires on average only 9\,709 function evaluations to reach the optimum, slightly fewer than SA and 13$\times$ fewer than PSO. In Case~8 (Hyperbolic Fit), BFGS converges in only 294 evaluations (0.02\,s), demonstrating the exceptional efficiency of gradient-based methods on smooth, low-dimensional landscapes.'''

content = content.replace(old_perf_text_1, new_perf_text_1)

old_perf_text_2 = r'''\textbf{SA occupies a middle ground}, with moderate computation times (1.6--6.2\,s) and a fixed evaluation budget of 16\,540 per case.'''

new_perf_text_2 = r'''\textbf{SA occupies a middle ground}, with moderate computation times (0.5--1.8\,s) and a fixed evaluation budget of 16\,540 per case.'''

content = content.replace(old_perf_text_2, new_perf_text_2)


old_perf_text_3 = r'''\textbf{PSO is the most expensive method}, requiring 150\,200 function evaluations per case (a fixed swarm size of 200 particles over 750 iterations). Despite this computational investment, PSO fails to produce feasible solutions in the majority of cases, yielding the worst cost-effectiveness ratio of the three methods.'''
new_perf_text_3 = r'''\textbf{PSO is the most expensive method}, requiring on average 133\,800 function evaluations per case due to dynamic termination criteria handling swarm stagnation. Despite this substantial computational investment, PSO struggles to produce the lowest surface areas among the methods.'''

content = content.replace(old_perf_text_3, new_perf_text_3)

with open(r'c:\Cranfield\Chenonceau_Task_6\Task6_Latex\Sections\4_Results.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated 4_Results.tex successfully.")
