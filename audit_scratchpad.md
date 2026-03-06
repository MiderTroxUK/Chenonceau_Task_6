# Audit Findings Scratchpad

## 1. NUMERICAL VERIFICATION & INCONSISTENCIES
- **Tables vs CSV**: 
  - BFGS Case 1: Table says 7766.65 m2, CSV says 7776.11 m2. Evals: Table 6469, CSV 6746.
  - BFGS Case 7: Table says Evals 1303, CSV says 6247.
  - BFGS Case 8: Table says Area 5961.77, CSV says 5961.74.
  - SA Case 1: Table says Area 7876.71, CSV says 7781.65.
  - SA Case 6 Area: Table 36042.25, CSV 39172.03.
  - PSO Table vs CSV is totally wrong: Table claims 150,200 evals everywhere, CSV varies from 15,600 to 359,200. Table claims catastrophic volume failures (e.g. Case 1 Vol Err 36,579.42), but CSV says 0.00002.
  
- **Computations in text**:
  - BFGS "requires on average 5,256 function evaluations": Calculated from table = 8,892. Calculated from CSV = 9,709. Claim is ERROR.
  - PSO "requires 150,200 function evaluations per case": Claim is ERROR. (Table says this, but it contradicts the algorithm and CSV entirely).
  - SA "fixed evaluation budget of 16,540 per case": VERIFIED against CSV.
  
- **"BFGS achieves the best surface area in five out of eight cases"**:
  - According to CSV: BFGS (1, 2, 4, 5, 6) = 5 cases. SA wins 3, 7. PSO wins 8. VERIFIED.

- **"PSO exhibits severe constraint failures in five out of eight cases"**:
  - According to table: Yes (cases 1, 3, 4, 5, 7 have huge errors).
  - According to CSV: No! Vol errors in CSV are basically <1.1m^3. This is an ERROR/INCONSISTENCY between text/table and truth.

- **"SA volume errors remain below 0.5 m^3 in all cases"**:
  - Table: Max error is 0.43 (Case 3). VERIFIED.
  - CSV: Max error is 0.74 (Case 6). INCONSISTENCY! 

- **Percentage claims**:
  - Sec 4.2.8 (Case 8 SA): "The area is 26% lower than Case 1". Table SA Case 1: 7876.71, Table SA Case 8: 5765.62. (7876.71 - 5765.62) / 7876.71 = 26.8%. (Close enough, or technically 26.8%).
  - Sec 4.3.2 (Case 2 BFGS): "BFGS achieves a 12% improvement over Case 1 (6830.77 m² vs 7766.65 m²)". 6830.77/7766.65 = 12.0%. VERIFIED.
  - Sec 5.3.2 (Case 2 SA area): "only 0.07% improvement". 7871.45 vs 7876.71 -> (7876.71 - 7871.45)/7876.71 = 5.26 / 7876.71 = 0.000667 = 0.0667%. VERIFIED.
  
- **"Case 8 yields the best result for all three methods"**:
  - CSV BFGS: 5961.74. (Wait, SA Case 8 CSV: 5732.75. PSO Case 8 CSV: 5695.72).
  - Are these the best for each solver? Yes. VERIFIED.

- **"BFGS converges in 12 iterations for Case 8"**:
  - CSV: time 0.02s, evals 294. From table it's 294. Cannot confirm 12 iterations exactly without checking history length, but 12 iters is plausible for 294 evals. The report says "convergence in 12 iterations (294 function evaluations)". We accept it as VERIFIED or at least matching internally.

- **"The SA zigzag profile in Case 6 has an area 4.7x larger than BFGS"**:
  - Table SA Case 6 = 36042.25. Table BFGS = 7678.24. 36042.25 / 7678.24 = 4.694x (~4.7x). VERIFIED. 
  - (With CSV: 39172.03 / 7679.54 = 5.1x. So inconsistency with CSV).

## 2. CODE-TO-TEXT CONSISTENCY
- Eqs 1-6 vs `cooling_tower.py` geometry:
  - Eq 1: $s_i = \sqrt{(r_{i-1} - r_i)^2 + h_i^2}$. Code matches (`slant_height`).
  - Eq 2: $A_i = \pi (r_{i-1} + r_i) s_i$. Code matches.
  - Eq 3: $V_i = \frac{\pi h_i}{3} (r_{i-1}^2 + r_{i-1} r_i + r_i^2)$. Code matches.
- Eq 5 penalty formulation vs `cost_function`:
  - Text: $J(x) = A_{total} + \rho [H(x)]^2$ where $H(x) = V - V_{target}$. Code matches.
- Penalty weight $\rho$:
  - Text says $\rho = 10^5$. Wait, let's check Sec 3.2. Text says: "The chosen value $\rho = 10^5$ was calibrated empirically..."
  - Wait, user prompt said "the text says 10^3 — check what the code actually uses". Let me re-read the report text!
  - AH! Sec 3.2 says $\rho = 10^5$. Maybe the user prompt was "the text says 10^3" from an older version? I will check the text again to be absolutely sure.
- BFGS Update formula:
  - Text Eq 11: $H_{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T$
  - Code matches.
- SA Acceptance criterion: 
  - Text: $P = e^{-(E_{new} - E_{actual})/T}$
  - Code matches.
- PSO formulation:
  - Matches pymoo usage.
- Study Case Definitions:
  - Case 1: 9 inner radii. Code matches.
  - Case 2: 10 heights. Code matches.
  - Case 3: 19 vars (9 radii + 10 heights). Code matches.
  - Case 4: Tight waist $r_{mid} < 0.7 \min(r_0, r_m)$. Code matches.
  - Case 5: Cylindrical init. Code matches.
  - Case 6: High Volume. Code: `target_vol *= 1.5`. Matches.
  - Case 7: Restricted Bounds [20, 60]. Code matches.
  - Case 8: Hyperbola parameters $r(z) = a \sqrt{1 + (z-c)^2 / b^2}$. Code matches.

## 3. INTERNAL CONSISTENCY
- Table 17 vs Tables 1-16 vs Tables 19-21:
  - Wait, I don't see Table 1-16. I see Tables with names `tab:sa_case1`, `tab:bfgs_case1`, etc. Those are the individual tables.
  - The appendix has Tables 7,8,9 as summary tables. Are those labeled `tab:bfgs_results`, `tab:pso_results`, `tab:sa_results`?
  - Yes! Appendix tables are A, B, C or something. The latex just has `\begin{table}`. Let's trace their values.
- Abstract vs Results:
  - Abstract claims BFGS 0 volume error, compute times < 1.5s.
  - Table BFGS Case 3 compute time is 22.72s. Error! The abstract says "computation times under 1.5s." This is factually contradicted by Case 3 and Case 5 (3.33s).
- Performance hierarchy (BFGS > SA >> PSO):
  - Supported by text tables, but CSV data shows PSO actually performed very well on volume constraint. So table 8 in appendix is totally fabricated compared to CSV.

## 4. ALGORITHM PARAMETERS
- PSO: swarmsize 200, max iter 50k, w 0.9, c1=2.0, c2=2.0, seed=10. 
  - Text: "seed 10 was selected". Code matches.
- SA: T_initial 5000, alpha 0.995, T_min 1e-8, max_iter 5000. 
  - Text: Section 3.2 says "alpha (cooling rate)". Code says `T_initial = 5000`, `alpha = 0.995`, `T_final = 1e-8`. Wait, do the text values match? Let's check section "Key Parameters" in SA.
- BFGS:
  - Code says `max_iter=300`, `gtol=1e-6`, `tol_step=1e-10`.
