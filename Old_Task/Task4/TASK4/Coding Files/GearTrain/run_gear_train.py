from __future__ import annotations

import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

from gear_train_bfgs import BFGSOptions, multi_start_bfgs


def wilson_ci_95(p: float, n: int) -> tuple[float, float]:
    """Wilson score interval (approx 95%) for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    z = 1.96
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def run_noise_point(
    sigma: float,
    base_opts: BFGSOptions,
    target_ratio: float,
    lb: np.ndarray,
    ub: np.ndarray,
    R: int,
    n_starts_per_rep: int,
    repair_radius: int,
    noise_avg: int,
    seed_base: int = 10_000,
    sigma_index: int = 0,
) -> dict:
    """Run R repetitions for a given sigma and return aggregate metrics."""
    conv_flags = []
    iters_list = []
    err_list = []
    f_list = []
    g_list = []

    for r in range(R):
        opts_noisy = BFGSOptions(**{
            **base_opts.__dict__,
            "noise_sigma": float(sigma),
            "noise_seed": seed_base + 100 * sigma_index + r,
            "noise_avg": int(noise_avg),
        })

        best_sigma = multi_start_bfgs(
            target_ratio=target_ratio,
            lb=lb,
            ub=ub,
            n_starts=n_starts_per_rep,
            seed=42,
            opts=opts_noisy,
            repair_radius=repair_radius,
        )

        conv_flags.append(bool(best_sigma["converged"]))
        iters_list.append(int(best_sigma["iters"]))
        err_list.append(float(best_sigma["err_int"]))
        f_list.append(float(best_sigma["f_best"]))
        g_list.append(float(best_sigma["history_grad_norm"][-1]))

    p = float(np.mean(conv_flags))
    ci_lo, ci_hi = wilson_ci_95(p, R)

    return {
        "sigma": float(sigma),
        "conv_rate": p,
        "conv_ci_lo": ci_lo,
        "conv_ci_hi": ci_hi,
        "med_iters": float(np.median(iters_list)),
        "med_err_int": float(np.median(err_list)),
        "med_f_best": float(np.median(f_list)),
        "med_grad_norm": float(np.median(g_list)),
    }


def main() -> None:
    lb = np.array([12, 12, 12, 12], dtype=float)
    ub = np.array([60, 60, 60, 60], dtype=float)
    target_ratio = 0.14427932477276006

    plot_dir = "PLOT"
    os.makedirs(plot_dir, exist_ok=True)

    # ------------------------
    # Baseline (noise-free)
    # ------------------------
    opts = BFGSOptions(
        max_iter=300,
        tol_grad=1e-8,
        tol_step=1e-12,
        fd_eps=1e-6,
        c1=1e-4,
        backtrack_beta=0.5,
        min_step=1e-12,
        verbose=False,
        noise_sigma=0.0,
        noise_seed=0,
        noise_avg=1,
    )

    best = multi_start_bfgs(
        target_ratio=target_ratio,
        lb=lb,
        ub=ub,
        n_starts=5000,
        seed=42,
        opts=opts,
        repair_radius=4,
    )

    x_cont = best["x_best"]
    x_int0 = best.get("x_int0", None)
    x_int = best["x_int"]
    f_best = float(best["f_best"])

    print("\n=== TASK 4 – BFGS (continuous relaxation) RESULT ===")
    print(f"Target ratio:           {target_ratio:.15f}")
    print(f"Best continuous x*:     {np.array2string(x_cont, precision=6, floatmode='fixed')}")
    print(f"Continuous objective:   {f_best:.6e}   (squared error)")

    print("\n--- After rounding/repair to integers (selection by integer error) ---")
    if x_int0 is not None:
        print(f"Rounded (pre-repair):   {x_int0.tolist()}")
    print(f"Integer (post-repair):  {x_int.tolist()}")
    print(f"Absolute error (int):   {best['err_int']:.6e}")

    print("\nExtra diagnostics:")
    print(f"Best run start index:   {best['start_index']}")
    print(f"Iterations:             {best['iters']}")
    print(f"Final ||grad||:         {best['history_grad_norm'][-1]:.6e}")
    print(f"Converged flag:         {best['converged']}")

    # Baseline convergence plots (cleaner)
    hist_f = best["history_f"]
    hist_g = best["history_grad_norm"]

    plt.figure()
    plt.plot(hist_f, marker="o", linewidth=1.8)
    plt.xlabel("Iteration")
    plt.ylabel(r"$f(x)=(r(x)-r^*)^2$")
    plt.title("BFGS Convergence (Objective)")
    plt.grid(True, which="both", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "task4_bfgs_convergence_objective.png"), dpi=220)

    plt.figure()
    plt.semilogy(hist_g, marker="o", linewidth=1.8)
    plt.xlabel("Iteration")
    plt.ylabel(r"$\|\nabla f(x)\|$")
    plt.title("BFGS Convergence (Gradient Norm)")
    plt.grid(True, which="both", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "task4_bfgs_convergence_gradnorm.png"), dpi=220)

    # ------------------------
    # Noise robustness experiment (clean & informative)
    # ------------------------
    reliability_threshold = 0.8  # strict

    print("\n[Noise experiment note]")
    print(f"- Noise-free optimum objective: f* ≈ {f_best:.1e}.")
    print("- Additive objective noise: f_noisy(x) = f(x) + sigma * N(0,1).")
    print("- If sigma >> f(x) near the optimum (~1e-20), then finite-difference gradients")
    print("  and Armijo line-search are dominated by noise, so BFGS loses reliability.\n")

    # Phase 1: coarse sweep
    R = 10
    n_starts_per_rep = 300
    repair_radius = 4
    noise_avg = 10  # stable

    coarse_sigmas = [0.0, 1e-15, 3e-15, 1e-14, 3e-14, 1e-13, 3e-13, 1e-12, 1e-11, 1e-10, 1e-8, 1e-6, 1e-4]
    results = []

    print("Coarse sweep:")
    for idx, sigma in enumerate(coarse_sigmas):
        out = run_noise_point(
            sigma=sigma,
            base_opts=opts,
            target_ratio=target_ratio,
            lb=lb,
            ub=ub,
            R=R,
            n_starts_per_rep=n_starts_per_rep,
            repair_radius=repair_radius,
            noise_avg=noise_avg,
            sigma_index=idx,
        )
        results.append(out)
        print(
            f" sigma={sigma:.1e} | rate={out['conv_rate']:.2f} "
            f"(~95% CI [{out['conv_ci_lo']:.2f},{out['conv_ci_hi']:.2f}]) | "
            f"med iters={out['med_iters']:.0f} | med int err={out['med_err_int']:.3e} | "
            f"med f_best={out['med_f_best']:.1e} | med ||g||={out['med_grad_norm']:.1e}"
        )

    # Identify first failure on coarse grid
    failure_sigma = None
    for out in results:
        if out["conv_rate"] < reliability_threshold and out["sigma"] > 0.0:
            failure_sigma = out["sigma"]
            break

    # Phase 2: refine around threshold (if found)
    # We refine in [sigma_prev, sigma_fail] using log-spaced points.
    if failure_sigma is not None:
        # find previous sigma before failure
        prev_sigma = None
        for out in results:
            if out["sigma"] < failure_sigma and out["sigma"] > 0.0:
                prev_sigma = out["sigma"]
        if prev_sigma is not None:
            refine_sigmas = np.geomspace(prev_sigma, failure_sigma, num=7)
            print("\nRefinement sweep around the transition:")
            for j, sigma in enumerate(refine_sigmas):
                out = run_noise_point(
                    sigma=float(sigma),
                    base_opts=opts,
                    target_ratio=target_ratio,
                    lb=lb,
                    ub=ub,
                    R=R,
                    n_starts_per_rep=n_starts_per_rep,
                    repair_radius=repair_radius,
                    noise_avg=noise_avg,
                    sigma_index=10_000 + j,
                )
                results.append(out)
                print(
                    f" sigma={sigma:.2e} | rate={out['conv_rate']:.2f} "
                    f"(~95% CI [{out['conv_ci_lo']:.2f},{out['conv_ci_hi']:.2f}])"
                )

    # Sort results by sigma
    results = sorted(results, key=lambda d: d["sigma"])

    # Recompute failure_sigma as first sigma where conv_rate < threshold
    failure_sigma = None
    for out in results:
        if out["sigma"] > 0.0 and out["conv_rate"] < reliability_threshold:
            failure_sigma = out["sigma"]
            break

    # Console conclusion
    print("\n=== Noise robustness conclusion ===")
    if failure_sigma is None:
        print(
            "BFGS remained reliable over the tested noise range "
            f"(convergence rate never dropped below {reliability_threshold:.1f})."
        )
    else:
        print(
            f"BFGS becomes unreliable at approximately noise sigma >= {failure_sigma:.2e} "
            f"(convergence rate < {reliability_threshold:.1f})."
        )

    # Prepare arrays for plots
    sig = np.array([d["sigma"] for d in results], dtype=float)
    rate = np.array([d["conv_rate"] for d in results], dtype=float)
    rate_lo = np.array([d["conv_ci_lo"] for d in results], dtype=float)
    rate_hi = np.array([d["conv_ci_hi"] for d in results], dtype=float)

    med_iters = np.array([d["med_iters"] for d in results], dtype=float)
    med_err = np.array([d["med_err_int"] for d in results], dtype=float)
    med_f = np.array([d["med_f_best"] for d in results], dtype=float)
    med_g = np.array([d["med_grad_norm"] for d in results], dtype=float)

    # Plot 1: convergence rate with confidence interval
    plt.figure()
    x = sig + 1e-30
    plt.semilogx(x, rate, marker="o", linewidth=1.8, label="Convergence rate")
    plt.fill_between(x, rate_lo, rate_hi, alpha=0.20, label="~95% CI (Wilson)")
    plt.axhline(reliability_threshold, linestyle="--", linewidth=1.5, label=f"Threshold = {reliability_threshold:.1f}")
    if failure_sigma is not None:
        plt.axvline(failure_sigma, linestyle="--", linewidth=1.5, label=f"Estimated failure ≈ {failure_sigma:.1e}")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Noise sigma (additive on objective)")
    plt.ylabel("Convergence rate over R runs")
    plt.title("BFGS Robustness: Convergence Rate vs Objective Noise")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "task4_bfgs_noise_convergence_rate.png"), dpi=220)

    # Plot 2: median integer error
    plt.figure()
    plt.semilogx(x, med_err, marker="o", linewidth=1.8)
    plt.xlabel("Noise sigma (additive on objective)")
    plt.ylabel("Median |ratio_int - target| after repair")
    plt.title("BFGS Robustness: Median Integer Error vs Objective Noise")
    plt.grid(True, which="both", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "task4_bfgs_noise_median_integer_error.png"), dpi=220)

    # Plot 3: median iterations
    plt.figure()
    plt.semilogx(x, med_iters, marker="o", linewidth=1.8)
    plt.xlabel("Noise sigma (additive on objective)")
    plt.ylabel("Median iterations (best run)")
    plt.title("BFGS Robustness: Median Iterations vs Objective Noise")
    plt.grid(True, which="both", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "task4_bfgs_noise_median_iterations.png"), dpi=220)

    # Plot 4 (NEW): median f_best (continuous objective) — tells the real story
    plt.figure()
    plt.semilogx(x, med_f, marker="o", linewidth=1.8)
    plt.xlabel("Noise sigma (additive on objective)")
    plt.ylabel("Median best continuous objective f_best")
    plt.title("BFGS Robustness: Median Best Objective vs Objective Noise")
    plt.grid(True, which="both", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "task4_bfgs_noise_median_fbest.png"), dpi=220)

    # Plot 5 (NEW): median grad norm — also very illustrative
    plt.figure()
    plt.semilogx(x, med_g, marker="o", linewidth=1.8)
    plt.xlabel("Noise sigma (additive on objective)")
    plt.ylabel(r"Median $\|\nabla f\|$ at termination (best run)")
    plt.title("BFGS Robustness: Median Final Gradient Norm vs Objective Noise")
    plt.grid(True, which="both", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "task4_bfgs_noise_median_gradnorm.png"), dpi=220)

    # Export CSV for report
    csv_path = os.path.join(plot_dir, "noise_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma", "conv_rate", "conv_ci_lo", "conv_ci_hi", "med_iters", "med_err_int", "med_f_best", "med_grad_norm"])
        for d in results:
            w.writerow([d["sigma"], d["conv_rate"], d["conv_ci_lo"], d["conv_ci_hi"], d["med_iters"], d["med_err_int"], d["med_f_best"], d["med_grad_norm"]])

if __name__ == "__main__":
    main()